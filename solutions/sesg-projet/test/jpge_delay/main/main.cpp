#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// 说明：此文件用于发送一帧 JPEG 到主机并等待主机按原格式回传，测量往返时延。
// 设备端流程：初始化相机 -> 配置 JPEG 通道 -> 发送一帧（使用 SesgJpegUdpStreamer）
// -> 开始计时 -> 等待通过 UDP 回传的完整 JPEG -> 停止计时并保存。

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sscma.h>
#include <video.h>

#include <sesg/stream_udp.h>

using namespace ma;

#define TAG "jpge_delay"

// 全局运行标志，支持信号中断（Ctrl+C）
static std::atomic<bool> g_running(true);
// 信号处理：把运行标志置为 false，主循环与阻塞调用会检测并退出
static void sig_handler(int) { g_running.store(false); }

// 命令行参数结构体：控制目标 IP/端口与本地接收端口以及 JPEG 通道配置
struct Args {
    std::string udp_ip = "192.168.2.101"; // 目标主机（回传脚本所在主机）
    int udp_port = 5001;   // 目标主机接收端口（主机监听端口）
    int recv_port = 5002;  // 本地接收回传数据的端口

    // JPEG 通道参数（用于 configureJpegChannel）
    int jpeg_ch = 1;
    int jpeg_w = 640;
    int jpeg_h = 480;
    int jpeg_fps = 1;
};

// 打印使用说明
static void print_usage(const char* argv0) {
    std::printf(
    "Usage: %s [udp_ip udp_port recv_port]\n\n"
        "Examples:\n"
        "  %s                        # 默认 192.168.2.101:5001, 本地接收 5002\n"
    "  %s 192.168.2.101 5001 5002 # 自定义地址/端口\n",
        argv0,
        argv0,
        argv0);
}

// 简单的整数解析器（用于解析命令行端口等）
static bool parse_int(const char* s, int* out) {
    if (!s || !out) return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = static_cast<int>(v);
    return true;
}

// 解析命令行参数，支持可选的目标 IP/端口和本地回传端口
static bool parse_args(int argc, char** argv, Args* a) {
    if (!a) return false;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        }
        positional.emplace_back(arg);
    }

    // 无位置参数：使用默认值
    if (positional.empty()) return true;

    // 三个位置参数：udp_ip udp_port recv_port
    if (positional.size() == 3) {
        a->udp_ip = positional[0];
        if (!parse_int(positional[1].c_str(), &a->udp_port)) {
            std::fprintf(stderr, "[错误] udp_port 必须是数字\n");
            return false;
        }
        if (!parse_int(positional[2].c_str(), &a->recv_port)) {
            std::fprintf(stderr, "[错误] recv_port 必须是数字\n");
            return false;
        }
        return true;
    }

    std::fprintf(stderr, "[错误] 参数数量不正确\n");
    print_usage(argv[0]);
    return false;
}

// UDP 帧头结构（与 udp_service 的统一格式一致，全部小端 32-bit）
struct UdpHeader {
    uint32_t magic;
    uint32_t width;
    uint32_t height;
    uint32_t payload_size;
    uint32_t det_count;
    uint32_t fmt;
    uint32_t fid;
};

// 用于分片重组的大帧缓存结构（按 fid 聚合多个 UDP 包）
struct Reassembly {
    bool active = false;
    uint32_t fid = 0;
    uint32_t total = 0;
    size_t received = 0;
    std::vector<uint8_t> buf;

    void start(uint32_t new_fid, uint32_t total_size) {
        active = true;
        fid = new_fid;
        total = total_size;
        received = 0;
        buf.clear();
        buf.reserve(total_size);
    }

    void add(const uint8_t* data, size_t len) {
        if (!active || received >= total) return;
        const size_t remain = total - received;
        const size_t copy_len = (len < remain) ? len : remain;
        buf.insert(buf.end(), data, data + copy_len);
        received += copy_len;
    }

    bool complete() const { return active && received >= total; }
};

// 从给定的 UDP socket 接收完整的一帧 JPEG（可能由多个 UDP 包分片组成）
// 返回 true 并通过 out_hdr/out_jpeg 填充完整帧，或返回 false（例如被中断）
static bool recv_one_frame(int sockfd, UdpHeader* out_hdr, std::vector<uint8_t>* out_jpeg) {
    if (!out_hdr || !out_jpeg) return false;

    constexpr size_t kMaxPkt = 65536;
    std::vector<uint8_t> pkt(kMaxPkt);
    Reassembly reasm;

    // 循环接收，直到收到完整帧或全局退出标志被置位
    while (g_running.load()) {
        sockaddr_in src_addr{};
        socklen_t addr_len = sizeof(src_addr);
        const ssize_t n = ::recvfrom(sockfd, pkt.data(), pkt.size(), 0,
                                     reinterpret_cast<sockaddr*>(&src_addr), &addr_len);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
                continue;
            }
            continue;
        }
        if (static_cast<size_t>(n) < sizeof(UdpHeader)) {
            continue;
        }

        UdpHeader hdr;
        std::memcpy(&hdr, pkt.data(), sizeof(UdpHeader));

        // 只处理 fmt==1 (JPEG) 且 payload_size 非 0 的帧
        if (hdr.payload_size == 0 || hdr.fmt != 1) {
            continue;
        }

        const uint8_t* payload = pkt.data() + sizeof(UdpHeader);
        const size_t payload_len = static_cast<size_t>(n) - sizeof(UdpHeader);

        // 新的 fid 则重置重组状态；否则追加数据并检测是否完成
        if (!reasm.active || hdr.fid != reasm.fid) {
            reasm.start(hdr.fid, hdr.payload_size);
        }

        reasm.add(payload, payload_len);
        if (reasm.complete()) {
            *out_hdr = hdr;
            *out_jpeg = std::move(reasm.buf);
            return true;
        }
    }

    return false;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return (argc > 1) ? 1 : 0;
    }

    Signal::install({SIGINT, SIGTERM, SIGQUIT}, [](int) { sig_handler(0); });

    std::printf("[%s] start | dst=%s:%d | recv_port=%d\n",
                TAG,
                args.udp_ip.c_str(),
                args.udp_port,
                args.recv_port);

    // 准备接收 socket（用于回传）
    const int recv_sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (recv_sock < 0) {
        MA_LOGE(TAG, "socket create failed");
        return 1;
    }

    sockaddr_in local_addr{};
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = INADDR_ANY;
    local_addr.sin_port = htons(static_cast<uint16_t>(args.recv_port));

    if (::bind(recv_sock, reinterpret_cast<sockaddr*>(&local_addr), sizeof(local_addr)) < 0) {
        MA_LOGE(TAG, "bind recv_port failed");
        ::close(recv_sock);
        return 1;
    }

    // 设置接收超时，便于 Ctrl+C 时及时退出
    timeval tv{};
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    ::setsockopt(recv_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);

            Camera::CtrlValue v;
            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "camera not found");
        ::close(recv_sock);
        return 1;
    }

    // 配置 JPEG 通道
    sesg::stream_udp::JpegChannelConfig jpeg_cfg;
    jpeg_cfg.ch = args.jpeg_ch;
    jpeg_cfg.width = args.jpeg_w;
    jpeg_cfg.height = args.jpeg_h;
    jpeg_cfg.fps = args.jpeg_fps;
    sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

    {
        Camera::CtrlValue v;
        v.i32 = 0;
        camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
    }

    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    auto camera_shared = std::shared_ptr<Camera>(camera, [](Camera*) {});
    sesg::stream_udp::SesgJpegUdpStreamer udp_streamer(args.udp_ip, (uint16_t)args.udp_port);

    if (!udp_streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
        MA_LOGE(TAG, "UDP streamer start failed");
        camera->stopStream();
        camera->deInit();
        ::close(recv_sock);
        return 1;
    }

    constexpr uint32_t MAGIC_JPEG = 0x4A504547;  // "JPEG"

    // 发送一张 JPEG
    bool sent = false;
    while (g_running.load()) {
        const double ret = udp_streamer.sendLatestRaw(MAGIC_JPEG, nullptr, 0, 0);
        if (ret >= 0.0) {
            sent = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!sent) {
        MA_LOGE(TAG, "send failed");
        udp_streamer.stop(/*deinit_vpss=*/false);
        camera->stopStream();
        camera->deInit();
        ::close(recv_sock);
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // 等待回传一张完整 JPEG
    UdpHeader hdr{};
    std::vector<uint8_t> jpeg;
    const bool got = recv_one_frame(recv_sock, &hdr, &jpeg);
    const auto t1 = std::chrono::steady_clock::now();

    udp_streamer.stop(/*deinit_vpss=*/false);
    camera->stopStream();
    camera->deInit();
    ::close(recv_sock);

    if (!got) {
        MA_LOGE(TAG, "recv failed");
        return 1;
    }

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("[%s] round_trip_ms=%lld\n", TAG, static_cast<long long>(elapsed_ms));

    const std::string out_path = "echo_frame_" + std::to_string(hdr.fid) + ".jpg";
    FILE* fp = std::fopen(out_path.c_str(), "wb");
    if (fp) {
        std::fwrite(jpeg.data(), 1, jpeg.size(), fp);
        std::fclose(fp);
        std::printf("[%s] saved %s (%zu bytes)\n", TAG, out_path.c_str(), jpeg.size());
    } else {
        MA_LOGE(TAG, "save failed");
    }

    std::printf("[%s] exit\n", TAG);
    return 0;
}
