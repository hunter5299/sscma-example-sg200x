#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <sscma.h>
#include <video.h>

#include <sesg/stream_udp.h>

using namespace ma;

#define TAG "jpeg_udp"

static std::atomic<bool> g_running(true);
static void sig_handler(int) { g_running.store(false); }

struct Args {
    // UDP 相关配置：默认启用，并使用默认目标
    bool enable_udp = true;
    std::string udp_ip = "192.168.2.101";
    int udp_port = 5001;

    // JPEG 通道配置（由 streamer 拉取 MA_PIXEL_FORMAT_JPEG）
    int jpeg_ch = 1;
    int jpeg_w = 640;
    int jpeg_h = 480;
    int jpeg_fps = 1;

    // 发送节流：0 表示全速；>0 表示间隔秒数
    double send_interval_s = 0.0;
};

static void print_usage(const char* argv0) {
    std::printf(
    "Usage: %s [interval_s] | [udp_ip udp_port] | [udp_ip udp_port interval_s]\n\n"
        "Examples:\n"
        "  %s                    # 默认发送到UDP（192.168.2.101:5001）\n"
    "  %s 1                  # 1秒一张（默认目标）\n"
    "  %s 0.1                # 0.1秒一张（默认目标）\n"
    "  %s 192.168.2.101 5001 # 覆盖默认目标\n"
    "  %s 192.168.2.101 5001 2 # 2秒一张（自定义目标）\n",
        argv0,
        argv0,
        argv0);
}

static bool parse_int(const char* s, int* out) {
    if (!s || !out) return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = static_cast<int>(v);
    return true;
}

static bool parse_double(const char* s, double* out) {
    if (!s || !out) return false;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end != '\0') return false;
    *out = v;
    return true;
}

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

    // 无参数：默认启用UDP，使用预设 IP/端口
    if (positional.empty()) {
        a->enable_udp = true;
        return true;
    }

    // ./jpeg_udp [interval_s]
    if (positional.size() == 1) {
        if (!parse_double(positional[0].c_str(), &a->send_interval_s)) {
            std::fprintf(stderr, "[错误] interval_s 必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
        return true;
    }

    // ./jpeg_udp [ip] [port]
    if (positional.size() == 2) {
        a->enable_udp = true;
        a->udp_ip = positional[0];
        if (!parse_int(positional[1].c_str(), &a->udp_port)) {
            std::fprintf(stderr, "[错误] 端口号必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
        return true;
    }

    // ./jpeg_udp [ip] [port] [interval_s]
    if (positional.size() == 3) {
        a->enable_udp = true;
        a->udp_ip = positional[0];
        if (!parse_int(positional[1].c_str(), &a->udp_port)) {
            std::fprintf(stderr, "[错误] 端口号必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
        if (!parse_double(positional[2].c_str(), &a->send_interval_s)) {
            std::fprintf(stderr, "[错误] interval_s 必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
        return true;
    }

    std::fprintf(stderr, "[错误] 参数数量不正确\n");
    print_usage(argv[0]);
    return false;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return (argc > 1) ? 1 : 0;
    }

    Signal::install({SIGINT, SIGTERM, SIGQUIT}, [](int) { sig_handler(0); });

    std::printf("[%s] start | UDP=%s -> %s:%d | JPEG(ch=%d)=%dx%d@%dfps | interval=%.3fs\n",
                TAG,
                args.enable_udp ? "ON" : "OFF",
                args.udp_ip.c_str(),
                args.udp_port,
                args.jpeg_ch,
                args.jpeg_w,
                args.jpeg_h,
                args.jpeg_fps,
                args.send_interval_s);

    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);

            // 让上层拿到 CPU 可直接访问的虚拟地址
            Camera::CtrlValue v;
            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "camera not found");
        return 1;
    }

    // 配置 JPEG 通道（必须在 startStream() 前）
    sesg::stream_udp::JpegChannelConfig jpeg_cfg;
    jpeg_cfg.ch = args.jpeg_ch;
    jpeg_cfg.width = args.jpeg_w;
    jpeg_cfg.height = args.jpeg_h;
    jpeg_cfg.fps = args.jpeg_fps;
    sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

    // 某些实现里 Ctrl 可能与“当前通道”相关，这里再设置一次，确保 JPEG 通道返回虚拟地址。
    {
        Camera::CtrlValue v;
        v.i32 = 0;
        camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
    }

    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // UDP 推流器（拉取 JPEG 帧到内部缓冲，主循环 sendLatestRaw 发送出去）
    auto camera_shared = std::shared_ptr<Camera>(camera, [](Camera*) {});
    sesg::stream_udp::SesgJpegUdpStreamer udp_streamer(args.udp_ip, (uint16_t)args.udp_port);

    if (args.enable_udp) {
        if (!udp_streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
            MA_LOGE(TAG, "UDP streamer start failed");
            camera->stopStream();
            camera->deInit();
            return 1;
        }
        std::printf("[%s] UDP streamer started\n", TAG);
    }

    // 发送格式仍使用 udp_service 的统一头 + JPEG payload（仅 det_count=0）
    constexpr uint32_t MAGIC_JPEG = 0x4A504547;  // "JPEG"

    auto stat_start = std::chrono::steady_clock::now();
    auto next_send_time = stat_start;

    while (g_running.load()) {
        if (args.enable_udp) {
            if (args.send_interval_s > 0.0) {
                const auto now = std::chrono::steady_clock::now();
                if (now < next_send_time) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            }

            // 无检测数据：det_buf=null, det_bytes=0, det_count=0
            const double ret = udp_streamer.sendLatestRaw(MAGIC_JPEG, nullptr, 0, 0);
            if (ret < 0.0) {
                // 还没取到新的 JPEG，稍微让出 CPU
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else if (args.send_interval_s > 0.0) {
                next_send_time = std::chrono::steady_clock::now() +
                                 std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                     std::chrono::duration<double>(args.send_interval_s));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - stat_start).count();
        if (elapsed_ms >= 1000) {
            std::printf("[%s] send_fps=%.2f\n", TAG, (double)udp_streamer.getSendFPS());
            stat_start = now;
        }
    }

    if (args.enable_udp) {
        udp_streamer.stop(/*deinit_vpss=*/false);
    }
    camera->stopStream();
    camera->deInit();

    std::printf("[%s] exit\n", TAG);
    return 0;
}
