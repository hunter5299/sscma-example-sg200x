#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <sscma.h>
#include <video.h>

#include <sesg/stream_udp.h>

using namespace ma;

#define TAG "groove_features"

static std::atomic<bool> g_running(true);
static void sig_handler(int) { g_running.store(false); }

// 与 Python 版本保持一致的 ROI / 参数
static constexpr int ROI_X = 425;
static constexpr int ROI_Y = 200;
static constexpr int ROI_X2 = 520;
static constexpr int ROI_Y2 = 300;

// 固定阈值与核长度（与现场算法保持一致）
static constexpr int FIXED_THRESHOLD = 80;
static constexpr int FIXED_KERNEL_LEN = 5;

// 运行参数配置
struct Args {
    bool enable_udp = true;
    std::string udp_ip = "192.168.2.101";
    int udp_port = 5001;

    int cam_w = 640;
    int cam_h = 480;

    int jpeg_ch = 1;
    int jpeg_w = 640;
    int jpeg_h = 480;
    int jpeg_fps = 10;
};

// 打印命令行使用说明
static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [udp_ip udp_port]\n\n"
        "Examples:\n"
    "  %s                      # 默认发送到UDP（192.168.2.101:5001）\n"
    "  %s 192.168.2.101 5001   # 发送到UDP（覆盖默认目标）\n",
        argv0,
        argv0,
        argv0);
}

// 解析整数字符串
static bool parse_int(const char* s, int* out) {
    if (!s || !out) return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = static_cast<int>(v);
    return true;
}

// 解析命令行参数
static bool parse_args(int argc, char** argv, Args* a) {
    if (!a) return false;

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return false;
        }
    }

    if (argc == 3) {
        a->enable_udp = true;
        a->udp_ip = argv[1];
        if (!parse_int(argv[2], &a->udp_port)) {
            std::fprintf(stderr, "[错误] 端口号必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
        return true;
    }

    if (argc == 1) {
        // 无参数：默认启用UDP，使用Args里预设的IP/端口
        a->enable_udp = true;
        return true;
    }

    std::fprintf(stderr, "[错误] 参数数量不正确\n");
    print_usage(argv[0]);
    return false;
}

// UDP发送的检测结果结构
struct UDPGrooveDet {
    // bbox: center-x/center-y/width/height
    // 坐标建议发送归一化(0..1)，便于接收端按 JPEG 分辨率绘制
    float bbox_x;
    float bbox_y;
    float bbox_w;
    float bbox_h;
    float score;
    int32_t id;
};

// 主循环：采集、处理、发送
static int run_video_loop(const Args& args) {
    // 安装信号处理，便于安全退出
    Signal::install({SIGINT, SIGTERM, SIGQUIT}, [](int) { sig_handler(0); });

    std::printf(
        "[%s] start (ROI=(%d,%d)-(%d,%d), thr=%d, klen=%d) | UDP=%s -> %s:%d\n",
        TAG,
        ROI_X,
        ROI_Y,
        ROI_X2,
        ROI_Y2,
        FIXED_THRESHOLD,
        FIXED_KERNEL_LEN,
        args.enable_udp ? "ON" : "OFF",
        args.udp_ip.c_str(),
        args.udp_port);

    // 获取设备与摄像头对象
    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    // 遍历并初始化摄像头
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);

            Camera::CtrlValue v;
            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, v);

            // 默认 640x480（与 Python ROI 假设一致）
            v.u16s[0] = static_cast<uint16_t>(args.cam_w);
            v.u16s[1] = static_cast<uint16_t>(args.cam_h);
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, v);

            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
            break;
        }
    }

    // 摄像头未找到则直接退出
    if (!camera) {
        MA_LOGE(TAG, "camera not found");
        return 1;
    }

    // 配置 JPEG 通道用于 UDP 推流（必须在 startStream() 之前调用）
    sesg::stream_udp::JpegChannelConfig jpeg_cfg;
    jpeg_cfg.ch = args.jpeg_ch;
    jpeg_cfg.width = args.jpeg_w;
    jpeg_cfg.height = args.jpeg_h;
    jpeg_cfg.fps = args.jpeg_fps;
    sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

    // 启动摄像头流
    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // 构造共享指针以交给UDP推流对象
    auto camera_shared = std::shared_ptr<Camera>(camera, [](Camera*) {});
    sesg::stream_udp::SesgJpegUdpStreamer udp_streamer(args.udp_ip, (uint16_t)args.udp_port);
    // 可选启动UDP推流
    if (args.enable_udp) {
        if (!udp_streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
            MA_LOGE(TAG, "UDP streamer start failed");
            camera->stopStream();
            camera->deInit();
            return 1;
        }
        std::printf("[%s] UDP streamer started (JPEG ch=%d %dx%d@%dfps)\n",
                    TAG,
                    jpeg_cfg.ch,
                    jpeg_cfg.width,
                    jpeg_cfg.height,
                    jpeg_cfg.fps);
    }

    // UDP数据包魔数
    constexpr uint32_t MAGIC_GROV = 0x47524F56; // "GROV"

    // 复制帧到安全缓冲区，避免帧释放后被复用
    std::vector<uint8_t> safe_frame;

    // 主循环：拉流、处理、发送
    while (g_running.load()) {
        // 取一帧图像
        ma_img_t frame;
        // 拉取RGB888帧，失败则稍作等待
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 校验帧数据有效性
        if (!frame.data || frame.size == 0 || frame.width == 0 || frame.height == 0) {
            camera->returnFrame(frame);
            continue;
        }

        // 拷贝到安全缓冲区
        if (safe_frame.size() < frame.size) safe_frame.resize(frame.size);
        std::memcpy(safe_frame.data(), frame.data, frame.size);
        camera->returnFrame(frame);

        // 将原始数据包装成OpenCV Mat（RGB888）
        ::cv::Mat rgb((int)frame.height, (int)frame.width, CV_8UC3, safe_frame.data());

        const auto t0 = std::chrono::steady_clock::now();

        // 1) ROI 裁剪（边界保护）
        const int x1 = std::max(0, std::min((int)frame.width, ROI_X));
        const int y1 = std::max(0, std::min((int)frame.height, ROI_Y));
        const int x2 = std::max(0, std::min((int)frame.width, ROI_X2));
        const int y2 = std::max(0, std::min((int)frame.height, ROI_Y2));

        // 准备结果缓存
        std::vector<UDPGrooveDet> udp_dets;
        int groove_count = 0;
        if (x2 > x1 && y2 > y1) {
            // 提取ROI区域
            ::cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
            ::cv::Mat roi_rgb = rgb(roi);

            // 2) 预处理：转灰度
            ::cv::Mat gray;
            ::cv::cvtColor(roi_rgb, gray, ::cv::COLOR_RGB2GRAY);

            // 3) 二值化（使用固定阈值）
            ::cv::Mat binary;
            ::cv::threshold(gray, binary, FIXED_THRESHOLD, 255, ::cv::THRESH_BINARY);
            //::cv::threshold(gray, binary, FIXED_THRESHOLD, 255, ::cv::THRESH_BINARY_INV);  
            // 4) 形态学处理（使用固定核大小），保持黑色横线为检测目标
            //    这里使用“横向”结构元素做开运算，用于去除竖向噪声并保留横向结构
            const int k_len = (FIXED_KERNEL_LEN < 1) ? 1 : FIXED_KERNEL_LEN;
            ::cv::Mat h_kernel = ::cv::getStructuringElement(::cv::MORPH_RECT, ::cv::Size(k_len, 1));

            ::cv::Mat morph;
            ::cv::morphologyEx(binary, morph, ::cv::MORPH_OPEN, h_kernel);
            //    如果横条中间有轻微断裂，使用横向闭运算把小缝隙连接起来，避免被分成两段
            const int gap_len = std::max(3, k_len); // 间隙连接核长度，避免过度粘连
            ::cv::Mat close_kernel = ::cv::getStructuringElement(::cv::MORPH_RECT, ::cv::Size(gap_len, 1));
            ::cv::morphologyEx(morph, morph, ::cv::MORPH_CLOSE, close_kernel);
            // 将黑色横线转换为白色前景，便于轮廓提取
            //::cv::bitwise_not(morph, morph);

            // 5) 查找并计数横线
            //    经过反色后，目标“黑色横线”会变成白色前景，便于轮廓提取
            std::vector<std::vector<::cv::Point>> contours;
            ::cv::findContours(morph, contours, ::cv::RETR_EXTERNAL, ::cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                const double area = ::cv::contourArea(contour);
                // 过滤小面积噪声（过小的轮廓通常为噪点）
                if (area < 40.0) continue;

                const ::cv::Rect r = ::cv::boundingRect(contour);
                const int w = r.width;
                const int h = r.height;

                // 面积阈值与长宽比筛选横线
                // 目标：过滤小横条，保留大横条，用面积阈值更直观
                const double min_area = 300.0; // 小于该面积的横线视为噪声
                if (area >= min_area) {
                    const float aspect_ratio = (float)w / (float)h;
                    if (aspect_ratio >= 1.5f) {
                        groove_count++;

                        // 发送 bbox：相对坐标(0..1)，以整帧为基准
                        const int abs_x = x1 + r.x;
                        const int abs_y = y1 + r.y;
                        const float cx = (float)abs_x + (float)w * 0.5f;
                        const float cy = (float)abs_y + (float)h * 0.5f;

                        UDPGrooveDet det{};
                        det.bbox_x = cx / (float)frame.width;
                        det.bbox_y = cy / (float)frame.height;
                        det.bbox_w = (float)w / (float)frame.width;
                        det.bbox_h = (float)h / (float)frame.height;
                        det.score = 1.0f;
                        det.id = groove_count;
                        udp_dets.push_back(det);
                    }
                }
            }
        }

        // 统计帧处理耗时
        const auto t1 = std::chrono::steady_clock::now();
        const double frame_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

        // 发送检测结果到UDP
        double udp_ms = -1.0;
        if (args.enable_udp) {
            udp_ms = udp_streamer.sendLatest(MAGIC_GROV, udp_dets);
        }

        // 打印统计信息
        if (args.enable_udp) {
            std::printf("[%s] count=%d frame_ms=%.2f udp_ms=%.2f\n", TAG, groove_count, frame_ms, udp_ms);
        } else {
            std::printf("[%s] count=%d frame_ms=%.2f\n", TAG, groove_count, frame_ms);
        }
    }

    // 退出前停止推流
    if (args.enable_udp) {
        udp_streamer.stop(/*deinit_vpss=*/false);
    }

    // 释放摄像头资源
    camera->stopStream();
    camera->deInit();

    std::printf("[%s] exit\n", TAG);
    return 0;
}

// 程序入口
int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return (argc > 1) ? 1 : 0;
    }
    return run_video_loop(args);
}
