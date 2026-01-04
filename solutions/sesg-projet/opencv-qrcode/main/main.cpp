#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <sscma.h>
#include <video.h>

#include <sesg/stream_udp.h>

using namespace ma;

#define TAG "opencv_qrcode"

static std::atomic<bool> g_running(true);
static void sig_handler(int) { g_running.store(false); }

struct UDPQrCodeResult {
    float bbox_x;
    float bbox_y;
    float bbox_w;
    float bbox_h;
    float score;
    char text[128];
};

struct Args {
    enum class Mode { VIDEO, IMAGE } mode = Mode::VIDEO;
    std::string image_path;

    bool enable_udp = false;
    std::string udp_ip = "192.168.2.101";
    int udp_port = 5001;

    int cam_w = 640;
    int cam_h = 480;

    int jpeg_ch = 1;
    int jpeg_w = 320;
    int jpeg_h = 240;
    int jpeg_fps = 10;
};

static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [udp_ip udp_port]\n"
        "       %s image <image_path>\n\n"
        "Examples:\n"
        "  %s                        # 视频流推理（无UDP推流）\n"
        "  %s 192.168.2.101 5001     # 视频流推理并发送到UDP\n"
        "  %s image /tmp/qr.jpg      # 本地图片推理\n",
        argv0, argv0, argv0, argv0, argv0);
}

static bool parse_int(const char* s, int* out) {
    if (!s || !out) return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    *out = static_cast<int>(v);
    return true;
}

static bool parse_args(int argc, char** argv, Args* a) {
    if (!a) return false;

    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return false;
        }

        if (arg1 == "image") {
            if (argc != 3) {
                std::fprintf(stderr, "错误：image 模式需要指定图片路径\n");
                print_usage(argv[0]);
                return false;
            }
            a->mode = Args::Mode::IMAGE;
            a->image_path = argv[2];
            return true;
        }
    }

    if (argc == 3) {
        a->mode = Args::Mode::VIDEO;
        a->enable_udp = true;
        a->udp_ip = argv[1];
        if (!parse_int(argv[2], &a->udp_port)) {
            std::fprintf(stderr, "错误：端口号必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
    } else if (argc == 1) {
        a->mode = Args::Mode::VIDEO;
        a->enable_udp = false;
    } else {
        std::fprintf(stderr, "错误：参数数量不正确\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

static bool decode_qr_gray_opencv(const ::cv::Mat& gray, std::string* out_text) {
    if (!out_text) return false;
    out_text->clear();
    if (gray.empty()) return false;

    try {
        static thread_local ::cv::QRCodeDetector detector;
        const std::string text = detector.detectAndDecode(gray);
        if (text.empty()) return false;
        *out_text = text;
        return true;
    } catch (const std::exception&) {
        return false;
    } catch (...) {
        return false;
    }
}

static bool decode_qr_gray_fast(const ::cv::Mat& gray_in, std::string* out_text) {
    if (!out_text) return false;
    out_text->clear();
    if (gray_in.empty()) return false;

    constexpr int kTargetWidth = 320;

    ::cv::Mat gray;
    if (gray_in.cols > kTargetWidth) {
        const double scale = (double)kTargetWidth / (double)gray_in.cols;
        ::cv::resize(gray_in, gray, ::cv::Size(), scale, scale, ::cv::INTER_AREA);
    } else {
        gray = gray_in;
    }

    return decode_qr_gray_opencv(gray, out_text);
}

static int run_image_mode(const std::string& image_path) {
    std::printf("[opencv-qrcode] 图片模式: %s\n", image_path.c_str());

    ::cv::Mat img = ::cv::imread(image_path, ::cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[错误] 无法读取图片: %s\n", image_path.c_str());
        return 1;
    }

    std::printf("[图片信息] 尺寸=%dx%d\n", img.cols, img.rows);

    ::cv::Mat gray;
    ::cv::cvtColor(img, gray, ::cv::COLOR_BGR2GRAY);

    auto start = std::chrono::steady_clock::now();

    std::string text;
    bool ok = decode_qr_gray_fast(gray, &text);

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (ok) {
        std::printf("[识别成功] 二维码内容: %s\n", text.c_str());
        std::printf("[性能] 耗时: %ld ms\n", elapsed_ms);
        return 0;
    }

    std::printf("[识别失败] 未检测到二维码\n");
    std::printf("[性能] 耗时: %ld ms\n", elapsed_ms);
    return 1;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return (argc > 1) ? 1 : 0;
    }

    if (args.mode == Args::Mode::IMAGE) {
        return run_image_mode(args.image_path);
    }

    Signal::install({SIGINT, SIGTERM, SIGQUIT}, [](int) { sig_handler(0); });

    std::printf(
        "[opencv-qrcode] 视频流模式 | UDP推流: %s, 目标地址=%s:%d, 摄像头=%dx%d, JPEG(通道=%d)=%dx%d@%dfps\n",
        args.enable_udp ? "启用" : "关闭",
        args.udp_ip.c_str(),
        args.udp_port,
        args.cam_w,
        args.cam_h,
        args.jpeg_ch,
        args.jpeg_w,
        args.jpeg_h,
        args.jpeg_fps);

    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);

            Camera::CtrlValue v;
            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, v);

            v.u16s[0] = static_cast<uint16_t>(args.cam_w);
            v.u16s[1] = static_cast<uint16_t>(args.cam_h);
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, v);

            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "未找到摄像头设备");
        return 1;
    }

    sesg::stream_udp::JpegChannelConfig jpeg_cfg;
    jpeg_cfg.ch = args.jpeg_ch;
    jpeg_cfg.width = args.jpeg_w;
    jpeg_cfg.height = args.jpeg_h;
    jpeg_cfg.fps = args.jpeg_fps;
    sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    auto camera_shared = std::shared_ptr<Camera>(camera, [](Camera*) {});
    sesg::stream_udp::SesgJpegUdpStreamer udp_streamer(args.udp_ip, (uint16_t)args.udp_port);

    if (args.enable_udp) {
        if (!udp_streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
            MA_LOGE(TAG, "UDP推流器启动失败");
            camera->stopStream();
            camera->deInit();
            return 1;
        }
        std::printf("[opencv-qrcode] UDP推流器已启动\n");
    }

    constexpr uint32_t MAGIC_QRCD = 0x51524344;

    std::vector<uint8_t> safe_frame;
    std::string last_text;

    int frame_count = 0;
    int decode_count = 0;
    int ok_count = 0;
    int new_text_count = 0;
    int64_t decode_ms_sum = 0;
    auto stat_start = std::chrono::steady_clock::now();
    auto last_decode = std::chrono::steady_clock::now();

    constexpr int kDecodeIntervalMs = 150;

    std::printf("[opencv-qrcode] 开始二维码识别循环...\n");

    while (g_running.load()) {
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (!frame.data || frame.size == 0 || frame.width == 0 || frame.height == 0) {
            camera->returnFrame(frame);
            continue;
        }

        if (safe_frame.size() < frame.size) {
            safe_frame.resize(frame.size);
        }
        std::memcpy(safe_frame.data(), frame.data, frame.size);
        camera->returnFrame(frame);

        ::cv::Mat rgb((int)frame.height, (int)frame.width, CV_8UC3, safe_frame.data());
        ::cv::Mat gray;
        ::cv::cvtColor(rgb, gray, ::cv::COLOR_RGB2GRAY);

        const auto now = std::chrono::steady_clock::now();
        const bool should_decode =
            (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_decode).count() >= kDecodeIntervalMs);

        std::string text;
        bool ok = false;
        if (should_decode) {
            last_decode = now;
            auto d0 = std::chrono::steady_clock::now();
            ok = decode_qr_gray_fast(gray, &text);
            auto d1 = std::chrono::steady_clock::now();
            const int64_t dms = std::chrono::duration_cast<std::chrono::milliseconds>(d1 - d0).count();
            decode_ms_sum += dms;
            decode_count++;
        }

        std::vector<UDPQrCodeResult> results;
        if (ok) {
            ok_count++;
            UDPQrCodeResult r{};
            r.bbox_x = 0.0f;
            r.bbox_y = 0.0f;
            r.bbox_w = 0.0f;
            r.bbox_h = 0.0f;
            r.score = 1.0f;
            std::strncpy(r.text, text.c_str(), sizeof(r.text) - 1);
            r.text[sizeof(r.text) - 1] = '\0';
            results.push_back(r);

            if (text != last_text) {
                new_text_count++;
                std::printf("[二维码] %s\n", text.c_str());
                last_text = text;
            }
        }

        if (args.enable_udp) {
            udp_streamer.sendLatest(MAGIC_QRCD, results);
        }

        frame_count++;
        const auto stat_now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stat_now - stat_start).count();

        if (elapsed >= 1000) {
            const double fps = (frame_count * 1000.0) / (double)elapsed;
            const double decode_fps = (decode_count * 1000.0) / (double)elapsed;
            const double avg_decode_ms = (decode_count > 0) ? ((double)decode_ms_sum / (double)decode_count) : 0.0;
            std::printf(
                "[性能] 采集FPS=%.2f | 解码FPS=%.2f | 平均解码耗时=%.1fms | 解码成功=%d次 | 新码=%d次\n",
                fps,
                decode_fps,
                avg_decode_ms,
                ok_count,
                new_text_count);

            frame_count = 0;
            decode_count = 0;
            ok_count = 0;
            new_text_count = 0;
            decode_ms_sum = 0;
            stat_start = stat_now;
        }
    }

    if (args.enable_udp) {
        udp_streamer.stop(/*deinit_vpss=*/false);
    }
    camera->stopStream();
    camera->deInit();

    std::printf("[opencv-qrcode] 程序正常退出\n");
    return 0;
}
