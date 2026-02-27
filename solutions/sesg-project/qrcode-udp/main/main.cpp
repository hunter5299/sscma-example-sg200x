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
#include <opencv2/imgcodecs.hpp>  // 用于imread等图像读写功能

#include <sscma.h>
#include <video.h>

#include <sesg/stream_udp.h>

#include <zxing/Binarizer.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
#include <zxing/Exception.h>
#include <zxing/MatSource.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/Result.h>
#include <zxing/common/HybridBinarizer.h>

using namespace ma;

#define TAG "qrcode_udp"

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
    // 运行模式
    enum class Mode { VIDEO, IMAGE } mode = Mode::VIDEO;  // 运行模式：视频流或图片
    std::string image_path;  // 图片模式下的图片路径
    
    // UDP 相关配置
    bool enable_udp = false;              // 默认关闭UDP推流
    std::string udp_ip = "192.168.2.101";  // UDP目标IP地址
    int udp_port = 5001;                   // UDP目标端口

    // 摄像头配置
    int cam_w = 640;   // 摄像头宽度
    int cam_h = 480;   // 摄像头高度

    // JPEG通道配置
    // 注意：在本仓库的 CameraSG200X 实现中，retrieveFrame(MA_PIXEL_FORMAT_JPEG) 固定读取 CH1。
    // 因此这里默认用 CH1；即使配置其它通道为 JPEG，上层 retrieveFrame(JPEG) 也不会去读。
    int jpeg_ch = 1;     // JPEG通道号
    int jpeg_w = 320;    // JPEG输出宽度
    int jpeg_h = 240;    // JPEG输出高度
    int jpeg_fps = 10;   // JPEG帧率
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

// 简化的参数解析：支持位置参数 argv[1]=ip, argv[2]=port 或 image模式
static bool parse_args(int argc, char** argv, Args* a) {
    if (!a) return false;
    
    // 检查帮助选项
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return false;
        }
        
        // 图片模式: ./qrcode-udp image <path>
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
    
    // 视频流模式：./qrcode-udp [ip] [port]
    if (argc == 3) {
        // 启用UDP模式
        a->mode = Args::Mode::VIDEO;
        a->enable_udp = true;
        a->udp_ip = argv[1];
        if (!parse_int(argv[2], &a->udp_port)) {
            std::fprintf(stderr, "错误：端口号必须是数字\n");
            print_usage(argv[0]);
            return false;
        }
    } else if (argc == 1) {
        // 无参数：仅本地推理
        a->mode = Args::Mode::VIDEO;
        a->enable_udp = false;
    } else {
        std::fprintf(stderr, "错误：参数数量不正确\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

static bool decode_qr_gray(const ::cv::Mat& gray, std::string* out_text) {
    if (!out_text) return false;
    out_text->clear();
    if (gray.empty()) return false;

    try {
        zxing::Ref<zxing::LuminanceSource> source = MatSource::create(gray);
        zxing::Ref<zxing::Binarizer> binarizer(new zxing::HybridBinarizer(source));
        zxing::Ref<zxing::BinaryBitmap> bitmap(new zxing::BinaryBitmap(binarizer));

        // TryHarder 会显著增加“找不到码”时的搜索开销，视频流场景默认关闭以保证实时性。
        zxing::DecodeHints hints(zxing::DecodeHints::DEFAULT_HINT);
        hints.setTryHarder(false);

        zxing::MultiFormatReader reader;
        zxing::Ref<zxing::Result> result = reader.decode(bitmap, hints);
        if (!result) return false;
        if (!result->getText()) return false;

        *out_text = result->getText()->getText();
        return !out_text->empty();
    } catch (const zxing::Exception&) {
        return false;
    } catch (const std::exception&) {
        return false;
    } catch (...) {
        return false;
    }
}

// 为实时视频做的“快速解码”包装：
// 1) 对大图先降采样（比如 640x480 -> 320x240），显著降低ZXing的搜索开销
// 2) 默认不启用TryHarder，避免无码时单帧秒级卡住
static bool decode_qr_gray_fast(const ::cv::Mat& gray_in, std::string* out_text) {
    if (!out_text) return false;
    out_text->clear();
    if (gray_in.empty()) return false;

    // 目标宽度：尽量用更小分辨率做全图解码
    constexpr int kTargetWidth = 320;

    ::cv::Mat gray;
    if (gray_in.cols > kTargetWidth) {
        const double scale = (double)kTargetWidth / (double)gray_in.cols;
        ::cv::resize(gray_in, gray, ::cv::Size(), scale, scale, ::cv::INTER_AREA);
    } else {
        gray = gray_in;
    }

    return decode_qr_gray(gray, out_text);
}

// 图片模式：读取本地图片并进行二维码识别
static int run_image_mode(const std::string& image_path) {
    std::printf("[qrcode-udp] 图片模式: %s\n", image_path.c_str());
    
    // 读取图片
    ::cv::Mat img = ::cv::imread(image_path, ::cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "[错误] 无法读取图片: %s\n", image_path.c_str());
        return 1;
    }
    
    std::printf("[图片信息] 尺寸=%dx%d\n", img.cols, img.rows);
    
    // 转换为灰度图
    ::cv::Mat gray;
    ::cv::cvtColor(img, gray, ::cv::COLOR_BGR2GRAY);
    
    // 开始计时
    auto start = std::chrono::steady_clock::now();
    
    // 执行二维码识别（使用与视频一致的“快速路径”）
    std::string text;
    bool ok = decode_qr_gray_fast(gray, &text);
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (ok) {
        std::printf("[识别成功] 二维码内容: %s\n", text.c_str());
        std::printf("[性能] 耗时: %ld ms\n", elapsed_ms);
        return 0;
    } else {
        std::printf("[识别失败] 未检测到二维码\n");
        std::printf("[性能] 耗时: %ld ms\n", elapsed_ms);
        return 1;
    }
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return (argc > 1) ? 1 : 0;
    }

    // 图片模式：直接处理图片后退出
    if (args.mode == Args::Mode::IMAGE) {
        return run_image_mode(args.image_path);
    }

    // === 以下是视频流模式 ===
    
    Signal::install({SIGINT, SIGTERM, SIGQUIT}, [](int) { sig_handler(0); });

    std::printf("[qrcode-udp] 视频流模式 | UDP推流: %s, 目标地址=%s:%d, 摄像头=%dx%d, JPEG(通道=%d)=%dx%d@%dfps\n",
                args.enable_udp ? "启用" : "关闭",
                args.udp_ip.c_str(),
                args.udp_port,
                args.cam_w,
                args.cam_h,
                args.jpeg_ch,
                args.jpeg_w,
                args.jpeg_h,
                args.jpeg_fps);

    // 获取设备实例
    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    // 遍历传感器列表，查找摄像头设备
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);

            Camera::CtrlValue v;
            // 设置通道号为0
            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, v);
            
            // 配置摄像头分辨率
            v.u16s[0] = static_cast<uint16_t>(args.cam_w);
            v.u16s[1] = static_cast<uint16_t>(args.cam_h);
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, v);

            // 设置返回CPU可直接访问的虚拟地址（非物理地址）
            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "未找到摄像头设备");
        return 1;
    }

    // 配置 JPEG 通道（必须在 startStream 前配置）
    sesg::stream_udp::JpegChannelConfig jpeg_cfg;
    jpeg_cfg.ch = args.jpeg_ch;
    jpeg_cfg.width = args.jpeg_w;
    jpeg_cfg.height = args.jpeg_h;
    jpeg_cfg.fps = args.jpeg_fps;
    sesg::stream_udp::SesgJpegUdpStreamer::configureJpegChannel(*camera, jpeg_cfg);

    // 启动摄像头数据流
    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // UDP推流器（可选）
    auto camera_shared = std::shared_ptr<Camera>(camera, [](Camera*) {});
    sesg::stream_udp::SesgJpegUdpStreamer udp_streamer(args.udp_ip, (uint16_t)args.udp_port);
    
    // 仅在启用UDP时启动推流器
    if (args.enable_udp) {
        if (!udp_streamer.start(camera_shared, jpeg_cfg, /*init_vpss=*/false)) {
            MA_LOGE(TAG, "UDP推流器启动失败");
            camera->stopStream();
            camera->deInit();
            return 1;
        }
        std::printf("[qrcode-udp] UDP推流器已启动\n");
    }

    constexpr uint32_t MAGIC_QRCD = 0x51524344;  // "QRCD" UDP数据包魔数

    std::vector<uint8_t> safe_frame;  // 安全的帧数据缓冲区
    std::string last_text;            // 上一次识别的二维码文本
    
    // 统计变量（分开统计“采集/处理”和“解码”）
    int frame_count = 0;   // 成功取到并处理（含跳过解码）的帧数
    int decode_count = 0;  // 实际执行解码的次数
    int ok_count = 0;      // 解码成功次数（同一个二维码也会重复计数）
    int new_text_count = 0;  // 新码次数（二维码文本变化才计数，更贴近“出现了几个码”）
    int64_t decode_ms_sum = 0;  // 解码耗时累计
    auto stat_start = std::chrono::steady_clock::now();
    auto last_decode = std::chrono::steady_clock::now();

    // 解码节流：视频场景不必每帧都跑ZXing（尤其无码时会非常耗时）
    // 150ms 一次接近你提到的“单张150ms”的量级；同时能把队列压力降下来。
    constexpr int kDecodeIntervalMs = 150;

    std::printf("[qrcode-udp] 开始二维码识别循环...\n");

    while (g_running.load()) {
        // 注意：这里的目标是“尽快取帧+归还”，避免底层缓存堆积。
        // 从摄像头获取RGB888格式的帧
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 减少等待时间
            continue;
        }

        // 检查帧数据有效性
        if (!frame.data || frame.size == 0 || frame.width == 0 || frame.height == 0) {
            camera->returnFrame(frame);
            continue;  // 移除sleep，直接继续
        }

        // 复制帧数据到安全缓冲区（避免returnFrame后数据失效）
        if (safe_frame.size() < frame.size) {
            safe_frame.resize(frame.size);
        }
        std::memcpy(safe_frame.data(), frame.data, frame.size);
        camera->returnFrame(frame);  // 尽快归还帧缓冲区

        // 将RGB888图像转换为灰度图（二维码识别需要）
        ::cv::Mat rgb((int)frame.height, (int)frame.width, CV_8UC3, safe_frame.data());
        ::cv::Mat gray;
        ::cv::cvtColor(rgb, gray, ::cv::COLOR_RGB2GRAY);

        // 是否执行解码（按时间节流）
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

        // 构建识别结果（只在本次确实解码且成功时发送/打印）
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

            // 仅当文本变化时打印/计“新码”
            if (text != last_text) {
                new_text_count++;
                std::printf("[二维码] %s\n", text.c_str());
                last_text = text;
            }
        }

        // 如果启用UDP，发送"最新 JPEG + 识别结果"
        if (args.enable_udp) {
            udp_streamer.sendLatest(MAGIC_QRCD, results);
        }

        // 更新统计
        frame_count++;
        const auto stat_now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stat_now - stat_start).count();

        // 每1秒打印一次统计信息：采集FPS、解码FPS、平均解码耗时
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

        // 不再sleep，直接处理下一帧以尽量把底层缓存“抽干”
    }

    // 清理资源
    if (args.enable_udp) {
        udp_streamer.stop(/*deinit_vpss=*/false);
    }
    camera->stopStream();
    camera->deInit();

    std::printf("[qrcode-udp] 程序正常退出\n");
    return 0;
}
