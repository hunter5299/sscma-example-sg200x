
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <sscma.h>
#include <video.h>

#include <udp_service/udp_service.h>

using namespace ma;

#define TAG "udp_detector"
#define UDP_IP "192.168.2.101"
#define UDP_PORT 5000
static constexpr uint32_t UDP_MAGIC = 0x55445044;

static std::atomic<bool> g_running(true);
void sig_handler(int) { g_running.store(false); }

static int g_stdout_fd = -1;
static int g_stderr_fd = -1;
static int g_null_fd = -1;

static void mute_stdio() {
    if (g_stdout_fd >= 0 || g_stderr_fd >= 0) return;
    g_stdout_fd = dup(STDOUT_FILENO);
    g_stderr_fd = dup(STDERR_FILENO);
    g_null_fd = open("/dev/null", O_WRONLY);
    if (g_null_fd >= 0) {
        dup2(g_null_fd, STDOUT_FILENO);
        dup2(g_null_fd, STDERR_FILENO);
    }
}

static void restore_stdio() {
    if (g_stdout_fd >= 0) {
        dup2(g_stdout_fd, STDOUT_FILENO);
        close(g_stdout_fd);
        g_stdout_fd = -1;
    }
    if (g_stderr_fd >= 0) {
        dup2(g_stderr_fd, STDERR_FILENO);
        close(g_stderr_fd);
        g_stderr_fd = -1;
    }
    if (g_null_fd >= 0) {
        close(g_null_fd);
        g_null_fd = -1;
    }
}

namespace detector_utils {

struct Detection {
    float x, y, w, h;
    float score;
    int32_t target;
};

class SquareDetector {
public:
    SquareDetector(ma::model::Detector* detector, int input_w, int input_h)
        : detector_(detector), input_w_(input_w), input_h_(input_h) {
        (void)input_w_;
        (void)input_h_;
    }

    void run(std::vector<Detection>& results) {
        detector_->run(nullptr);

        auto sscma_results = detector_->getResults();
        results.clear();
        for (auto& r : sscma_results) {
            Detection det;
            det.x = r.x;
            det.y = r.y;
            det.w = r.w;
            det.h = r.h;
            det.score = r.score;
            det.target = r.target;
            results.push_back(det);
        }
    }

    void setThreshold(float threshold) {
        detector_->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
    }

private:
    ma::model::Detector* detector_;
    int input_w_;
    int input_h_;
};

} // namespace detector_utils

static bool looks_like_ipv4(const char* s) {
    if (!s || !*s) return false;
    int dots = 0;
    for (const char* p = s; *p; ++p) {
        if (*p == '.') {
            dots++;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    }
    return dots == 3;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s model.cvimodel [threshold] [udp_ip] [udp_port]\n", argv[0]);
        printf("  %s model.cvimodel [udp_ip] [udp_port]\n", argv[0]);
        printf("  threshold: default 0.5\n");
        printf("  udp_ip: default %s\n", UDP_IP);
        printf("  udp_port: default %d\n", UDP_PORT);
        exit(-1);
    }

    const char* model_path = argv[1];

    float threshold = 0.5f;
    const char* udp_ip = UDP_IP;
    int udp_port = UDP_PORT;

    if (argc == 3 && looks_like_ipv4(argv[2])) {
        udp_ip = argv[2];
    } else if (argc == 4 && looks_like_ipv4(argv[2])) {
        udp_ip = argv[2];
        udp_port = atoi(argv[3]);
    } else {
        if (argc >= 3) threshold = atof(argv[2]);
        if (argc >= 4) udp_ip = argv[3];
        if (argc >= 5) udp_port = atoi(argv[4]);
    }

    printf("UDP detector\n");
    printf("model: %s\n", model_path);
    printf("threshold: %.2f\n", threshold);
    printf("udp: %s:%d\n", udp_ip, udp_port);

    auto* engine = new ma::engine::EngineCVI();
    if (engine->init() != MA_OK || engine->load(model_path) != MA_OK) {
        MA_LOGE(TAG, "engine init/load failed");
        delete engine;
        return 1;
    }

    ma::Model* model = ma::ModelFactory::create(engine);
    if (!model) {
        MA_LOGE(TAG, "ModelFactory::create failed (this demo does not bypass SSCMA)");
        return 1;
    }
    if (model->getInputType() != MA_INPUT_TYPE_IMAGE || model->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
        MA_LOGE(TAG, "model not supported (require detector bbox model)");
        return 1;
    }

    const ma_img_t* model_input = static_cast<const ma_img_t*>(model->getInput());
    int input_w = model_input->width;
    int input_h = model_input->height;

    if (input_w != input_h) {
        MA_LOGE(TAG, "Square input required (%dx%d)", input_w, input_h);
        return 1;
    }

    auto* detector = static_cast<ma::model::Detector*>(model);
    detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
    auto* square_detector = new detector_utils::SquareDetector(detector, input_w, input_h);

    mute_stdio();
    Device* device = Device::getInstance();
    Camera* camera = nullptr;
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);
            Camera::CtrlValue val;

            val.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, val);
            val.u16s[0] = input_w;
            val.u16s[1] = input_h;
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, val);
            val.i32 = 1;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, val);

            val.i32 = 1;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, val);
            val.u16s[0] = 320;
            val.u16s[1] = 240;
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, val);
            val.i32 = (int)MA_PIXEL_FORMAT_JPEG;
            camera->commandCtrl(Camera::CtrlType::kFormat, Camera::CtrlMode::kWrite, val);
            val.i32 = 10;
            camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, val);

            break;
        }
    }
    if (!camera) { 
        restore_stdio();
        MA_LOGE(TAG, "No camera"); 
        return 1; 
    }

    camera->startStream(Camera::StreamMode::kRefreshOnReturn);
    restore_stdio();
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    udp_service::SenderOptions udp_opt;
    udp_opt.max_udp_packet = 1400;
    udp_service::UDPSender udp_sender(udp_ip, udp_port, udp_opt);
    udp_sender.start();

    struct JpegFrameOwned {
        std::unique_ptr<uint8_t[]> buf;
        size_t size = 0;
        int w = 0;
        int h = 0;
    };

    std::mutex jpeg_mu;
    JpegFrameOwned latest_jpeg;
    std::atomic<bool> jpeg_running{false};
    std::thread jpeg_thread;

    jpeg_running.store(true);
    jpeg_thread = std::thread([&]() {
        ma_img_t jpeg;
        while (g_running.load() && jpeg_running.load()) {
            if (camera->retrieveFrame(jpeg, MA_PIXEL_FORMAT_JPEG) != MA_OK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            const size_t jpeg_size = jpeg.size;
            const int jpeg_w = jpeg.width;
            const int jpeg_h = jpeg.height;

            std::unique_ptr<uint8_t[]> jpeg_buf(new uint8_t[jpeg_size]);
            memcpy(jpeg_buf.get(), reinterpret_cast<const void*>(jpeg.data), jpeg_size);
            camera->returnFrame(jpeg);

            {
                std::lock_guard<std::mutex> lk(jpeg_mu);
                latest_jpeg.buf = std::move(jpeg_buf);
                latest_jpeg.size = jpeg_size;
                latest_jpeg.w = jpeg_w;
                latest_jpeg.h = jpeg_h;
            }
        }
    });

    std::vector<detector_utils::Detection> cached_detections;

    auto fps_start = std::chrono::high_resolution_clock::now();
    int fps_frame_count = 0;

    while (g_running.load()) {
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        fps_frame_count++;

        ma_tensor_t tensor = {
            .size = frame.size,
            .is_physical = true,
            .is_variable = false,
        };
        tensor.data.data = reinterpret_cast<void*>(frame.data);
        engine->setInput(0, tensor);

        std::vector<detector_utils::Detection> results;
        square_detector->run(results);
        cached_detections = results;

        camera->returnFrame(frame);

        JpegFrameOwned jpeg_owned;
        {
            std::lock_guard<std::mutex> lk(jpeg_mu);
            if (latest_jpeg.buf) {
                jpeg_owned = std::move(latest_jpeg);
            }
        }

        if (jpeg_owned.buf && jpeg_owned.size > 0) {
            udp_sender.sendJpegOwned(UDP_MAGIC,
                                     std::move(jpeg_owned.buf),
                                     jpeg_owned.size,
                                     jpeg_owned.w,
                                     jpeg_owned.h,
                                     cached_detections);
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count();
        if (elapsed_ms >= 2000) {
            const float fps = (elapsed_ms > 0) ? (fps_frame_count * 1000.0f / (float)elapsed_ms) : 0.0f;
            printf("[stats] fps=%.1f dets=%zu\n", fps, cached_detections.size());
            for (size_t i = 0; i < cached_detections.size(); ++i) {
                const auto& d = cached_detections[i];
                printf("[det %zu] xywh=(%.3f,%.3f,%.3f,%.3f) score=%.3f target=%d\n",
                       i, d.x, d.y, d.w, d.h, d.score, d.target);
            }
            fps_start = now;
            fps_frame_count = 0;
        }
    }

    jpeg_running.store(false);
    if (jpeg_thread.joinable()) {
        jpeg_thread.join();
    }
    udp_sender.stop();
    camera->stopStream();
    for (auto& sensor : device->getSensors()) sensor->deInit();
    
    if (square_detector) delete square_detector;
    ma::ModelFactory::remove(model);
    delete engine;

    return 0;
}
