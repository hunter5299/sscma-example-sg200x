#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <atomic>

#include <sscma.h>
#include <video.h>
#include "rtsp_demo.h"
#include <opencv2/opencv.hpp>

#include "ocr_pipeline.h"
#include "text_recognizer.h"
#include "mqtt_publisher.h"

using namespace ma;
using namespace ppocr;

#define TAG "ppocr-reader"

// Default configuration
static struct {
    // Model paths
    std::string det_model_path = "/userdata/local/models/ppocr_det_cv181x_mix.cvimodel";
    std::string rec_model_path = "/userdata/local/models/ppocr_rec_cv181x_bf16.cvimodel";
    std::string dict_path = "/userdata/local/dict/ppocr_keys_v1.txt";

    // MQTT configuration
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/ppocr/texts";

    // Video configuration
    int inference_width = 640;
    int inference_height = 480;
    int inference_fps = 15;
    int stream_width = 640;
    int stream_height = 480;
    int stream_fps = 15;

    // Runtime flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;

    // Test mode
    std::string test_rec_image;  // If set, test recognizer with this image and exit
} g_config;

// Global state
static std::atomic<bool> g_running(true);
static OcrPipeline* g_pipeline = nullptr;
static MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("PP-OCRv3 Text Reader for ReCamera\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --det-model PATH     Detection model (default: %s)\n", g_config.det_model_path.c_str());
    printf("  --rec-model PATH     Recognition model (default: %s)\n", g_config.rec_model_path.c_str());
    printf("  --dict PATH          Dictionary file (default: %s)\n", g_config.dict_path.c_str());
    printf("  --mqtt-host HOST     MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT     MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC   MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-rtsp            Disable RTSP streaming\n");
    printf("  --no-mqtt            Disable MQTT publishing\n");
    printf("  --test-rec PATH      Test recognizer with an image file and exit\n");
    printf("  -v, --verbose        Enable verbose logging\n");
    printf("  -h, --help           Show this help\n");
    printf("\n");
    printf("RTSP Stream: rtsp://<device_ip>:8554/live0\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"det-model", required_argument, 0, 1},
        {"rec-model", required_argument, 0, 2},
        {"dict", required_argument, 0, 3},
        {"mqtt-host", required_argument, 0, 4},
        {"mqtt-port", required_argument, 0, 5},
        {"mqtt-topic", required_argument, 0, 6},
        {"no-rtsp", no_argument, 0, 7},
        {"no-mqtt", no_argument, 0, 8},
        {"test-rec", required_argument, 0, 9},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 1: g_config.det_model_path = optarg; break;
            case 2: g_config.rec_model_path = optarg; break;
            case 3: g_config.dict_path = optarg; break;
            case 4: g_config.mqtt_host = optarg; break;
            case 5: g_config.mqtt_port = std::stoi(optarg); break;
            case 6: g_config.mqtt_topic = optarg; break;
            case 7: g_config.enable_rtsp = false; break;
            case 8: g_config.enable_mqtt = false; break;
            case 9: g_config.test_rec_image = optarg; break;
            case 'v': g_config.verbose = true; break;
            case 'h': print_usage(argv[0]); exit(0);
            default: print_usage(argv[0]); return false;
        }
    }
    return true;
}

static bool init_pipeline() {
    g_pipeline = new OcrPipeline();
    if (!g_pipeline->init(g_config.det_model_path, g_config.rec_model_path, g_config.dict_path)) {
        MA_LOGE(TAG, "Failed to initialize OCR pipeline");
        return false;
    }
    MA_LOGI(TAG, "OCR pipeline initialized");
    return true;
}

static bool init_camera() {
    Device* device = Device::getInstance();

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;

            // Channel 0 for inference
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Inference resolution
            value.u16s[0] = g_config.inference_width;
            value.u16s[1] = g_config.inference_height;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // Virtual address mode (not physical)
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d)", g_config.inference_width, g_config.inference_height);
            return true;
        }
    }

    MA_LOGE(TAG, "No camera found");
    return false;
}

static bool init_video_streaming() {
    if (!g_config.enable_rtsp) {
        MA_LOGI(TAG, "RTSP streaming disabled");
        return true;
    }

    if (initVideo() != 0) {
        MA_LOGE(TAG, "Failed to initialize video subsystem");
        return false;
    }

    video_ch_param_t stream_param;
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);

    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);
    initRtsp((0x01 << VIDEO_CH2));

    MA_LOGI(TAG, "RTSP streaming initialized (%dx%d @ %dfps)",
            g_config.stream_width, g_config.stream_height, g_config.stream_fps);
    return true;
}

static bool init_mqtt() {
    if (!g_config.enable_mqtt) {
        MA_LOGI(TAG, "MQTT publishing disabled");
        return true;
    }

    g_mqtt_publisher = new MqttPublisher();
    MqttConfig mqtt_config;
    mqtt_config.host = g_config.mqtt_host;
    mqtt_config.port = g_config.mqtt_port;
    mqtt_config.topic = g_config.mqtt_topic;

    if (!g_mqtt_publisher->init(mqtt_config)) {
        MA_LOGE(TAG, "Failed to initialize MQTT publisher");
        return false;
    }
    return true;
}

static void cleanup() {
    if (g_camera) {
        g_camera->stopStream();
    }

    if (g_config.enable_rtsp) {
        deinitRtsp();
        deinitVideo();
    }

    if (g_mqtt_publisher) {
        g_mqtt_publisher->deinit();
        delete g_mqtt_publisher;
        g_mqtt_publisher = nullptr;
    }

    if (g_pipeline) {
        delete g_pipeline;
        g_pipeline = nullptr;
    }

    MA_LOGI(TAG, "Cleanup completed");
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    // Drain stale frames from camera queue — keep only the latest.
    // The camera pushes at 15fps into a 15-slot FIFO, but OCR only
    // consumes at ~2fps, so old frames accumulate and cause multi-second
    // overlay lag vs the real-time RTSP video.
    {
        ma_img_t newer;
        int skipped = 0;
        while (g_camera->retrieveFrame(newer, MA_PIXEL_FORMAT_RGB888) == MA_OK) {
            g_camera->returnFrame(frame);
            frame = newer;
            skipped++;
        }
        if (skipped > 0 && g_config.verbose) {
            MA_LOGD(TAG, "Skipped %d stale frames", skipped);
        }
    }

    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // Run OCR pipeline
    OcrTimings timings;
    std::vector<OcrResult> results = g_pipeline->process(&frame, timings);
    int frame_w = frame.width;
    int frame_h = frame.height;

    // Return frame to camera
    g_camera->returnFrame(frame);

    // Publish via MQTT
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        g_mqtt_publisher->publishResults(timestamp_ms, g_frame_id, results, timings, frame_w, frame_h);
    }

    // Log results
    if (g_config.verbose || !results.empty()) {
        MA_LOGI(TAG, "Frame %u: %zu texts, det=%.0fms rec=%.0fms total=%.0fms",
                g_frame_id, results.size(),
                timings.detection_ms, timings.recognition_ms, timings.total_ms);

        for (const auto& r : results) {
            MA_LOGI(TAG, "  [%.2f] \"%s\"", r.rec_confidence, r.text.c_str());
        }
    }

    g_frame_id++;
}

// Test recognizer with a reference image file (no camera needed)
static int run_test_rec(const std::string& image_path) {
    MA_LOGI(TAG, "=== Test Recognizer Mode ===");
    MA_LOGI(TAG, "Image: %s", image_path.c_str());
    MA_LOGI(TAG, "Model: %s", g_config.rec_model_path.c_str());
    MA_LOGI(TAG, "Dict: %s", g_config.dict_path.c_str());

    // Load image
    ::cv::Mat img = ::cv::imread(image_path, ::cv::IMREAD_COLOR);
    if (img.empty()) {
        MA_LOGE(TAG, "Failed to load image: %s", image_path.c_str());
        return 1;
    }
    MA_LOGI(TAG, "Image loaded: %dx%d channels=%d", img.cols, img.rows, img.channels());

    // OpenCV loads as BGR, convert to RGB
    ::cv::Mat rgb;
    ::cv::cvtColor(img, rgb, ::cv::COLOR_BGR2RGB);

    // Initialize recognizer only
    TextRecognizer recognizer;
    if (!recognizer.init(g_config.rec_model_path, g_config.dict_path)) {
        MA_LOGE(TAG, "Failed to init recognizer");
        return 1;
    }

    // Run recognition
    auto t0 = std::chrono::high_resolution_clock::now();
    RecognitionResult result = recognizer.recognize(rgb.data, rgb.cols, rgb.rows);
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("=== Recognition Result ===\n");
    printf("Text: \"%s\"\n", result.text.c_str());
    printf("Confidence: %.4f\n", result.confidence);
    printf("Time: %.1f ms\n", ms);

    return 0;
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) return 1;

    // Test mode: run recognizer on a single image and exit
    if (!g_config.test_rec_image.empty()) {
        return run_test_rec(g_config.test_rec_image);
    }

    MA_LOGI(TAG, "Starting PP-OCRv3 Text Reader");
    MA_LOGI(TAG, "Detection model: %s", g_config.det_model_path.c_str());
    MA_LOGI(TAG, "Recognition model: %s", g_config.rec_model_path.c_str());
    MA_LOGI(TAG, "Dictionary: %s", g_config.dict_path.c_str());

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (!init_pipeline()) { cleanup(); return 1; }
    if (!init_camera()) { cleanup(); return 1; }
    if (!init_video_streaming()) { cleanup(); return 1; }
    if (!init_mqtt()) { cleanup(); return 1; }

    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    MA_LOGI(TAG, "PP-OCRv3 reader running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());

    while (g_running.load()) {
        process_frame();
        // Yield CPU briefly to prevent starving other processes (SSH, etc.)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    cleanup();
    MA_LOGI(TAG, "PP-OCRv3 Text Reader terminated");
    return 0;
}
