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

#include "face_detector.h"
#include "attribute_analyzer.h"
#include "mqtt_publisher.h"
#include "face_blur.h"

using namespace ma;
using namespace face_analysis;

#define TAG "face-analysis"

// Default configuration
static struct {
    // Model paths (auto-detects FairFace vs InsightFace format)
    std::string face_model = "/userdata/local/models/yolo-face_mixfp16.cvimodel";
    std::string genderage_model = "/userdata/local/models/genderage_int8.cvimodel";
    std::string emotion_model = "/userdata/local/models/emotion_bf16.cvimodel";

    // Detection parameters
    float face_threshold = 0.4f;

    // MQTT configuration
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/face-analysis/results";

    // Video configuration
    int inference_width = 640;
    int inference_height = 480;
    int inference_fps = 10;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Blur configuration
    int max_regions = 12;

    // Runtime flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool enable_blur = true;
    bool verbose = false;
} g_config;

// Global state
static std::atomic<bool> g_running(true);
static FaceDetector* g_face_detector = nullptr;
static AttributeAnalyzer* g_attribute_analyzer = nullptr;
static MqttPublisher* g_mqtt_publisher = nullptr;
static FaceBlur* g_face_blur = nullptr;
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("Face Analysis for ReCamera\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -f, --face-model PATH     Face detection model (default: %s)\n", g_config.face_model.c_str());
    printf("  -g, --genderage-model PATH GenderAge model (default: %s)\n", g_config.genderage_model.c_str());
    printf("  -e, --emotion-model PATH  Emotion model (default: %s)\n", g_config.emotion_model.c_str());
    printf("  -t, --threshold FLOAT     Face detection threshold (default: %.2f)\n", g_config.face_threshold);
    printf("  -m, --mqtt-host HOST      MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  -p, --mqtt-port PORT      MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --no-rtsp                 Disable RTSP streaming\n");
    printf("  --no-mqtt                 Disable MQTT publishing\n");
    printf("  --no-blur                 Disable face blur on RTSP stream\n");
    printf("  --max-regions N           Max blur regions (1-16, default: %d)\n", g_config.max_regions);
    printf("  -v, --verbose             Enable verbose logging\n");
    printf("  -h, --help                Show this help message\n");
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"face-model", required_argument, 0, 'f'},
        {"genderage-model", required_argument, 0, 'g'},
        {"emotion-model", required_argument, 0, 'e'},
        {"threshold", required_argument, 0, 't'},
        {"mqtt-host", required_argument, 0, 'm'},
        {"mqtt-port", required_argument, 0, 'p'},
        {"no-rtsp", no_argument, 0, 1},
        {"no-mqtt", no_argument, 0, 2},
        {"no-blur", no_argument, 0, 3},
        {"max-regions", required_argument, 0, 4},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:g:e:t:m:p:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                g_config.face_model = optarg;
                break;
            case 'g':
                g_config.genderage_model = optarg;
                break;
            case 'e':
                g_config.emotion_model = optarg;
                break;
            case 't':
                g_config.face_threshold = std::stof(optarg);
                break;
            case 'm':
                g_config.mqtt_host = optarg;
                break;
            case 'p':
                g_config.mqtt_port = std::stoi(optarg);
                break;
            case 1:
                g_config.enable_rtsp = false;
                break;
            case 2:
                g_config.enable_mqtt = false;
                break;
            case 3:
                g_config.enable_blur = false;
                break;
            case 4:
                g_config.max_regions = std::stoi(optarg);
                break;
            case 'v':
                g_config.verbose = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                return false;
        }
    }
    return true;
}

static bool init_models() {
    // Initialize face detector
    g_face_detector = new FaceDetector();
    if (!g_face_detector->init(g_config.face_model)) {
        MA_LOGE(TAG, "Failed to initialize face detector");
        return false;
    }
    g_face_detector->setThreshold(g_config.face_threshold);
    MA_LOGI(TAG, "Face detector initialized (input: %dx%d)",
            g_face_detector->getInputWidth(), g_face_detector->getInputHeight());

    // Initialize attribute analyzer
    g_attribute_analyzer = new AttributeAnalyzer();
    if (!g_attribute_analyzer->init(g_config.genderage_model, g_config.emotion_model)) {
        MA_LOGE(TAG, "Failed to initialize attribute analyzer");
        return false;
    }
    MA_LOGI(TAG, "Attribute analyzer initialized (GenderAge: %s, Emotion: %s)",
            g_attribute_analyzer->isGenderAgeReady() ? "yes" : "no",
            g_attribute_analyzer->isEmotionReady() ? "yes" : "no");

    return true;
}

static bool init_camera() {
    Device* device = Device::getInstance();

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            g_camera = static_cast<Camera*>(sensor);
            g_camera->init(0);

            Camera::CtrlValue value;

            // Set channel 0 for inference
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, value);

            // Set inference resolution
            value.u16s[0] = g_config.inference_width;
            value.u16s[1] = g_config.inference_height;
            g_camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, value);

            // Disable physical address mode - attribute analysis needs CPU access to frame data
            value.i32 = 0;
            g_camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, value);

            MA_LOGI(TAG, "Camera initialized (%dx%d @ %dfps for inference)",
                    g_config.inference_width, g_config.inference_height, g_config.inference_fps);
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

    // Setup H.264 streaming channel (CH2)
    video_ch_param_t stream_param;
    stream_param.format = VIDEO_FORMAT_H264;
    stream_param.width = g_config.stream_width;
    stream_param.height = g_config.stream_height;
    stream_param.fps = g_config.stream_fps;
    setupVideo(VIDEO_CH2, &stream_param);

    // Register RTSP handler
    registerVideoFrameHandler(VIDEO_CH2, 0, fpStreamingSendToRtsp, NULL);

    // Initialize RTSP server
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

static bool init_blur() {
    if (!g_config.enable_blur) {
        return true;
    }

    if (!g_config.enable_rtsp) {
        MA_LOGW(TAG, "Face blur requires RTSP streaming, ignoring --blur");
        g_config.enable_blur = false;
        return true;
    }

    g_face_blur = new FaceBlur();
    g_face_blur->setMaxRegions(g_config.max_regions);
    if (!g_face_blur->init(g_config.stream_width, g_config.stream_height)) {
        MA_LOGE(TAG, "Failed to initialize face blur");
        delete g_face_blur;
        g_face_blur = nullptr;
        return false;
    }

    MA_LOGI(TAG, "Face blur enabled");
    return true;
}

static void cleanup() {
    if (g_face_blur) {
        g_face_blur->deinit();
        delete g_face_blur;
        g_face_blur = nullptr;
    }

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

    if (g_attribute_analyzer) {
        delete g_attribute_analyzer;
        g_attribute_analyzer = nullptr;
    }

    if (g_face_detector) {
        delete g_face_detector;
        g_face_detector = nullptr;
    }

    MA_LOGI(TAG, "Cleanup completed");
}

static void process_frame() {
    ma_img_t frame;
    if (g_camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // Step 1: Face detection
    auto detect_start = std::chrono::high_resolution_clock::now();
    std::vector<FaceInfo> faces = g_face_detector->detect(&frame);
    auto detect_end = std::chrono::high_resolution_clock::now();
    auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

    // Step 2: Feed face detections to blur overlay
    if (g_config.enable_blur && g_face_blur) {
        g_face_blur->onDetection(faces);
    }

    // Step 3: Attribute analysis for each face
    auto analyze_start = std::chrono::high_resolution_clock::now();
    std::vector<AnalyzedFace> analyzed_faces = g_attribute_analyzer->analyzeAll(&frame, faces);
    auto analyze_end = std::chrono::high_resolution_clock::now();
    auto analyze_time = std::chrono::duration_cast<std::chrono::milliseconds>(analyze_end - analyze_start).count();

    // Return frame to camera
    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Step 4: Publish results via MQTT
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        g_mqtt_publisher->publishResults(timestamp_ms, g_frame_id, analyzed_faces, static_cast<float>(total_time));
    }

    // Log results
    if (g_config.verbose || !analyzed_faces.empty()) {
        MA_LOGI(TAG, "Frame %u: %zu faces, detect=%lldms, analyze=%lldms, total=%lldms",
                g_frame_id, analyzed_faces.size(), detect_time, analyze_time, total_time);

        for (const auto& face : analyzed_faces) {
            if (face.attributes.is_fairface) {
                MA_LOGI(TAG, "  Face[%d]: age=%s, gender=%s(%.2f), race=%s(%.2f), emotion=%s(%.2f)",
                        face.face.id,
                        face.attributes.age_label.c_str(),
                        face.attributes.gender.c_str(),
                        face.attributes.gender_confidence,
                        face.attributes.race_label.c_str(),
                        face.attributes.race_confidence,
                        getEmotionName(face.attributes.emotion),
                        face.attributes.emotion_confidence);
            } else {
                MA_LOGI(TAG, "  Face[%d]: age=%s, gender=%s(%.2f), emotion=%s(%.2f)",
                        face.face.id,
                        face.attributes.age_label.c_str(),
                        face.attributes.gender.c_str(),
                        face.attributes.gender_confidence,
                        getEmotionName(face.attributes.emotion),
                        face.attributes.emotion_confidence);
            }
        }
    }

    g_frame_id++;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (!parse_args(argc, argv)) {
        return 1;
    }

    MA_LOGI(TAG, "Starting Face Analysis Application");
    MA_LOGI(TAG, "Face model: %s", g_config.face_model.c_str());
    MA_LOGI(TAG, "GenderAge model: %s", g_config.genderage_model.c_str());
    MA_LOGI(TAG, "Emotion model: %s", g_config.emotion_model.c_str());

    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize components
    if (!init_models()) {
        MA_LOGE(TAG, "Model initialization failed");
        cleanup();
        return 1;
    }

    if (!init_camera()) {
        MA_LOGE(TAG, "Camera initialization failed");
        cleanup();
        return 1;
    }

    if (!init_video_streaming()) {
        MA_LOGE(TAG, "Video streaming initialization failed");
        cleanup();
        return 1;
    }

    if (!init_mqtt()) {
        MA_LOGE(TAG, "MQTT initialization failed");
        cleanup();
        return 1;
    }

    // Start camera streaming
    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    // Start video streaming if enabled
    if (g_config.enable_rtsp) {
        startVideo();
    }

    // Initialize blur AFTER video pipeline is started (RGN needs VPSS channel running)
    if (!init_blur()) {
        MA_LOGW(TAG, "Face blur initialization failed, continuing without blur");
        g_config.enable_blur = false;
    }

    MA_LOGI(TAG, "Face analysis running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:554/live");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());

    // Main processing loop
    while (g_running.load()) {
        process_frame();
    }

    // Cleanup
    cleanup();

    MA_LOGI(TAG, "Face Analysis Application terminated");
    return 0;
}
