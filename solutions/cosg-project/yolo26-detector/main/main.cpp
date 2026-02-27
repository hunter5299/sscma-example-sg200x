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

#include "yolo26_detector.h"
#include "person_tracker.h"
#include "mqtt_publisher.h"

using namespace ma;
using namespace yolo26;

#define TAG "yolo26-detector"

// Default configuration
static struct {
    // Model path
    std::string model_path = "/userdata/local/models/yolo26n_cv181x_int8.cvimodel";

    // Detection parameters
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;

    // MQTT configuration
    std::string mqtt_host = "localhost";
    int mqtt_port = 1883;
    std::string mqtt_topic = "recamera/yolo26/detections";

    // Video configuration
    int inference_width = 640;
    int inference_height = 640;
    int inference_fps = 15;
    int stream_width = 1280;
    int stream_height = 720;
    int stream_fps = 15;

    // Tracking configuration
    bool enable_tracking = true;
    float dwell_speed_threshold = 10.0f;    // px/s - below this = stationary
    float dwell_min_duration = 1.5f;        // seconds for ENGAGED state
    float dwell_assistance_threshold = 20.0f; // seconds for ASSISTANCE state

    // Runtime flags
    bool enable_rtsp = true;
    bool enable_mqtt = true;
    bool verbose = false;
} g_config;

// Global state
static std::atomic<bool> g_running(true);
static Yolo26Detector* g_detector = nullptr;
static PersonTracker* g_tracker = nullptr;
static MqttPublisher* g_mqtt_publisher = nullptr;
static Camera* g_camera = nullptr;
static uint32_t g_frame_id = 0;
static std::chrono::steady_clock::time_point g_start_time;

static void signal_handler(int sig) {
    MA_LOGI(TAG, "Received signal %d, shutting down...", sig);
    g_running.store(false);
}

static void print_usage(const char* prog) {
    printf("YOLO26 Object Detector for ReCamera\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model PATH          Model path (default: %s)\n", g_config.model_path.c_str());
    printf("  -c, --conf-threshold FLOAT Confidence threshold (default: %.2f)\n", g_config.conf_threshold);
    printf("  -n, --nms-threshold FLOAT  NMS threshold (default: %.2f)\n", g_config.nms_threshold);
    printf("  --mqtt-host HOST          MQTT broker host (default: %s)\n", g_config.mqtt_host.c_str());
    printf("  --mqtt-port PORT          MQTT broker port (default: %d)\n", g_config.mqtt_port);
    printf("  --mqtt-topic TOPIC        MQTT topic (default: %s)\n", g_config.mqtt_topic.c_str());
    printf("  --no-tracking             Disable person tracking\n");
    printf("  --dwell-speed FLOAT       Dwell speed threshold in px/s (default: %.1f)\n", g_config.dwell_speed_threshold);
    printf("  --dwell-engaged FLOAT     Time for ENGAGED state in sec (default: %.1f)\n", g_config.dwell_min_duration);
    printf("  --dwell-assist FLOAT      Time for ASSISTANCE state in sec (default: %.1f)\n", g_config.dwell_assistance_threshold);
    printf("  --no-rtsp                 Disable RTSP streaming\n");
    printf("  --no-mqtt                 Disable MQTT publishing\n");
    printf("  -v, --verbose             Enable verbose logging\n");
    printf("  -h, --help                Show this help message\n");
    printf("\n");
    printf("RTSP Stream: rtsp://<device_ip>:8554/live0\n");
    printf("\n");
    printf("Dwell States:\n");
    printf("  TRANSIENT  - Person is moving\n");
    printf("  DWELLING   - Person stopped < %.1fs\n", g_config.dwell_min_duration);
    printf("  ENGAGED    - Person stopped %.1f-%.1fs\n", g_config.dwell_min_duration, g_config.dwell_assistance_threshold);
    printf("  ASSISTANCE - Person stopped > %.1fs\n", g_config.dwell_assistance_threshold);
}

static bool parse_args(int argc, char** argv) {
    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"conf-threshold", required_argument, 0, 'c'},
        {"nms-threshold", required_argument, 0, 'n'},
        {"mqtt-host", required_argument, 0, 1},
        {"mqtt-port", required_argument, 0, 2},
        {"mqtt-topic", required_argument, 0, 3},
        {"no-tracking", no_argument, 0, 4},
        {"dwell-speed", required_argument, 0, 5},
        {"dwell-engaged", required_argument, 0, 6},
        {"dwell-assist", required_argument, 0, 7},
        {"no-rtsp", no_argument, 0, 8},
        {"no-mqtt", no_argument, 0, 9},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:n:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                g_config.model_path = optarg;
                break;
            case 'c':
                g_config.conf_threshold = std::stof(optarg);
                break;
            case 'n':
                g_config.nms_threshold = std::stof(optarg);
                break;
            case 1:
                g_config.mqtt_host = optarg;
                break;
            case 2:
                g_config.mqtt_port = std::stoi(optarg);
                break;
            case 3:
                g_config.mqtt_topic = optarg;
                break;
            case 4:
                g_config.enable_tracking = false;
                break;
            case 5:
                g_config.dwell_speed_threshold = std::stof(optarg);
                break;
            case 6:
                g_config.dwell_min_duration = std::stof(optarg);
                break;
            case 7:
                g_config.dwell_assistance_threshold = std::stof(optarg);
                break;
            case 8:
                g_config.enable_rtsp = false;
                break;
            case 9:
                g_config.enable_mqtt = false;
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

static bool init_detector() {
    g_detector = new Yolo26Detector();
    if (!g_detector->init(g_config.model_path)) {
        MA_LOGE(TAG, "Failed to initialize YOLO26 detector");
        return false;
    }

    g_detector->setConfThreshold(g_config.conf_threshold);
    g_detector->setNmsThreshold(g_config.nms_threshold);

    MA_LOGI(TAG, "YOLO26 detector initialized (input: %dx%d)",
            g_detector->getInputWidth(), g_detector->getInputHeight());

    return true;
}

static bool init_tracker() {
    if (!g_config.enable_tracking) {
        MA_LOGI(TAG, "Person tracking disabled");
        return true;
    }

    g_tracker = new PersonTracker();

    TrackerConfig tracker_config;
    tracker_config.dwell_speed_threshold = g_config.dwell_speed_threshold;
    tracker_config.dwell_min_duration = g_config.dwell_min_duration;
    tracker_config.dwell_assistance_threshold = g_config.dwell_assistance_threshold;
    tracker_config.frame_width = g_config.inference_width;
    tracker_config.frame_height = g_config.inference_height;

    g_tracker->setConfig(tracker_config);

    MA_LOGI(TAG, "Person tracker initialized");
    MA_LOGI(TAG, "  Dwell speed threshold: %.1f px/s", g_config.dwell_speed_threshold);
    MA_LOGI(TAG, "  Engaged duration: %.1fs", g_config.dwell_min_duration);
    MA_LOGI(TAG, "  Assistance duration: %.1fs", g_config.dwell_assistance_threshold);

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

            // Disable physical address mode - CPU-based image processing needs virtual addresses
            // Physical mode causes segfault when ma::cv::convert() tries to access physical memory
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

    if (g_tracker) {
        delete g_tracker;
        g_tracker = nullptr;
    }

    if (g_detector) {
        delete g_detector;
        g_detector = nullptr;
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

    // Calculate elapsed time since start (for tracker)
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time
    ).count();
    float current_time_sec = elapsed / 1000.0f;

    // Run YOLO26 detection
    std::vector<Detection> detections = g_detector->detect(&frame);

    // Return frame to camera
    g_camera->returnFrame(frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    // Publish results via MQTT
    if (g_config.enable_mqtt && g_mqtt_publisher) {
        if (g_config.enable_tracking && g_tracker) {
            // Update tracker with detections
            auto tracked_persons = g_tracker->update(detections, current_time_sec);
            auto state_counts = g_tracker->getStateCounts();

            // Publish tracking results
            g_mqtt_publisher->publishTrackingResults(timestamp_ms, g_frame_id,
                                                      tracked_persons, state_counts,
                                                      static_cast<float>(inference_time));

            // Log tracking results
            if (g_config.verbose || !tracked_persons.empty()) {
                MA_LOGI(TAG, "Frame %u: %zu persons tracked, inference=%lldms",
                        g_frame_id, tracked_persons.size(), inference_time);
                MA_LOGI(TAG, "  Zone: total=%d, browsing=%d, engaged=%d, assistance=%d",
                        state_counts.total, state_counts.browsing,
                        state_counts.engaged, state_counts.assistance);

                for (const auto& person : tracked_persons) {
                    MA_LOGI(TAG, "  [T%d] %s (%.1f%%) speed=%.1f px/s, dwell=%.1fs",
                            person.track_id,
                            getDwellStateName(person.dwell_state),
                            person.detection.confidence * 100.0f,
                            person.speed_px_s,
                            person.dwell_duration_sec);
                }
            }
        } else {
            // Publish basic detection results (no tracking)
            g_mqtt_publisher->publishResults(timestamp_ms, g_frame_id, detections,
                                              static_cast<float>(inference_time));

            // Log detection results
            if (g_config.verbose || !detections.empty()) {
                MA_LOGI(TAG, "Frame %u: %zu detections, inference=%lldms",
                        g_frame_id, detections.size(), inference_time);

                for (const auto& det : detections) {
                    MA_LOGI(TAG, "  [%d] %s (%.1f%%) at (%.2f, %.2f, %.2f, %.2f)",
                            det.id,
                            Yolo26Detector::getClassName(det.class_id),
                            det.confidence * 100.0f,
                            det.x, det.y, det.w, det.h);
                }
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

    MA_LOGI(TAG, "Starting YOLO26 Object Detector");
    MA_LOGI(TAG, "Model: %s", g_config.model_path.c_str());
    MA_LOGI(TAG, "Confidence threshold: %.2f", g_config.conf_threshold);
    MA_LOGI(TAG, "NMS threshold: %.2f", g_config.nms_threshold);

    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Record start time for tracker
    g_start_time = std::chrono::steady_clock::now();

    // Initialize components
    if (!init_detector()) {
        MA_LOGE(TAG, "Detector initialization failed");
        cleanup();
        return 1;
    }

    if (!init_tracker()) {
        MA_LOGE(TAG, "Tracker initialization failed");
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
    // Note: g_camera->startStream() internally calls startVideo() via CAMERA_INIT()
    // which initializes VI, VPSS, and VENC. Do NOT call startVideo() again
    // as it will cause VPSS double initialization and kernel crash.
    g_camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    MA_LOGI(TAG, "YOLO26 detector running...");
    MA_LOGI(TAG, "RTSP stream: rtsp://<device_ip>:8554/live0");
    MA_LOGI(TAG, "MQTT topic: %s", g_config.mqtt_topic.c_str());
    if (g_config.enable_tracking) {
        MA_LOGI(TAG, "Person tracking: ENABLED");
    } else {
        MA_LOGI(TAG, "Person tracking: DISABLED");
    }

    // Main processing loop
    while (g_running.load()) {
        process_frame();
    }

    // Cleanup
    cleanup();

    MA_LOGI(TAG, "YOLO26 Object Detector terminated");
    return 0;
}
