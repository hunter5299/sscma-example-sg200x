#include "mqtt_publisher.h"

#include <sstream>
#include <iomanip>

#define TAG "MqttPublisher"

#include <sscma.h>

namespace ppocr {

MqttPublisher::MqttPublisher()
    : client_(nullptr),
      connected_(false),
      initialized_(false) {}

MqttPublisher::~MqttPublisher() {
    deinit();
}

bool MqttPublisher::init(const MqttConfig& config) {
    if (initialized_.load()) return true;

    config_ = config;
    mosquitto_lib_init();

    client_ = mosquitto_new(config_.client_id.c_str(), true, this);
    if (!client_) {
        MA_LOGE(TAG, "Failed to create MQTT client");
        return false;
    }

    mosquitto_connect_callback_set(client_, onConnectCallback);
    mosquitto_disconnect_callback_set(client_, onDisconnectCallback);

    if (!config_.username.empty() && !config_.password.empty()) {
        mosquitto_username_pw_set(client_, config_.username.c_str(), config_.password.c_str());
    }

    mosquitto_reconnect_delay_set(client_, 2, 30, true);

    int rc = mosquitto_loop_start(client_);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Failed to start MQTT loop: %d", rc);
        mosquitto_destroy(client_);
        client_ = nullptr;
        return false;
    }

    rc = mosquitto_connect(client_, config_.host.c_str(), config_.port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGW(TAG, "Initial connection failed (will retry): %d", rc);
    }

    initialized_.store(true);
    MA_LOGI(TAG, "MQTT publisher initialized: %s:%d topic=%s",
            config_.host.c_str(), config_.port, config_.topic.c_str());

    return true;
}

void MqttPublisher::deinit() {
    if (!initialized_.load()) return;

    if (client_) {
        if (connected_.load()) mosquitto_disconnect(client_);
        mosquitto_loop_stop(client_, true);
        mosquitto_destroy(client_);
        client_ = nullptr;
    }

    mosquitto_lib_cleanup();
    initialized_.store(false);
    connected_.store(false);
}

void MqttPublisher::onConnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    auto* self = static_cast<MqttPublisher*>(obj);
    if (self) self->onConnect(rc);
}

void MqttPublisher::onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    auto* self = static_cast<MqttPublisher*>(obj);
    if (self) self->onDisconnect(rc);
}

void MqttPublisher::onConnect(int rc) {
    if (rc == 0) {
        connected_.store(true);
        MA_LOGI(TAG, "Connected to MQTT broker");
    } else {
        MA_LOGE(TAG, "Connection failed: %d", rc);
    }
}

void MqttPublisher::onDisconnect(int rc) {
    connected_.store(false);
    if (rc != 0) MA_LOGW(TAG, "Unexpected disconnect: %d", rc);
}

bool MqttPublisher::publish(const std::string& topic, const std::string& payload) {
    if (!initialized_.load() || !client_) return false;

    int rc = mosquitto_publish(client_, nullptr, topic.c_str(),
                                static_cast<int>(payload.size()),
                                payload.data(), config_.qos, config_.retain);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Publish failed: %d", rc);
        return false;
    }
    return true;
}

// Escape special JSON characters in a string
static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string MqttPublisher::buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<OcrResult>& results,
                                            const OcrTimings& timings,
                                            int frame_width, int frame_height) {
    std::ostringstream json;
    json << std::fixed;

    float inv_w = (frame_width > 0) ? (1.0f / frame_width) : 1.0f;
    float inv_h = (frame_height > 0) ? (1.0f / frame_height) : 1.0f;

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":{";
    json << "\"detection\":" << std::setprecision(1) << timings.detection_ms << ",";
    json << "\"recognition\":" << std::setprecision(1) << timings.recognition_ms << ",";
    json << "\"total\":" << std::setprecision(1) << timings.total_ms;
    json << "},";
    json << "\"text_count\":" << results.size() << ",";
    json << "\"frame_width\":1,";
    json << "\"frame_height\":1,";
    json << "\"texts\":[";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        if (i > 0) json << ",";
        json << "{";
        json << "\"id\":" << i << ",";

        // Box as 4 normalized points [[x,y],[x,y],[x,y],[x,y]]
        json << "\"box\":[";
        for (int p = 0; p < 4; ++p) {
            if (p > 0) json << ",";
            json << "[" << std::setprecision(4) << r.box.points[p][0] * inv_w
                 << "," << r.box.points[p][1] * inv_h << "]";
        }
        json << "],";

        json << "\"text\":\"" << jsonEscape(r.text) << "\",";
        json << "\"confidence\":" << std::setprecision(3) << r.rec_confidence << ",";
        json << "\"det_confidence\":" << std::setprecision(3) << r.det_confidence;
        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

bool MqttPublisher::publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                                    const std::vector<OcrResult>& results,
                                    const OcrTimings& timings,
                                    int frame_width, int frame_height) {
    std::string payload = buildResultJson(timestamp_ms, frame_id, results, timings, frame_width, frame_height);
    return publish(config_.topic, payload);
}

}  // namespace ppocr
