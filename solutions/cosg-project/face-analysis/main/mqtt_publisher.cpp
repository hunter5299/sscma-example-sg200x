#include "mqtt_publisher.h"

#include <sstream>
#include <iomanip>
#include <ctime>

#define TAG "MqttPublisher"

// Use SSCMA logging macros
#include <sscma.h>

namespace face_analysis {

MqttPublisher::MqttPublisher()
    : client_(nullptr),
      connected_(false),
      initialized_(false) {}

MqttPublisher::~MqttPublisher() {
    deinit();
}

bool MqttPublisher::init(const MqttConfig& config) {
    if (initialized_.load()) {
        MA_LOGW(TAG, "Already initialized");
        return true;
    }

    config_ = config;

    // Initialize mosquitto library
    mosquitto_lib_init();

    // Create client
    client_ = mosquitto_new(config_.client_id.c_str(), true, this);
    if (!client_) {
        MA_LOGE(TAG, "Failed to create MQTT client");
        return false;
    }

    // Set callbacks
    mosquitto_connect_callback_set(client_, onConnectCallback);
    mosquitto_disconnect_callback_set(client_, onDisconnectCallback);

    // Set credentials if provided
    if (!config_.username.empty() && !config_.password.empty()) {
        mosquitto_username_pw_set(client_, config_.username.c_str(),
                                   config_.password.c_str());
    }

    // Set reconnect delay
    mosquitto_reconnect_delay_set(client_, 2, 30, true);

    // Start network loop thread
    int rc = mosquitto_loop_start(client_);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Failed to start MQTT loop: %d", rc);
        mosquitto_destroy(client_);
        client_ = nullptr;
        return false;
    }

    // Connect to broker
    rc = mosquitto_connect(client_, config_.host.c_str(), config_.port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGW(TAG, "Initial connection failed (will retry): %d", rc);
        // Don't return false - mosquitto will auto-reconnect
    }

    initialized_.store(true);
    MA_LOGI(TAG, "MQTT publisher initialized: %s:%d", config_.host.c_str(), config_.port);
    MA_LOGI(TAG, "Publishing to topic: %s", config_.topic.c_str());

    return true;
}

void MqttPublisher::deinit() {
    if (!initialized_.load()) {
        return;
    }

    if (client_) {
        if (connected_.load()) {
            mosquitto_disconnect(client_);
        }
        mosquitto_loop_stop(client_, true);
        mosquitto_destroy(client_);
        client_ = nullptr;
    }

    mosquitto_lib_cleanup();
    initialized_.store(false);
    connected_.store(false);

    MA_LOGI(TAG, "MQTT publisher deinitialized");
}

void MqttPublisher::onConnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    MqttPublisher* self = static_cast<MqttPublisher*>(obj);
    if (self) {
        self->onConnect(rc);
    }
}

void MqttPublisher::onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc) {
    MqttPublisher* self = static_cast<MqttPublisher*>(obj);
    if (self) {
        self->onDisconnect(rc);
    }
}

void MqttPublisher::onConnect(int rc) {
    if (rc == 0) {
        connected_.store(true);
        MA_LOGI(TAG, "Connected to MQTT broker");
    } else {
        MA_LOGE(TAG, "Connection failed with code: %d", rc);
    }
}

void MqttPublisher::onDisconnect(int rc) {
    connected_.store(false);
    if (rc != 0) {
        MA_LOGW(TAG, "Unexpected disconnect: %d", rc);
    } else {
        MA_LOGI(TAG, "Disconnected from MQTT broker");
    }
}

bool MqttPublisher::publish(const std::string& topic, const std::string& payload) {
    if (!initialized_.load() || !client_) {
        return false;
    }

    if (!connected_.load()) {
        MA_LOGV(TAG, "Not connected, message will be queued");
    }

    int rc = mosquitto_publish(client_, nullptr, topic.c_str(),
                                static_cast<int>(payload.size()),
                                payload.data(), config_.qos, config_.retain);

    if (rc != MOSQ_ERR_SUCCESS) {
        MA_LOGE(TAG, "Publish failed: rc=%d, connected=%d, topic=%s, size=%d",
                rc, connected_.load() ? 1 : 0, topic.c_str(), (int)payload.size());
        return false;
    }

    MA_LOGD(TAG, "Published %d bytes to %s", (int)payload.size(), topic.c_str());
    return true;
}

std::string MqttPublisher::buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                            const std::vector<AnalyzedFace>& faces,
                                            float inference_time_ms) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);

    json << "{";
    json << "\"timestamp\":" << timestamp_ms << ",";
    json << "\"frame_id\":" << frame_id << ",";
    json << "\"inference_time_ms\":" << inference_time_ms << ",";
    json << "\"face_count\":" << faces.size() << ",";
    json << "\"faces\":[";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        const auto& attrs = face.attributes;

        if (i > 0) json << ",";

        json << "{";
        json << "\"id\":" << face.face.id << ",";

        // Bounding box (normalized coordinates)
        json << "\"bbox\":{";
        json << "\"x\":" << face.face.x << ",";
        json << "\"y\":" << face.face.y << ",";
        json << "\"w\":" << face.face.w << ",";
        json << "\"h\":" << face.face.h;
        json << "},";

        json << "\"confidence\":" << face.face.score << ",";

        // Age (format depends on model)
        if (attrs.is_fairface) {
            json << "\"age_bin\":" << attrs.age_bin << ",";
        } else {
            json << "\"age\":" << attrs.age_continuous << ",";
        }
        json << "\"age_label\":\"" << attrs.age_label << "\",";
        json << "\"age_confidence\":" << attrs.age_confidence << ",";

        // Gender
        json << "\"gender\":\"" << attrs.gender << "\",";
        json << "\"gender_confidence\":" << attrs.gender_confidence << ",";

        // Race (FairFace only)
        if (attrs.race_bin >= 0) {
            json << "\"race\":\"" << attrs.race_label << "\",";
            json << "\"race_confidence\":" << attrs.race_confidence << ",";
        }

        // Emotion
        json << "\"emotion\":\"" << getEmotionName(attrs.emotion) << "\",";
        json << "\"emotion_confidence\":" << attrs.emotion_confidence << ",";

        // All emotion probabilities (7 classes)
        json << "\"emotion_probs\":{";
        json << "\"angry\":" << attrs.emotion_probs[0] << ",";
        json << "\"disgust\":" << attrs.emotion_probs[1] << ",";
        json << "\"fear\":" << attrs.emotion_probs[2] << ",";
        json << "\"happy\":" << attrs.emotion_probs[3] << ",";
        json << "\"sad\":" << attrs.emotion_probs[4] << ",";
        json << "\"surprise\":" << attrs.emotion_probs[5] << ",";
        json << "\"neutral\":" << attrs.emotion_probs[6];
        json << "}";

        json << "}";
    }

    json << "]";
    json << "}";

    return json.str();
}

bool MqttPublisher::publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                                    const std::vector<AnalyzedFace>& faces,
                                    float inference_time_ms) {
    std::string payload = buildResultJson(timestamp_ms, frame_id, faces, inference_time_ms);
    return publish(config_.topic, payload);
}

}  // namespace face_analysis
