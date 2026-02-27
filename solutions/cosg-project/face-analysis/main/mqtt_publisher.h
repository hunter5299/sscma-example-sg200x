#ifndef _MQTT_PUBLISHER_H_
#define _MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

#include "attribute_analyzer.h"

namespace face_analysis {

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-face-analysis";
    std::string topic = "recamera/face-analysis/results";
    int qos = 0;
    bool retain = false;
};

class MqttPublisher {
public:
    MqttPublisher();
    ~MqttPublisher();

    // Initialize and connect to MQTT broker
    bool init(const MqttConfig& config);

    // Disconnect and cleanup
    void deinit();

    // Check if connected
    bool isConnected() const { return connected_.load(); }

    // Publish face analysis results
    // timestamp_ms: frame timestamp in milliseconds
    // frame_id: sequential frame counter
    // faces: analyzed faces with attributes
    // inference_time_ms: total inference time for this frame
    bool publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                        const std::vector<AnalyzedFace>& faces,
                        float inference_time_ms);

    // Publish raw JSON message
    bool publish(const std::string& topic, const std::string& payload);

private:
    // MQTT callbacks
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);

    void onConnect(int rc);
    void onDisconnect(int rc);

    // Build JSON result string
    std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<AnalyzedFace>& faces,
                                 float inference_time_ms);

private:
    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace face_analysis

#endif  // _MQTT_PUBLISHER_H_
