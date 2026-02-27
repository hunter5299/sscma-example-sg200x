#ifndef _MQTT_PUBLISHER_H_
#define _MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

#include "yolo26_detector.h"
#include "person_tracker.h"

namespace yolo26 {

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-yolo26-detector";
    std::string topic = "recamera/yolo26/detections";
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

    // Publish detection results (basic format)
    // timestamp_ms: frame timestamp in milliseconds
    // frame_id: sequential frame counter
    // detections: list of detected objects
    // inference_time_ms: total inference time for this frame
    bool publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                        const std::vector<Detection>& detections,
                        float inference_time_ms);

    // Publish tracking results with person state info
    // persons: tracked persons with velocity and dwell state
    // counts: zone occupancy summary
    bool publishTrackingResults(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<TrackedPerson>& persons,
                                 const StateCount& counts,
                                 float inference_time_ms);

    // Publish raw JSON message
    bool publish(const std::string& topic, const std::string& payload);

private:
    // MQTT callbacks
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);

    void onConnect(int rc);
    void onDisconnect(int rc);

    // Build JSON result string (basic detection format)
    std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<Detection>& detections,
                                 float inference_time_ms);

    // Build JSON result string (tracking format with person state)
    std::string buildTrackingJson(uint64_t timestamp_ms, uint32_t frame_id,
                                   const std::vector<TrackedPerson>& persons,
                                   const StateCount& counts,
                                   float inference_time_ms);

private:
    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace yolo26

#endif  // _MQTT_PUBLISHER_H_
