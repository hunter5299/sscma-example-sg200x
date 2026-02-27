#ifndef _PPOCR_MQTT_PUBLISHER_H_
#define _PPOCR_MQTT_PUBLISHER_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#include <mosquitto.h>

#include "ocr_pipeline.h"

namespace ppocr {

struct MqttConfig {
    std::string host = "localhost";
    int port = 1883;
    std::string username;
    std::string password;
    std::string client_id = "recamera-ppocr-reader";
    std::string topic = "recamera/ppocr/texts";
    int qos = 0;
    bool retain = false;
};

class MqttPublisher {
public:
    MqttPublisher();
    ~MqttPublisher();

    bool init(const MqttConfig& config);
    void deinit();
    bool isConnected() const { return connected_.load(); }

    bool publishResults(uint64_t timestamp_ms, uint32_t frame_id,
                        const std::vector<OcrResult>& results,
                        const OcrTimings& timings,
                        int frame_width, int frame_height);

    bool publish(const std::string& topic, const std::string& payload);

private:
    static void onConnectCallback(struct mosquitto* mosq, void* obj, int rc);
    static void onDisconnectCallback(struct mosquitto* mosq, void* obj, int rc);
    void onConnect(int rc);
    void onDisconnect(int rc);

    std::string buildResultJson(uint64_t timestamp_ms, uint32_t frame_id,
                                 const std::vector<OcrResult>& results,
                                 const OcrTimings& timings,
                                 int frame_width, int frame_height);

    struct mosquitto* client_;
    MqttConfig config_;
    std::atomic<bool> connected_;
    std::atomic<bool> initialized_;
};

}  // namespace ppocr

#endif  // _PPOCR_MQTT_PUBLISHER_H_
