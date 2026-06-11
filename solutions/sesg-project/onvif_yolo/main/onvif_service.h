#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace onvif {

// 设备静态信息（用于 ONVIF GetDeviceInformation 等响应）。
struct DeviceInfo {
    std::string manufacturer = "Seeed Studio";
    std::string model        = "reCamera";
    std::string firmware     = "1.0-onvif-yolo";
    std::string serial       = "RECAMERA-0001";
    std::string hardware     = "SG2002";
};

// ONVIF 服务配置。
struct OnvifConfig {
    std::string device_ip;            // 对外可达 IP（用于回填到 XAddr / Stream URI）
    int onvif_port   = 8080;          // ONVIF SOAP HTTP 服务端口（避开设备 80 端口 Web UI）
    int rtsp_port    = 8554;          // RTSP 推流端口
    std::string rtsp_session = "live"; // RTSP path：rtsp://<ip>:<rtsp_port>/<session>
    uint32_t width   = 1280;
    uint32_t height  = 720;
    uint32_t fps     = 25;
    DeviceInfo info;
};

// 单进程内提供两类 ONVIF 能力：
//  1) WS-Discovery：监听 UDP 组播 239.255.255.250:3702，响应 Probe，让客户端发现本机。
//  2) Device/Media SOAP：基于 mongoose 的 HTTP 服务，处理 GetCapabilities /
//     GetDeviceInformation / GetProfiles / GetStreamUri 等核心接口。
// 两者均在后台线程运行，start() 后立即返回。
class OnvifService {
public:
    OnvifService();
    ~OnvifService();

    OnvifService(const OnvifService&) = delete;
    OnvifService& operator=(const OnvifService&) = delete;

    bool start(const OnvifConfig& cfg);
    void stop();
    bool isRunning() const { return running_.load(); }

private:
    void discoveryLoop();   // WS-Discovery UDP 线程
    void httpLoop();        // mongoose HTTP/SOAP 线程

    OnvifConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread disc_thread_;
    std::thread http_thread_;
    int disc_fd_ = -1;
};

} // namespace onvif
