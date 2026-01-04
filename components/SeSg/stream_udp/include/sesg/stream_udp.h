#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sscma.h>

#include <sesg/udp_service.h>

// VPSS 封装可选：默认不启用，避免把 sophgo/VPSS 链路强制拉进所有使用者。
// 如需启用，请在构建时打开 CMake 选项：SESG_STREAM_UDP_ENABLE_VPSS=ON
#if defined(SESG_STREAM_UDP_ENABLE_VPSS)
#include <sesg/vpss.h>
#endif

namespace sesg::stream_udp {

struct JpegChannelConfig {
    int ch = 1;
    int width = 320;
    int height = 240;
    int fps = 10;
};

struct JpegFrameOwned {
    std::unique_ptr<uint8_t[]> buf;
    size_t size = 0;
    int w = 0;
    int h = 0;
};

class SesgJpegUdpStreamer {
public:
    SesgJpegUdpStreamer(const std::string& ip, uint16_t port, udp_service::SenderOptions opt = {});
    ~SesgJpegUdpStreamer();

    SesgJpegUdpStreamer(const SesgJpegUdpStreamer&) = delete;
    SesgJpegUdpStreamer& operator=(const SesgJpegUdpStreamer&) = delete;

    // 建议在 camera->startStream() 之前调用
    static void configureJpegChannel(ma::Camera& camera, const JpegChannelConfig& cfg);

    // 若 init_vpss=true：内部会调用 sesg_vpss_init()。
    // 注意：vpss 初始化不是幂等的，如果你的系统已经初始化过 VPSS，请传 false。
    bool start(std::shared_ptr<ma::Camera> camera, const JpegChannelConfig& cfg, bool init_vpss = false);

    // 若 deinit_vpss=true：内部会调用 sesg_vpss_deinit()。
    // 注意：如果 VPSS 被其它模块共享，请不要 deinit。
    void stop(bool deinit_vpss = false);

    bool isRunning() const;

    // 在主循环里调用：发送“最近一帧 JPEG + 检测结果”。
    template <class T>
    double sendLatest(uint32_t magic, const std::vector<T>& detections) {
        JpegFrameOwned jpeg_owned;
        {
            std::lock_guard<std::mutex> lk(jpeg_mu_);
            if (latest_jpeg_.buf) {
                jpeg_owned = std::move(latest_jpeg_);
            }
        }

        if (!jpeg_owned.buf || jpeg_owned.size == 0) {
            return -1.0;
        }

        return sender_.sendJpegOwned(magic,
                                    std::move(jpeg_owned.buf),
                                    jpeg_owned.size,
                                    jpeg_owned.w,
                                    jpeg_owned.h,
                                    detections);
    }

    double sendLatestRaw(uint32_t magic, const void* det_buf, size_t det_bytes, uint32_t det_count);

    float getSendFPS() const;

private:
    void jpegLoop();

    sesg::udp_service::SesgUDPSender sender_;

    std::shared_ptr<ma::Camera> camera_;
    JpegChannelConfig cfg_;

    std::atomic<bool> running_{false};

    std::mutex jpeg_mu_;
    JpegFrameOwned latest_jpeg_;
    std::thread jpeg_thread_;

    bool vpss_inited_ = false;
};

} // namespace sesg::stream_udp
