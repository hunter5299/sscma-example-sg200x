#include <sesg/stream_udp.h>

#include <chrono>

namespace sesg::stream_udp {

SesgJpegUdpStreamer::SesgJpegUdpStreamer(const std::string& ip, uint16_t port, udp_service::SenderOptions opt)
    : sender_(ip.c_str(), static_cast<int>(port), opt) {}

SesgJpegUdpStreamer::~SesgJpegUdpStreamer() {
    stop(false);
}

void SesgJpegUdpStreamer::configureJpegChannel(ma::Camera& camera, const JpegChannelConfig& cfg) {
    ma::Camera::CtrlValue val;

    val.i32 = cfg.ch;
    camera.commandCtrl(ma::Camera::CtrlType::kChannel, ma::Camera::CtrlMode::kWrite, val);

    val.u16s[0] = static_cast<uint16_t>(cfg.width);
    val.u16s[1] = static_cast<uint16_t>(cfg.height);
    camera.commandCtrl(ma::Camera::CtrlType::kWindow, ma::Camera::CtrlMode::kWrite, val);

    val.i32 = static_cast<int>(MA_PIXEL_FORMAT_JPEG);
    camera.commandCtrl(ma::Camera::CtrlType::kFormat, ma::Camera::CtrlMode::kWrite, val);

    val.i32 = cfg.fps;
    camera.commandCtrl(ma::Camera::CtrlType::kFps, ma::Camera::CtrlMode::kWrite, val);
}

bool SesgJpegUdpStreamer::start(std::shared_ptr<ma::Camera> camera, const JpegChannelConfig& cfg, bool init_vpss) {
    if (running_.load()) {
        return true;
    }

    camera_ = std::move(camera);
    cfg_ = cfg;

    if (!camera_) {
        return false;
    }

    if (init_vpss) {
#if defined(SESG_STREAM_UDP_ENABLE_VPSS)
        const int ret = sesg_vpss_init();
        if (ret != 0) {
            return false;
        }
        vpss_inited_ = true;
#else
        (void)init_vpss;
        return false;
#endif
    }

    sender_.start();

    running_.store(true);
    jpeg_thread_ = std::thread(&SesgJpegUdpStreamer::jpegLoop, this);

    return true;
}

void SesgJpegUdpStreamer::stop(bool deinit_vpss) {
    if (!running_.exchange(false)) {
        return;
    }

    if (jpeg_thread_.joinable()) {
        jpeg_thread_.join();
    }

    sender_.stop();

    {
        std::lock_guard<std::mutex> lk(jpeg_mu_);
        latest_jpeg_ = {};
    }

    camera_.reset();

    if (deinit_vpss && vpss_inited_) {
#if defined(SESG_STREAM_UDP_ENABLE_VPSS)
        sesg_vpss_deinit();
        vpss_inited_ = false;
#else
        (void)deinit_vpss;
#endif
    }
}

bool SesgJpegUdpStreamer::isRunning() const {
    return running_.load();
}

double SesgJpegUdpStreamer::sendLatestRaw(uint32_t magic, const void* det_buf, size_t det_bytes, uint32_t det_count) {
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

    return sender_.sendJpegOwnedRaw(magic,
                                   std::move(jpeg_owned.buf),
                                   jpeg_owned.size,
                                   jpeg_owned.w,
                                   jpeg_owned.h,
                                   det_buf,
                                   det_bytes,
                                   det_count);
}

float SesgJpegUdpStreamer::getSendFPS() const {
    return sender_.getSendFPS();
}

void SesgJpegUdpStreamer::jpegLoop() {
    ma_img_t jpeg;
    while (running_.load()) {
        if (!camera_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (camera_->retrieveFrame(jpeg, MA_PIXEL_FORMAT_JPEG) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        std::unique_ptr<uint8_t[]> jpeg_buf(reinterpret_cast<uint8_t*>(jpeg.data));
        const size_t jpeg_size = jpeg.size;
        const int jpeg_w = jpeg.width;
        const int jpeg_h = jpeg.height;

        // 归还相机队列占用（但不释放我们接管的 buffer）
        jpeg.data = nullptr;
        jpeg.size = 0;
        camera_->returnFrame(jpeg);

        {
            std::lock_guard<std::mutex> lk(jpeg_mu_);
            latest_jpeg_.buf = std::move(jpeg_buf);
            latest_jpeg_.size = jpeg_size;
            latest_jpeg_.w = jpeg_w;
            latest_jpeg_.h = jpeg_h;
        }
    }
}

} // namespace sesg::stream_udp
