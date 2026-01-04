#include <udp_service/udp_service.h>

#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

extern "C" {
#include <cvi_sys.h>
}

namespace udp_service {

struct UDPSender::FrameBuffer {
    std::unique_ptr<uint8_t[]> owned;
    size_t owned_size = 0;
    bool use_owned = false;

    std::vector<uint8_t> copied; // for physical->cpu copy
    size_t copied_size = 0;

    std::vector<uint8_t> det_bytes;
    uint32_t det_count = 0;

    int width = 0;
    int height = 0;
    PayloadFormat format = PayloadFormat::RGB888;
    uint32_t magic = 0;

    std::atomic<bool> ready{false};

    const uint8_t* payloadPtr() const {
        return use_owned ? owned.get() : copied.data();
    }

    size_t payloadSize() const {
        return use_owned ? owned_size : copied_size;
    }

    void resetPayload() {
        use_owned = false;
        owned.reset();
        owned_size = 0;
        copied_size = 0;
    }
};

UDPSender::UDPSender(const char* ip, int port, SenderOptions opt)
    : ip_(ip), port_(port), sock_(-1), opt_(opt), frame_buf_(new FrameBuffer[2]), write_idx_(0), running_(false),
      send_fps_(0) {
    for (int i = 0; i < 2; ++i) {
        frame_buf_[i].copied.resize(opt_.max_copy_bytes);
    }
}

UDPSender::~UDPSender() {
    stop();
}

float UDPSender::getSendFPS() const {
    return send_fps_.load();
}

void UDPSender::start() {
    if (running_.load()) {
        return;
    }

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0) {
        printf("[UDP] socket \u521b\u5efa\u5931\u8d25\n");
        return;
    }

    running_.store(true);
    thread_ = std::thread(&UDPSender::threadFunc, this);
    printf("[UDP] \u670d\u52a1\u542f\u52a8: %s:%d\n", ip_, port_);
}

void UDPSender::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    if (thread_.joinable()) {
        thread_.join();
    }

    if (sock_ >= 0) {
        close(sock_);
        sock_ = -1;
    }
    printf("[UDP] \u670d\u52a1\u505c\u6b62\n");
}

static inline void appendBytes(std::vector<uint8_t>& out, const void* data, size_t len) {
    if (!data || len == 0) {
        return;
    }
    const size_t old = out.size();
    out.resize(old + len);
    memcpy(out.data() + old, data, len);
}

double UDPSender::sendJpegOwnedRaw(uint32_t magic,
                                  std::unique_ptr<uint8_t[]> jpeg_data,
                                  size_t jpeg_size,
                                  int width,
                                  int height,
                                  const void* det_bytes,
                                  size_t det_bytes_size,
                                  uint32_t det_count) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int w_idx = write_idx_.load();
    FrameBuffer& buf = frame_buf_[w_idx];

    if (buf.ready.load()) {
        return 0;
    }

    if (!jpeg_data || jpeg_size == 0) {
        return 0;
    }

    buf.magic = magic;
    buf.width = width;
    buf.height = height;
    buf.format = PayloadFormat::JPEG;

    buf.owned = std::move(jpeg_data);
    buf.owned_size = jpeg_size;
    buf.use_owned = true;

    buf.det_count = det_count;
    buf.det_bytes.clear();
    appendBytes(buf.det_bytes, det_bytes, det_bytes_size);

    buf.ready.store(true);
    write_idx_.store(1 - w_idx);

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

double UDPSender::sendRgb888PhysicalRaw(uint32_t magic,
                                       void* frame_phy_addr,
                                       size_t frame_size,
                                       int width,
                                       int height,
                                       const void* det_bytes,
                                       size_t det_bytes_size,
                                       uint32_t det_count) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int w_idx = write_idx_.load();
    FrameBuffer& buf = frame_buf_[w_idx];

    if (buf.ready.load()) {
        return 0;
    }

    if (!frame_phy_addr || frame_size == 0) {
        return 0;
    }

    const uint64_t phy_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(frame_phy_addr));
    void* vir_addr = CVI_SYS_Mmap(phy_addr, frame_size);
    if (!vir_addr) {
        return 0;
    }

    const size_t copy_size = (frame_size < opt_.max_copy_bytes) ? frame_size : opt_.max_copy_bytes;
    if (buf.copied.size() < opt_.max_copy_bytes) {
        buf.copied.resize(opt_.max_copy_bytes);
    }
    memcpy(buf.copied.data(), vir_addr, copy_size);
    CVI_SYS_Munmap(vir_addr, frame_size);

    buf.magic = magic;
    buf.width = width;
    buf.height = height;
    buf.format = PayloadFormat::RGB888;

    buf.owned.reset();
    buf.owned_size = 0;
    buf.use_owned = false;
    buf.copied_size = copy_size;

    buf.det_count = det_count;
    buf.det_bytes.clear();
    appendBytes(buf.det_bytes, det_bytes, det_bytes_size);

    buf.ready.store(true);
    write_idx_.store(1 - w_idx);

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

void UDPSender::threadFunc() {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, ip_, &addr.sin_addr);

    std::vector<uint8_t> prefix;
    std::vector<uint8_t> cross_buf;
    cross_buf.resize(opt_.max_udp_packet);

    int send_count = 0;
    auto fps_start = std::chrono::high_resolution_clock::now();

    while (running_.load()) {
        int read_idx = 1 - write_idx_.load();
        FrameBuffer& buf = frame_buf_[read_idx];

        if (buf.ready.load()) {
            const uint8_t* payload_ptr = buf.payloadPtr();
            const size_t payload_size = buf.payloadSize();

            // prefix = magic + header(5*u32) + fid + det_bytes
            prefix.clear();
            prefix.reserve(4 + 20 + 4 + buf.det_bytes.size());

            appendBytes(prefix, &buf.magic, sizeof(uint32_t));

            uint32_t header[5];
            header[0] = static_cast<uint32_t>(buf.width);
            header[1] = static_cast<uint32_t>(buf.height);
            header[2] = static_cast<uint32_t>(payload_size);
            header[3] = buf.det_count;
            header[4] = static_cast<uint32_t>(buf.format);
            appendBytes(prefix, header, sizeof(header));

            uint32_t fid = frame_counter_.fetch_add(1);
            appendBytes(prefix, &fid, sizeof(fid));

            appendBytes(prefix, buf.det_bytes.data(), buf.det_bytes.size());

            const size_t prefix_len = prefix.size();
            const size_t total_size = prefix_len + payload_size;
            const size_t MAX_PKT = opt_.max_udp_packet;

            for (size_t i = 0; i < total_size; i += MAX_PKT) {
                const size_t len = (i + MAX_PKT < total_size) ? MAX_PKT : (total_size - i);

                if (i + len <= prefix_len) {
                    sendto(sock_, prefix.data() + i, len, 0, (sockaddr*)&addr, sizeof(addr));
                } else if (i >= prefix_len) {
                    sendto(sock_, payload_ptr + (i - prefix_len), len, 0, (sockaddr*)&addr, sizeof(addr));
                } else {
                    const size_t prefix_part = prefix_len - i;
                    const size_t payload_part = len - prefix_part;

                    if (cross_buf.size() < len) {
                        cross_buf.resize(len);
                    }

                    memcpy(cross_buf.data(), prefix.data() + i, prefix_part);
                    memcpy(cross_buf.data() + prefix_part, payload_ptr, payload_part);
                    sendto(sock_, cross_buf.data(), len, 0, (sockaddr*)&addr, sizeof(addr));
                }
            }

            buf.resetPayload();
            buf.det_bytes.clear();
            buf.det_count = 0;
            buf.ready.store(false);
            send_count++;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count();
        if (elapsed >= 2000) {
            send_fps_.store(send_count * 1000.0f / elapsed);
            send_count = 0;
            fps_start = now;
        }
    }
}

} // namespace udp_service
