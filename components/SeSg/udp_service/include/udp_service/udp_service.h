#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace udp_service {

enum class PayloadFormat : uint32_t {
    RGB888 = 0,
    JPEG = 1,
};

struct SenderOptions {
    // 单个 UDP 包最大 payload（prefix 或 payload 的分片）
    size_t max_udp_packet = 60000;

    // 物理帧映射后拷贝到 CPU buffer 的最大字节数
    size_t max_copy_bytes = 1024 * 1024;
};

// 发送端统一头（device 端一般为 little-endian）：
// magic(u32)
// width(u32), height(u32), payload_size(u32), det_count(u32), fmt(u32)
// fid(u32)
// det_bytes (det_count * sizeof(T))
// payload bytes
class UDPSender {
public:
    UDPSender(const char* ip, int port, SenderOptions opt = {});
    ~UDPSender();

    void start();
    void stop();

    float getSendFPS() const;

    //  JPEG buffer
    double sendJpegOwnedRaw(uint32_t magic,
                            std::unique_ptr<uint8_t[]> jpeg_data,
                            size_t jpeg_size,
                            int width,
                            int height,
                            const void* det_bytes,
                            size_t det_bytes_size,
                            uint32_t det_count);

    //  RGB888  CVI_SYS_Mmap/Munmap 
    double sendRgb888PhysicalRaw(uint32_t magic,
                                 void* frame_phy_addr,
                                 size_t frame_size,
                                 int width,
                                 int height,
                                 const void* det_bytes,
                                 size_t det_bytes_size,
                                 uint32_t det_count);

    template <class T>
    double sendJpegOwned(uint32_t magic,
                         std::unique_ptr<uint8_t[]> jpeg_data,
                         size_t jpeg_size,
                         int width,
                         int height,
                         const std::vector<T>& detections) {
        static_assert(std::is_trivially_copyable<T>::value, "Detection type must be trivially copyable");
        const uint32_t det_count = static_cast<uint32_t>(detections.size());
        const void* det_ptr = detections.empty() ? nullptr : static_cast<const void*>(detections.data());
        const size_t det_bytes = detections.size() * sizeof(T);
        return sendJpegOwnedRaw(magic, std::move(jpeg_data), jpeg_size, width, height, det_ptr, det_bytes, det_count);
    }

    template <class T>
    double sendRgb888Physical(uint32_t magic,
                              void* frame_phy_addr,
                              size_t frame_size,
                              int width,
                              int height,
                              const std::vector<T>& detections) {
        static_assert(std::is_trivially_copyable<T>::value, "Detection type must be trivially copyable");
        const uint32_t det_count = static_cast<uint32_t>(detections.size());
        const void* det_ptr = detections.empty() ? nullptr : static_cast<const void*>(detections.data());
        const size_t det_bytes = detections.size() * sizeof(T);
        return sendRgb888PhysicalRaw(magic, frame_phy_addr, frame_size, width, height, det_ptr, det_bytes, det_count);
    }

private:
    struct FrameBuffer;

    void threadFunc();

    const char* ip_;
    int port_;
    int sock_;
    SenderOptions opt_;

    std::unique_ptr<FrameBuffer[]> frame_buf_;
    std::atomic<int> write_idx_;

    std::atomic<bool> running_;
    std::thread thread_;

    std::atomic<float> send_fps_;
    std::atomic<uint32_t> frame_counter_{0};
};

} // namespace udp_service
