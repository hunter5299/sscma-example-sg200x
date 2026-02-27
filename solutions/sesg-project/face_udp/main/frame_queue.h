/**
 * @file frame_queue.h
 * @brief 线程安全的帧队列，用于采集线程与推理线程间传递数据
 */

#ifndef FRAME_QUEUE_H
#define FRAME_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sscma.h>

/**
 * @struct FrameData
 * @brief 封装一帧图像数据（拷贝到用户空间）
 */
struct FrameData {
    uint8_t* data;           // 图像数据指针
    size_t size;             // 数据大小
    int width;               // 图像宽度
    int height;              // 图像高度
    int64_t timestamp_ms;    // 采集时间戳（毫秒）
    bool valid;              // 是否有效

    FrameData() : data(nullptr), size(0), width(0), height(0), timestamp_ms(0), valid(false) {}
    
    // 从 ma_img_t 构造，深拷贝数据
    FrameData(const ma_img_t& frame, void* vir_addr, int64_t ts) 
        : size(frame.size), width(frame.width), height(frame.height), timestamp_ms(ts), valid(true) {
        data = new uint8_t[size];
        std::memcpy(data, vir_addr, size);
    }
    
    // 移动构造
    FrameData(FrameData&& other) noexcept 
        : data(other.data), size(other.size), width(other.width), height(other.height),
          timestamp_ms(other.timestamp_ms), valid(other.valid) {
        other.data = nullptr;
        other.valid = false;
    }
    
    // 移动赋值
    FrameData& operator=(FrameData&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            width = other.width;
            height = other.height;
            timestamp_ms = other.timestamp_ms;
            valid = other.valid;
            other.data = nullptr;
            other.valid = false;
        }
        return *this;
    }
    
    // 禁止拷贝
    FrameData(const FrameData&) = delete;
    FrameData& operator=(const FrameData&) = delete;
    
    ~FrameData() {
        delete[] data;
    }
};

/**
 * @class FrameQueue
 * @brief 有界线程安全队列，支持阻塞和超时操作
 */
class FrameQueue {
public:
    explicit FrameQueue(size_t max_size = 2)
        : max_size_(max_size), stopped_(false) {}

    /**
     * @brief 压入一帧（队列满时丢弃最旧的帧）
     */
    void push(FrameData&& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (queue_.size() >= max_size_) {
            queue_.pop();  // 丢弃旧帧保持实时性
        }
        queue_.push(std::move(frame));
        cond_.notify_one();
    }

    /**
     * @brief 弹出一帧（阻塞等待）
     */
    bool pop(FrameData& frame, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms > 0) {
            if (!cond_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                [this] { return !queue_.empty() || stopped_; })) {
                return false;
            }
        } else {
            cond_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        }

        if (stopped_ && queue_.empty()) {
            return false;
        }

        frame = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cond_.notify_all();
    }

    bool isStopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<FrameData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    size_t max_size_;
    bool stopped_;
};

#endif // FRAME_QUEUE_H
