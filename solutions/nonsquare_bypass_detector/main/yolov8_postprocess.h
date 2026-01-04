/**
 * @file yolov8_postprocess.h
 * @brief 通用 YOLOv8 后处理，支持任意分辨率（正方形和非正方形）
 */

#ifndef YOLOV8_POSTPROCESS_H
#define YOLOV8_POSTPROCESS_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace yolov8 {

// 检测框结构
struct BBox {
    float x;       // 中心 x (归一化 0-1)
    float y;       // 中心 y (归一化 0-1)
    float w;       // 宽度 (归一化 0-1)
    float h;       // 高度 (归一化 0-1)
    float score;   // 置信度
    int target;    // 类别 ID
};

// 输出 tensor 信息
struct TensorInfo {
    void* data;
    int dims[4];      // [N, C, H, W]
    float scale;      // 量化 scale
    int zero_point;   // 量化 zero_point
    bool is_int8;     // 是否为 INT8 量化
};

/**
 * @brief 通用 YOLOv8 后处理类
 *
 * 支持：
 * - 任意输入分辨率（正方形、非正方形）
 * - INT8 量化模型
 * - 标准 YOLOv8 6 输出格式 [box0, box1, box2, cls0, cls1, cls2]
 */
class YoloV8PostProcessor {
public:
    YoloV8PostProcessor() = default;

    void setThreshold(float score_threshold, float nms_threshold) {
        score_threshold_ = score_threshold;
        nms_threshold_ = nms_threshold;
    }

    std::vector<BBox> process(
        const TensorInfo box_outputs[3],
        const TensorInfo cls_outputs[3],
        int img_width,
        int img_height
    );

private:
    float score_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static float inverseSigmoid(float y) {
        return -std::log(1.0f / y - 1.0f);
    }

    static float dequantize(int8_t val, float scale, int zero_point) {
        return (static_cast<float>(val) - zero_point) * scale;
    }

    static void computeDFL(float* tensor, int dfl_len, float* box);
    static void nms(std::vector<BBox>& boxes, float iou_threshold);
    static float computeIoU(const BBox& a, const BBox& b);
};

} // namespace yolov8

#endif // YOLOV8_POSTPROCESS_H
