/**
 * @file yolov8_postprocess.cpp
 * @brief 通用 YOLOv8 后处理实现
 */

#include "yolov8_postprocess.h"
#include <cstring>

namespace yolov8 {

void YoloV8PostProcessor::computeDFL(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_sum = 0;
        float acc_sum = 0;
        
        // 计算 softmax 分母
        for (int i = 0; i < dfl_len; i++) {
            exp_sum += std::exp(tensor[i + b * dfl_len]);
        }
        
        // 计算加权平均
        for (int i = 0; i < dfl_len; i++) {
            float softmax_val = std::exp(tensor[i + b * dfl_len]) / exp_sum;
            acc_sum += softmax_val * i;
        }
        box[b] = acc_sum;
    }
}

float YoloV8PostProcessor::computeIoU(const BBox& a, const BBox& b) {
    // 转换为角点坐标
    float a_x1 = a.x - a.w / 2, a_y1 = a.y - a.h / 2;
    float a_x2 = a.x + a.w / 2, a_y2 = a.y + a.h / 2;
    float b_x1 = b.x - b.w / 2, b_y1 = b.y - b.h / 2;
    float b_x2 = b.x + b.w / 2, b_y2 = b.y + b.h / 2;
    
    // 计算交集
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    
    // 计算并集
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;
    
    return inter_area / (union_area + 1e-6f);
}

void YoloV8PostProcessor::nms(std::vector<BBox>& boxes, float iou_threshold) {
    // 按置信度排序
    std::sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BBox> result;
    
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        
        result.push_back(boxes[i]);
        
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            
            // 只对同类别做 NMS
            if (boxes[i].target == boxes[j].target) {
                if (computeIoU(boxes[i], boxes[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    boxes = std::move(result);
}

std::vector<BBox> YoloV8PostProcessor::process(
    const TensorInfo box_outputs[3],
    const TensorInfo cls_outputs[3],
    int img_width,
    int img_height
) {
    std::vector<BBox> results;
    
    // 从第一个 cls 输出获取类别数
    int num_class = cls_outputs[0].dims[1];
    
    // DFL 长度 (通常是 16)
    int dfl_len = box_outputs[0].dims[1] / 4;
    
    // 预计算逆 sigmoid 阈值
    float score_threshold_non_sigmoid = inverseSigmoid(score_threshold_);
    
    // 处理 3 个尺度 (P3, P4, P5)
    for (int scale = 0; scale < 3; scale++) {
        int grid_h = box_outputs[scale].dims[2];
        int grid_w = box_outputs[scale].dims[3];
        int grid_len = grid_h * grid_w;
        
        // 计算 stride（分别计算宽高方向）
        int stride_h = img_height / grid_h;
        int stride_w = img_width / grid_w;
        
        // 遍历所有 grid 位置
        for (int j = 0; j < grid_h; j++) {
            for (int k = 0; k < grid_w; k++) {
                int grid_offset = j * grid_w + k;
                
                // 找到最大置信度的类别
                int best_class = -1;
                float best_score_raw = -1e9f;
                
                if (cls_outputs[scale].is_int8) {
                    int8_t* cls_data = static_cast<int8_t*>(cls_outputs[scale].data);
                    int8_t max_val = -128;
                    
                    int offset = grid_offset;
                    for (int c = 0; c < num_class; c++) {
                        int8_t val = cls_data[offset];
                        if (val > max_val) {
                            max_val = val;
                            best_class = c;
                        }
                        offset += grid_len;
                    }
                    
                    // 使用 cls 输出的量化参数反量化
                    best_score_raw = dequantize(max_val, cls_outputs[scale].scale, cls_outputs[scale].zero_point);
                } else {
                    float* cls_data = static_cast<float*>(cls_outputs[scale].data);
                    
                    int offset = grid_offset;
                    for (int c = 0; c < num_class; c++) {
                        float val = cls_data[offset];
                        if (val > best_score_raw) {
                            best_score_raw = val;
                            best_class = c;
                        }
                        offset += grid_len;
                    }
                }
                
                // 阈值过滤
                if (best_class < 0 || best_score_raw <= score_threshold_non_sigmoid) {
                    continue;
                }
                
                // 解析 box
                float before_dfl[64];  // dfl_len * 4, 最大支持 dfl_len=16
                
                if (box_outputs[scale].is_int8) {
                    int8_t* box_data = static_cast<int8_t*>(box_outputs[scale].data);
                    int offset = grid_offset;
                    for (int b = 0; b < dfl_len * 4; b++) {
                        // 使用 box 输出的量化参数反量化
                        before_dfl[b] = dequantize(box_data[offset], box_outputs[scale].scale, box_outputs[scale].zero_point);
                        offset += grid_len;
                    }
                } else {
                    float* box_data = static_cast<float*>(box_outputs[scale].data);
                    int offset = grid_offset;
                    for (int b = 0; b < dfl_len * 4; b++) {
                        before_dfl[b] = box_data[offset];
                        offset += grid_len;
                    }
                }
                
                // DFL 解码
                float rect[4];
                computeDFL(before_dfl, dfl_len, rect);
                
                // 计算边界框（支持非正方形，分别用 stride_w 和 stride_h）
                float x1 = (-rect[0] + k + 0.5f) * stride_w;
                float y1 = (-rect[1] + j + 0.5f) * stride_h;
                float x2 = (rect[2] + k + 0.5f) * stride_w;
                float y2 = (rect[3] + j + 0.5f) * stride_h;
                
                float w = x2 - x1;
                float h = y2 - y1;
                float cx = x1 + w / 2.0f;
                float cy = y1 + h / 2.0f;
                
                // 创建检测结果（归一化坐标）
                BBox box;
                box.x = cx / img_width;
                box.y = cy / img_height;
                box.w = w / img_width;
                box.h = h / img_height;
                box.score = sigmoid(best_score_raw);
                box.target = best_class;
                
                results.push_back(box);
            }
        }
    }
    
    // NMS
    nms(results, nms_threshold_);
    
    return results;
}

} // namespace yolov8
