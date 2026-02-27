#include "yolo26_detector.h"

#include <algorithm>
#include <cstring>
#include <cmath>

#define TAG "Yolo26Detector"

namespace yolo26 {

// COCO class names
const char* COCO_CLASSES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

constexpr int Yolo26Detector::GRID_SIZES[3];
constexpr int Yolo26Detector::STRIDES[3];

Yolo26Detector::Yolo26Detector()
    : engine_(nullptr),
      conf_threshold_(0.25f),
      nms_threshold_(0.45f),
      input_width_(640),
      input_height_(640),
      initialized_(false),
      detection_id_counter_(0) {}

Yolo26Detector::~Yolo26Detector() {
    // Engine cleanup handled by unique_ptr
}

bool Yolo26Detector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Initializing YOLO26 detector with model: %s", model_path.c_str());

    engine_ = std::make_unique<ma::engine::EngineCVI>();
    ma_err_t ret = engine_->init();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Failed to initialize CVI engine");
        return false;
    }

    ret = engine_->load(model_path.c_str());
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Failed to load model: %s", model_path.c_str());
        return false;
    }

    // Verify we have 6 outputs (cv2.x and cv3.x for 3 scales)
    int output_count = engine_->getOutputSize();
    if (output_count != NUM_OUTPUTS) {
        MA_LOGW(TAG, "Expected %d outputs, got %d", NUM_OUTPUTS, output_count);
    }

    // Get input dimensions from the model
    input_tensor_ = engine_->getInput(0);
    ma_shape_t input_shape = engine_->getInputShape(0);

    // Determine if NHWC or NCHW format
    bool is_nhwc = (input_shape.dims[3] == 3 || input_shape.dims[3] == 1);
    if (is_nhwc) {
        input_height_ = input_shape.dims[1];
        input_width_ = input_shape.dims[2];
    } else {
        input_height_ = input_shape.dims[2];
        input_width_ = input_shape.dims[3];
    }

    // Initialize internal image structure
    img_.width = input_width_;
    img_.height = input_height_;
    img_.format = MA_PIXEL_FORMAT_RGB888;
    img_.size = input_width_ * input_height_ * 3;
    img_.data = input_tensor_.data.u8;

    MA_LOGI(TAG, "YOLO26 detector initialized");
    MA_LOGI(TAG, "  Input size: %dx%d", input_width_, input_height_);
    MA_LOGI(TAG, "  Output count: %d", output_count);
    MA_LOGI(TAG, "  Confidence threshold: %.2f", conf_threshold_);
    MA_LOGI(TAG, "  NMS threshold: %.2f", nms_threshold_);

    initialized_ = true;
    return true;
}

void Yolo26Detector::setConfThreshold(float threshold) {
    conf_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

void Yolo26Detector::setNmsThreshold(float threshold) {
    nms_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

const char* Yolo26Detector::getClassName(int class_id) {
    if (class_id >= 0 && class_id < NUM_CLASSES) {
        return COCO_CLASSES[class_id];
    }
    return "unknown";
}

void Yolo26Detector::letterboxPreprocess(const ma_img_t* src) {
    // Calculate scale to fit source image into model input while preserving aspect ratio
    float scale_w = static_cast<float>(input_width_) / src->width;
    float scale_h = static_cast<float>(input_height_) / src->height;
    float scale = std::min(scale_w, scale_h);

    int new_width = static_cast<int>(src->width * scale);
    int new_height = static_cast<int>(src->height * scale);

    // Calculate padding to center the image
    int pad_left = (input_width_ - new_width) / 2;
    int pad_top = (input_height_ - new_height) / 2;

    // Store letterbox info for coordinate transformation
    letterbox_info_.scale = scale;
    letterbox_info_.pad_left = pad_left;
    letterbox_info_.pad_top = pad_top;
    letterbox_info_.new_width = new_width;
    letterbox_info_.new_height = new_height;
    letterbox_info_.orig_width = src->width;
    letterbox_info_.orig_height = src->height;

    // Ensure letterbox buffer is allocated
    size_t buffer_size = input_width_ * input_height_ * 3;
    if (letterbox_buffer_.size() != buffer_size) {
        letterbox_buffer_.resize(buffer_size);
    }

    // Fill with gray (128, 128, 128) - standard letterbox padding color
    std::memset(letterbox_buffer_.data(), 128, buffer_size);

    // Resize source image to letterbox buffer using nearest neighbor
    // This is similar to ma::cv::convert but with offset for letterbox
    uint32_t beta_w = (static_cast<uint32_t>(src->width) << 16) / new_width;
    uint32_t beta_h = (static_cast<uint32_t>(src->height) << 16) / new_height;

    const uint8_t* src_data = src->data;
    uint8_t* dst_data = letterbox_buffer_.data();

    if (src->format == MA_PIXEL_FORMAT_RGB888) {
        // RGB888 to RGB888 with letterbox
        for (int i = 0; i < new_height; ++i) {
            int src_y = (i * beta_h) >> 16;
            int dst_row = (i + pad_top) * input_width_;

            for (int j = 0; j < new_width; ++j) {
                int src_x = (j * beta_w) >> 16;
                int src_idx = (src_y * src->width + src_x) * 3;
                int dst_idx = (dst_row + j + pad_left) * 3;

                dst_data[dst_idx + 0] = src_data[src_idx + 0];
                dst_data[dst_idx + 1] = src_data[src_idx + 1];
                dst_data[dst_idx + 2] = src_data[src_idx + 2];
            }
        }
    } else {
        // For other formats, use ma::cv::convert to a temp buffer first, then letterbox
        // Create a temporary image for the scaled content
        ma_img_t scaled_img;
        scaled_img.width = new_width;
        scaled_img.height = new_height;
        scaled_img.format = MA_PIXEL_FORMAT_RGB888;
        scaled_img.size = new_width * new_height * 3;
        scaled_img.rotate = MA_PIXEL_ROTATE_0;

        std::vector<uint8_t> scaled_buffer(scaled_img.size);
        scaled_img.data = scaled_buffer.data();

        ma::cv::convert(src, &scaled_img);

        // Copy scaled image to letterbox buffer with padding
        for (int i = 0; i < new_height; ++i) {
            int dst_row = (i + pad_top) * input_width_;
            int src_row = i * new_width;

            for (int j = 0; j < new_width; ++j) {
                int src_idx = (src_row + j) * 3;
                int dst_idx = (dst_row + j + pad_left) * 3;

                dst_data[dst_idx + 0] = scaled_buffer[src_idx + 0];
                dst_data[dst_idx + 1] = scaled_buffer[src_idx + 1];
                dst_data[dst_idx + 2] = scaled_buffer[src_idx + 2];
            }
        }
    }

    // Copy letterbox buffer to model input tensor
    std::memcpy(input_tensor_.data.u8, letterbox_buffer_.data(), buffer_size);
}

void Yolo26Detector::transformCoordinates(std::vector<Detection>& detections) {
    // Transform coordinates from letterbox (model input) space to original image space
    // Detection coordinates are normalized [0-1] relative to input_width_ x input_height_

    for (auto& det : detections) {
        // Convert normalized coords to pixel coords in letterbox space
        float px = det.x * input_width_;
        float py = det.y * input_height_;
        float pw = det.w * input_width_;
        float ph = det.h * input_height_;

        // Remove padding offset (convert to scaled image space)
        px -= letterbox_info_.pad_left;
        py -= letterbox_info_.pad_top;

        // Scale back to original image space
        px /= letterbox_info_.scale;
        py /= letterbox_info_.scale;
        pw /= letterbox_info_.scale;
        ph /= letterbox_info_.scale;

        // Convert back to normalized coordinates relative to original image
        det.x = px / letterbox_info_.orig_width;
        det.y = py / letterbox_info_.orig_height;
        det.w = pw / letterbox_info_.orig_width;
        det.h = ph / letterbox_info_.orig_height;

        // Clip center coordinates to valid range [0, 1]
        det.x = std::max(0.0f, std::min(1.0f, det.x));
        det.y = std::max(0.0f, std::min(1.0f, det.y));

        // Clip width/height to ensure box stays within image bounds
        float half_w = det.w / 2.0f;
        float half_h = det.h / 2.0f;
        float x1 = std::max(0.0f, det.x - half_w);
        float y1 = std::max(0.0f, det.y - half_h);
        float x2 = std::min(1.0f, det.x + half_w);
        float y2 = std::min(1.0f, det.y + half_h);

        det.x = (x1 + x2) / 2.0f;
        det.y = (y1 + y2) / 2.0f;
        det.w = x2 - x1;
        det.h = y2 - y1;
    }
}

std::vector<Detection> Yolo26Detector::detect(ma_img_t* img) {
    std::vector<Detection> detections;

    if (!initialized_ || !img) {
        return detections;
    }

    // Preprocess with letterbox: resize preserving aspect ratio, pad to model input size
    letterboxPreprocess(img);

    // Handle INT8 quantization offset if needed
    if (input_tensor_.type == MA_TENSOR_TYPE_S8) {
        for (size_t i = 0; i < input_tensor_.size; i++) {
            input_tensor_.data.u8[i] -= 128;
        }
    }

    // Run inference
    ma_err_t ret = engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Inference failed with error: %d", ret);
        return detections;
    }

    // Decode outputs
    decodeOutputs(detections);

    // Apply NMS
    applyNMS(detections);

    // Transform coordinates from letterbox space to original image space
    transformCoordinates(detections);

    // Assign detection IDs
    for (auto& det : detections) {
        det.id = detection_id_counter_++;
    }

    return detections;
}

void Yolo26Detector::decodeOutputs(std::vector<Detection>& results) {
    results.clear();

    /**
     * YOLO26 output format (6 outputs):
     *   - cv2.x tensors: shape 1×4×H×W (bbox coordinates)
     *   - cv3.x tensors: shape 1×80×H×W (class scores)
     *
     * OPTIMIZATION: Use inverseSigmoid to pre-compute threshold,
     * compare raw values instead of calling sigmoid 672,000 times.
     */

    // Pre-compute threshold in logit space to avoid sigmoid calls
    const float score_threshold_logit = inverseSigmoid(conf_threshold_);

    // Map output indices to their purpose based on shape
    int bbox_indices[3] = {-1, -1, -1};
    int cls_indices[3] = {-1, -1, -1};

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        ma_tensor_t tensor = engine_->getOutput(i);
        int n = (tensor.shape.size > 0) ? tensor.shape.dims[0] : 1;
        int c = (tensor.shape.size > 1) ? tensor.shape.dims[1] : 1;
        int h = (tensor.shape.size > 2) ? tensor.shape.dims[2] : 1;
        int w = (tensor.shape.size > 3) ? tensor.shape.dims[3] : 1;
        (void)n;

        int scale_idx = -1;
        if (h == 80 && w == 80) scale_idx = 0;
        else if (h == 40 && w == 40) scale_idx = 1;
        else if (h == 20 && w == 20) scale_idx = 2;

        if (scale_idx >= 0) {
            if (c == 4) {
                bbox_indices[scale_idx] = i;
            } else if (c == NUM_CLASSES) {
                cls_indices[scale_idx] = i;
            }
        }
    }

    // Process each scale
    for (int scale = 0; scale < 3; scale++) {
        int bbox_idx = bbox_indices[scale];
        int cls_idx = cls_indices[scale];

        if (bbox_idx < 0 || cls_idx < 0) [[unlikely]] {
            MA_LOGW(TAG, "Missing output for scale %d (bbox=%d, cls=%d)",
                    scale, bbox_idx, cls_idx);
            continue;
        }

        int stride = STRIDES[scale];
        int grid_size = GRID_SIZES[scale];
        int spatial_size = grid_size * grid_size;

        ma_tensor_t bbox_tensor = engine_->getOutput(bbox_idx);
        ma_tensor_t cls_tensor = engine_->getOutput(cls_idx);

        bool is_int8 = (bbox_tensor.type == MA_TENSOR_TYPE_S8);

        // Get quantization parameters for INT8 tensors
        float bbox_scale = 1.0f;
        int32_t bbox_zp = 0;
        float cls_scale = 1.0f;
        int32_t cls_zp = 0;

        if (is_int8) {
            bbox_scale = bbox_tensor.quant_param.scale;
            bbox_zp = bbox_tensor.quant_param.zero_point;
            cls_scale = cls_tensor.quant_param.scale;
            cls_zp = cls_tensor.quant_param.zero_point;
        }

        // Process each grid cell
        for (int j = 0; j < grid_size; j++) {
            for (int k = 0; k < grid_size; k++) {
                int offset = j * grid_size + k;

                // Find best class using raw logit values
                int best_class = -1;
                float max_logit = score_threshold_logit;

                if (is_int8) {
                    // INT8: dequantize class scores
                    int8_t* cls_data_s8 = cls_tensor.data.s8;
                    for (int c = 0; c < NUM_CLASSES; c++) {
                        float logit = (static_cast<float>(cls_data_s8[c * spatial_size + offset]) - cls_zp) * cls_scale;
                        if (logit > max_logit) [[unlikely]] {
                            max_logit = logit;
                            best_class = c;
                        }
                    }
                } else {
                    // F32: direct access
                    float* cls_data = cls_tensor.data.f32;
                    for (int c = 0; c < NUM_CLASSES; c++) {
                        float logit = cls_data[c * spatial_size + offset];
                        if (logit > max_logit) [[unlikely]] {
                            max_logit = logit;
                            best_class = c;
                        }
                    }
                }

                if (best_class < 0) [[likely]] {
                    continue;
                }

                float confidence = sigmoid(max_logit);

                // Decode bbox (direct coordinates, no DFL)
                float bx, by, bw, bh;
                if (is_int8) {
                    int8_t* bbox_data_s8 = bbox_tensor.data.s8;
                    bx = (static_cast<float>(bbox_data_s8[0 * spatial_size + offset]) - bbox_zp) * bbox_scale;
                    by = (static_cast<float>(bbox_data_s8[1 * spatial_size + offset]) - bbox_zp) * bbox_scale;
                    bw = (static_cast<float>(bbox_data_s8[2 * spatial_size + offset]) - bbox_zp) * bbox_scale;
                    bh = (static_cast<float>(bbox_data_s8[3 * spatial_size + offset]) - bbox_zp) * bbox_scale;
                } else {
                    float* bbox_data = bbox_tensor.data.f32;
                    bx = bbox_data[0 * spatial_size + offset];
                    by = bbox_data[1 * spatial_size + offset];
                    bw = bbox_data[2 * spatial_size + offset];
                    bh = bbox_data[3 * spatial_size + offset];
                }

                // Convert to normalized image coordinates
                Detection det;
                det.x = (k + 0.5f + bx) * stride / static_cast<float>(input_width_);
                det.y = (j + 0.5f + by) * stride / static_cast<float>(input_height_);
                det.w = std::exp(bw) * stride / static_cast<float>(input_width_);
                det.h = std::exp(bh) * stride / static_cast<float>(input_height_);
                det.confidence = confidence;
                det.class_id = best_class;
                det.id = 0;

                // Clip to valid range
                det.x = std::max(0.0f, std::min(1.0f, det.x));
                det.y = std::max(0.0f, std::min(1.0f, det.y));
                det.w = std::max(0.0f, std::min(1.0f, det.w));
                det.h = std::max(0.0f, std::min(1.0f, det.h));

                results.push_back(det);
            }
        }
    }
}

void Yolo26Detector::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty()) {
        return;
    }

    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);

    auto computeIoU = [](const Detection& a, const Detection& b) -> float {
        // Convert center format to corner format
        float ax1 = a.x - a.w / 2.0f;
        float ay1 = a.y - a.h / 2.0f;
        float ax2 = a.x + a.w / 2.0f;
        float ay2 = a.y + a.h / 2.0f;

        float bx1 = b.x - b.w / 2.0f;
        float by1 = b.y - b.h / 2.0f;
        float bx2 = b.x + b.w / 2.0f;
        float by2 = b.y + b.h / 2.0f;

        float inter_x1 = std::max(ax1, bx1);
        float inter_y1 = std::max(ay1, by1);
        float inter_x2 = std::min(ax2, bx2);
        float inter_y2 = std::min(ay2, by2);

        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;

        float a_area = a.w * a.h;
        float b_area = b.w * b.h;
        float union_area = a_area + b_area - inter_area;

        return (union_area > 0) ? inter_area / union_area : 0.0f;
    };

    // Apply NMS per class
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            // Only compare same class
            if (detections[i].class_id != detections[j].class_id) continue;

            float iou = computeIoU(detections[i], detections[j]);
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    // Remove suppressed detections
    std::vector<Detection> kept;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            kept.push_back(detections[i]);
        }
    }
    detections = std::move(kept);
}

}  // namespace yolo26
