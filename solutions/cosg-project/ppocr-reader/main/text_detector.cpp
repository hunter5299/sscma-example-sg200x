#include "text_detector.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>

#define TAG "TextDetector"

namespace ppocr {

TextDetector::TextDetector()
    : engine_(nullptr),
      input_width_(480),
      input_height_(480),
      orig_width_(0),
      orig_height_(0),
      scale_(1.0f),
      pad_left_(0),
      pad_top_(0),
      tensor_stride_(0),
      det_threshold_(0.3f),
      box_threshold_(0.5f),
      unclip_ratio_(1.6f),
      min_box_size_(10),
      initialized_(false) {}

TextDetector::~TextDetector() {}

bool TextDetector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Initializing text detector: %s", model_path.c_str());

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

    // Get input shape
    input_tensor_ = engine_->getInput(0);
    ma_shape_t input_shape = engine_->getInputShape(0);

    bool is_nhwc = (input_shape.dims[3] == 3 || input_shape.dims[3] == 1);
    if (is_nhwc) {
        input_height_ = input_shape.dims[1];
        input_width_ = input_shape.dims[2];
    } else {
        input_height_ = input_shape.dims[2];
        input_width_ = input_shape.dims[3];
    }

    int output_count = engine_->getOutputSize();

    // Detect row alignment: if tensor size > W*H*3, model has aligned_input
    size_t row_bytes = input_width_ * 3;
    tensor_stride_ = input_tensor_.size / input_height_;

    MA_LOGI(TAG, "Text detector initialized");
    MA_LOGI(TAG, "  Input: %dx%d, type=%d, size=%zu", input_width_, input_height_,
            input_tensor_.type, input_tensor_.size);
    if (tensor_stride_ != row_bytes) {
        MA_LOGI(TAG, "  Aligned input: row=%zu stride=%zu pad=%zu",
                row_bytes, tensor_stride_, tensor_stride_ - row_bytes);
    }
    MA_LOGI(TAG, "  Outputs: %d", output_count);

    initialized_ = true;
    return true;
}

void TextDetector::preprocess(const ma_img_t* src) {
    orig_width_ = src->width;
    orig_height_ = src->height;

    // Calculate letterbox scale
    float scale_w = static_cast<float>(input_width_) / src->width;
    float scale_h = static_cast<float>(input_height_) / src->height;
    scale_ = std::min(scale_w, scale_h);

    int new_width = static_cast<int>(src->width * scale_);
    int new_height = static_cast<int>(src->height * scale_);

    pad_left_ = (input_width_ - new_width) / 2;
    pad_top_ = (input_height_ - new_height) / 2;

    size_t buffer_size = input_width_ * input_height_ * 3;
    if (letterbox_buffer_.size() != buffer_size) {
        letterbox_buffer_.resize(buffer_size);
    }

    // Fill with gray (128)
    std::memset(letterbox_buffer_.data(), 128, buffer_size);

    // Use OpenCV resize (INTER_AREA for downscaling, INTER_LINEAR for upscaling)
    cv::Mat src_mat(src->height, src->width, CV_8UC3, src->data);
    cv::Mat resized;
    int interp = (new_width < src->width) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(src_mat, resized, cv::Size(new_width, new_height), 0, 0, interp);

    // Copy resized image into letterbox buffer at correct offset
    uint8_t* dst_data = letterbox_buffer_.data();
    for (int i = 0; i < new_height; ++i) {
        int dst_offset = ((i + pad_top_) * input_width_ + pad_left_) * 3;
        std::memcpy(dst_data + dst_offset, resized.ptr<uint8_t>(i), new_width * 3);
    }

    // Copy to model input — handle potential row alignment padding
    // If model was compiled with --aligned_input, tensor stride > row_bytes
    size_t row_bytes = input_width_ * 3;

    if (tensor_stride_ == row_bytes) {
        // No alignment padding, flat copy
        std::memcpy(input_tensor_.data.u8, letterbox_buffer_.data(), buffer_size);
    } else {
        // Row-aligned: copy row by row with proper stride
        uint8_t* dst = input_tensor_.data.u8;
        const uint8_t* src_buf = letterbox_buffer_.data();
        std::memset(dst, 128, input_tensor_.size);
        for (int y = 0; y < input_height_; ++y) {
            std::memcpy(dst + y * tensor_stride_, src_buf + y * row_bytes, row_bytes);
        }
    }
}

void TextDetector::unclipPolygon(float points[4][2], float unclip_ratio) {
    // Calculate polygon area and perimeter
    float area = 0.0f;
    float perimeter = 0.0f;

    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1];
        float dx = points[j][0] - points[i][0];
        float dy = points[j][1] - points[i][1];
        perimeter += std::sqrt(dx * dx + dy * dy);
    }
    area = std::abs(area) / 2.0f;

    if (perimeter < 1e-6f) return;

    float distance = area * unclip_ratio / perimeter;

    // Expand each edge outward by distance
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) {
        cx += points[i][0];
        cy += points[i][1];
    }
    cx /= 4.0f;
    cy /= 4.0f;

    for (int i = 0; i < 4; ++i) {
        float dx = points[i][0] - cx;
        float dy = points[i][1] - cy;
        float d = std::sqrt(dx * dx + dy * dy);
        if (d > 1e-6f) {
            float expand = distance / d;
            points[i][0] += dx * expand;
            points[i][1] += dy * expand;
        }
    }
}

void TextDetector::postprocess(std::vector<TextBox>& boxes) {
    // Get the probability map output
    ma_tensor_t output = engine_->getOutput(0);

    // DBNet output: 1x1xHxW probability map
    int out_h = 0, out_w = 0;
    if (output.shape.size == 4) {
        out_h = output.shape.dims[2];
        out_w = output.shape.dims[3];
    } else if (output.shape.size == 3) {
        out_h = output.shape.dims[1];
        out_w = output.shape.dims[2];
    } else {
        MA_LOGE(TAG, "Unexpected output shape size: %d", output.shape.size);
        return;
    }

    // Dequantize to float probability map
    int map_size = out_h * out_w;
    std::vector<float> prob_map(map_size);

    if (output.type == MA_TENSOR_TYPE_S8) {
        float scale = output.quant_param.scale;
        int32_t zp = output.quant_param.zero_point;
        for (int i = 0; i < map_size; ++i) {
            prob_map[i] = (static_cast<float>(output.data.s8[i]) - zp) * scale;
        }
    } else {
        // F32
        std::memcpy(prob_map.data(), output.data.f32, map_size * sizeof(float));
    }

    // Create binary mask using threshold
    cv::Mat prob_mat(out_h, out_w, CV_32F, prob_map.data());
    cv::Mat binary_mat;
    cv::threshold(prob_mat, binary_mat, det_threshold_, 1.0f, cv::THRESH_BINARY);

    // Convert to 8-bit for contour finding
    cv::Mat binary_u8;
    binary_mat.convertTo(binary_u8, CV_8UC1, 255);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_u8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Process each contour
    for (const auto& contour : contours) {
        if (contour.size() < 3) continue;

        // Get min-area rotated rect
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float w = std::min(rect.size.width, rect.size.height);
        float h = std::max(rect.size.width, rect.size.height);

        if (w < min_box_size_ || h < min_box_size_) continue;

        // Calculate box score (mean probability within contour)
        cv::Mat mask = cv::Mat::zeros(out_h, out_w, CV_8UC1);
        std::vector<std::vector<cv::Point>> single_contour = {contour};
        cv::fillPoly(mask, single_contour, cv::Scalar(255));

        float score = cv::mean(prob_mat, mask)[0];
        if (score < box_threshold_) continue;

        // Get 4 corner points
        cv::Point2f corners[4];
        rect.points(corners);

        TextBox box;
        box.score = score;

        // Map from output space to input space (they should be the same for PP-OCR det)
        float scale_x = static_cast<float>(input_width_) / out_w;
        float scale_y = static_cast<float>(input_height_) / out_h;

        for (int i = 0; i < 4; ++i) {
            box.points[i][0] = corners[i].x * scale_x;
            box.points[i][1] = corners[i].y * scale_y;
        }

        // Unclip (expand box slightly)
        unclipPolygon(box.points, unclip_ratio_);

        // Map from letterbox space to original image space
        for (int i = 0; i < 4; ++i) {
            box.points[i][0] = (box.points[i][0] - pad_left_) / scale_;
            box.points[i][1] = (box.points[i][1] - pad_top_) / scale_;

            // Clip to image bounds
            box.points[i][0] = std::max(0.0f, std::min(static_cast<float>(orig_width_), box.points[i][0]));
            box.points[i][1] = std::max(0.0f, std::min(static_cast<float>(orig_height_), box.points[i][1]));
        }

        // Sort points: top-left, top-right, bottom-right, bottom-left
        // Find top-left (smallest x+y) and sort clockwise
        float cx = 0, cy = 0;
        for (int i = 0; i < 4; ++i) {
            cx += box.points[i][0];
            cy += box.points[i][1];
        }
        cx /= 4.0f;
        cy /= 4.0f;

        // Assign to quadrants relative to center
        float sorted[4][2];
        int tl = -1, tr = -1, br = -1, bl = -1;
        for (int i = 0; i < 4; ++i) {
            bool is_left = box.points[i][0] < cx;
            bool is_top = box.points[i][1] < cy;
            if (is_left && is_top) tl = i;
            else if (!is_left && is_top) tr = i;
            else if (!is_left && !is_top) br = i;
            else bl = i;
        }

        // Fallback: if quadrant assignment failed, use raw order
        if (tl < 0 || tr < 0 || br < 0 || bl < 0) {
            // Sort by y first, then x
            std::vector<int> idx = {0, 1, 2, 3};
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if (std::abs(box.points[a][1] - box.points[b][1]) < 5)
                    return box.points[a][0] < box.points[b][0];
                return box.points[a][1] < box.points[b][1];
            });
            tl = idx[0]; tr = idx[1]; br = idx[3]; bl = idx[2];
        }

        sorted[0][0] = box.points[tl][0]; sorted[0][1] = box.points[tl][1];
        sorted[1][0] = box.points[tr][0]; sorted[1][1] = box.points[tr][1];
        sorted[2][0] = box.points[br][0]; sorted[2][1] = box.points[br][1];
        sorted[3][0] = box.points[bl][0]; sorted[3][1] = box.points[bl][1];

        std::memcpy(box.points, sorted, sizeof(sorted));
        boxes.push_back(box);
    }
}

std::vector<TextBox> TextDetector::detect(ma_img_t* img) {
    std::vector<TextBox> boxes;

    if (!initialized_ || !img) return boxes;

    preprocess(img);

    // Handle INT8 offset
    if (input_tensor_.type == MA_TENSOR_TYPE_S8) {
        for (size_t i = 0; i < input_tensor_.size; i++) {
            input_tensor_.data.u8[i] -= 128;
        }
    }

    ma_err_t ret = engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Detection inference failed: %d", ret);
        return boxes;
    }

    postprocess(boxes);

    return boxes;
}

}  // namespace ppocr
