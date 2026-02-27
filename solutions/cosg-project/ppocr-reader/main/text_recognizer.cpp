#include "text_recognizer.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <opencv2/opencv.hpp>

#define TAG "TextRecognizer"

namespace ppocr {

TextRecognizer::TextRecognizer()
    : engine_(nullptr),
      input_width_(320),
      input_height_(48),
      dict_size_(0),
      tensor_stride_(0),
      initialized_(false) {}

TextRecognizer::~TextRecognizer() {}

bool TextRecognizer::loadDictionary(const std::string& dict_path) {
    std::ifstream f(dict_path);
    if (!f.is_open()) {
        MA_LOGE(TAG, "Failed to open dictionary: %s", dict_path.c_str());
        return false;
    }

    dictionary_.clear();

    // PP-OCRv3 CTC class layout: [blank, char1, char2, ..., charN, space]
    // Index 0 = CTC blank (added here)
    // Index 1..N = characters from dictionary file
    // Index N+1 = space (from use_space_char=True in PP-OCR config)
    //
    // The dict file itself varies by language:
    //   ch: ppocr_keys_v1.txt  - 6623 chars (no space line)
    //   en: en_dict.txt        - 94 chars + space as last line
    // PP-OCR Python only strips \n and \r\n (NOT spaces), so we match that.
    dictionary_.push_back("");  // index 0: CTC blank placeholder

    std::string line;
    int file_lines = 0;
    while (std::getline(f, line)) {
        // Strip only \r and \n (matching PaddleOCR behavior, preserving spaces)
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        dictionary_.push_back(line);  // indices 1..N (include space if present)
        file_lines++;
    }

    // PP-OCR always appends space (use_space_char=True in config)
    dictionary_.push_back(" ");

    dict_size_ = static_cast<int>(dictionary_.size());

    MA_LOGI(TAG, "Dictionary loaded: %d classes (blank + %d chars + space)",
            dict_size_, dict_size_ - 2);

    return true;
}

bool TextRecognizer::init(const std::string& model_path, const std::string& dict_path) {
    MA_LOGI(TAG, "Initializing text recognizer: %s", model_path.c_str());

    if (!loadDictionary(dict_path)) {
        return false;
    }

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

    // Detect row alignment: if tensor size > W*H*3, model has aligned_input
    size_t row_bytes = input_width_ * 3;
    tensor_stride_ = input_tensor_.size / input_height_;

    MA_LOGI(TAG, "Text recognizer initialized (input: %dx%d, dict: %d, type=%d, size=%zu)",
            input_width_, input_height_, dict_size_, input_tensor_.type, input_tensor_.size);
    if (tensor_stride_ != row_bytes) {
        MA_LOGI(TAG, "  Aligned input: row=%zu stride=%zu pad=%zu",
                row_bytes, tensor_stride_, tensor_stride_ - row_bytes);
    }

    initialized_ = true;
    return true;
}

void TextRecognizer::preprocess(const uint8_t* rgb_data, int width, int height) {
    size_t buffer_size = input_width_ * input_height_ * 3;
    if (resize_buffer_.size() != buffer_size) {
        resize_buffer_.resize(buffer_size);
    }

    // Fill with gray (128 -> maps to 0 after normalization [-1,1])
    std::memset(resize_buffer_.data(), 128, buffer_size);

    // PP-OCR standard preprocessing:
    // 1. Resize height to input_height_ (48), scale width proportionally
    // 2. If resulting width > input_width_ (320), clamp to 320 (squish)
    // 3. Pad right with gray if needed
    float ratio = static_cast<float>(input_height_) / height;
    int new_w = static_cast<int>(width * ratio);
    if (new_w > input_width_) new_w = input_width_;
    int new_h = input_height_;

    // Use OpenCV resize (INTER_AREA for downscaling, INTER_LINEAR for upscaling)
    cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(rgb_data));
    cv::Mat resized;
    int interp = (new_w < width) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, interp);

    // Copy resized data into buffer (left-aligned, gray padding on right)
    for (int y = 0; y < new_h; ++y) {
        std::memcpy(resize_buffer_.data() + y * input_width_ * 3,
                    resized.ptr<uint8_t>(y),
                    new_w * 3);
    }

    // Copy to model input — handle potential row alignment padding
    size_t row_bytes = input_width_ * 3;

    if (tensor_stride_ == row_bytes) {
        // No alignment padding, flat copy
        std::memcpy(input_tensor_.data.u8, resize_buffer_.data(), buffer_size);
    } else {
        // Row-aligned: copy row by row with proper stride
        uint8_t* dst = input_tensor_.data.u8;
        const uint8_t* src_buf = resize_buffer_.data();
        std::memset(dst, 128, input_tensor_.size);
        for (int y = 0; y < input_height_; ++y) {
            std::memcpy(dst + y * tensor_stride_, src_buf + y * row_bytes, row_bytes);
        }
    }
}

RecognitionResult TextRecognizer::ctcDecode(const ma_tensor_t& output) {
    RecognitionResult result;
    result.confidence = 0.0f;

    // PP-OCRv3 rec output: 1 x T x C (or [1, T, C, 1] from cvimodel)
    int time_steps = 0;
    int num_classes = 0;

    if (output.shape.size == 4) {
        if (output.shape.dims[3] == 1) {
            time_steps = output.shape.dims[1];
            num_classes = output.shape.dims[2];
        } else {
            time_steps = output.shape.dims[2];
            num_classes = output.shape.dims[3];
        }
    } else if (output.shape.size == 3) {
        time_steps = output.shape.dims[1];
        num_classes = output.shape.dims[2];
    } else if (output.shape.size == 2) {
        time_steps = output.shape.dims[0];
        num_classes = output.shape.dims[1];
    } else {
        MA_LOGE(TAG, "Unexpected rec output shape: %d dims", output.shape.size);
        return result;
    }

    // CTC greedy decode: blank is at index 0
    int blank_idx = 0;
    int prev_idx = blank_idx;
    float total_conf = 0.0f;
    int char_count = 0;

    for (int t = 0; t < time_steps; ++t) {
        int best_idx = 0;
        float best_val = -1e30f;

        if (output.type == MA_TENSOR_TYPE_S8) {
            float scale = output.quant_param.scale;
            int32_t zp = output.quant_param.zero_point;
            for (int c = 0; c < num_classes; ++c) {
                float val = (static_cast<float>(output.data.s8[t * num_classes + c]) - zp) * scale;
                if (val > best_val) {
                    best_val = val;
                    best_idx = c;
                }
            }
        } else {
            // F32 (BF16 models are auto-converted to F32 by CVI runtime)
            const float* row = output.data.f32 + t * num_classes;
            for (int c = 0; c < num_classes; ++c) {
                if (row[c] > best_val) {
                    best_val = row[c];
                    best_idx = c;
                }
            }
        }

        // Skip blank and repeated indices
        if (best_idx != blank_idx && best_idx != prev_idx) {
            if (best_idx >= 0 && best_idx < static_cast<int>(dictionary_.size())) {
                result.text += dictionary_[best_idx];
                total_conf += best_val;
                char_count++;
            }
        }

        prev_idx = best_idx;
    }

    if (char_count > 0) {
        result.confidence = total_conf / char_count;
    }

    return result;
}

RecognitionResult TextRecognizer::recognize(const uint8_t* rgb_data, int width, int height) {
    RecognitionResult result;
    result.confidence = 0.0f;

    if (!initialized_ || !rgb_data || width <= 0 || height <= 0) {
        return result;
    }

    preprocess(rgb_data, width, height);

    // Handle INT8 input offset
    if (input_tensor_.type == MA_TENSOR_TYPE_S8) {
        for (size_t i = 0; i < input_tensor_.size; i++) {
            input_tensor_.data.u8[i] -= 128;
        }
    }

    ma_err_t ret = engine_->run();
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Recognition inference failed: %d", ret);
        return result;
    }

    ma_tensor_t output = engine_->getOutput(0);

    // Debug: log output tensor info on first call
    static bool logged_output_info = false;
    if (!logged_output_info) {
        MA_LOGI(TAG, "Rec output: type=%d, dims=%d, shape=[%d,%d,%d,%d]",
                output.type, output.shape.size,
                output.shape.size >= 1 ? output.shape.dims[0] : 0,
                output.shape.size >= 2 ? output.shape.dims[1] : 0,
                output.shape.size >= 3 ? output.shape.dims[2] : 0,
                output.shape.size >= 4 ? output.shape.dims[3] : 0);
        if (output.type == MA_TENSOR_TYPE_S8) {
            MA_LOGI(TAG, "Rec output quant: scale=%.6f, zp=%d",
                    output.quant_param.scale, output.quant_param.zero_point);
        }
        logged_output_info = true;
    }

    result = ctcDecode(output);

    return result;
}

}  // namespace ppocr
