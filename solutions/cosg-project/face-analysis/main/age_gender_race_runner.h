#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sscma.h>

namespace face_analysis {

struct AgeGenderRaceResult {
    bool ok = false;
    int gender = -1;        // 1=Male, 0=Female
    int age = -1;           // FairFace: 0..8 (age bins), InsightFace: 0..100 (years)
    int race = -1;          // 0..6 (FairFace only, -1 if unavailable)
    float gender_score = 0.f;
    float age_score = 0.f;
    float race_score = 0.f;
    bool is_fairface = false;  // true=FairFace (age bins+race), false=InsightFace (continuous age)
};

// Age/Gender/Race inference (SG2002 / CVIMODEL)
// Supports two model formats (auto-detected):
//   - FairFace: 18 outputs (7 race + 2 gender + 9 age bins)
//   - InsightFace: 3 outputs (2 gender logits + 1 continuous age)
// No OpenCV dependency: crop/resize uses pure CPU bilinear interpolation
class AgeGenderRaceRunner {
public:
    AgeGenderRaceRunner() = default;
    ~AgeGenderRaceRunner() = default;

    bool init(const std::string& model_path);

    int inputSize() const { return input_size_; }

    void setPreprocess(float mean0, float mean1, float mean2, float scale0, float scale1, float scale2) {
        mean_[0] = mean0;
        mean_[1] = mean1;
        mean_[2] = mean2;
        scale_[0] = scale0;
        scale_[1] = scale1;
        scale_[2] = scale2;
    }

    void setCropScale(float s) { crop_scale_ = s; }

    bool infer(const uint8_t* rgb888, int src_w, int src_h,
               float x1, float y1, float x2, float y2,
               AgeGenderRaceResult& out);

    bool infer(const uint8_t* rgb888, int src_w, int src_h, int src_stride_bytes,
               float x1, float y1, float x2, float y2,
               AgeGenderRaceResult& out);

private:
    static float bf16_to_fp32(uint16_t v);
    static float fp16_to_fp32(uint16_t v);
    static uint16_t fp32_to_bf16(float v);
    static uint16_t fp32_to_fp16(float v);
    static size_t elem_size(ma_tensor_type_t t);
    static size_t shape_numel(const ma_shape_t& s);

    float read_val(const ma_tensor_t& t, int idx) const;

    void alignCropRgb(const uint8_t* src, int src_w, int src_h,
                      int src_stride_bytes,
                      float x1, float y1, float x2, float y2,
                      uint8_t* dst, int dst_size) const;

    bool prepareInputTensor();
    void packInput(const uint8_t* rgb_hwc_u8);
    bool parseOutputs(AgeGenderRaceResult& out);
    bool parseOutputsFairFace(AgeGenderRaceResult& out);
    bool parseOutputsInsightFace(AgeGenderRaceResult& out);

    enum class ModelFormat { Unknown, FairFace, InsightFace };
    ModelFormat detectModelFormat() const;

private:
    int input_size_ = 224;
    ModelFormat model_format_ = ModelFormat::Unknown;
    float crop_scale_ = 1.3f;

    float mean_[3] = {0.0f, 0.0f, 0.0f};
    float scale_[3] = {1.0f, 1.0f, 1.0f};

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    std::vector<uint8_t> input_rgb_;

    ma_tensor_type_t input_type_ = MA_TENSOR_TYPE_NONE;
    bool input_is_chw_ = false;
    int input_c_ = 3;
    int input_h_ = 224;
    int input_w_ = 224;
    size_t input_numel_ = 0;

    std::vector<uint8_t> input_u8_;
    std::vector<int8_t> input_s8_;
    std::vector<uint16_t> input_u16_;
    std::vector<float> input_f32_;

    ma_tensor_t input_tensor_cache_{};
    bool inited_ = false;
};

}  // namespace face_analysis
