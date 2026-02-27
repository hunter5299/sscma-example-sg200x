#include "emotion_runner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace face_analysis {

float EmotionRunner::bf16_to_fp32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

float EmotionRunner::fp16_to_fp32(uint16_t v) {
    const uint32_t sign = (uint32_t)(v & 0x8000u) << 16;
    const uint32_t exp = (v & 0x7C00u) >> 10;
    const uint32_t mant = (v & 0x03FFu);

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 0;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                e++;
            }
            m &= 0x03FFu;
            const uint32_t exp32 = (127 - 15 - e) << 23;
            const uint32_t mant32 = m << 13;
            out = sign | exp32 | mant32;
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t exp32 = (exp + (127 - 15)) << 23;
        const uint32_t mant32 = mant << 13;
        out = sign | exp32 | mant32;
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t EmotionRunner::fp32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return (uint16_t)(u >> 16);
}

uint16_t EmotionRunner::fp32_to_fp16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 31) & 1;
    int exp = (int)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)(sign << 15);
        mant |= 0x800000;
        const int shift = 14 - exp;
        return (uint16_t)((sign << 15) | (mant >> shift));
    }
    if (exp >= 31) {
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    return (uint16_t)((sign << 15) | ((uint16_t)exp << 10) | (uint16_t)(mant >> 13));
}

size_t EmotionRunner::elem_size(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_F32: return 4;
        case MA_TENSOR_TYPE_F16: return 2;
        case MA_TENSOR_TYPE_BF16: return 2;
        case MA_TENSOR_TYPE_S8: return 1;
        case MA_TENSOR_TYPE_U8: return 1;
        default: return 0;
    }
}

size_t EmotionRunner::shape_numel(const ma_shape_t& s) {
    if (s.size <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < s.size; ++i) {
        if (s.dims[i] <= 0) return 0;
        n *= (size_t)s.dims[i];
    }
    return n;
}

float EmotionRunner::read_val(const ma_tensor_t& t, int idx) const {
    switch (t.type) {
        case MA_TENSOR_TYPE_F32:
            return t.data.f32[idx];
        case MA_TENSOR_TYPE_F16:
            return fp16_to_fp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_BF16:
            return bf16_to_fp32(t.data.u16[idx]);
        case MA_TENSOR_TYPE_S8: {
            const float scale = t.quant_param.scale;
            const int zp = t.quant_param.zero_point;
            return (t.data.s8[idx] - zp) * scale;
        }
        case MA_TENSOR_TYPE_U8: {
            const float scale = t.quant_param.scale;
            const int zp = t.quant_param.zero_point;
            return ((int)t.data.u8[idx] - zp) * scale;
        }
        default:
            return 0.f;
    }
}

bool EmotionRunner::init(const std::string& model_path) {
    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) return false;
    if (engine_->load(model_path) != MA_OK) return false;
    if (!prepareInputTensor()) return false;

    input_rgb_.assign((size_t)input_h_ * (size_t)input_w_ * 3, 0);
    inited_ = true;
    return true;
}

bool EmotionRunner::prepareInputTensor() {
    input_tensor_cache_ = engine_->getInput(0);
    input_type_ = input_tensor_cache_.type;

    const ma_shape_t s = engine_->getInputShape(0);
    if (s.size != 4) return false;

    if (s.dims[1] == 3 || s.dims[1] == 1) {
        input_is_chw_ = true;
        input_c_ = s.dims[1];
        input_h_ = s.dims[2];
        input_w_ = s.dims[3];
    } else if (s.dims[3] == 3 || s.dims[3] == 1) {
        input_is_chw_ = false;
        input_h_ = s.dims[1];
        input_w_ = s.dims[2];
        input_c_ = s.dims[3];
    } else {
        input_is_chw_ = true;
        input_c_ = s.dims[1];
        input_h_ = s.dims[2];
        input_w_ = s.dims[3];
    }

    input_size_ = std::min(input_w_, input_h_);
    input_numel_ = shape_numel(s);
    if (input_numel_ == 0) return false;

    input_u8_.clear();
    input_s8_.clear();
    input_u16_.clear();
    input_f32_.clear();

    switch (input_type_) {
        case MA_TENSOR_TYPE_U8:
            input_u8_.assign(input_numel_, 0);
            break;
        case MA_TENSOR_TYPE_S8:
            input_s8_.assign(input_numel_, 0);
            break;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16:
            input_u16_.assign(input_numel_, 0);
            break;
        case MA_TENSOR_TYPE_F32:
            input_f32_.assign(input_numel_, 0.f);
            break;
        default:
            return false;
    }

    return true;
}

void EmotionRunner::alignCropRgb(const uint8_t* src, int src_w, int src_h,
                                float x1, float y1, float x2, float y2,
                                uint8_t* dst, int dst_size) const {
    const float bw = std::max(1.0f, x2 - x1);
    const float bh = std::max(1.0f, y2 - y1);
    const float cx = 0.5f * (x1 + x2);
    const float cy = 0.5f * (y1 + y2);

    const float box = std::max(bw, bh) * std::max(1.0f, crop_scale_);
    const float scale = (box > 1e-6f) ? (dst_size / box) : 1.0f;

    const float dst_c = (dst_size - 1) * 0.5f;

    auto sample = [&](float sx, float sy, int c) -> uint8_t {
        if (sx < 0.f || sy < 0.f || sx > (float)(src_w - 1) || sy > (float)(src_h - 1)) return 0;
        const int x0 = (int)std::floor(sx);
        const int y0 = (int)std::floor(sy);
        const int x1i = std::min(x0 + 1, src_w - 1);
        const int y1i = std::min(y0 + 1, src_h - 1);
        const float ax = sx - x0;
        const float ay = sy - y0;

        const uint8_t* p00 = src + (y0 * src_w + x0) * 3;
        const uint8_t* p10 = src + (y0 * src_w + x1i) * 3;
        const uint8_t* p01 = src + (y1i * src_w + x0) * 3;
        const uint8_t* p11 = src + (y1i * src_w + x1i) * 3;

        const float v00 = (float)p00[c];
        const float v10 = (float)p10[c];
        const float v01 = (float)p01[c];
        const float v11 = (float)p11[c];

        const float v0 = v00 + (v10 - v00) * ax;
        const float v1 = v01 + (v11 - v01) * ax;
        const float v = v0 + (v1 - v0) * ay;
        const int iv = (int)std::lround(v);
        return (uint8_t)std::clamp(iv, 0, 255);
    };

    for (int dy = 0; dy < dst_size; ++dy) {
        for (int dx = 0; dx < dst_size; ++dx) {
            const float sx = (dx - dst_c) / scale + cx;
            const float sy = (dy - dst_c) / scale + cy;
            uint8_t* outp = dst + (dy * dst_size + dx) * 3;
            outp[0] = sample(sx, sy, 0);
            outp[1] = sample(sx, sy, 1);
            outp[2] = sample(sx, sy, 2);
        }
    }
}

void EmotionRunner::packInput(const uint8_t* rgb_hwc_u8) {
    const int H = input_h_;
    const int W = input_w_;
    const int C = input_c_;

    const float qscale = input_tensor_cache_.quant_param.scale;
    const int qzp = input_tensor_cache_.quant_param.zero_point;

    auto to_real = [&](uint8_t px, int c) -> float {
        const float v = (float)px;
        return (v - mean_[c]) * scale_[c];
    };

    auto store_real = [&](size_t idx, float real) {
        switch (input_type_) {
            case MA_TENSOR_TYPE_F32:
                input_f32_[idx] = real;
                break;
            case MA_TENSOR_TYPE_BF16:
                input_u16_[idx] = fp32_to_bf16(real);
                break;
            case MA_TENSOR_TYPE_F16:
                input_u16_[idx] = fp32_to_fp16(real);
                break;
            case MA_TENSOR_TYPE_S8: {
                const float inv = (qscale > 0.f) ? (1.0f / qscale) : 0.f;
                int q = (int)std::lround(real * inv) + qzp;
                q = std::clamp(q, -128, 127);
                input_s8_[idx] = (int8_t)q;
                break;
            }
            case MA_TENSOR_TYPE_U8: {
                const float inv = (qscale > 0.f) ? (1.0f / qscale) : 0.f;
                int q = (int)std::lround(real * inv) + qzp;
                q = std::clamp(q, 0, 255);
                input_u8_[idx] = (uint8_t)q;
                break;
            }
            default:
                break;
        }
    };

    if (input_is_chw_) {
        const size_t plane = (size_t)H * (size_t)W;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const uint8_t* p = rgb_hwc_u8 + ((size_t)y * (size_t)W + (size_t)x) * 3;
                for (int c = 0; c < C; ++c) {
                    store_real((size_t)c * plane + (size_t)y * (size_t)W + (size_t)x, to_real(p[c], c));
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const uint8_t* p = rgb_hwc_u8 + ((size_t)y * (size_t)W + (size_t)x) * 3;
                for (int c = 0; c < C; ++c) {
                    store_real(((size_t)y * (size_t)W + (size_t)x) * (size_t)C + (size_t)c, to_real(p[c], c));
                }
            }
        }
    }
}

static void softmax_argmax7(const std::vector<float>& logits, int& idx, float& prob) {
    idx = -1;
    prob = 0.f;
    if (logits.size() != 7) return;
    float m = -std::numeric_limits<float>::infinity();
    for (float v : logits) m = std::max(m, v);
    float sum = 0.f;
    for (float v : logits) sum += std::exp(v - m);
    if (sum <= 0.f) return;

    int best = 0;
    float bestp = 0.f;
    for (int i = 0; i < 7; ++i) {
        const float p = std::exp(logits[(size_t)i] - m) / sum;
        if (p > bestp) {
            bestp = p;
            best = i;
        }
    }
    idx = best;
    prob = bestp;
}

bool EmotionRunner::parseOutputs(EmotionResult& out) {
    const int out_n = engine_->getOutputSize();
    if (out_n <= 0) return false;

    auto tensor_numel = [&](const ma_tensor_t& t) -> size_t {
        size_t n = shape_numel(t.shape);
        if (n > 0) return n;
        const size_t es = elem_size(t.type);
        return (es > 0) ? (t.size / es) : 0;
    };

    const ma_tensor_t t = engine_->getOutput(0);
    const int n = (int)tensor_numel(t);
    if (n < 7) return false;

    std::vector<float> logits;
    logits.reserve(7);
    for (int i = 0; i < 7; ++i) logits.push_back(read_val(t, i));

    int idx = -1;
    float p = 0.f;
    softmax_argmax7(logits, idx, p);

    out.ok = (idx >= 0);
    out.emotion = idx;
    out.score = p;
    return out.ok;
}

bool EmotionRunner::infer(const uint8_t* rgb888, int src_w, int src_h,
                         float x1, float y1, float x2, float y2,
                         EmotionResult& out) {
    out = EmotionResult{};
    if (!inited_ || !engine_ || !rgb888) return false;
    if (src_w <= 0 || src_h <= 0) return false;
    if ((x2 - x1) < 10.f || (y2 - y1) < 10.f) return false;

    alignCropRgb(rgb888, src_w, src_h, x1, y1, x2, y2, input_rgb_.data(), input_size_);
    packInput(input_rgb_.data());

    ma_tensor_t tensor = {
        .size = input_numel_ * elem_size(input_type_),
        .is_physical = false,
        .is_variable = false,
    };
    switch (input_type_) {
        case MA_TENSOR_TYPE_F32:
            tensor.data.data = input_f32_.data();
            break;
        case MA_TENSOR_TYPE_F16:
        case MA_TENSOR_TYPE_BF16:
            tensor.data.data = input_u16_.data();
            break;
        case MA_TENSOR_TYPE_S8:
            tensor.data.data = input_s8_.data();
            break;
        case MA_TENSOR_TYPE_U8:
            tensor.data.data = input_u8_.data();
            break;
        default:
            return false;
    }

    engine_->setInput(0, tensor);
    if (engine_->run() != MA_OK) return false;
    return parseOutputs(out);
}

}  // namespace face_analysis
