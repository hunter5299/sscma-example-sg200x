#include "age_gender_race_runner.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>

namespace udp_face {

namespace {
bool debug_preprocess_enabled() {
    const char* v = std::getenv("UDP_FACE_DEBUG_PREPROCESS");
    if (!v || !*v) return false;
    return std::atoi(v) != 0;
}

void debug_print_patch_stats_once(const uint8_t* patch_rgb_hwc, int H, int W,
                                 int src_w, int src_h, int stride_bytes,
                                 float x1, float y1, float x2, float y2,
                                 bool input_is_chw, ma_tensor_type_t input_type) {
    static std::atomic<bool> printed{false};
    if (!debug_preprocess_enabled()) return;
    if (printed.exchange(true)) return;

    if (!patch_rgb_hwc || H <= 0 || W <= 0) return;

    // 统计 patch 像素（RGB，0..255）
    double sum[3] = {0, 0, 0};
    double sumsq[3] = {0, 0, 0};
    int minv[3] = {255, 255, 255};
    int maxv[3] = {0, 0, 0};

    const size_t n = (size_t)H * (size_t)W;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = patch_rgb_hwc + i * 3u;
        for (int c = 0; c < 3; ++c) {
            const int v = (int)p[c];
            sum[c] += v;
            sumsq[c] += (double)v * (double)v;
            minv[c] = std::min(minv[c], v);
            maxv[c] = std::max(maxv[c], v);
        }
    }

    auto mean_std = [&](int c, double& mean, double& stdv) {
        mean = sum[c] / (double)n;
        const double var = sumsq[c] / (double)n - mean * mean;
        stdv = std::sqrt(std::max(0.0, var));
    };
    double m0, s0, m1, s1, m2, s2;
    mean_std(0, m0, s0);
    mean_std(1, m1, s1);
    mean_std(2, m2, s2);

    const char* type_str = "UNKNOWN";
    switch (input_type) {
        case MA_TENSOR_TYPE_F32: type_str = "F32"; break;
        case MA_TENSOR_TYPE_F16: type_str = "F16"; break;
        case MA_TENSOR_TYPE_BF16: type_str = "BF16"; break;
        case MA_TENSOR_TYPE_S8: type_str = "S8"; break;
        case MA_TENSOR_TYPE_U8: type_str = "U8"; break;
        default: break;
    }

    printf("\n[DEBUG][AGR] preprocess enabled (UDP_FACE_DEBUG_PREPROCESS=1)\n");
    printf("[DEBUG][AGR] src: %dx%d stride=%d bytes, crop xyxy=(%.1f,%.1f)-(%.1f,%.1f)\n",
           src_w, src_h, stride_bytes, x1, y1, x2, y2);
    printf("[DEBUG][AGR] patch: %dx%d RGB(HWC,u8)\n", W, H);
    printf("[DEBUG][AGR] patch ch0: min=%d max=%d mean=%.2f std=%.2f\n", minv[0], maxv[0], m0, s0);
    printf("[DEBUG][AGR] patch ch1: min=%d max=%d mean=%.2f std=%.2f\n", minv[1], maxv[1], m1, s1);
    printf("[DEBUG][AGR] patch ch2: min=%d max=%d mean=%.2f std=%.2f\n", minv[2], maxv[2], m2, s2);
    printf("[DEBUG][AGR] engine input: layout=%s type=%s\n", input_is_chw ? "NCHW" : "NHWC", type_str);
    printf("[DEBUG][AGR] patch first 6 px RGB: ");
    for (int i = 0; i < 6; ++i) {
        const uint8_t* p = patch_rgb_hwc + (size_t)i * 3u;
        printf("(%u,%u,%u)%s", p[0], p[1], p[2], (i == 5) ? "\n" : " ");
    }
    printf("\n");
}
} // namespace

float AgeGenderRaceRunner::bf16_to_fp32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

float AgeGenderRaceRunner::fp16_to_fp32(uint16_t v) {
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

uint16_t AgeGenderRaceRunner::fp32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return (uint16_t)(u >> 16);
}

uint16_t AgeGenderRaceRunner::fp32_to_fp16(float v) {
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

size_t AgeGenderRaceRunner::elem_size(ma_tensor_type_t t) {
    switch (t) {
        case MA_TENSOR_TYPE_F32: return 4;
        case MA_TENSOR_TYPE_F16: return 2;
        case MA_TENSOR_TYPE_BF16: return 2;
        case MA_TENSOR_TYPE_S8: return 1;
        case MA_TENSOR_TYPE_U8: return 1;
        default: return 0;
    }
}

size_t AgeGenderRaceRunner::shape_numel(const ma_shape_t& s) {
    if (s.size <= 0) return 0;
    size_t n = 1;
    for (int i = 0; i < s.size; ++i) {
        if (s.dims[i] <= 0) return 0;
        n *= (size_t)s.dims[i];
    }
    return n;
}

float AgeGenderRaceRunner::read_val(const ma_tensor_t& t, int idx) const {
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

bool AgeGenderRaceRunner::init(const std::string& model_path) {
    engine_ = std::make_unique<ma::engine::EngineCVI>();
    if (engine_->init() != MA_OK) return false;
    if (engine_->load(model_path) != MA_OK) return false;
    if (!prepareInputTensor()) return false;

    input_rgb_.assign((size_t)input_h_ * (size_t)input_w_ * 3, 0);
    inited_ = true;
    return true;
}

bool AgeGenderRaceRunner::prepareInputTensor() {
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

void AgeGenderRaceRunner::alignCropRgb(const uint8_t* src, int src_w, int src_h,
                                      int src_stride_bytes,
                                      float x1, float y1, float x2, float y2,
                                      uint8_t* dst, int dst_size) const {
    if (src_stride_bytes <= 0) src_stride_bytes = src_w * 3;
    const int min_stride = src_w * 3;
    if (src_stride_bytes < min_stride) src_stride_bytes = min_stride;

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

        const uint8_t* p00 = src + (size_t)y0 * (size_t)src_stride_bytes + (size_t)x0 * 3u;
        const uint8_t* p10 = src + (size_t)y0 * (size_t)src_stride_bytes + (size_t)x1i * 3u;
        const uint8_t* p01 = src + (size_t)y1i * (size_t)src_stride_bytes + (size_t)x0 * 3u;
        const uint8_t* p11 = src + (size_t)y1i * (size_t)src_stride_bytes + (size_t)x1i * 3u;

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

void AgeGenderRaceRunner::packInput(const uint8_t* rgb_hwc_u8) {
    const int H = input_h_;
    const int W = input_w_;
    const int C = input_c_;

    const float qscale = input_tensor_cache_.quant_param.scale;
    const int qzp = input_tensor_cache_.quant_param.zero_point;

    // 备注（输入打包/预处理策略）：
    // AGR cvimodel 的输入类型在不同导出方式下可能不同：
    // - 如果模型在 TPU 侧 fuse 了 preprocess（减均值/归一化等），输入通常是 U8/S8，
    //   且 quant_param 可能表现为“默认值”（例如 scale=1,zp=0 或 scale<=0）。
    //   这类模型期望喂「原始像素」，由 TPU 在内部做预处理。
    // - 如果模型输入是 F32/F16/BF16，则通常期望我们在 CPU 侧做 (px-mean)*scale。
    //
    // 关键坑：当输入是 U8 且 quant_param 看起来是默认值时，
    // 若我们仍做 (px-mean)*scale 的 float 预处理再量化回 U8，很容易把输入压成接近全 0，
    // 导致模型输出“几乎恒定”。
    //
    // 因此：当输入是 U8/S8 且 quant 参数看起来是“未配置/默认”时，走 raw passthrough。
    const bool input_is_int = (input_type_ == MA_TENSOR_TYPE_U8 || input_type_ == MA_TENSOR_TYPE_S8);
    const bool quant_param_defaultish = (!std::isfinite(qscale) || qscale <= 0.f || (std::fabs(qscale - 1.0f) < 1e-6f && qzp == 0));
    const bool use_raw_passthrough = input_is_int && quant_param_defaultish;

    static std::atomic<bool> printed_quant{false};
    if (debug_preprocess_enabled() && !printed_quant.exchange(true)) {
        const char* tname = (input_type_ == MA_TENSOR_TYPE_U8)  ? "U8"
                           : (input_type_ == MA_TENSOR_TYPE_S8)  ? "S8"
                           : (input_type_ == MA_TENSOR_TYPE_F16) ? "F16"
                           : (input_type_ == MA_TENSOR_TYPE_BF16)? "BF16"
                           : (input_type_ == MA_TENSOR_TYPE_F32) ? "F32"
                           : "UNKNOWN";
        printf("[DEBUG][AGR] input tensor: type=%s quant(scale=%.8f zp=%d) mean=[%.3f %.3f %.3f] scale=[%.6f %.6f %.6f]\n",
               tname, qscale, qzp, mean_[0], mean_[1], mean_[2], scale_[0], scale_[1], scale_[2]);
        printf("[DEBUG][AGR] packInput mode: %s\n", use_raw_passthrough ? "raw_passthrough" : "normalize_then_quantize");
    }

    auto to_real = [&](uint8_t px, int c) -> float {
        const float v = (float)px;
        if (use_raw_passthrough) return v;
        return (v - mean_[c]) * scale_[c];
    };

    auto store_raw = [&](size_t idx, uint8_t px) {
        switch (input_type_) {
            case MA_TENSOR_TYPE_U8:
                input_u8_[idx] = px;
                break;
            case MA_TENSOR_TYPE_S8:
                // 常见约定：signed input 把 [0,255] 平移到 [-128,127]
                input_s8_[idx] = (int8_t)std::clamp((int)px - 128, -128, 127);
                break;
            default:
                break;
        }
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
                    const size_t idx = (size_t)c * plane + (size_t)y * (size_t)W + (size_t)x;
                    if (use_raw_passthrough) store_raw(idx, p[c]);
                    else store_real(idx, to_real(p[c], c));
                }
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const uint8_t* p = rgb_hwc_u8 + ((size_t)y * (size_t)W + (size_t)x) * 3;
                for (int c = 0; c < C; ++c) {
                    const size_t idx = ((size_t)y * (size_t)W + (size_t)x) * (size_t)C + (size_t)c;
                    if (use_raw_passthrough) store_raw(idx, p[c]);
                    else store_real(idx, to_real(p[c], c));
                }
            }
        }
    }

    // 只在开启调试时打印一次 pack 后的前几个元素，确认数值不是全 0/全常数
    static std::atomic<bool> printed_pack{false};
    if (debug_preprocess_enabled() && !printed_pack.exchange(true)) {
        const int k = 12;
        printf("[DEBUG][AGR] packed input first %d elems: ", k);
        for (int i = 0; i < k; ++i) {
            float v = 0.f;
            switch (input_type_) {
                case MA_TENSOR_TYPE_F32:
                    v = input_f32_[(size_t)i];
                    break;
                case MA_TENSOR_TYPE_BF16:
                    v = bf16_to_fp32(input_u16_[(size_t)i]);
                    break;
                case MA_TENSOR_TYPE_F16:
                    v = fp16_to_fp32(input_u16_[(size_t)i]);
                    break;
                case MA_TENSOR_TYPE_S8: {
                    const float scale = input_tensor_cache_.quant_param.scale;
                    const int zp = input_tensor_cache_.quant_param.zero_point;
                    if (!std::isfinite(scale) || scale <= 0.f) v = (float)input_s8_[(size_t)i];
                    else v = ((int)input_s8_[(size_t)i] - zp) * scale;
                    break;
                }
                case MA_TENSOR_TYPE_U8: {
                    const float scale = input_tensor_cache_.quant_param.scale;
                    const int zp = input_tensor_cache_.quant_param.zero_point;
                    if (!std::isfinite(scale) || scale <= 0.f) v = (float)input_u8_[(size_t)i];
                    else v = ((int)input_u8_[(size_t)i] - zp) * scale;
                    break;
                }
                default:
                    v = 0.f;
                    break;
            }
            printf("%.4f%s", v, (i == k - 1) ? "\n" : " ");
        }
    }
}

static void softmax_argmax(const std::vector<float>& logits, int& idx, float& prob) {
    idx = -1;
    prob = 0.f;
    if (logits.empty()) return;
    float m = -std::numeric_limits<float>::infinity();
    for (float v : logits) m = std::max(m, v);
    float sum = 0.f;
    for (float v : logits) sum += std::exp(v - m);
    if (sum <= 0.f) return;
    int best = 0;
    float bestp = 0.f;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float p = std::exp(logits[i] - m) / sum;
        if (p > bestp) {
            bestp = p;
            best = (int)i;
        }
    }
    idx = best;
    prob = bestp;
}

bool AgeGenderRaceRunner::parseOutputs(AgeGenderRaceResult& out) {
    const int out_n = engine_->getOutputSize();
    if (out_n <= 0) return false;

    auto tensor_numel = [&](const ma_tensor_t& t) -> size_t {
        size_t n = shape_numel(t.shape);
        if (n > 0) return n;
        const size_t es = elem_size(t.type);
        return (es > 0) ? (t.size / es) : 0;
    };

    auto read_logits = [&](const ma_tensor_t& t, int n) -> std::vector<float> {
        std::vector<float> v;
        v.reserve((size_t)n);
        for (int i = 0; i < n; ++i) v.push_back(read_val(t, i));
        return v;
    };

    std::vector<float> race_logits, gender_logits, age_logits;

    if (out_n == 1) {
        const ma_tensor_t t = engine_->getOutput(0);
        const int n = (int)tensor_numel(t);
        if (n >= 18) {
            for (int i = 0; i < 7; ++i) race_logits.push_back(read_val(t, i));
            for (int i = 0; i < 2; ++i) gender_logits.push_back(read_val(t, 7 + i));
            for (int i = 0; i < 9; ++i) age_logits.push_back(read_val(t, 9 + i));
        } else {
            return false;
        }
    } else {
        for (int oi = 0; oi < out_n; ++oi) {
            const ma_tensor_t t = engine_->getOutput(oi);
            const int n = (int)tensor_numel(t);
            if (n == 7 && race_logits.empty()) {
                race_logits = read_logits(t, 7);
            } else if (n == 2 && gender_logits.empty()) {
                gender_logits = read_logits(t, 2);
            } else if (n == 9 && age_logits.empty()) {
                age_logits = read_logits(t, 9);
            }
        }
        if (race_logits.size() != 7 || gender_logits.size() != 2 || age_logits.size() != 9) {
            return false;
        }
    }

    int race = -1, gender = -1, age = -1;
    float race_p = 0.f, gender_p = 0.f, age_p = 0.f;
    softmax_argmax(race_logits, race, race_p);
    softmax_argmax(gender_logits, gender, gender_p);
    softmax_argmax(age_logits, age, age_p);

    // 在 debug 模式下打印原始模型输出，确认是否每次都一样（预处理问题）还是 argmax 映射有问题
    static std::atomic<bool> printed_outputs{false};
    if (debug_preprocess_enabled() && !printed_outputs.exchange(true)) {
        printf("\n[DEBUG][AGR] model raw outputs (first inference):\n");
        printf("  race_logits  (7): ");
        for (size_t i = 0; i < race_logits.size(); ++i) printf("%.4f ", race_logits[i]);
        printf("-> argmax=%d (%.3f)\n", race, race_p);
        printf("  gender_logits(2): ");
        for (size_t i = 0; i < gender_logits.size(); ++i) printf("%.4f ", gender_logits[i]);
        printf("-> argmax=%d (%.3f)\n", gender, gender_p);
        printf("  age_logits   (9): ");
        for (size_t i = 0; i < age_logits.size(); ++i) printf("%.4f ", age_logits[i]);
        printf("-> argmax=%d (%.3f)\n", age, age_p);
    }

    out.ok = (race >= 0 && gender >= 0 && age >= 0);
    out.race = race;
    // 备注（gender 语义对齐）：
    // - FairFace 风格的模型里，gender head 常见约定为：0=Male, 1=Female。
    // - 但本项目 UDP 协议约定是：1=Male, 0=Female。
    // 为了保证“设备端 stdout / UDP 包 / Python 接收端”三者一致，这里必须做一次映射。
    if (gender == 0) out.gender = 1;
    else if (gender == 1) out.gender = 0;
    else out.gender = -1;
    out.age = age;
    out.race_score = race_p;
    out.gender_score = gender_p;
    out.age_score = age_p;
    return out.ok;
}

bool AgeGenderRaceRunner::infer(const uint8_t* rgb888, int src_w, int src_h,
                               float x1, float y1, float x2, float y2,
                               AgeGenderRaceResult& out) {
    return infer(rgb888, src_w, src_h, src_w * 3, x1, y1, x2, y2, out);
}

bool AgeGenderRaceRunner::infer(const uint8_t* rgb888, int src_w, int src_h, int src_stride_bytes,
                               float x1, float y1, float x2, float y2,
                               AgeGenderRaceResult& out) {
    out = AgeGenderRaceResult{};
    if (!inited_ || !engine_ || !rgb888) return false;
    if (src_w <= 0 || src_h <= 0) return false;
    if ((x2 - x1) < 10.f || (y2 - y1) < 10.f) return false;

    if (src_stride_bytes <= 0) src_stride_bytes = src_w * 3;
    alignCropRgb(rgb888, src_w, src_h, src_stride_bytes, x1, y1, x2, y2, input_rgb_.data(), input_size_);

    debug_print_patch_stats_once(
        input_rgb_.data(), input_size_, input_size_,
        src_w, src_h, src_stride_bytes,
        x1, y1, x2, y2,
        input_is_chw_, input_type_);

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

}  // namespace udp_face
