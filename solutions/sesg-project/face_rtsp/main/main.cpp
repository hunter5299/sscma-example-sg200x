/**
 * @file main.cpp
 * @brief RTSP 人脸分析：YOLO(人脸检测) + Age/Gender/Race + Emotion + RTSP 推流
 *
 * 特性：
 * - 不依赖 OpenCV（对齐裁剪/缩放用纯 CPU 实现）
 * - 相机 CH0：RGB888 物理地址 -> YOLO 推理 + 属性推理
 * - 相机 CH2：H264 -> RTSP 推流（低 CPU/低内存/低延时）
 *
 * 用法：
 *   ./face_rtsp yolo_face.cvimodel age_gender_race.cvimodel emotion.cvimodel [single|multi] [threshold] [skip] [rtsp_port] [rtsp_session]
 */

#include <chrono>
#include <signal.h>
#include <cstring>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <cctype>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include <sscma.h>
#include <video.h>

#include <sesg/stream_rtsp.h>

extern "C" {
#include <app_ipcam_sys.h>
#include <app_ipcam_venc.h>
#include <app_ipcam_vpss.h>
}

#include "detector_utils.h"
#include "face_types.h"
#include "age_gender_race_runner.h"
#include "emotion_runner.h"

using namespace ma;

#define TAG "face_rtsp"

static constexpr int DEFAULT_RTSP_PORT = 8554;
static constexpr const char* DEFAULT_RTSP_SESSION = "live";

struct LowLatencyDefaults {
    // VB pool block 数：越小内存越低、延时越低，但更容易掉帧。
    uint32_t vb_blk_num = 3;

    // VPSS 输出队列深度：越小延时越低。
    uint32_t vpss_depth = 1;

    // 编码参数：GOP 越小首开更快、拖影更少，但码率/画质会变。
    uint32_t gop = 25;
    uint32_t bitrate_kbps = 1500;
    VENC_RC_MODE_E rc_mode = VENC_RC_MODE_H264CBR;
};

static inline void append_filled_rect_px(std::vector<sesg::stream_rtsp::OverlayRect>& out, uint32_t canvas_w,
                                        uint32_t canvas_h, int x, int y, int w, int h, uint32_t argb) {
    if (canvas_w == 0 || canvas_h == 0) return;
    if (w <= 0 || h <= 0) return;
    if (x >= (int)canvas_w || y >= (int)canvas_h) return;
    if (x + w <= 0 || y + h <= 0) return;

    x = std::max(0, x);
    y = std::max(0, y);
    const int x2 = std::min<int>((int)canvas_w, x + w);
    const int y2 = std::min<int>((int)canvas_h, y + h);
    const int ww = x2 - x;
    const int hh = y2 - y;
    if (ww <= 0 || hh <= 0) return;

    sesg::stream_rtsp::OverlayRect r;
    r.x = (float)x / (float)canvas_w;
    r.y = (float)y / (float)canvas_h;
    r.w = (float)ww / (float)canvas_w;
    r.h = (float)hh / (float)canvas_h;
    r.argb = argb;
    // 利用“画框厚度=段宽/段高”的方式，让 outline 退化成近似填充。
    r.thickness = (uint16_t)std::max(1, std::min(ww, hh));
    out.push_back(r);
}

static inline uint8_t seven_seg_mask(int digit) {
    // bits: A B C D E F G (A=0)
    // 0: A B C D E F
    // 1: B C
    // 2: A B G E D
    // 3: A B C D G
    // 4: F G B C
    // 5: A F G C D
    // 6: A F E D C G
    // 7: A B C
    // 8: A B C D E F G
    // 9: A B C D F G
    static const uint8_t kMask[10] = {
        0b0111111,
        0b0000110,
        0b1011011,
        0b1001111,
        0b1100110,
        0b1101101,
        0b1111101,
        0b0000111,
        0b1111111,
        0b1101111,
    };
    if (digit < 0 || digit > 9) return 0;
    return kMask[digit];
}

static inline void append_digit_7seg(std::vector<sesg::stream_rtsp::OverlayRect>& out, uint32_t canvas_w,
                                     uint32_t canvas_h, int x, int y, int digit, uint32_t argb,
                                     int digit_w = 12, int digit_h = 20, int seg_t = 2) {
    const uint8_t m = seven_seg_mask(digit);
    if (m == 0) return;

    const int t = std::max(1, seg_t);
    const int w = std::max(6, digit_w);
    const int h = std::max(10, digit_h);
    const int hlen = std::max(1, w - 2 * t);
    const int vlen = std::max(1, (h - 3 * t) / 2);

    // A
    if (m & (1 << 0)) append_filled_rect_px(out, canvas_w, canvas_h, x + t, y + 0, hlen, t, argb);
    // B
    if (m & (1 << 1)) append_filled_rect_px(out, canvas_w, canvas_h, x + w - t, y + t, t, vlen, argb);
    // C
    if (m & (1 << 2)) append_filled_rect_px(out, canvas_w, canvas_h, x + w - t, y + t + vlen + t, t, vlen, argb);
    // D
    if (m & (1 << 3)) append_filled_rect_px(out, canvas_w, canvas_h, x + t, y + h - t, hlen, t, argb);
    // E
    if (m & (1 << 4)) append_filled_rect_px(out, canvas_w, canvas_h, x + 0, y + t + vlen + t, t, vlen, argb);
    // F
    if (m & (1 << 5)) append_filled_rect_px(out, canvas_w, canvas_h, x + 0, y + t, t, vlen, argb);
    // G
    if (m & (1 << 6)) append_filled_rect_px(out, canvas_w, canvas_h, x + t, y + t + vlen, hlen, t, argb);
}

static void apply_low_latency_defaults(video_ch_index_t ch, uint32_t fps, const LowLatencyDefaults& d) {
    // VB pool
    if (auto* sys = app_ipcam_Sys_Param_Get()) {
        if (ch < sys->vb_pool_num) {
            sys->vb_pool[ch].vb_blk_num = d.vb_blk_num;
        }
    }

    // VPSS depth
    if (auto* vpss = app_ipcam_Vpss_Param_Get()) {
        if (vpss->u32GrpCnt > 0) {
            auto& grp0 = vpss->astVpssGrpCfg[0];
            if (ch < VPSS_MAX_PHY_CHN_NUM) {
                grp0.astVpssChnAttr[ch].u32Depth = d.vpss_depth;
            }
        }
    }

    // VENC params (only meaningful for encoded channels)
    if (auto* venc = app_ipcam_Venc_Param_Get()) {
        if (ch < venc->s32VencChnCnt) {
            auto& vcfg = venc->astVencChnCfg[ch];
            vcfg.u32Gop = d.gop;
            vcfg.enRcMode = d.rc_mode;
            vcfg.u32BitRate = d.bitrate_kbps;
            vcfg.u32MaxBitRate = d.bitrate_kbps;
            vcfg.statTime = 1;
            vcfg.u32SrcFrameRate = fps;
            vcfg.u32DstFrameRate = fps;
        }
    }
}

static std::atomic<bool> g_running(true);
void sig_handler(int) { g_running.store(false); }

// 主循环日志降频：每 N 次推理打印一次（不是每 N 帧）。
// 例如：推理频率≈20fps 时，N=20 约等于 1 秒打印一次。
// 该值支持通过命令行参数 [log_every_n_infer] 调整。
static int g_log_every_n_infer = 20;

static inline float sigmoidf_safe(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static inline float bf16_to_fp32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline float fp16_to_fp32(uint16_t v) {
    // IEEE 754 half -> float
    const uint32_t sign = (uint32_t)(v & 0x8000u) << 16;
    const uint32_t exp = (v & 0x7C00u) >> 10;
    const uint32_t mant = (v & 0x03FFu);

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // subnormal
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
        // inf/nan
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        // normal
        const uint32_t exp32 = (exp + (127 - 15)) << 23;
        const uint32_t mant32 = mant << 13;
        out = sign | exp32 | mant32;
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static inline float det_iou_center_xywh(const detector_utils::Detection& a, const detector_utils::Detection& b) {
    // a/b: normalized center (x,y,w,h)
    const float ax1 = a.x - a.w * 0.5f;
    const float ay1 = a.y - a.h * 0.5f;
    const float ax2 = a.x + a.w * 0.5f;
    const float ay2 = a.y + a.h * 0.5f;
    const float bx1 = b.x - b.w * 0.5f;
    const float by1 = b.y - b.h * 0.5f;
    const float bx2 = b.x + b.w * 0.5f;
    const float by2 = b.y + b.h * 0.5f;

    const float inter_x1 = std::max(ax1, bx1);
    const float inter_y1 = std::max(ay1, by1);
    const float inter_x2 = std::min(ax2, bx2);
    const float inter_y2 = std::min(ay2, by2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float area_a = a.w * a.h;
    const float area_b = b.w * b.h;
    const float union_area = area_a + area_b - inter_area;
    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

static std::vector<detector_utils::Detection> nms_center_xywh(std::vector<detector_utils::Detection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<detector_utils::Detection> out;
    std::vector<bool> sup(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (sup[j]) continue;
            if (det_iou_center_xywh(dets[i], dets[j]) > iou_thresh) sup[j] = true;
        }
    }
    return out;
}

static std::vector<detector_utils::Detection> parse_single_output_center_xywh(ma::engine::Engine* engine, float threshold,
                                                                             float nms_thresh, int img_w, int img_h) {
    if (!engine || engine->getOutputSize() < 1) return {};

    auto out0 = engine->getOutput(0);
    auto shape = engine->getOutputShape(0);
    // Expect [1,5,N,(1)] or [1,N,5,(1)]
    if (shape.size < 3) return {};

    const int d1 = shape.dims[1];
    const int d2 = shape.dims[2];
    int num_boxes = 0;
    bool prefer_ch_first = false;
    if (d1 == 5) {
        num_boxes = d2;
        prefer_ch_first = true;
    } else if (d2 == 5) {
        num_boxes = d1;
        prefer_ch_first = false;
    } else {
        return {};
    }

    const int sample_n = std::min(256, num_boxes);
    std::vector<detector_utils::Detection> dets;
    dets.reserve((size_t)num_boxes);

    static bool debug_printed = false;

    auto type_name = [&]() -> const char* {
        switch (out0.type) {
            case MA_TENSOR_TYPE_F32: return "FLOAT32";
            case MA_TENSOR_TYPE_F16: return "FLOAT16";
            case MA_TENSOR_TYPE_BF16: return "BF16";
            case MA_TENSOR_TYPE_S8: return "INT8";
            default: return "UNKNOWN";
        }
    };

    auto plausible = [&](float max_v, int nonzero) {
        // score usually in [0,1] (sigmoid) or logits within a small range
        return nonzero > sample_n / 10 && max_v > 0.01f && max_v < 20.0f;
    };

    auto read_val = [&](int idx) -> float {
        switch (out0.type) {
            case MA_TENSOR_TYPE_F32: {
                float* data = (float*)out0.data.data;
                return data[idx];
            }
            case MA_TENSOR_TYPE_F16: {
                uint16_t* data = (uint16_t*)out0.data.data;
                return fp16_to_fp32(data[idx]);
            }
            case MA_TENSOR_TYPE_BF16: {
                uint16_t* data = (uint16_t*)out0.data.data;
                return bf16_to_fp32(data[idx]);
            }
            case MA_TENSOR_TYPE_S8: {
                int8_t* data = (int8_t*)out0.data.data;
                const float scale = out0.quant_param.scale;
                const int zp = out0.quant_param.zero_point;
                return (data[idx] - zp) * scale;
            }
            default:
                return 0.0f;
        }
    };

    auto get_ch_first = [&](int c, int i) -> float { return read_val(c * num_boxes + i); };
    auto get_box_first = [&](int c, int i) -> float { return read_val(i * 5 + c); };

    float smax_a = -1e9f, smax_b = -1e9f;
    int snz_a = 0, snz_b = 0;
    for (int i = 0; i < sample_n; ++i) {
        float va = get_ch_first(4, i);
        float vb = get_box_first(4, i);
        smax_a = std::max(smax_a, va);
        smax_b = std::max(smax_b, vb);
        if (fabsf(va) > 1e-12f) snz_a++;
        if (fabsf(vb) > 1e-12f) snz_b++;
    }

    bool use_ch_first = prefer_ch_first;
    const bool a_ok = plausible(smax_a, snz_a);
    const bool b_ok = plausible(smax_b, snz_b);
    if (a_ok && !b_ok) use_ch_first = true;
    else if (!a_ok && b_ok) use_ch_first = false;

    if (!debug_printed) {
        debug_printed = true;
        printf("\n[调试] 单输出解析 (%s):\n", type_name());
        if (shape.size >= 4)
            printf("  Shape: [%d, %d, %d, %d]\n", shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]);
        else
            printf("  Shape: [%d, %d, %d]\n", shape.dims[0], shape.dims[1], shape.dims[2]);
        printf("  采样score统计 (按 [5,N]): max=%.6f nonzero=%d/%d\n", smax_a, snz_a, sample_n);
        printf("  采样score统计 (按 [N,5]): max=%.6f nonzero=%d/%d\n", smax_b, snz_b, sample_n);
        printf("  选择布局: %s\n\n", use_ch_first ? "[1,5,N,(1)]" : "[1,N,5,(1)]");
    }

    if (out0.type != MA_TENSOR_TYPE_F32 && out0.type != MA_TENSOR_TYPE_F16 && out0.type != MA_TENSOR_TYPE_BF16 &&
        out0.type != MA_TENSOR_TYPE_S8) {
        return {};
    }

    auto get = [&](int c, int i) -> float { return use_ch_first ? get_ch_first(c, i) : get_box_first(c, i); };

    for (int i = 0; i < num_boxes; ++i) {
        float x_center = get(0, i);
        float y_center = get(1, i);
        float w = get(2, i);
        float h = get(3, i);
        float score_raw = get(4, i);

        float score = score_raw;
        if (score_raw < -0.1f || score_raw > 1.1f) score = sigmoidf_safe(score_raw);
        if (score < threshold) continue;

        detector_utils::Detection d;
        d.x = x_center / (float)img_w;
        d.y = y_center / (float)img_h;
        d.w = w / (float)img_w;
        d.h = h / (float)img_h;
        d.score = score;
        d.target = 0;

        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.w) || !std::isfinite(d.h)) continue;
        if (d.w <= 0.0f || d.h <= 0.0f) continue;
        if (d.w > 2.0f || d.h > 2.0f) continue;

        dets.push_back(d);
    }

    return nms_center_xywh(dets, nms_thresh);
}

// ========== 主函数 ==========
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage:\n");
        printf("  %s yolo_face.cvimodel age_gender_race.cvimodel emotion.cvimodel [single|multi] [threshold] [skip] [rtsp_port] [rtsp_session] [log_every_n_infer]\n", argv[0]);
        printf("  single|multi: YOLO head type (default multi; use single for single-output models)\n");
        printf("  threshold: detection threshold (multi default 0.5; single default 0.7)\n");
        printf("  skip: infer every N frames (multi default 3; single default 1)\n");
        printf("  rtsp_port/rtsp_session: enable RTSP when port is set (default 8554/live)\n");
        printf("  log_every_n_infer: log every N inferences (default 20; 0 disables)\n");
        printf("\nExamples:\n");
        printf("  %s yolo.cvimodel agr.cvimodel emo.cvimodel\n", argv[0]);
        printf("  %s yolo.cvimodel agr.cvimodel emo.cvimodel single\n", argv[0]);
        printf("  %s yolo.cvimodel agr.cvimodel emo.cvimodel single 0.7 1 8554 live 20\n", argv[0]);
        exit(-1);
    }

    const char* yolo_model_path = argv[1];
    const char* agr_model_path = argv[2];
    const char* emotion_model_path = argv[3];

    enum class YoloHeadMode { Multi, Single };

    // 默认参数（按你的习惯：single 默认更严格阈值+不跳帧；并且默认不启 UDP）
    YoloHeadMode head_mode = YoloHeadMode::Multi;
    float threshold = 0.5f;
    int skip_interval = 3;
    int rtsp_port = 0;
    const char* rtsp_session = nullptr;
    bool enable_rtsp = false;

    // 参数顺序（新）：yolo agr emotion [single|multi] [threshold] [skip] [rtsp_port] [rtsp_session] [log_every_n_infer]
    // 兼容：如果第三个参数不是 single|multi，则按旧顺序把它当作 threshold。
    int argi = 4;
    if (argc >= 5) {
        std::string maybe_mode = argv[4];
        std::string lower = maybe_mode;
        for (auto& c : lower) c = std::tolower(static_cast<unsigned char>(c));
        if (lower == "single" || lower == "multi") {
            head_mode = (lower == "single") ? YoloHeadMode::Single : YoloHeadMode::Multi;
            argi = 5;
        }
    }

    if (head_mode == YoloHeadMode::Single) {
        threshold = 0.7f;
        skip_interval = 1;
    }

    if (argc > argi) threshold = atof(argv[argi++]);
    if (argc > argi) skip_interval = atoi(argv[argi++]);
    if (argc > argi) rtsp_port = atoi(argv[argi++]);
    if (argc > argi) rtsp_session = argv[argi++];
    if (argc > argi) g_log_every_n_infer = atoi(argv[argi++]);

    if (rtsp_port > 0) enable_rtsp = true;
    if (rtsp_port <= 0) rtsp_port = DEFAULT_RTSP_PORT;
    if (!rtsp_session || std::strlen(rtsp_session) == 0) rtsp_session = DEFAULT_RTSP_SESSION;

    if (skip_interval < 1) skip_interval = 1;
    if (g_log_every_n_infer < 0) g_log_every_n_infer = 0;

    // ==================== 初始化引擎 ====================
    printf("\n========== RTSP 人脸分析 ==========\n");
    printf("YOLO 模型: %s\n", yolo_model_path);
    printf("AgeGenderRace 模型: %s\n", agr_model_path);
    printf("Emotion 模型: %s\n", emotion_model_path);
    printf("阈值: %.2f\n", threshold);
    if (skip_interval == 1) {
        printf("跳帧: 关闭（每帧都推理）\n");
    } else {
        printf("跳帧: 每%d帧推理1次\n", skip_interval);
    }
    if (enable_rtsp) {
        printf("RTSP: :%d/%s\n", rtsp_port, rtsp_session);
    } else {
        printf("RTSP: 关闭（未提供 rtsp_port）\n");
    }
    if (head_mode == YoloHeadMode::Single) {
        printf("YOLO 头: single (单输出模型)\n");
    } else {
        printf("YOLO 头: multi (多输出模型)\n");
    }
    printf("================================\n\n");

    auto* yolo_engine = new ma::engine::EngineCVI();
    if (yolo_engine->init() != MA_OK || yolo_engine->load(yolo_model_path) != MA_OK) {
        MA_LOGE(TAG, "yolo engine init/load failed");
        return 1;
    }

    // ==================== 输入尺寸 ====================
    int input_w = 0, input_h = 0;
    {
        auto s = yolo_engine->getInputShape(0);
        if (s.size == 4) {
            // [1,H,W,C] or [1,C,H,W]
            if (s.dims[3] == 3 || s.dims[3] == 1) {
                input_h = s.dims[1];
                input_w = s.dims[2];
            } else {
                input_h = s.dims[2];
                input_w = s.dims[3];
            }
        }
        if (input_w <= 0 || input_h <= 0) {
            input_w = 640;
            input_h = 640;
        }
    }

    // ==================== 创建检测器（multi: SSCMA detector；single: 自定义后处理） ====================
    ma::Model* yolo_model = nullptr;
    ma::model::Detector* detector = nullptr;
    detector_utils::SquareDetector* square_detector = nullptr;

    if (head_mode == YoloHeadMode::Multi) {
        yolo_model = ma::ModelFactory::create(yolo_engine);
        if (!yolo_model) {
            MA_LOGE(TAG, "ModelFactory::create failed. If your model is single-output, please append `single` as the last argument.");
            return 1;
        }
        if (yolo_model->getInputType() != MA_INPUT_TYPE_IMAGE || yolo_model->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
            MA_LOGE(TAG, "yolo model not supported (require detector bbox model)");
            return 1;
        }
        detector = static_cast<ma::model::Detector*>(yolo_model);
        detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
        square_detector = new detector_utils::SquareDetector(detector, input_w, input_h);
        printf("[检测器] YOLO 检测器已创建 (SSCMA 多输出)\n");
    } else {
        printf("[检测器] YOLO 检测器已创建 (单输出自定义解析)\n");
    }

    // ==================== AgeGenderRace 模型 ====================
    udp_face::AgeGenderRaceRunner agr;
    if (!agr.init(agr_model_path)) {
        MA_LOGE(TAG, "AgeGenderRace init failed");
        return 1;
    }
    // (px/255 - mean)/std => (px - 255*mean)/(255*std)
    // mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    agr.setPreprocess(
        123.675f, 116.28f, 103.53f,
        1.0f / (255.0f * 0.229f), 1.0f / (255.0f * 0.224f), 1.0f / (255.0f * 0.225f));
    agr.setCropScale(1.3f);
    printf("[ATTR] AgeGenderRace init, input=%d\n", agr.inputSize());

    // ==================== Emotion 模型 ====================
    udp_face::EmotionRunner emotion;
    if (!emotion.init(emotion_model_path)) {
        MA_LOGE(TAG, "Emotion init failed");
        return 1;
    }
    emotion.setPreprocess(0.f, 0.f, 0.f, 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f);
    emotion.setCropScale(1.3f);
    printf("[ATTR] Emotion init, input=%d\n", emotion.inputSize());

    // ==================== 初始化相机 ====================
    Device* device = Device::getInstance();
    Camera* camera = nullptr;
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);
            Camera::CtrlValue val;

            // CH0: 维持原逻辑（RGB888 + 物理地址），用于推理
            val.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, val);
            val.u16s[0] = input_w;
            val.u16s[1] = input_h;
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, val);
            val.i32 = 1;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, val);

            break;
        }
    }
    if (!camera) { 
        MA_LOGE(TAG, "No camera"); 
        return 1; 
    }

    // ==================== 低延时 + 低资源默认配置 ====================
    // 说明：这些参数直接作用于 app_ipcam 的全局参数结构体，必须在 startVideo() 前设置。
    // camera->startStream() 内部会调用 startVideo()。
    LowLatencyDefaults ll;
    // 推理通道：尽量低延时
    apply_low_latency_defaults(VIDEO_CH0, 25, ll);
    // 推流通道：H264 CBR + 小 GOP
    apply_low_latency_defaults(VIDEO_CH2, 25, ll);

    // ==================== 启动 RTSP（复用同一套 VI/VPSS/VENC） ====================
    sesg::stream_rtsp::SesgRtspVpssStreamer streamer;
    const uint32_t overlay_canvas_w = 1280;
    const uint32_t overlay_canvas_h = 720;
    if (enable_rtsp) {
        sesg::stream_rtsp::ServerConfig server_cfg;
        server_cfg.port = rtsp_port;

        sesg::stream_rtsp::ChannelConfig ch2;
        ch2.ch = VIDEO_CH2;
        ch2.format = VIDEO_FORMAT_H264;
        ch2.width = 1280;
        ch2.height = 720;
        ch2.fps = 25;
        // 设备端 OSD：把推理框叠加到 RTSP 画面（需要构建时打开 SESG_STREAM_RTSP_ENABLE_RGN）。
        ch2.enable_rgn_overlay = true;
        // 为了保持 URL 形态：rtsp://<ip>:<port>/<session>
        ch2.path = rtsp_session;

        // consumer_index=1：避免覆盖 camera 可能使用的 index=0。
        if (!streamer.startAttached(server_cfg, {ch2}, 1)) {
            MA_LOGE(TAG, "streamer.startAttached failed: port=%d session=%s", rtsp_port, rtsp_session);
            enable_rtsp = false;
        }
    }

    // 启动 camera（内部会 startVideo()）：CH0 推理 + CH2 硬编（由上面的 setupVideo 启用）
    camera->startStream(Camera::StreamMode::kRefreshOnReturn);
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // ==================== 主循环 ====================
    if (skip_interval == 1) {
        printf("\n[主线程] 启动：每帧推理\n\n");
    } else {
        printf("\n[主线程] 启动：每 %d 帧推理 1 次\n\n", skip_interval);
    }

    int frame_counter = 0;
    int infer_count = 0;
    
    // 性能统计
    // 下面统计都按“最近2秒窗口”计算，到点打印后会清零。
    double win_total_infer_ms = 0;
    double win_total_pre_ms = 0;
    double win_total_inf_ms = 0;
    double win_total_post_ms = 0;
    int win_infer_count = 0;

    double win_total_mmap_ms = 0;
    double win_total_munmap_ms = 0;
    double win_total_agr_ms = 0;
    double win_total_emo_ms = 0;
    int win_attr_frame_count = 0;
    int win_attr_face_count = 0;
    int win_agr_ok_count = 0;
    int win_emo_ok_count = 0;

    // RTSP 发送统计由 VENC->RTSP 回调内部完成（此处不重复统计，避免引入额外拷贝/队列）。
    
    auto fps_start = std::chrono::high_resolution_clock::now();
    int fps_frame_count = 0;

    // 缓存上次检测结果
    std::vector<udp_face::FaceResult> cached_faces;

    while (g_running.load()) {
        // CH0: RGB 帧用于推理（物理地址）
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        frame_counter++;
        fps_frame_count++;

        int frame_w = frame.width > 0 ? frame.width : input_w;
        int frame_h = frame.height > 0 ? frame.height : input_h;

        const bool do_inference = (skip_interval <= 1) ? true : (frame_counter % skip_interval == 0);

        // ========== 推理 ==========
        if (do_inference) {
            // 1) YOLO 输入直接使用物理地址（避免 memcpy），交给 CVI 引擎做 DMA 读取
            ma_tensor_t tensor = {
                .size        = frame.size,
                .is_physical = true,
                .is_variable = false,
            };
            tensor.data.data = reinterpret_cast<void*>(frame.data);
            yolo_engine->setInput(0, tensor);

            std::vector<detector_utils::Detection> results;
            detector_utils::PerfStats perf;

            if (head_mode == YoloHeadMode::Single) {
                auto t0 = std::chrono::high_resolution_clock::now();
                yolo_engine->run();
                auto t1 = std::chrono::high_resolution_clock::now();
                results = parse_single_output_center_xywh(yolo_engine, threshold, 0.45f, input_w, input_h);
                auto t2 = std::chrono::high_resolution_clock::now();
                perf.preprocess_ms = 0.0;
                perf.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                perf.postprocess_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            } else {
                perf = square_detector->run(frame.data, frame.size, results);
            }
            
            // 将 bbox + 属性合成 FaceResult（属性推理走虚拟地址）
            cached_faces.clear();

            // 2) Attribute inference (AGR + Emotion) needs CPU RGB: mmap physical address
            uint64_t phy_addr = (uint64_t)frame.data;
            void* vir_addr = nullptr;
            {
                const auto t0 = std::chrono::high_resolution_clock::now();
                vir_addr = CVI_SYS_Mmap(phy_addr, frame.size);
                const auto t1 = std::chrono::high_resolution_clock::now();
                win_total_mmap_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            const uint8_t* rgb_ptr = (const uint8_t*)vir_addr;
            if (vir_addr) win_attr_frame_count++;

            int stride_bytes = frame_w * 3;
            if (frame_h > 0 && frame.size >= (size_t)frame_h) {
                const size_t s = frame.size / (size_t)frame_h;
                if (s >= (size_t)frame_w * 3) stride_bytes = (int)s;
            }

            for (size_t i = 0; i < results.size() && i < (size_t)udp_face::MAX_DETECTIONS; ++i) {
                const auto& d = results[i];
                udp_face::FaceResult fr;
                fr.x = d.x;
                fr.y = d.y;
                fr.w = d.w;
                fr.h = d.h;
                fr.score = d.score;
                fr.target = d.target;
                fr.gender = -1;
                fr.age = -1;
                fr.race = -1;
                fr.emotion = -1;
                fr.gender_score = 0.f;
                fr.age_score = 0.f;
                fr.race_score = 0.f;
                fr.emotion_score = 0.f;

                if (vir_addr) {
                    const float cx = d.x;
                    const float cy = d.y;
                    const float bw = d.w;
                    const float bh = d.h;
                    const float x1 = (cx - bw * 0.5f) * frame_w;
                    const float y1 = (cy - bh * 0.5f) * frame_h;
                    const float x2 = (cx + bw * 0.5f) * frame_w;
                    const float y2 = (cy + bh * 0.5f) * frame_h;

                    win_attr_face_count++;

                    udp_face::AgeGenderRaceResult ar;
                    {
                        const auto t0 = std::chrono::high_resolution_clock::now();
                        const bool ok = agr.infer(rgb_ptr, frame_w, frame_h, stride_bytes, x1, y1, x2, y2, ar) && ar.ok;
                        const auto t1 = std::chrono::high_resolution_clock::now();
                        win_total_agr_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                        if (ok) {
                            win_agr_ok_count++;
                            fr.gender = ar.gender;
                            fr.age = ar.age;
                            fr.race = ar.race;
                            fr.gender_score = ar.gender_score;
                            fr.age_score = ar.age_score;
                            fr.race_score = ar.race_score;
                        }
                    }

                    udp_face::EmotionResult er;
                    {
                        const auto t0 = std::chrono::high_resolution_clock::now();
                        const bool ok = emotion.infer(rgb_ptr, frame_w, frame_h, x1, y1, x2, y2, er) && er.ok;
                        const auto t1 = std::chrono::high_resolution_clock::now();
                        win_total_emo_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                        if (ok) {
                            win_emo_ok_count++;
                            fr.emotion = er.emotion;
                            fr.emotion_score = er.score;
                        }
                    }
                }

                cached_faces.push_back(fr);
            }

            if (vir_addr) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                CVI_SYS_Munmap(vir_addr, frame.size);
                const auto t1 = std::chrono::high_resolution_clock::now();
                win_total_munmap_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            // 3) 设备端叠加：把人脸框同步到 CH2 的 RGN/OSD。
            // FaceResult: center(x,y,w,h)；OverlayRect: left-top(x,y,w,h)。
            if (enable_rtsp) {
                std::vector<sesg::stream_rtsp::OverlayRect> rects;
                rects.reserve(cached_faces.size());
                for (const auto& fr : cached_faces) {
                    const float x1 = fr.x - fr.w * 0.5f;
                    const float y1 = fr.y - fr.h * 0.5f;

                    sesg::stream_rtsp::OverlayRect r;
                    r.x = std::clamp(x1, 0.0f, 1.0f);
                    r.y = std::clamp(y1, 0.0f, 1.0f);
                    r.w = std::clamp(fr.w, 0.0f, 1.0f);
                    r.h = std::clamp(fr.h, 0.0f, 1.0f);
                    r.argb = 0xFFFF0000u; // red
                    r.thickness = 2;

                    // 极端情况下 bbox 可能越界，简单裁剪避免无效输入。
                    if (r.w <= 0.0f || r.h <= 0.0f) continue;
                    if (r.x >= 1.0f || r.y >= 1.0f) continue;
                    if (r.x + r.w <= 0.0f || r.y + r.h <= 0.0f) continue;

                    rects.push_back(r);

                    // Age: draw two digits near bbox top-left (if available).
                    if (fr.age >= 0) {
                        const int age = std::min(99, fr.age);
                        const int tens = age / 10;
                        const int ones = age % 10;

                        const int px = (int)(r.x * overlay_canvas_w);
                        const int py = (int)(r.y * overlay_canvas_h);
                        const int margin = 4;
                        const int digit_w = 12;
                        const int digit_h = 20;
                        const int seg_t = 2;
                        const int gap = 3;

                        // gender hint by color (optional): male=blue, female=magenta, unknown=yellow
                        uint32_t c = 0xFFFFFFFFu; // default white
                        if (fr.gender == 1) c = 0xFF00A0FFu;
                        else if (fr.gender == 0) c = 0xFFFF00FFu;
                        else c = 0xFFFFFF00u;

                        append_digit_7seg(rects, overlay_canvas_w, overlay_canvas_h, px + margin, py + margin, tens, c,
                                          digit_w, digit_h, seg_t);
                        append_digit_7seg(rects, overlay_canvas_w, overlay_canvas_h,
                                          px + margin + digit_w + gap, py + margin, ones, c, digit_w, digit_h, seg_t);
                    }
                }
                (void)streamer.updateOverlayRects(VIDEO_CH2, rects);
            }

            infer_count++;
            win_infer_count++;
            
            // 统计
            win_total_infer_ms += perf.total();
            win_total_pre_ms += perf.preprocess_ms;
            win_total_inf_ms += perf.inference_ms;
            win_total_post_ms += perf.postprocess_ms;
            
                 // 3) 打印很影响性能：按 g_log_every_n_infer 降频（0=关闭）
                 if (g_log_every_n_infer > 0 && (infer_count % g_log_every_n_infer) == 0) {
                  printf("[推理 #%d] 总计=%.1f ms (前处理=%.1f, 推理=%.1f, 后处理=%.1f) | 检测=%zu\n",
                      infer_count, perf.total(),
                      perf.preprocess_ms, perf.inference_ms, perf.postprocess_ms,
                      results.size());
                 }
        }

        camera->returnFrame(frame);

        // RTSP 推流在独立线程中进行

        // ========== FPS统计 ==========
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count();
        if (elapsed >= 2000) {
            float video_fps = fps_frame_count * 1000.0f / elapsed;
            
            printf("\n========== 性能统计 (最近2秒) ==========\n");
            printf("观感 FPS: %.1f\n", video_fps);
            printf("窗口帧数: %d | 窗口推理: %d\n", fps_frame_count, win_infer_count);

            // YOLO (exclude AGR/Emotion and mmap)
            if (win_infer_count > 0) {
                printf("\n平均 YOLO 耗时 (窗口内):\n");
                printf("  前处理: %.1f ms\n", win_total_pre_ms / win_infer_count);
                printf("  推理:   %.1f ms\n", win_total_inf_ms / win_infer_count);
                printf("  后处理: %.1f ms\n", win_total_post_ms / win_infer_count);
                printf("  总计:   %.1f ms\n", win_total_infer_ms / win_infer_count);
            }

            // AGR + Emotion + mmap
            if (win_attr_frame_count > 0) {
                printf("\n平均 属性/表情 额外耗时 (窗口内):\n");
                printf("  mmap:   %.1f ms\n", win_total_mmap_ms / win_attr_frame_count);
                printf("  munmap: %.1f ms\n", win_total_munmap_ms / win_attr_frame_count);
                printf("  AGR总计: %.1f ms/帧 | 每脸: %.1f ms/脸 (faces=%d, ok=%d)\n",
                       win_total_agr_ms / win_attr_frame_count,
                       (win_attr_face_count > 0) ? (win_total_agr_ms / win_attr_face_count) : 0.0,
                       win_attr_face_count,
                       win_agr_ok_count);
                printf("  EMO总计: %.1f ms/帧 | 每脸: %.1f ms/脸 (faces=%d, ok=%d)\n",
                       win_total_emo_ms / win_attr_frame_count,
                       (win_attr_face_count > 0) ? (win_total_emo_ms / win_attr_face_count) : 0.0,
                       win_attr_face_count,
                       win_emo_ok_count);
            }

            if (enable_rtsp) {
                printf("\nRTSP 推流:\n");
                printf("  url: rtsp://<device-ip>:%d/%s\n", rtsp_port, rtsp_session);
                printf("  enc: H264 CBR %u kbps | GOP=%u | vb_blk=%u | vpss_depth=%u\n",
                       ll.bitrate_kbps, ll.gop, ll.vb_blk_num, ll.vpss_depth);
            }
            printf("======================================\n\n");

            // reset window counters
            win_total_infer_ms = win_total_pre_ms = win_total_inf_ms = win_total_post_ms = 0;
            win_infer_count = 0;
            win_total_mmap_ms = win_total_munmap_ms = 0;
            win_total_agr_ms = win_total_emo_ms = 0;
            win_attr_frame_count = win_attr_face_count = 0;
            win_agr_ok_count = win_emo_ok_count = 0;
            fps_frame_count = 0;
            fps_start = now;
        }
    }

    // ==================== 清理 ====================
    printf("\n[停止] 清理资源...\n");

    streamer.stopAttached();

    camera->stopStream();
    for (auto& sensor : device->getSensors()) sensor->deInit();
    
    if (square_detector) delete square_detector;
    if (yolo_model) ma::ModelFactory::remove(yolo_model);
    delete yolo_engine;

    printf("[完成]\n");
    return 0;
}
