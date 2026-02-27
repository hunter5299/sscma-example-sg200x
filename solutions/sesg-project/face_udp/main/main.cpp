/**
 * @file main.cpp
 * @brief UDP 人脸分析：YOLO(人脸检测) + Age/Gender/Race(FairFace) + Emotion + UDP 推送
 *
 * 特性：
 * - 不依赖 OpenCV（对齐裁剪/缩放用纯 CPU 实现）
 * - 相机 CH0：RGB888 物理地址 -> YOLO 推理 + 属性/表情推理
 * - 相机 CH1：JPEG -> UDP 推送（带人脸框 + race/gender/age/emotion）
 */

#include <chrono>
#include <signal.h>
#include <cstring>
#include <cstdlib>
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

#include "detector_utils.h"
#include "face_types.h"
#include "age_gender_race_runner.h"
#include "emotion_runner.h"
#include <udp_service/udp_service.h>

using namespace ma;

#define TAG "face_udp"
#define UDP_IP "192.168.2.101"
#define UDP_PORT 5001

static constexpr uint32_t UDP_MAGIC = 0xFACEBEEF;

// 备注（UDP 协议）：
// - sender（本程序）会把 JPEG（CH1）与检测/属性结果一起打包通过 UDP 发送。
// - FaceResult 的 C++ 结构体在 face_types.h 中定义，当前协议为 56 bytes：
//   '<fffffiiiiiffff'
//   (x,y,w,h,score,target,gender,age,race,emotion, gender_score,age_score,race_score,emotion_score)
// - 仓库里历史上也存在 40 bytes 的旧协议；Python 接收端已做自动判定，避免解包错位。

static std::atomic<bool> g_running(true);
void sig_handler(int) { g_running.store(false); }

// 主循环日志降频：每 N 次推理打印一次（不是每 N 帧）。
// 例如：推理频率≈20fps 时，N=20 约等于 1 秒打印一次。
// 该值支持通过命令行参数 [log_every_n_infer] 调整。
static int g_log_every_n_infer = 20;

// 备注：debug 开关
// - 运行时传最后一个参数 debug=1，会在程序内 setenv("UDP_FACE_DEBUG_PREPROCESS","1")
// - 该 env 变量用于打开 AGR runner 内部的调试输出（例如输入 patch 统计、packed input 预览、logits）。

static bool env_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    if (v[0] == '\0') return true;
    std::string s(v);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

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

static const char* agr_gender_label(int gender) {
    // 备注：此处的 gender 使用 UDP 协议语义（与 C++ 打包/与 Python 接收端一致）：
    // 1=Male, 0=Female。
    if (gender == 1) return "Male";
    if (gender == 0) return "Female";
    return "N/A";
}

static const char* agr_age_bin_label(int age_bin) {
    static const char* kBins[9] = {"0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"};
    if (age_bin >= 0 && age_bin < 9) return kBins[age_bin];
    return "N/A";
}

static const char* agr_age_range_label(int age_bin) {
    // 备注：模型输出为 9-bin（0..8），但用户侧更喜欢“10 年一档”的粗粒度区间。
    // 这里做一个简单合并，方便肉眼检查（不改变 UDP 协议里传的 age bin id）。
    // 用户侧更关心“10年一档”，这里把 9-bin 映射成粗粒度区间
    // 0:0-2,1:3-9 -> 0-10
    // 2 -> 10-20, 3 -> 20-30, 4 -> 30-40, 5 -> 40-50, 6 -> 50-60, 7 -> 60-70, 8 -> 70+
    if (age_bin == 0 || age_bin == 1) return "0-10";
    if (age_bin == 2) return "10-20";
    if (age_bin == 3) return "20-30";
    if (age_bin == 4) return "30-40";
    if (age_bin == 5) return "40-50";
    if (age_bin == 6) return "50-60";
    if (age_bin == 7) return "60-70";
    if (age_bin == 8) return "70+";
    return "N/A";
}

static const char* agr_race_label(int race) {
    // 备注：race label 顺序与 FairFace 常用定义一致（与 Python 接收端一致）。
    static const char* kLabels[7] = {
        "White",
        "Black",
        "Latino_Hispanic",
        "East_Asian",
        "Southeast_Asian",
        "Indian",
        "Middle_Eastern",
    };
    if (race >= 0 && race < 7) return kLabels[race];
    return "N/A";
}

static const char* emo_label(int emo) {
    // 备注：与 Python 接收端的 emo_labels 对齐
    static const char* kEmo[7] = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"};
    if (emo >= 0 && emo < 7) return kEmo[emo];
    return "N/A";
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
        printf("  %s yolo_face.cvimodel age_gender_race.cvimodel emotion.cvimodel [single|multi] [threshold] [skip] [udp_ip] [udp_port] [log_every_n_infer] [debug]\n", argv[0]);
        printf("  single|multi: YOLO 头类型 (默认 multi；传 single 启用单输出模型路径)\n");
        printf("  threshold: 检测阈值 (multi 默认0.5；single 默认0.7)\n");
        printf("  skip: 每N帧推理1次 (multi 默认3；single 默认1=不跳帧)\n");
        printf("  udp_ip/udp_port: 只有同时提供才启动 UDP（否则不发送 JPEG/不启用CH1）\n");
        printf("  log_every_n_infer: 每N次推理打印一次性能日志 (默认20；传0关闭)\n");
        printf("  debug: 启用调试输出 (传1开启；默认0关闭)\n");
        printf("\n示例:\n");
        printf("  %s yolo.cvimodel age_gender_race.cvimodel emotion.cvimodel\n", argv[0]);
        printf("  %s yolo.cvimodel age_gender_race.cvimodel emotion.cvimodel single\n", argv[0]);
        printf("  %s yolo.cvimodel age_gender_race.cvimodel emotion.cvimodel single 0.7 1 192.168.1.100 5001 1 1\n", argv[0]);
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
    const char* udp_ip = nullptr;
    int udp_port = 0;
    bool enable_udp = false;

    // 备注（参数解析）：
    // 参数顺序：yolo agr emotion [single|multi] [threshold] [skip] [udp_ip] [udp_port] [log_every_n_infer] [debug]
    // - single/multi 可选；如果不传，默认 multi
    // - 为兼容历史调用：如果 argv[4] 不是 single|multi，会把它当作 threshold
    // - 末尾参数兼容两种：
    //   a) ... udp_port debug
    //   b) ... udp_port log_every_n_infer debug
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
    if (argc > argi) udp_ip = argv[argi++];
    if (argc > argi) udp_port = atoi(argv[argi++]);

    // 兼容两种尾部参数写法：
    // - ... udp_port debug
    // - ... udp_port log_every_n_infer debug
    bool debug_preprocess = false;
    if (argc > argi) {
        const int v0 = atoi(argv[argi++]);
        if (argc > argi) {
            // 两个参数都提供：第一个是 log_every_n_infer，第二个是 debug
            g_log_every_n_infer = v0;
            debug_preprocess = (atoi(argv[argi++]) != 0);
        } else {
            // 只提供一个参数：按你的诉求，当作 debug
            debug_preprocess = (v0 != 0);
        }
    }

    if (udp_ip && udp_port > 0) enable_udp = true;

    if (skip_interval < 1) skip_interval = 1;
    if (g_log_every_n_infer < 0) g_log_every_n_infer = 0;

    // ==================== 初始化引擎 ====================
    printf("\n========== UDP 人脸分析 ==========\n");
    printf("YOLO 模型: %s\n", yolo_model_path);
    printf("AgeGenderRace 模型: %s\n", agr_model_path);
    printf("Emotion 模型: %s\n", emotion_model_path);
    printf("阈值: %.2f\n", threshold);
    if (skip_interval == 1) {
        printf("跳帧: 关闭（每帧都推理）\n");
    } else {
        printf("跳帧: 每%d帧推理1次\n", skip_interval);
    }
    if (enable_udp) {
        printf("UDP: %s:%d\n", udp_ip, udp_port);
    } else {
        printf("UDP: 关闭（未提供 udp_ip/udp_port）\n");
    }
    if (head_mode == YoloHeadMode::Single) {
        printf("YOLO 头: single (单输出模型)\n");
    } else {
        printf("YOLO 头: multi (多输出模型)\n");
    }
    printf("UDP 协议: FaceResult=%zu bytes (Python fmt: <fffffiiiiiffff)\n", sizeof(udp_face::FaceResult));
    printf("================================\n\n");

    // 备注：debug 参数开启后
    // 1) 主程序会打印更多信息（例如帧 stride、face0 结果等）
    // 2) 让 AGR runner 也能打印（它内部仍使用 UDP_FACE_DEBUG_PREPROCESS 作为开关）
    if (debug_preprocess) {
        setenv("UDP_FACE_DEBUG_PREPROCESS", "1", 1);
        printf("[DEBUG] Debug mode enabled (argv debug=1)\n");
        if (g_log_every_n_infer == 0) g_log_every_n_infer = 1;
    }

    auto* yolo_engine = new ma::engine::EngineCVI();
    if (yolo_engine->init() != MA_OK || yolo_engine->load(yolo_model_path) != MA_OK) {
        fprintf(stderr, "[ERROR] yolo engine init/load failed: %s\n", yolo_model_path);
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

    // ==================== Age/Gender/Race 模型 ====================
    udp_face::AgeGenderRaceRunner agr;
    if (!agr.init(agr_model_path)) {
        MA_LOGE(TAG, "AgeGenderRace init failed");
        return 1;
    }
    // 对齐 face_2/c++： (px/255 - mean)/std => (px - 255*mean)/(255*std)
    // mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    agr.setPreprocess(
        123.675f, 116.28f, 103.53f,
        1.0f / (255.0f * 0.229f), 1.0f / (255.0f * 0.224f), 1.0f / (255.0f * 0.225f));
    agr.setCropScale(1.3f);
    printf("[属性] AgeGenderRace 已初始化，输入=%d\n", agr.inputSize());

    // ==================== Emotion 模型 ====================
    udp_face::EmotionRunner emotion;
    if (!emotion.init(emotion_model_path)) {
        MA_LOGE(TAG, "Emotion init failed");
        return 1;
    }
    // 对齐 face_2/c++：灰度/归一化到 0..1。此处按 RGB 同步做 /255。
    emotion.setPreprocess(0.f, 0.f, 0.f, 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f);
    emotion.setCropScale(1.3f);
    printf("[表情] Emotion 已初始化，输入=%d\n", emotion.inputSize());

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

            // CH1: JPEG（走 VENC），用于 UDP 推送（不走 RTSP）
            if (enable_udp) {
                val.i32 = 1;
                camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, val);
                // JPEG 分辨率建议设小一些，确保单帧尽量 < 60KB，避免 UDP 分包丢包带来的花屏/解码失败
                val.u16s[0] = 320;
                val.u16s[1] = 240;
                camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, val);
                val.i32 = (int)MA_PIXEL_FORMAT_JPEG;
                camera->commandCtrl(Camera::CtrlType::kFormat, Camera::CtrlMode::kWrite, val);
                // 注意：如果这里的 fps 高于你的主循环处理能力，会导致 VENC 内部链表缓存积压并打印
                // “LL cache is full and drop data”。建议把 CH1 fps 设置为 <= 实际处理 fps。
                val.i32 = 10;
                camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, val);
            }

            break;
        }
    }
    if (!camera) { 
        MA_LOGE(TAG, "No camera"); 
        return 1; 
    }

    camera->startStream(Camera::StreamMode::kRefreshOnReturn);
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // ==================== 启动UDP服务（仅在 enable_udp 时） ====================
    std::unique_ptr<udp_service::UDPSender> udp_sender;
    if (enable_udp) {
        udp_sender = std::make_unique<udp_service::UDPSender>(udp_ip, udp_port);
        udp_sender->start();
    }

    // ==================== CH1 JPEG 拉流线程（避免主循环阻塞） ====================
    // CameraSG200X::retrieveFrame(JPEG) 内部会用 1000/fps 作为 fetch 超时，
    // 若在主循环里每次都调用，会把主循环“卡”到接近 CH1 fps（例如 fps=10 时会卡到 ~7-10fps）。
    struct JpegFrameOwned {
        std::unique_ptr<uint8_t[]> buf;
        size_t size = 0;
        int w = 0;
        int h = 0;
    };

    std::mutex jpeg_mu;
    JpegFrameOwned latest_jpeg;
    std::atomic<bool> jpeg_running{false};
    std::thread jpeg_thread;

    // JPEG 拉流（CH1）统计：count + retrieveFrame 等待时长（包含阻塞等待）
    std::atomic<uint64_t> jpeg_fetch_count{0};
    std::atomic<uint64_t> jpeg_fetch_wait_us{0};

    if (udp_sender) {
        jpeg_running.store(true);
        jpeg_thread = std::thread([&]() {
            ma_img_t jpeg;
            while (g_running.load() && jpeg_running.load()) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                if (camera->retrieveFrame(jpeg, MA_PIXEL_FORMAT_JPEG) != MA_OK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                const auto t1 = std::chrono::high_resolution_clock::now();
                jpeg_fetch_count.fetch_add(1);
                jpeg_fetch_wait_us.fetch_add(
                    (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

                std::unique_ptr<uint8_t[]> jpeg_buf(reinterpret_cast<uint8_t*>(jpeg.data));
                const size_t jpeg_size = jpeg.size;
                const int jpeg_w = jpeg.width;
                const int jpeg_h = jpeg.height;

                // 归还相机队列占用（但不释放我们接管的 buffer）
                jpeg.data = nullptr;
                jpeg.size = 0;
                camera->returnFrame(jpeg);

                {
                    std::lock_guard<std::mutex> lk(jpeg_mu);
                    latest_jpeg.buf = std::move(jpeg_buf);
                    latest_jpeg.size = jpeg_size;
                    latest_jpeg.w = jpeg_w;
                    latest_jpeg.h = jpeg_h;
                }
            }
        });
    }

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

    double win_total_udp_ms = 0;
    int win_udp_send_count = 0;
    uint64_t win_udp_send_bytes = 0;

    uint64_t last_jpeg_fetch_count = 0;
    uint64_t last_jpeg_fetch_wait_us = 0;
    
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

            // 2) GenderAge 需要 CPU 读 RGB 数据：把物理地址 mmap 成虚拟地址
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

            // RGB888 常见存在行对齐 padding：优先用 size/height 推断 stride（bytes per row）
            int stride_bytes = frame_w * 3;
            if (frame_h > 0 && frame.size >= (size_t)frame_h) {
                const size_t s = frame.size / (size_t)frame_h;
                if (s >= (size_t)frame_w * 3) stride_bytes = (int)s;
            }

            static std::atomic<bool> printed_frame{false};
            if (debug_preprocess && vir_addr && !printed_frame.exchange(true)) {
                const int px = 8;
                printf("[DEBUG] frame0: w=%d h=%d size=%zu stride=%d bytes\n", frame_w, frame_h, frame.size, stride_bytes);
                printf("[DEBUG] frame0 first %d px RGB: ", px);
                for (int i = 0; i < px; ++i) {
                    const int off = i * 3;
                    printf("(%u,%u,%u) ", rgb_ptr[off + 0], rgb_ptr[off + 1], rgb_ptr[off + 2]);
                }
                printf("\n");
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
                    // SSCMA bbox 为中心点格式（x,y,w,h）且为归一化
                    const float cx = d.x;
                    const float cy = d.y;
                    const float bw = d.w;
                    const float bh = d.h;
                    // 转为像素坐标 xyxy，供 align/crop 使用
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
                            // UDP 协议语义与 runner 一致：1=Male, 0=Female
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

            infer_count++;
            win_infer_count++;

            // 不考虑性能：每次推理都把 face0 的 AGR 直接打印出来（便于与 Python 接收端对照）。
            if (!cached_faces.empty()) {
                const auto& f0 = cached_faces[0];
                printf("[AGR] face0: gender=%s(%d,%.3f) age=%s(%d,%.3f) race=%s(%d,%.3f)\n",
                       agr_gender_label(f0.gender), f0.gender, f0.gender_score,
                       agr_age_bin_label(f0.age), f0.age, f0.age_score,
                       agr_race_label(f0.race), f0.race, f0.race_score);
            }
            
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

                if (!cached_faces.empty()) {
                    const auto& f = cached_faces[0];
                    printf("  face0: xywh=(%.3f,%.3f,%.3f,%.3f) det=%.3f tgt=%d | stride=%d\n",
                           f.x, f.y, f.w, f.h, f.score, f.target, stride_bytes);
                          printf("  AGR: gender=%s(%.3f) age=%s(%.3f) race=%s(%.3f) | EMO: emo=%s(%d,%.3f)\n",
                          agr_gender_label(f.gender), f.gender_score,
                          agr_age_range_label(f.age), f.age_score,
                          agr_race_label(f.race), f.race_score,
                              emo_label(f.emotion), f.emotion, f.emotion_score);
                }
            }
        }

        camera->returnFrame(frame);

        // CH1: JPEG 帧用于 UDP 推送（不走 RTSP）
        if (udp_sender) {
            JpegFrameOwned jpeg_owned;
            {
                std::lock_guard<std::mutex> lk(jpeg_mu);
                if (latest_jpeg.buf) {
                    jpeg_owned = std::move(latest_jpeg);
                }
            }

            if (jpeg_owned.buf && jpeg_owned.size > 0) {
                double udp_ms = udp_sender->sendJpegOwned(UDP_MAGIC,
                                                         std::move(jpeg_owned.buf),
                                                         jpeg_owned.size,
                                                         jpeg_owned.w,
                                                         jpeg_owned.h,
                                                         cached_faces);
                win_udp_send_count++;
                win_udp_send_bytes += jpeg_owned.size;
                if (udp_ms > 0) win_total_udp_ms += udp_ms;
            }
        }

        // ========== FPS统计 ==========
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count();
        if (elapsed >= 2000) {
            float video_fps = fps_frame_count * 1000.0f / elapsed;
            float udp_fps = udp_sender ? udp_sender->getSendFPS() : 0.0f;

            const uint64_t cur_jpeg_fetch_count = jpeg_fetch_count.load();
            const uint64_t cur_jpeg_fetch_wait_us = jpeg_fetch_wait_us.load();
            const uint64_t win_jpeg_fetch_count = cur_jpeg_fetch_count - last_jpeg_fetch_count;
            const uint64_t win_jpeg_fetch_wait_us = cur_jpeg_fetch_wait_us - last_jpeg_fetch_wait_us;
            last_jpeg_fetch_count = cur_jpeg_fetch_count;
            last_jpeg_fetch_wait_us = cur_jpeg_fetch_wait_us;
            const double win_jpeg_fetch_avg_ms = (win_jpeg_fetch_count > 0)
                                                    ? ((double)win_jpeg_fetch_wait_us / 1000.0 / (double)win_jpeg_fetch_count)
                                                    : 0.0;
            const double win_jpeg_fetch_fps = (double)win_jpeg_fetch_count * 1000.0 / (double)elapsed;
            
            printf("\n========== 性能统计 (最近2秒) ==========\n");
            printf("观感 FPS: %.1f | UDP FPS: %.1f\n", video_fps, udp_fps);
            printf("窗口帧数: %d | 窗口推理: %d | 窗口UDP发送: %d\n", fps_frame_count, win_infer_count, win_udp_send_count);

            // YOLO（不含 GenderAge/mmap 等额外开销）
            if (win_infer_count > 0) {
                printf("\n平均 YOLO 耗时 (窗口内):\n");
                printf("  前处理: %.1f ms\n", win_total_pre_ms / win_infer_count);
                printf("  推理:   %.1f ms\n", win_total_inf_ms / win_infer_count);
                printf("  后处理: %.1f ms\n", win_total_post_ms / win_infer_count);
                printf("  总计:   %.1f ms\n", win_total_infer_ms / win_infer_count);
            }

                 // 属性/表情 + mmap（不在 perf 里统计）
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

            // CH1 JPEG 拉流（会阻塞等待，现已移到线程）
            if (udp_sender) {
                printf("\nJPEG 拉流 (CH1, 窗口内):\n");
                printf("  fetch_fps: %.1f | fetch_avg_wait: %.1f ms\n", win_jpeg_fetch_fps, win_jpeg_fetch_avg_ms);
            }

            // UDP 发送（分包/发送耗时）
            if (udp_sender) {
                const double udp_avg_ms = (win_udp_send_count > 0) ? (win_total_udp_ms / win_udp_send_count) : 0.0;
                const double udp_kbps = (double)win_udp_send_bytes * 8.0 / (double)elapsed;  // kbps
                printf("\nUDP 发送 (窗口内):\n");
                printf("  avg_send: %.1f ms | frames=%d | throughput=%.1f kbps\n", udp_avg_ms, win_udp_send_count, udp_kbps);
            }
            printf("======================================\n\n");

            // reset window counters
            win_total_infer_ms = win_total_pre_ms = win_total_inf_ms = win_total_post_ms = 0;
            win_infer_count = 0;
            win_total_mmap_ms = win_total_munmap_ms = 0;
            win_total_agr_ms = 0;
            win_total_emo_ms = 0;
            win_attr_frame_count = win_attr_face_count = 0;
            win_agr_ok_count = win_emo_ok_count = 0;
            win_total_udp_ms = 0;
            win_udp_send_count = 0;
            win_udp_send_bytes = 0;

            fps_frame_count = 0;
            fps_start = now;
        }
    }

    // ==================== 清理 ====================
    printf("\n[停止] 清理资源...\n");

    if (jpeg_thread.joinable()) {
        jpeg_running.store(false);
        jpeg_thread.join();
    }
    
    if (udp_sender) udp_sender->stop();
    camera->stopStream();
    for (auto& sensor : device->getSensors()) sensor->deInit();
    
    if (square_detector) delete square_detector;
    if (yolo_model) ma::ModelFactory::remove(yolo_model);
    delete yolo_engine;

    printf("[完成]\n");
    return 0;
}
