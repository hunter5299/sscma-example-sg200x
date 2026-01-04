/**
 * @file camera_test.cpp
 * @brief 摄像头实时检测测试 - 用于验证单输出 YOLO 模型的实际检测效果
 * 
 * 特性：
 * - 支持单输出模型（直接构造 YoloV8）
 * - 支持多输出模型（使用 ModelFactory）
 * - 摄像头实时检测
 * - UDP 推送检测结果（可选）
 * 
 * 用法:
 *   ./camera_test <模型路径> [模型类型] [阈值] [udp_ip] [udp_port]
 * 
 * 模型类型:
 *   auto = 自动检测 (默认)
 *   single = 单输出模型（直接构造 YoloV8）
 *   multi = 多输出模型（使用 ModelFactory）
 * 
 * 示例:
 *   ./camera_test yolo-face.cvimodel single 0.5
 *   ./camera_test yolo-face.cvimodel single 0.5 192.168.1.100 5000
 */

#include <iostream>
//
// Clean rebuild of camera_test with custom postprocess for single-output YOLO.
//

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <signal.h>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include <sscma.h>
#include <video.h>
#include <core/model/ma_model_yolov8.h>

using namespace ma;

static std::atomic<bool> g_running(true);
void sig_handler(int) { g_running.store(false); }

enum class ModelMode { Auto, Single, Multi };

struct Box {
    float x, y, w, h, score;
    int target;
};

static float box_iou(const Box& a, const Box& b) {
    // Box is in normalized top-left coordinates: (x, y, w, h)
    float ax1 = a.x;
    float ay1 = a.y;
    float ax2 = a.x + a.w;
    float ay2 = a.y + a.h;
    float bx1 = b.x;
    float by1 = b.y;
    float bx2 = b.x + b.w;
    float by2 = b.y + b.h;
    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - inter_area;
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

static std::vector<Box> nms(std::vector<Box>& boxes, float iou_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<Box> res;
    std::vector<bool> sup(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (sup[i]) continue;
        res.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (sup[j]) continue;
            if (box_iou(boxes[i], boxes[j]) > iou_thresh) sup[j] = true;
        }
    }
    return res;
}

static inline float sigmoidf_safe(float x) {
    // numerically safe sigmoid
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static std::vector<Box> parse_single_output(ma::engine::Engine* engine, float threshold, float nms_thresh, int img_w, int img_h) {
    auto out = engine->getOutput(0);
    auto shape = engine->getOutputShape(0);
    // Expect something like [1,5,8400,1] or [1,8400,5,1]
    if (shape.size < 3) {
        return {};
    }

    const int d1 = shape.dims[1];
    const int d2 = shape.dims[2];
    int num_boxes = 0;
    bool prefer_ch_first = false;
    if (d1 == 5) {
        num_boxes = d2;
        prefer_ch_first = true; // [1,5,N,(1)]
    } else if (d2 == 5) {
        num_boxes = d1;
        prefer_ch_first = false; // [1,N,5,(1)]
    } else {
        return {};
    }
    std::vector<Box> boxes;

    static bool debug_printed = false;

    if (out.type == MA_TENSOR_TYPE_F32) {
        float* data = (float*)out.data.data;
        const int sample_n = std::min(256, num_boxes);

        auto get_ch_first = [&](int c, int i) -> float {
            // [1,5,N,(1)] contiguous: c-major, then i
            return data[c * num_boxes + i];
        };
        auto get_box_first = [&](int c, int i) -> float {
            // [1,N,5,(1)] contiguous: i-major, then c
            return data[i * 5 + c];
        };

        auto compute_score_stats = [&](bool ch_first, float& min_v, float& max_v, float& mean_v, int& nonzero) {
            min_v = 1e9f;
            max_v = -1e9f;
            double sum = 0.0;
            nonzero = 0;
            for (int i = 0; i < sample_n; ++i) {
                float v = ch_first ? get_ch_first(4, i) : get_box_first(4, i);
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
                sum += v;
                if (fabsf(v) > 1e-12f) nonzero++;
            }
            mean_v = (float)(sum / sample_n);
        };

        float smin_a, smax_a, smean_a;
        float smin_b, smax_b, smean_b;
        int snz_a, snz_b;
        compute_score_stats(true, smin_a, smax_a, smean_a, snz_a);
        compute_score_stats(false, smin_b, smax_b, smean_b, snz_b);

        // Choose layout
        bool use_ch_first = prefer_ch_first;
        auto plausible = [&](float min_v, float max_v, int nonzero) {
            // score should not be huge; may be [0,1] after sigmoid
            return nonzero > sample_n / 10 && max_v > 0.01f && max_v < 5.0f;
        };
        bool a_ok = plausible(smin_a, smax_a, snz_a);
        bool b_ok = plausible(smin_b, smax_b, snz_b);
        if (a_ok && !b_ok) use_ch_first = true;
        else if (!a_ok && b_ok) use_ch_first = false;

        if (!debug_printed) {
            printf("\n[调试] 单输出解析 (FLOAT32):\n");
            if (shape.size >= 4)
                printf("  Shape: [%d, %d, %d, %d]\n", shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]);
            else
                printf("  Shape: [%d, %d, %d]\n", shape.dims[0], shape.dims[1], shape.dims[2]);
            printf("  采样score统计 (按 [5,N]): min=%.6f max=%.6f mean=%.6f nonzero=%d/%d\n", smin_a, smax_a, smean_a, snz_a, sample_n);
            printf("  采样score统计 (按 [N,5]): min=%.6f max=%.6f mean=%.6f nonzero=%d/%d\n", smin_b, smax_b, smean_b, snz_b, sample_n);
            printf("  选择布局: %s\n", use_ch_first ? "[1,5,N,(1)]" : "[1,N,5,(1)]");
            debug_printed = true;
        }

        auto get = [&](int c, int i) -> float {
            return use_ch_first ? get_ch_first(c, i) : get_box_first(c, i);
        };

        for (int i = 0; i < num_boxes; ++i) {
            float x_center = get(0, i);
            float y_center = get(1, i);
            float w = get(2, i);
            float h = get(3, i);
            float score_raw = get(4, i);

            // Some exported models already apply sigmoid; only apply if it's clearly not in [0,1]
            float score = score_raw;
            if (score_raw < -0.1f || score_raw > 1.1f) {
                score = sigmoidf_safe(score_raw);
            }
            if (score < threshold) continue;

            float x1 = (x_center - w * 0.5f) / (float)img_w;
            float y1 = (y_center - h * 0.5f) / (float)img_h;
            float box_w = w / (float)img_w;
            float box_h = h / (float)img_h;

            // basic sanity: drop invalid boxes
            if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(box_w) || !std::isfinite(box_h)) continue;
            if (box_w <= 0.0f || box_h <= 0.0f) continue;
            if (box_w > 2.0f || box_h > 2.0f) continue;

            Box b{ x1, y1, box_w, box_h, score, 0 };
            boxes.push_back(b);
        }
    } else if (out.type == MA_TENSOR_TYPE_S8) {
        int8_t* data = (int8_t*)out.data.data;
        float scale = out.quant_param.scale;
        int zp = out.quant_param.zero_point;
        const int sample_n = std::min(256, num_boxes);

        auto deq = [&](int8_t v) -> float { return (v - zp) * scale; };
        auto get_ch_first = [&](int c, int i) -> float {
            return deq(data[c * num_boxes + i]);
        };
        auto get_box_first = [&](int c, int i) -> float {
            return deq(data[i * 5 + c]);
        };

        float smin_a = 1e9f, smax_a = -1e9f, smean_a = 0.0f;
        float smin_b = 1e9f, smax_b = -1e9f, smean_b = 0.0f;
        int snz_a = 0, snz_b = 0;
        for (int i = 0; i < sample_n; ++i) {
            float va = get_ch_first(4, i);
            float vb = get_box_first(4, i);
            smin_a = std::min(smin_a, va); smax_a = std::max(smax_a, va); smean_a += va; if (fabsf(va) > 1e-12f) snz_a++;
            smin_b = std::min(smin_b, vb); smax_b = std::max(smax_b, vb); smean_b += vb; if (fabsf(vb) > 1e-12f) snz_b++;
        }
        smean_a /= sample_n;
        smean_b /= sample_n;

        bool use_ch_first = prefer_ch_first;
        auto plausible = [&](float max_v, int nonzero) { return nonzero > sample_n / 10 && max_v > -2.0f && max_v < 20.0f; };
        bool a_ok = plausible(smax_a, snz_a);
        bool b_ok = plausible(smax_b, snz_b);
        if (a_ok && !b_ok) use_ch_first = true;
        else if (!a_ok && b_ok) use_ch_first = false;

        if (!debug_printed) {
            printf("\n[调试] 单输出解析 (INT8):\n");
            printf("  scale=%.8f zp=%d\n", scale, zp);
            printf("  采样score(解量化) (按 [5,N]): min=%.6f max=%.6f mean=%.6f nonzero=%d/%d\n", smin_a, smax_a, smean_a, snz_a, sample_n);
            printf("  采样score(解量化) (按 [N,5]): min=%.6f max=%.6f mean=%.6f nonzero=%d/%d\n", smin_b, smax_b, smean_b, snz_b, sample_n);
            printf("  选择布局: %s\n", use_ch_first ? "[1,5,N,(1)]" : "[1,N,5,(1)]");
            debug_printed = true;
        }

        auto get = [&](int c, int i) -> float { return use_ch_first ? get_ch_first(c, i) : get_box_first(c, i); };
        for (int i = 0; i < num_boxes; ++i) {
            float x_center = get(0, i);
            float y_center = get(1, i);
            float w = get(2, i);
            float h = get(3, i);
            float score_raw = get(4, i);
            float score = score_raw;
            if (score_raw < -0.1f || score_raw > 1.1f) {
                score = sigmoidf_safe(score_raw);
            }
            if (score < threshold) continue;

            float x1 = (x_center - w * 0.5f) / (float)img_w;
            float y1 = (y_center - h * 0.5f) / (float)img_h;
            float box_w = w / (float)img_w;
            float box_h = h / (float)img_h;
            if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(box_w) || !std::isfinite(box_h)) continue;
            if (box_w <= 0.0f || box_h <= 0.0f) continue;
            if (box_w > 2.0f || box_h > 2.0f) continue;
            boxes.push_back({x1, y1, box_w, box_h, score, 0});
        }
    }
    
    return nms(boxes, nms_thresh);
}

class SimpleUDPSender {
public:
    SimpleUDPSender(const char* ip, int port) : ip_(ip), port_(port), sock_(-1) {}
    bool init() {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) return false;
        memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(port_);
        if (inet_pton(AF_INET, ip_, &addr_.sin_addr) <= 0) return false;
        return true;
    }
    void send(const char* data, size_t len) {
        if (sock_ < 0) return;
        sendto(sock_, data, len, 0, (struct sockaddr*)&addr_, sizeof(addr_));
    }
    ~SimpleUDPSender() {
        if (sock_ >= 0) close(sock_);
    }
private:
    const char* ip_;
    int port_;
    int sock_;
    struct sockaddr_in addr_;
};

int main(int argc, char** argv) {
    printf("\n========== 摄像头检测测试 ==========\n");
    if (argc < 2) {
        printf("用法: %s <模型路径> [mode: auto|single|multi] [threshold] [udp_ip] [udp_port]\n", argv[0]);
        return 0;
    }

    const char* model_path = argv[1];
    ModelMode mode = ModelMode::Auto;
    float threshold = 0.5f;
    const char* udp_ip = nullptr;
    int udp_port = 5000;

    if (argc >= 3) {
        std::string m = argv[2];
        std::transform(m.begin(), m.end(), m.begin(), ::tolower);
        if (m == "single") mode = ModelMode::Single;
        else if (m == "multi") mode = ModelMode::Multi;
    }
    if (argc >= 4) threshold = atof(argv[3]);
    if (argc >= 5) udp_ip = argv[4];
    if (argc >= 6) udp_port = atoi(argv[5]);

    auto* engine = new ma::engine::EngineCVI();
    if (engine->init() != MA_OK || engine->load(model_path) != MA_OK) {
        printf("引擎初始化或模型加载失败\n");
        delete engine;
        return 1;
    }

    printf("模型文件: %s\n", model_path);
    printf("模式: %s\n", mode == ModelMode::Single ? "single" : mode == ModelMode::Multi ? "multi" : "auto");
    printf("阈值: %.2f\n", threshold);
    if (udp_ip) printf("UDP: %s:%d\n", udp_ip, udp_port);

    printf("输入数量: %zu\n", engine->getInputSize());
    for (size_t i = 0; i < engine->getInputSize(); ++i) {
        auto s = engine->getInputShape(i);
        printf("  输入[%zu]: [", i);
        for (size_t j = 0; j < s.size; ++j) printf("%d%s", s.dims[j], j + 1 == s.size ? "" : ",");
        printf("]\n");
    }
    printf("输出数量: %zu\n", engine->getOutputSize());
    for (size_t i = 0; i < engine->getOutputSize(); ++i) {
        auto s = engine->getOutputShape(i);
        auto o = engine->getOutput(i);
        printf("  输出[%zu]: [", i);
        for (size_t j = 0; j < s.size; ++j) printf("%d%s", s.dims[j], j + 1 == s.size ? "" : ",");
        printf("] type=%d\n", o.type);
    }

    bool is_single_output = (engine->getOutputSize() == 1);
    ma::Model* model = nullptr;
    ma::model::Detector* detector = nullptr;

    if (!is_single_output) {
        auto try_factory = [&]() { return ma::ModelFactory::create(engine); };
        auto try_single_model = [&]() {
            ma::Model* m = new ma::model::YoloV8(engine);
            if (!m || m->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
                if (m) ma::ModelFactory::remove(m);
                return (ma::Model*)nullptr;
            }
            return m;
        };
        if (mode == ModelMode::Single) model = try_single_model();
        else if (mode == ModelMode::Multi) model = try_factory();
        else {
            model = try_factory();
            if (!model) model = try_single_model();
        }
        if (!model) {
            printf("模型创建失败\n");
            delete engine;
            return 1;
        }
        detector = static_cast<ma::model::Detector*>(model);
        detector->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold);
    }

    int input_w = 0, input_h = 0;
    if (detector) {
        const ma_img_t* mi = static_cast<const ma_img_t*>(detector->getInput());
        input_w = mi->width;
        input_h = mi->height;
    } else {
        auto s = engine->getInputShape(0);
        if (s.size == 4) {
            if (s.dims[3] == 3 || s.dims[3] == 1) {
                input_h = s.dims[1];
                input_w = s.dims[2];
            } else {
                input_h = s.dims[2];
                input_w = s.dims[3];
            }
        }
    }
    printf("模型输入: %dx%d\n", input_w, input_h);

    Device* device = Device::getInstance();
    Camera* camera = nullptr;
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);
            Camera::CtrlValue val;
            val.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, val);
            val.u16s[0] = input_w; val.u16s[1] = input_h;
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, val);
            val.i32 = 1;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, val);
            break;
        }
    }
    if (!camera) {
        printf("未找到相机\n");
        if (model) ma::ModelFactory::remove(model);
        delete engine;
        return 1;
    }
    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    SimpleUDPSender* udp_sender = nullptr;
    if (udp_ip) {
        udp_sender = new SimpleUDPSender(udp_ip, udp_port);
        if (!udp_sender->init()) { delete udp_sender; udp_sender = nullptr; }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    int frame_count = 0, detect_count = 0;
    double total_pre = 0, total_inf = 0, total_post = 0;
    auto fps_start = std::chrono::high_resolution_clock::now();
    int fps_frames = 0;

    while (g_running.load()) {
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        frame_count++;
        fps_frames++;

        ma_tensor_t tensor{};
        tensor.size = frame.size;
        tensor.is_physical = true;
        tensor.is_variable = false;
        tensor.data.data = reinterpret_cast<void*>(frame.data);
        engine->setInput(0, tensor);

        std::vector<Box> results;
        ma_perf_t perf{0};

        if (is_single_output) {
            auto t0 = std::chrono::high_resolution_clock::now();
            engine->run();
            auto t1 = std::chrono::high_resolution_clock::now();
            results = parse_single_output(engine, threshold, 0.45f, input_w, input_h);
            auto t2 = std::chrono::high_resolution_clock::now();
            perf.preprocess = 0;
            perf.inference = std::chrono::duration<double, std::milli>(t1 - t0).count();
            perf.postprocess = std::chrono::duration<double, std::milli>(t2 - t1).count();
        } else {
            detector->run(nullptr);
            auto rs = detector->getResults();
            for (auto& r : rs) results.push_back({r.x, r.y, r.w, r.h, r.score, r.target});
            perf = detector->getPerf();
        }

        size_t det_num = results.size();
        if (det_num > 0) detect_count++;
        total_pre += perf.preprocess;
        total_inf += perf.inference;
        total_post += perf.postprocess;

        bool should_print = (det_num > 0) || (frame_count % 30 == 0);
        if (should_print) {
            if (det_num > 0) {
                printf("[帧 #%d] ✓ 检测到 %zu 个目标 | 耗时: %.1f+%.1f+%.1f=%.1fms\n",
                       frame_count, det_num, perf.preprocess, perf.inference, perf.postprocess,
                       perf.preprocess + perf.inference + perf.postprocess);
                int idx = 0;
                for (auto& r : results) {
                    printf("  [%d] x=%.3f y=%.3f w=%.3f h=%.3f score=%.3f target=%d\n",
                           idx++, r.x, r.y, r.w, r.h, r.score, r.target);
                }
            } else {
                printf("[帧 #%d] × 未检测到目标 (阈值=%.2f) | 耗时: %.1fms\n",
                       frame_count, threshold, perf.preprocess + perf.inference + perf.postprocess);
            }
            if (udp_sender && det_num > 0) {
                char msg[128];
                int len = snprintf(msg, sizeof(msg), "DETECTIONS:%zu\n", det_num);
                udp_sender->send(msg, len);
            }
        }

        camera->returnFrame(frame);

        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count();
        if (elapsed >= 5000) {
            float fps = fps_frames * 1000.0f / elapsed;
            printf("\n========== 性能统计 (最近5秒) ==========\n");
            printf("FPS: %.1f\n", fps);
            printf("总帧数: %d | 检测帧数: %d (%.1f%%)\n", frame_count, detect_count, detect_count * 100.0f / frame_count);
            if (frame_count > 0) {
                printf("\n平均耗时:\n");
                printf("  前处理: %.1f ms\n", total_pre / frame_count);
                printf("  推理:   %.1f ms\n", total_inf / frame_count);
                printf("  后处理: %.1f ms\n", total_post / frame_count);
                printf("  总计:   %.1f ms\n", (total_pre + total_inf + total_post) / frame_count);
            }
            printf("======================================\n\n");
            fps_frames = 0;
            fps_start = now;
        }
    }

    camera->stopStream();
    for (auto& s : device->getSensors()) s->deInit();
    if (model) ma::ModelFactory::remove(model);
    delete engine;
    if (udp_sender) delete udp_sender;
    printf("[完成] 总帧数: %d | 检测帧数: %d\n", frame_count, detect_count);
    return 0;
}
