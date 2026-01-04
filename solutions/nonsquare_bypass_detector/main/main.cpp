#include <iostream>
#include <chrono>
#include <thread>
#include <string>

#include <sscma.h>
#include <video.h>

// 用于在 ModelFactory 不支持时，直接构造 YOLOv8 以便获得输入尺寸
#include <core/model/ma_model_yolov8.h>

#include "yolov8_postprocess.h"

using namespace ma;

#define TAG "nonsquare_bypass_detector"

static void print_usage(const char* prog) {
    printf("Usage: %s model.cvimodel [threshold] [skip]\n", prog);
    printf("  threshold: 置信度阈值 (默认 0.5)\n");
    printf("  skip: 每N帧推理1次 (默认 1)\n");
    printf("\n说明：本示例绕过 SSCMA Detector 后处理，直接使用 EngineCVI 的输出做 YOLOv8(6输出) 自定义后处理，支持非正方形输入。\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    float threshold = (argc >= 3) ? atof(argv[2]) : 0.5f;
    int skip_interval = (argc >= 4) ? atoi(argv[3]) : 1;

    auto* engine = new ma::engine::EngineCVI();
    if (engine->init() != MA_OK || engine->load(model_path) != MA_OK) {
        MA_LOGE(TAG, "engine init/load failed");
        return 1;
    }

    ma::Model* model = ma::ModelFactory::create(engine);
    if (!model) {
        // ModelFactory 可能因模型元信息不完整而失败，这里仅对 YOLOv8 做兜底
        std::string path(model_path);
        if (path.find("yolov8") != std::string::npos) {
            MA_LOGW(TAG, "ModelFactory::create failed, fallback to direct ma::model::YoloV8 construction");
            model = new ma::model::YoloV8(engine);
        }
    }

    if (!model) {
        MA_LOGE(TAG, "model not supported (require YOLOv8 6-output for this demo)");
        return 1;
    }

    if (model->getInputType() != MA_INPUT_TYPE_IMAGE) {
        MA_LOGE(TAG, "model input type not supported (require image)");
        return 1;
    }

    const ma_img_t* model_input = static_cast<const ma_img_t*>(model->getInput());
    const int input_w = model_input->width;
    const int input_h = model_input->height;

    if (engine->getOutputSize() != 6) {
        MA_LOGE(TAG, "this demo expects YOLOv8 6 outputs, but engine->getOutputSize()=%d", engine->getOutputSize());
        return 1;
    }

    MA_LOGI(TAG, "model: %s", model_path);
    MA_LOGI(TAG, "input: %dx%d", input_w, input_h);
    MA_LOGI(TAG, "threshold: %.2f, skip: %d", threshold, skip_interval);

    // init camera
    Device* device = Device::getInstance();
    Camera* camera = nullptr;

    Signal::install({SIGINT, SIGSEGV, SIGABRT, SIGTRAP, SIGTERM, SIGHUP, SIGQUIT, SIGPIPE}, [device](int sig) {
        std::cout << "Caught signal " << sig << std::endl;
        for (auto& sensor : device->getSensors()) {
            sensor->deInit();
        }
        exit(0);
    });

    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            camera->init(0);
            Camera::CtrlValue v;

            v.i32 = 0;
            camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, v);
            v.u16s[0] = input_w;
            v.u16s[1] = input_h;
            camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, v);

            v.i32 = 1;
            camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);
            break;
        }
    }

    if (!camera) {
        MA_LOGE(TAG, "No camera found");
        return 1;
    }

    camera->startStream(Camera::StreamMode::kRefreshOnReturn);

    yolov8::YoloV8PostProcessor postprocessor;
    postprocessor.setThreshold(threshold, 0.45f);

    int frame_counter = 0;
    int infer_counter = 0;

    while (true) {
        ma_img_t frame;
        if (camera->retrieveFrame(frame, MA_PIXEL_FORMAT_RGB888) != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        frame_counter++;
        const bool do_infer = (frame_counter % skip_interval == 0);
        if (!do_infer) {
            camera->returnFrame(frame);
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        ma_tensor_t tensor = {
            .size = frame.size,
            .is_physical = true,
            .is_variable = false,
        };
        tensor.data.data = reinterpret_cast<void*>(frame.data);
        engine->setInput(0, tensor);

        auto t1 = std::chrono::high_resolution_clock::now();
        engine->run();
        auto t2 = std::chrono::high_resolution_clock::now();

        yolov8::TensorInfo box_outputs[3];
        yolov8::TensorInfo cls_outputs[3];

        for (int i = 0; i < 3; i++) {
            auto box_out = engine->getOutput(i);
            auto cls_out = engine->getOutput(i + 3);

            box_outputs[i].data = box_out.data.data;
            box_outputs[i].dims[0] = box_out.shape.dims[0];
            box_outputs[i].dims[1] = box_out.shape.dims[1];
            box_outputs[i].dims[2] = box_out.shape.dims[2];
            box_outputs[i].dims[3] = box_out.shape.dims[3];
            box_outputs[i].scale = box_out.quant_param.scale;
            box_outputs[i].zero_point = box_out.quant_param.zero_point;
            box_outputs[i].is_int8 = (box_out.type == MA_TENSOR_TYPE_S8);

            cls_outputs[i].data = cls_out.data.data;
            cls_outputs[i].dims[0] = cls_out.shape.dims[0];
            cls_outputs[i].dims[1] = cls_out.shape.dims[1];
            cls_outputs[i].dims[2] = cls_out.shape.dims[2];
            cls_outputs[i].dims[3] = cls_out.shape.dims[3];
            cls_outputs[i].scale = cls_out.quant_param.scale;
            cls_outputs[i].zero_point = cls_out.quant_param.zero_point;
            cls_outputs[i].is_int8 = (cls_out.type == MA_TENSOR_TYPE_S8);
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        auto results = postprocessor.process(box_outputs, cls_outputs, input_w, input_h);
        auto t4 = std::chrono::high_resolution_clock::now();

        camera->returnFrame(frame);

        infer_counter++;

        auto ms_setinput = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto ms_infer = std::chrono::duration<double, std::milli>(t2 - t1).count();
        auto ms_extract = std::chrono::duration<double, std::milli>(t3 - t2).count();
        auto ms_post = std::chrono::duration<double, std::milli>(t4 - t3).count();
        auto ms_total = std::chrono::duration<double, std::milli>(t4 - t0).count();

        printf("[Infer #%d] setInput=%.1f ms, run=%.1f ms, extract=%.1f ms, post=%.1f ms, total=%.1f ms | det=%zu\n",
               infer_counter, ms_setinput, ms_infer, ms_extract, ms_post, ms_total, results.size());
    }

    // unreachable
    camera->stopStream();
    for (auto& sensor : device->getSensors()) sensor->deInit();

    ma::ModelFactory::remove(model);
    delete engine;
    return 0;
}
