/**
 * @file main.cpp
 * @brief 模型测试程序 - 纯测试工具
 * 
 * 测试 cvimodel 模型的加载、推理时间、输入输出结构
 * 不需要 OpenCV，使用纯色测试数据
 * 
 * 使用方法:
 *   ./cls_test <模型路径> [模型类型]
 * 
 * 模型类型:
 *   0 = 自动检测 (默认)
 *   3 = YOLOv5
 *   4 = 分类模型 (IMCLS)
 *   6 = YOLOv8
 *   9 = YOLO11
 */

#include <iostream>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <sscma.h>

// 包含具体模型类（用于直接构造绕过 ModelFactory 验证）
#include <core/model/ma_model_yolov8.h>
#include <core/model/ma_model_yolo11.h>

// 自定义 YOLOv8 后处理（支持任意分辨率）
#include "yolov8_postprocess.h"

using namespace ma;

#define TAG "model_test"

/**
 * @brief 计算 forward_list 的大小
 */
template<typename T>
size_t forward_list_size(const std::forward_list<T>& list) {
    size_t count = 0;
    for (auto it = list.begin(); it != list.end(); ++it) {
        ++count;
    }
    return count;
}

/**
 * @brief 打印分隔线
 */
void printSeparator(const char* title = nullptr) {
    std::cout << "\n";
    if (title) {
        std::cout << "========== " << title << " ==========" << std::endl;
    } else {
        std::cout << "======================================" << std::endl;
    }
}

/**
 * @brief 获取数据类型名称
 */
const char* getTypeName(ma_tensor_type_t type) {
    switch (type) {
        case MA_TENSOR_TYPE_U8: return "UINT8";
        case MA_TENSOR_TYPE_S8: return "INT8";
        case MA_TENSOR_TYPE_U16: return "UINT16";
        case MA_TENSOR_TYPE_S16: return "INT16";
        case MA_TENSOR_TYPE_U32: return "UINT32";
        case MA_TENSOR_TYPE_S32: return "INT32";
        case MA_TENSOR_TYPE_F32: return "FLOAT32";
        case MA_TENSOR_TYPE_F64: return "FLOAT64";
        case MA_TENSOR_TYPE_BF16: return "BF16";
        case MA_TENSOR_TYPE_F16: return "FLOAT16";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    printSeparator("模型测试工具 Model Tester");
    
    if (argc < 2) {
        std::cout << "\n用法: " << argv[0] << " <模型路径> [模型类型]" << std::endl;
        std::cout << "\n模型类型:" << std::endl;
        std::cout << "  0 = 自动检测 (默认)" << std::endl;
        std::cout << "  3 = YOLOv5" << std::endl;
        std::cout << "  4 = 分类模型" << std::endl;
        std::cout << "  6 = YOLOv8" << std::endl;
        std::cout << "  9 = YOLO11" << std::endl;
        std::cout << "\n示例:" << std::endl;
        std::cout << "  " << argv[0] << " model.cvimodel" << std::endl;
        std::cout << "  " << argv[0] << " yolov8.cvimodel 6" << std::endl;
        return 0;
    }
    
    const char* model_path = argv[1];
    int model_type_hint = (argc >= 3) ? atoi(argv[2]) : 0;
    
    std::cout << "\n模型文件: " << model_path << std::endl;
    if (model_type_hint > 0) {
        std::cout << "指定类型: " << model_type_hint << std::endl;
    }
    
    // ==================== 初始化引擎 ====================
    printSeparator("Step 1: 初始化引擎");
    
    ma_err_t ret = MA_OK;
    auto* engine = new ma::engine::EngineCVI();
    
    auto t0 = std::chrono::high_resolution_clock::now();
    ret = engine->init();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    if (ret != MA_OK) {
        std::cerr << "[FAIL] 引擎初始化失败, 错误码: " << ret << std::endl;
        delete engine;
        return 1;
    }
    std::cout << "[OK] 引擎初始化成功, 耗时: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms" << std::endl;
    
    // ==================== 加载模型 ====================
    printSeparator("Step 2: 加载模型");
    
    t0 = std::chrono::high_resolution_clock::now();
    ret = engine->load(model_path);
    t1 = std::chrono::high_resolution_clock::now();
    
    if (ret != MA_OK) {
        std::cerr << "[FAIL] 模型加载失败, 错误码: " << ret << std::endl;
        delete engine;
        return 1;
    }
    std::cout << "[OK] 模型加载成功, 耗时: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " ms" << std::endl;
    
    // ==================== 模型结构 ====================
    printSeparator("Step 3: 模型结构");
    
    std::cout << "\n输入数量: " << engine->getInputSize() << std::endl;
    
    int input_width = 0, input_height = 0, input_channels = 0;
    
    for (size_t i = 0; i < engine->getInputSize(); i++) {
        auto shape = engine->getInputShape(i);
        auto in = engine->getInput(i);
        
        std::cout << "  输入[" << i << "]: shape=[";
        for (size_t j = 0; j < shape.size; j++) {
            std::cout << shape.dims[j];
            if (j < shape.size - 1) std::cout << ",";
        }
        std::cout << "], type=" << getTypeName(in.type);
        std::cout << ", scale=" << in.quant_param.scale;
        std::cout << ", zp=" << in.quant_param.zero_point << std::endl;
        
        // 检测格式: NCHW 或 NHWC
        if (shape.size == 4) {
            // 判断是 NCHW 还是 NHWC
            // NCHW: dims[1] 通常是 3 (channels), dims[2/3] 是 H/W (较大)
            // NHWC: dims[3] 通常是 3 (channels), dims[1/2] 是 H/W (较大)
            if (shape.dims[1] == 3 || shape.dims[1] == 1) {
                // NCHW 格式
                input_channels = shape.dims[1];
                input_height = shape.dims[2];
                input_width = shape.dims[3];
            } else if (shape.dims[3] == 3 || shape.dims[3] == 1) {
                // NHWC 格式
                input_height = shape.dims[1];
                input_width = shape.dims[2];
                input_channels = shape.dims[3];
            } else {
                // 默认假设 NCHW
                input_channels = shape.dims[1];
                input_height = shape.dims[2];
                input_width = shape.dims[3];
            }
        }
    }
    
    std::cout << "\n输出数量: " << engine->getOutputSize() << std::endl;
    
    for (size_t i = 0; i < engine->getOutputSize(); i++) {
        auto shape = engine->getOutputShape(i);
        auto out = engine->getOutput(i);
        
        std::cout << "  输出[" << i << "]: shape=[";
        for (size_t j = 0; j < shape.size; j++) {
            std::cout << shape.dims[j];
            if (j < shape.size - 1) std::cout << ",";
        }
        std::cout << "], type=" << getTypeName(out.type);
        std::cout << ", scale=" << out.quant_param.scale;
        std::cout << ", zp=" << out.quant_param.zero_point << std::endl;
    }
    
    std::cout << "\n推断输入尺寸: " << input_width << "x" << input_height << "x" << input_channels << std::endl;
    
    // ==================== 创建模型实例 ====================
    printSeparator("Step 4: 创建模型实例");
    
    // 尝试自动创建
    ma::Model* model = ma::ModelFactory::create(engine, model_type_hint);
    int effective_type = model_type_hint;
    
    if (model == nullptr && model_type_hint == 0) {
        // 根据文件名猜测
        std::string path = model_path;
        if (path.find("yolov8") != std::string::npos) {
            effective_type = 6;
            std::cout << "[Info] 从文件名推断: YOLOv8" << std::endl;
        } else if (path.find("yolo11") != std::string::npos) {
            effective_type = 9;
            std::cout << "[Info] 从文件名推断: YOLO11" << std::endl;
        } else if (path.find("yolov5") != std::string::npos) {
            effective_type = 3;
            std::cout << "[Info] 从文件名推断: YOLOv5" << std::endl;
        } else if (path.find("cls") != std::string::npos) {
            effective_type = 4;
            std::cout << "[Info] 从文件名推断: 分类模型" << std::endl;
        }
        model = ma::ModelFactory::create(engine, effective_type);
    }
    
    // 如果 ModelFactory 失败，尝试直接构造
    if (model == nullptr && (effective_type == 6 || effective_type == 9)) {
        std::cout << "[Info] ModelFactory 失败，尝试直接构造..." << std::endl;
        if (effective_type == 6) {
            model = new ma::model::YoloV8(engine);
        } else if (effective_type == 9) {
            model = new ma::model::Yolo11(engine);
        }
    }
    
    if (model == nullptr) {
        std::cerr << "[FAIL] 无法创建模型实例" << std::endl;
        std::cerr << "  请尝试指定模型类型: " << argv[0] << " " << model_path << " <类型>" << std::endl;
        delete engine;
        return 1;
    }
    
    std::cout << "[OK] 模型实例创建成功" << std::endl;
    std::cout << "  模型类型: " << model->getType() << std::endl;
    std::cout << "  输入类型: " << model->getInputType() << std::endl;
    std::cout << "  输出类型: " << model->getOutputType() << std::endl;
    
    // ==================== 准备测试数据 ====================
    printSeparator("Step 5: 准备测试数据");
    
    // 创建纯色测试数据 (RGB)
    size_t data_size = input_width * input_height * 3;
    uint8_t* test_data = (uint8_t*)malloc(data_size);
    
    // 填充灰色 (128, 128, 128)
    memset(test_data, 128, data_size);
    
    ma_img_t img;
    img.data = test_data;
    img.size = data_size;
    img.width = input_width;
    img.height = input_height;
    img.format = MA_PIXEL_FORMAT_RGB888;
    img.rotate = MA_PIXEL_ROTATE_0;
    
    std::cout << "[OK] 测试数据: " << input_width << "x" << input_height << ", " << data_size << " bytes" << std::endl;
    
    // ==================== 运行推理测试 ====================
    printSeparator("Step 6: 推理测试");
    
    const int warmup_runs = 3;
    const int test_runs = 10;
    
    std::cout << "\n预热 " << warmup_runs << " 次..." << std::endl;
    for (int i = 0; i < warmup_runs; i++) {
        if (model->getOutputType() == MA_OUTPUT_TYPE_CLASS) {
            static_cast<ma::model::Classifier*>(model)->run(&img);
        } else if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX) {
            static_cast<ma::model::Detector*>(model)->run(&img);
        }
    }
    
    std::cout << "测试 " << test_runs << " 次..." << std::endl;
    
    double total_preprocess = 0, total_inference = 0, total_postprocess = 0;
    int detection_count = 0;
    
    // 判断是否需要使用自定义 YOLOv8 后处理
    // 当输入不是正方形时，SSCMA 后处理有 bug，使用自定义后处理
    bool use_custom_yolov8 = (effective_type == 6 && input_width != input_height);
    
    if (use_custom_yolov8) {
        std::cout << "[Info] 非正方形输入，使用自定义 YOLOv8 后处理" << std::endl;
    }
    
    for (int i = 0; i < test_runs; i++) {
        if (model->getOutputType() == MA_OUTPUT_TYPE_CLASS) {
            auto* classifier = static_cast<ma::model::Classifier*>(model);
            classifier->run(&img);
            auto perf = model->getPerf();
            total_preprocess += perf.preprocess;
            total_inference += perf.inference;
            total_postprocess += perf.postprocess;
        } else if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX && use_custom_yolov8) {
            // 使用自定义 YOLOv8 后处理
            auto t_pre_start = std::chrono::high_resolution_clock::now();
            
            // 构建输入并预处理
            ma_tensor_t input_tensor = {
                .size = img.size,
                .is_physical = false,
                .is_variable = false,
            };
            input_tensor.data.data = img.data;
            engine->setInput(0, input_tensor);
            
            auto t_infer_start = std::chrono::high_resolution_clock::now();
            engine->run();
            auto t_post_start = std::chrono::high_resolution_clock::now();
            
            // 自定义后处理
            yolov8::TensorInfo box_outputs[3];
            yolov8::TensorInfo cls_outputs[3];
            
            for (int j = 0; j < 3; j++) {
                auto box_out = engine->getOutput(j);
                auto cls_out = engine->getOutput(j + 3);
                
                box_outputs[j].data = box_out.data.data;
                box_outputs[j].dims[0] = box_out.shape.dims[0];
                box_outputs[j].dims[1] = box_out.shape.dims[1];
                box_outputs[j].dims[2] = box_out.shape.dims[2];
                box_outputs[j].dims[3] = box_out.shape.dims[3];
                box_outputs[j].scale = box_out.quant_param.scale;
                box_outputs[j].zero_point = box_out.quant_param.zero_point;
                box_outputs[j].is_int8 = (box_out.type == MA_TENSOR_TYPE_S8);
                
                cls_outputs[j].data = cls_out.data.data;
                cls_outputs[j].dims[0] = cls_out.shape.dims[0];
                cls_outputs[j].dims[1] = cls_out.shape.dims[1];
                cls_outputs[j].dims[2] = cls_out.shape.dims[2];
                cls_outputs[j].dims[3] = cls_out.shape.dims[3];
                cls_outputs[j].scale = cls_out.quant_param.scale;
                cls_outputs[j].zero_point = cls_out.quant_param.zero_point;
                cls_outputs[j].is_int8 = (cls_out.type == MA_TENSOR_TYPE_S8);
            }
            
            yolov8::YoloV8PostProcessor postprocessor;
            postprocessor.setThreshold(0.5f, 0.45f);
            auto results = postprocessor.process(box_outputs, cls_outputs, input_width, input_height);
            detection_count = results.size();
            
            auto t_end = std::chrono::high_resolution_clock::now();
            
            total_preprocess += std::chrono::duration<double, std::milli>(t_infer_start - t_pre_start).count();
            total_inference += std::chrono::duration<double, std::milli>(t_post_start - t_infer_start).count();
            total_postprocess += std::chrono::duration<double, std::milli>(t_end - t_post_start).count();
        } else if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX) {
            // 使用 SSCMA 原生后处理
            auto* detector = static_cast<ma::model::Detector*>(model);
            detector->run(&img);
            detection_count = forward_list_size(detector->getResults());
            auto perf = model->getPerf();
            total_preprocess += perf.preprocess;
            total_inference += perf.inference;
            total_postprocess += perf.postprocess;
        }
    }
    
    // ==================== 性能结果 ====================
    printSeparator("测试结果");
    
    std::cout << "\n模型: " << model_path << std::endl;
    std::cout << "输入: " << input_width << "x" << input_height << std::endl;
    std::cout << "测试次数: " << test_runs << std::endl;
    
    std::cout << "\n平均耗时:" << std::endl;
    std::cout << "  预处理:   " << (total_preprocess / test_runs) << " ms" << std::endl;
    std::cout << "  推理:     " << (total_inference / test_runs) << " ms" << std::endl;
    std::cout << "  后处理:   " << (total_postprocess / test_runs) << " ms" << std::endl;
    std::cout << "  总计:     " << ((total_preprocess + total_inference + total_postprocess) / test_runs) << " ms" << std::endl;
    
    double fps = 1000.0 / ((total_preprocess + total_inference + total_postprocess) / test_runs);
    std::cout << "\n理论 FPS: " << fps << std::endl;
    
    if (model->getOutputType() == MA_OUTPUT_TYPE_BBOX) {
        std::cout << "\n检测结果数: " << detection_count << " (纯色测试图，可能为0或误检)" << std::endl;
    }
    
    // ==================== 清理 ====================
    printSeparator();
    
    free(test_data);
    ma::ModelFactory::remove(model);
    delete engine;
    
    std::cout << "测试完成!\n" << std::endl;
    
    return 0;
}
