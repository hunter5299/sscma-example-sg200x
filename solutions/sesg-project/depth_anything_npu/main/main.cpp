#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sscma.h>
#include <sys/stat.h>
#include <video.h>

using namespace ma;

namespace {

const char* tensor_type_name(ma_tensor_type_t type) {
    switch (type) {
        case MA_TENSOR_TYPE_U8: return "U8";
        case MA_TENSOR_TYPE_S8: return "S8";
        case MA_TENSOR_TYPE_U16: return "U16";
        case MA_TENSOR_TYPE_S16: return "S16";
        case MA_TENSOR_TYPE_U32: return "U32";
        case MA_TENSOR_TYPE_S32: return "S32";
        case MA_TENSOR_TYPE_F16: return "F16";
        case MA_TENSOR_TYPE_F32: return "F32";
        case MA_TENSOR_TYPE_F64: return "F64";
        case MA_TENSOR_TYPE_BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

void print_shape(const char* label, const ma_tensor_t& tensor) {
    std::cout << label << ": shape=[";
    for (uint32_t i = 0; i < tensor.shape.size; ++i) {
        std::cout << tensor.shape.dims[i];
        if (i + 1 < tensor.shape.size) {
            std::cout << ",";
        }
    }
    std::cout << "], type=" << tensor_type_name(tensor.type)
              << ", bytes=" << tensor.size
              << ", scale=" << tensor.quant_param.scale
              << ", zp=" << tensor.quant_param.zero_point << std::endl;
}

size_t shape_elements(const ma_shape_t& shape) {
    if (shape.size == 0) {
        return 0;
    }
    size_t n = 1;
    for (uint32_t i = 0; i < shape.size; ++i) {
        n *= static_cast<size_t>(shape.dims[i]);
    }
    return n;
}

float bf16_to_float(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<float> tensor_to_float(const ma_tensor_t& tensor) {
    size_t elements = shape_elements(tensor.shape);
    if (elements == 0) {
        if (tensor.type == MA_TENSOR_TYPE_F32) {
            elements = tensor.size / sizeof(float);
        } else if (tensor.type == MA_TENSOR_TYPE_BF16 || tensor.type == MA_TENSOR_TYPE_U16 || tensor.type == MA_TENSOR_TYPE_S16) {
            elements = tensor.size / sizeof(uint16_t);
        } else {
            elements = tensor.size;
        }
    }

    std::vector<float> values(elements);
    switch (tensor.type) {
        case MA_TENSOR_TYPE_F32: {
            const float* p = reinterpret_cast<const float*>(tensor.data.data);
            std::copy(p, p + elements, values.begin());
            break;
        }
        case MA_TENSOR_TYPE_BF16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(tensor.data.data);
            for (size_t i = 0; i < elements; ++i) {
                values[i] = bf16_to_float(p[i]);
            }
            break;
        }
        case MA_TENSOR_TYPE_U8: {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(tensor.data.data);
            for (size_t i = 0; i < elements; ++i) {
                values[i] = (static_cast<int>(p[i]) - tensor.quant_param.zero_point) * tensor.quant_param.scale;
            }
            break;
        }
        case MA_TENSOR_TYPE_S8: {
            const int8_t* p = reinterpret_cast<const int8_t*>(tensor.data.data);
            for (size_t i = 0; i < elements; ++i) {
                values[i] = (static_cast<int>(p[i]) - tensor.quant_param.zero_point) * tensor.quant_param.scale;
            }
            break;
        }
        default:
            throw std::runtime_error(std::string("unsupported output tensor type: ") + tensor_type_name(tensor.type));
    }
    return values;
}

bool infer_hw(const ma_shape_t& shape, int& h, int& w) {
    if (shape.size >= 4) {
        h = shape.dims[shape.size - 2];
        w = shape.dims[shape.size - 1];
        return h > 0 && w > 0;
    }
    if (shape.size == 3) {
        h = shape.dims[1];
        w = shape.dims[2];
        return h > 0 && w > 0;
    }
    if (shape.size == 2) {
        h = shape.dims[0];
        w = shape.dims[1];
        return h > 0 && w > 0;
    }
    return false;
}

::cv::Mat make_side_by_side(const std::vector<float>& depth, int h, int w, const ::cv::Mat& preview_bgr, float& min_v, float& max_v) {
    min_v = std::numeric_limits<float>::infinity();
    max_v = -std::numeric_limits<float>::infinity();
    for (float v : depth) {
        if (std::isfinite(v)) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (!std::isfinite(min_v) || !std::isfinite(max_v) || std::abs(max_v - min_v) < 1e-6f) {
        min_v = 0.0f;
        max_v = 1.0f;
    }

    ::cv::Mat gray(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = depth[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
            float norm = (v - min_v) / (max_v - min_v);
            norm = std::min(1.0f, std::max(0.0f, norm));
            gray.at<uint8_t>(y, x) = static_cast<uint8_t>(norm * 255.0f + 0.5f);
        }
    }

    ::cv::Mat color;
    ::cv::applyColorMap(gray, color, ::cv::COLORMAP_INFERNO);

    ::cv::Mat preview;
    ::cv::resize(preview_bgr, preview, ::cv::Size(w, h), 0, 0, ::cv::INTER_AREA);

    ::cv::Mat combined;
    ::cv::hconcat(preview, color, combined);
    return combined;
}

void write_depth_outputs(const std::vector<float>& depth, int h, int w, const ::cv::Mat& preview_bgr, const std::string& output_prefix) {
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : depth) {
        if (std::isfinite(v)) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (!std::isfinite(min_v) || !std::isfinite(max_v) || std::abs(max_v - min_v) < 1e-6f) {
        min_v = 0.0f;
        max_v = 1.0f;
    }

    ::cv::Mat gray(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = depth[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
            float norm = (v - min_v) / (max_v - min_v);
            norm = std::min(1.0f, std::max(0.0f, norm));
            gray.at<uint8_t>(y, x) = static_cast<uint8_t>(norm * 255.0f + 0.5f);
        }
    }

    ::cv::Mat color;
    ::cv::applyColorMap(gray, color, ::cv::COLORMAP_INFERNO);

    ::cv::Mat preview;
    ::cv::resize(preview_bgr, preview, ::cv::Size(w, h), 0, 0, ::cv::INTER_AREA);

    ::cv::Mat combined;
    ::cv::hconcat(preview, color, combined);

    const std::string gray_path = output_prefix + "_gray.png";
    const std::string color_path = output_prefix + "_color.png";
    const std::string side_path = output_prefix + "_side_by_side.png";
    if (!::cv::imwrite(gray_path, gray) || !::cv::imwrite(color_path, color) || !::cv::imwrite(side_path, combined)) {
        throw std::runtime_error("failed to write one or more output images");
    }

    std::cout << std::fixed << std::setprecision(6)
              << "depth_min=" << min_v << ", depth_max=" << max_v << std::endl;
    std::cout << "wrote: " << gray_path << std::endl;
    std::cout << "wrote: " << color_path << std::endl;
    std::cout << "wrote: " << side_path << std::endl;
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " model.cvimodel input_image output_prefix [repeat]\n"
              << "       " << prog << " --camera model.cvimodel output_dir [frames] [skip]\n"
              << "Example: " << prog << " depth_anything_v2_vits_224_int8_min8.cvimodel test.jpg depth_out 3\n"
              << "Example: " << prog << " --camera depth_anything_v2_vits_224_int8_min8.cvimodel ./live_depth 20 1\n";
}

std::string frame_path(const std::string& output_dir, int index) {
    std::ostringstream oss;
    oss << output_dir << "/frame_" << std::setw(4) << std::setfill('0') << index << ".png";
    return oss.str();
}

int run_camera_mode(const std::string& model_path, const std::string& output_dir, int frames_to_save, int skip_interval) {
    mkdir(output_dir.c_str(), 0755);

    auto* engine = new ma::engine::EngineCVI();
    ma_err_t ret = engine->init();
    if (ret != MA_OK) {
        std::cerr << "engine init failed: " << ret << std::endl;
        delete engine;
        return 1;
    }
    ret = engine->load(model_path.c_str());
    if (ret != MA_OK) {
        std::cerr << "model load failed: " << ret << std::endl;
        delete engine;
        return 1;
    }

    ma_tensor_t model_input = engine->getInput(0);
    print_shape("input[0]", model_input);

    int input_h = 224;
    int input_w = 224;
    if (model_input.shape.size == 4) {
        if (model_input.shape.dims[3] == 3) {
            input_h = model_input.shape.dims[1];
            input_w = model_input.shape.dims[2];
        } else {
            input_h = model_input.shape.dims[2];
            input_w = model_input.shape.dims[3];
        }
    }

    Device* device = Device::getInstance();
    Camera* camera = nullptr;
    for (auto& sensor : device->getSensors()) {
        if (sensor->getType() == ma::Sensor::Type::kCamera) {
            camera = static_cast<Camera*>(sensor);
            break;
        }
    }
    if (!camera) {
        std::cerr << "camera not found" << std::endl;
        delete engine;
        return 1;
    }

    camera->init(0);
    Camera::CtrlValue v;
    v.i32 = 1;
    camera->commandCtrl(Camera::CtrlType::kChannel, Camera::CtrlMode::kWrite, v);
    v.u16s[0] = 640;
    v.u16s[1] = 480;
    camera->commandCtrl(Camera::CtrlType::kWindow, Camera::CtrlMode::kWrite, v);
    v.i32 = MA_PIXEL_FORMAT_JPEG;
    camera->commandCtrl(Camera::CtrlType::kFormat, Camera::CtrlMode::kWrite, v);
    v.i32 = 5;
    camera->commandCtrl(Camera::CtrlType::kFps, Camera::CtrlMode::kWrite, v);
    v.i32 = 0;
    camera->commandCtrl(Camera::CtrlType::kPhysical, Camera::CtrlMode::kWrite, v);

    ret = camera->startStream(Camera::StreamMode::kRefreshOnReturn);
    if (ret != MA_OK) {
        std::cerr << "camera startStream failed: " << ret << std::endl;
        camera->deInit();
        delete engine;
        return 1;
    }

    int seen = 0;
    int saved = 0;
    double total_run_ms = 0.0;
    while (saved < frames_to_save) {
        ma_img_t frame{};
        ret = camera->retrieveFrame(frame, MA_PIXEL_FORMAT_JPEG);
        if (ret != MA_OK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        ++seen;
        if ((seen % skip_interval) != 0) {
            camera->returnFrame(frame);
            continue;
        }

        if (frame.data == nullptr || frame.size == 0) {
            std::cerr << "bad camera frame: size=" << frame.size << std::endl;
            camera->returnFrame(frame);
            continue;
        }

        std::vector<uint8_t> jpeg(frame.data, frame.data + frame.size);
        ::cv::Mat decoded_bgr = ::cv::imdecode(jpeg, ::cv::IMREAD_COLOR);
        if (decoded_bgr.empty()) {
            std::cerr << "failed to decode jpeg frame: size=" << frame.size << std::endl;
            camera->returnFrame(frame);
            continue;
        }

        ::cv::Mat resized_bgr;
        ::cv::resize(decoded_bgr, resized_bgr, ::cv::Size(input_w, input_h), 0, 0, ::cv::INTER_AREA);
        ::cv::Mat resized_rgb;
        ::cv::cvtColor(resized_bgr, resized_rgb, ::cv::COLOR_BGR2RGB);
        if (!resized_rgb.isContinuous()) {
            resized_rgb = resized_rgb.clone();
        }

        const size_t expected_input = static_cast<size_t>(input_w) * static_cast<size_t>(input_h) * 3;
        ma_tensor_t input_tensor{};
        input_tensor.size = expected_input;
        input_tensor.is_physical = false;
        input_tensor.is_variable = false;
        input_tensor.data.data = resized_rgb.data;
        ret = engine->setInput(0, input_tensor);
        if (ret != MA_OK) {
            std::cerr << "setInput failed: " << ret << std::endl;
            camera->returnFrame(frame);
            break;
        }

        auto r0 = std::chrono::steady_clock::now();
        ret = engine->run();
        auto r1 = std::chrono::steady_clock::now();
        const double run_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
        if (ret != MA_OK) {
            std::cerr << "engine run failed: " << ret << std::endl;
            camera->returnFrame(frame);
            break;
        }
        total_run_ms += run_ms;

        ma_tensor_t output = engine->getOutput(0);
        int out_h = 0;
        int out_w = 0;
        if (!infer_hw(output.shape, out_h, out_w)) {
            out_h = input_h;
            out_w = input_w;
        }

        std::vector<float> depth = tensor_to_float(output);
        const size_t expected_output = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        if (depth.size() > expected_output) {
            depth.resize(expected_output);
        }

        float min_v = 0.0f;
        float max_v = 0.0f;
        ::cv::Mat combined = make_side_by_side(depth, out_h, out_w, resized_bgr, min_v, max_v);
        const std::string out_path = frame_path(output_dir, saved);
        if (!::cv::imwrite(out_path, combined)) {
            std::cerr << "failed to write " << out_path << std::endl;
            camera->returnFrame(frame);
            break;
        }

        std::cout << "saved=" << saved << ", run_ms=" << run_ms
                  << ", depth_min=" << min_v << ", depth_max=" << max_v
                  << ", path=" << out_path << std::endl;
        ++saved;
        camera->returnFrame(frame);
    }

    camera->stopStream();
    camera->deInit();
    delete engine;

    if (saved == 0) {
        return 1;
    }
    std::cout << "saved_frames=" << saved << ", avg_npu_run_ms=" << (total_run_ms / saved) << std::endl;
    return saved == frames_to_save ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--camera") {
        if (argc < 4) {
            usage(argv[0]);
            return 1;
        }
        const std::string model_path = argv[2];
        const std::string output_dir = argv[3];
        const int frames_to_save = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 20;
        const int skip_interval = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 1;
        return run_camera_mode(model_path, output_dir, frames_to_save, skip_interval);
    }

    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_prefix = argv[3];
    const int repeat = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 1;

    try {
        ::cv::Mat input_bgr = ::cv::imread(image_path, ::cv::IMREAD_COLOR);
        if (input_bgr.empty()) {
            std::cerr << "failed to read image: " << image_path << std::endl;
            return 1;
        }

        auto* engine = new ma::engine::EngineCVI();
        auto t0 = std::chrono::steady_clock::now();
        ma_err_t ret = engine->init();
        auto t1 = std::chrono::steady_clock::now();
        if (ret != MA_OK) {
            std::cerr << "engine init failed: " << ret << std::endl;
            delete engine;
            return 1;
        }

        ret = engine->load(model_path.c_str());
        auto t2 = std::chrono::steady_clock::now();
        if (ret != MA_OK) {
            std::cerr << "model load failed: " << ret << std::endl;
            delete engine;
            return 1;
        }

        std::cout << "engine_init_ms=" << std::chrono::duration<double, std::milli>(t1 - t0).count() << std::endl;
        std::cout << "model_load_ms=" << std::chrono::duration<double, std::milli>(t2 - t1).count() << std::endl;
        std::cout << "input_count=" << engine->getInputSize() << ", output_count=" << engine->getOutputSize() << std::endl;

        ma_tensor_t model_input = engine->getInput(0);
        print_shape("input[0]", model_input);

        int input_h = 224;
        int input_w = 224;
        if (model_input.shape.size == 4) {
            if (model_input.shape.dims[3] == 3) {
                input_h = model_input.shape.dims[1];
                input_w = model_input.shape.dims[2];
            } else {
                input_h = model_input.shape.dims[2];
                input_w = model_input.shape.dims[3];
            }
        }
        if (input_h <= 0 || input_w <= 0) {
            input_h = 224;
            input_w = 224;
        }

        ::cv::Mat resized_bgr;
        ::cv::resize(input_bgr, resized_bgr, ::cv::Size(input_w, input_h), 0, 0, ::cv::INTER_AREA);

        ::cv::Mat resized_rgb;
        ::cv::cvtColor(resized_bgr, resized_rgb, ::cv::COLOR_BGR2RGB);
        if (!resized_rgb.isContinuous()) {
            resized_rgb = resized_rgb.clone();
        }

        ma_tensor_t input_tensor{};
        input_tensor.size = static_cast<size_t>(input_w) * static_cast<size_t>(input_h) * 3;
        input_tensor.is_physical = false;
        input_tensor.is_variable = false;
        input_tensor.data.data = resized_rgb.data;

        ret = engine->setInput(0, input_tensor);
        if (ret != MA_OK) {
            std::cerr << "setInput failed: " << ret << std::endl;
            delete engine;
            return 1;
        }

        double last_run_ms = 0.0;
        for (int i = 0; i < repeat; ++i) {
            auto r0 = std::chrono::steady_clock::now();
            ret = engine->run();
            auto r1 = std::chrono::steady_clock::now();
            last_run_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
            std::cout << "run[" << i << "]_ms=" << last_run_ms << ", ret=" << ret << std::endl;
            if (ret != MA_OK) {
                delete engine;
                return 1;
            }
        }

        if (engine->getOutputSize() < 1) {
            std::cerr << "model produced no outputs" << std::endl;
            delete engine;
            return 1;
        }

        ma_tensor_t output = engine->getOutput(0);
        print_shape("output[0]", output);
        int out_h = 0;
        int out_w = 0;
        if (!infer_hw(output.shape, out_h, out_w)) {
            out_h = input_h;
            out_w = input_w;
        }

        std::vector<float> depth = tensor_to_float(output);
        const size_t expected = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        if (depth.size() < expected) {
            std::cerr << "output elements too small: got " << depth.size() << ", expected " << expected << std::endl;
            delete engine;
            return 1;
        }
        if (depth.size() > expected) {
            depth.resize(expected);
        }

        write_depth_outputs(depth, out_h, out_w, resized_bgr, output_prefix);
        std::cout << "last_npu_run_ms=" << last_run_ms << std::endl;

        delete engine;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
