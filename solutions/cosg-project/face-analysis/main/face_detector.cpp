#include "face_detector.h"

#include <algorithm>

#define TAG "FaceDetector"

namespace face_analysis {

FaceDetector::FaceDetector()
    : engine_(nullptr),
      detector_(nullptr),
      threshold_(0.5f),
      input_width_(640),
      input_height_(640),
      initialized_(false),
      face_id_counter_(0) {}

FaceDetector::~FaceDetector() {
    if (detector_) {
        ma::ModelFactory::remove(detector_);
        detector_ = nullptr;
    }
}

bool FaceDetector::init(const std::string& model_path) {
    MA_LOGI(TAG, "Initializing face detector with model: %s", model_path.c_str());

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

    // Log tensor info
    int num_inputs = engine_->getInputSize();
    int num_outputs = engine_->getOutputSize();
    MA_LOGI(TAG, "Model loaded: %d inputs, %d outputs", num_inputs, num_outputs);

    for (int i = 0; i < num_inputs; i++) {
        auto shape = engine_->getInputShape(i);
        auto tensor = engine_->getInput(i);
        MA_LOGI(TAG, "  Input[%d]: dims=[%d,%d,%d,%d] type=%d",
                i, shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3], tensor.type);
    }
    for (int i = 0; i < num_outputs; i++) {
        auto shape = engine_->getOutputShape(i);
        auto tensor = engine_->getOutput(i);
        MA_LOGI(TAG, "  Output[%d]: dims=[%d,%d,%d,%d] type=%d size=%zu",
                i, shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3],
                tensor.type, tensor.size);
    }

    // ModelFactory handles SCRFD, YOLO multi-output, and YOLO single-output
    ma::Model* model = ma::ModelFactory::create(engine_.get());
    if (model == nullptr) {
        MA_LOGE(TAG, "ModelFactory::create returned nullptr - model format not recognized");
        return false;
    }

    if (model->getOutputType() != MA_OUTPUT_TYPE_BBOX) {
        MA_LOGE(TAG, "Model output type is not BBOX, got: %d", model->getOutputType());
        ma::ModelFactory::remove(model);
        return false;
    }

    detector_ = static_cast<ma::model::Detector*>(model);

    // Get input dimensions
    const ma_img_t* model_input = static_cast<const ma_img_t*>(detector_->getInput());
    if (model_input) {
        input_width_ = model_input->width;
        input_height_ = model_input->height;
    }

    MA_LOGI(TAG, "Face detector initialized, input size: %dx%d", input_width_, input_height_);
    initialized_ = true;
    return true;
}

void FaceDetector::setThreshold(float threshold) {
    threshold_ = std::max(0.0f, std::min(1.0f, threshold));
    if (detector_) {
        detector_->setConfig(MA_MODEL_CFG_OPT_THRESHOLD, threshold_);
    }
}

std::vector<FaceInfo> FaceDetector::detect(ma_img_t* img) {
    std::vector<FaceInfo> faces;

    if (!initialized_ || detector_ == nullptr || img == nullptr) {
        return faces;
    }

    ma_err_t ret = detector_->run(img);
    if (ret != MA_OK) {
        MA_LOGE(TAG, "Detection failed with error: %d", ret);
        return faces;
    }

    auto results = detector_->getResults();
    for (const auto& bbox : results) {
        if (bbox.score >= threshold_) {
            FaceInfo face;
            face.x = bbox.x;
            face.y = bbox.y;
            face.w = bbox.w;
            face.h = bbox.h;
            face.score = bbox.score;
            face.id = face_id_counter_++;
            faces.push_back(face);
        }
    }

    return faces;
}

}  // namespace face_analysis
