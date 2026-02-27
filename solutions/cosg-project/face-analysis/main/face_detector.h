#ifndef _FACE_DETECTOR_H_
#define _FACE_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include <sscma.h>

namespace face_analysis {

struct FaceInfo {
    float x;        // Normalized x coordinate [0-1]
    float y;        // Normalized y coordinate [0-1]
    float w;        // Normalized width [0-1]
    float h;        // Normalized height [0-1]
    float score;    // Detection confidence [0-1]
    int id;         // Face ID for tracking
};

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    // Initialize the detector with model path
    bool init(const std::string& model_path);

    // Set detection threshold
    void setThreshold(float threshold);

    // Detect faces in the given image
    std::vector<FaceInfo> detect(ma_img_t* img);

    // Get input dimensions required by the model
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

    bool isInitialized() const { return initialized_; }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma::model::Detector* detector_;
    float threshold_;
    int input_width_;
    int input_height_;
    bool initialized_;
    int face_id_counter_;
};

}  // namespace face_analysis

#endif  // _FACE_DETECTOR_H_
