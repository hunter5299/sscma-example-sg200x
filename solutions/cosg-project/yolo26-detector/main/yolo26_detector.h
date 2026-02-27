#ifndef _YOLO26_DETECTOR_H_
#define _YOLO26_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <cmath>

#include <sscma.h>

namespace yolo26 {

// Detection result structure
struct Detection {
    float x;            // Normalized center x [0-1]
    float y;            // Normalized center y [0-1]
    float w;            // Normalized width [0-1]
    float h;            // Normalized height [0-1]
    float confidence;   // Detection confidence [0-1]
    int class_id;       // Class index (0-79 for COCO)
    int id;             // Detection ID for tracking
};

// Letterbox info for coordinate transformation
struct LetterboxInfo {
    float scale;        // Scale factor applied to original image
    int pad_left;       // Left padding in pixels
    int pad_top;        // Top padding in pixels
    int new_width;      // Scaled image width (before padding)
    int new_height;     // Scaled image height (before padding)
    int orig_width;     // Original image width
    int orig_height;    // Original image height
};

// COCO class names (80 classes)
extern const char* COCO_CLASSES[80];

/**
 * YOLO26 Detector with custom post-processing
 *
 * YOLO26 output format (NMS-free, no DFL):
 *   - one2one_cv2.0: 1×4×80×80 (bbox x,y,w,h for large objects, stride 8)
 *   - one2one_cv2.1: 1×4×40×40 (bbox for medium objects, stride 16)
 *   - one2one_cv2.2: 1×4×20×20 (bbox for small objects, stride 32)
 *   - one2one_cv3.0: 1×80×80×80 (80-class scores for large objects)
 *   - one2one_cv3.1: 1×80×40×40 (80-class scores for medium objects)
 *   - one2one_cv3.2: 1×80×20×20 (80-class scores for small objects)
 */
class Yolo26Detector {
public:
    Yolo26Detector();
    ~Yolo26Detector();

    // Initialize with model path
    bool init(const std::string& model_path);

    // Set detection parameters
    void setConfThreshold(float threshold);
    void setNmsThreshold(float threshold);

    // Run detection on image
    std::vector<Detection> detect(ma_img_t* img);

    // Get input dimensions
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

    // Check if initialized
    bool isInitialized() const { return initialized_; }

    // Get class name
    static const char* getClassName(int class_id);

private:
    // Letterbox preprocessing: resize with aspect ratio preserved, pad to model input size
    void letterboxPreprocess(const ma_img_t* src);

    // Transform detection coordinates from letterbox space to original image space
    void transformCoordinates(std::vector<Detection>& detections);

    // Custom post-processing for YOLO26
    void decodeOutputs(std::vector<Detection>& results);

    // Apply Non-Maximum Suppression
    void applyNMS(std::vector<Detection>& detections);

    // Sigmoid activation
    inline float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Inverse sigmoid for threshold pre-computation
    inline float inverseSigmoid(float y) const {
        return -std::log(1.0f / y - 1.0f);
    }

private:
    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma_tensor_t input_tensor_;  // Input tensor reference
    ma_img_t img_;              // Internal image buffer for preprocessing
    std::vector<uint8_t> letterbox_buffer_;  // Buffer for letterbox preprocessing
    LetterboxInfo letterbox_info_;           // Current letterbox transformation info
    float conf_threshold_;
    float nms_threshold_;
    int input_width_;
    int input_height_;
    bool initialized_;
    int detection_id_counter_;

    // Output tensor indices (in order: cv2.0, cv3.0, cv2.1, cv3.1, cv2.2, cv3.2)
    static constexpr int NUM_OUTPUTS = 6;
    static constexpr int NUM_CLASSES = 80;

    // Grid sizes for each scale
    static constexpr int GRID_SIZES[3] = {80, 40, 20};
    static constexpr int STRIDES[3] = {8, 16, 32};
};

}  // namespace yolo26

#endif  // _YOLO26_DETECTOR_H_
