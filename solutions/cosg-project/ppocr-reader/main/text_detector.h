#ifndef _TEXT_DETECTOR_H_
#define _TEXT_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>

#include <sscma.h>

namespace ppocr {

// 4-point polygon box (clockwise from top-left)
struct TextBox {
    float points[4][2];  // [4][x,y] in pixel coordinates (original image space)
    float score;         // Detection confidence
};

class TextDetector {
public:
    TextDetector();
    ~TextDetector();

    bool init(const std::string& model_path);
    std::vector<TextBox> detect(ma_img_t* img);

    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

private:
    // Resize input preserving aspect ratio with letterbox padding
    void preprocess(const ma_img_t* src);

    // DBNet post-processing: probability map -> text boxes
    void postprocess(std::vector<TextBox>& boxes);

    // Unclip a polygon by expanding it by a ratio
    void unclipPolygon(float points[4][2], float unclip_ratio);

    // Get min-area bounding rect for contour and convert to 4 points
    bool contourToBox(const std::vector<std::vector<int>>& contour, TextBox& box);

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma_tensor_t input_tensor_;
    std::vector<uint8_t> letterbox_buffer_;

    int input_width_;
    int input_height_;
    int orig_width_;
    int orig_height_;
    float scale_;
    int pad_left_;
    int pad_top_;
    size_t tensor_stride_;  // Row stride in tensor (may differ from width*3 if aligned_input)

    float det_threshold_;   // Binary threshold for probability map (default: 0.3)
    float box_threshold_;   // Minimum box score (default: 0.5)
    float unclip_ratio_;    // Box expansion ratio (default: 1.6)
    int min_box_size_;      // Minimum box side length in pixels (default: 3)

    bool initialized_;
};

}  // namespace ppocr

#endif  // _TEXT_DETECTOR_H_
