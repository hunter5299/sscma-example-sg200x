#ifndef _OCR_PIPELINE_H_
#define _OCR_PIPELINE_H_

#include <vector>
#include <string>

#include <sscma.h>

#include "text_detector.h"
#include "text_recognizer.h"

namespace ppocr {

struct OcrResult {
    TextBox box;
    std::string text;
    float det_confidence;
    float rec_confidence;
};

struct OcrTimings {
    float detection_ms;
    float recognition_ms;
    float total_ms;
};

class OcrPipeline {
public:
    OcrPipeline();
    ~OcrPipeline();

    bool init(const std::string& det_model_path,
              const std::string& rec_model_path,
              const std::string& dict_path);

    // Run full OCR pipeline on a camera frame
    std::vector<OcrResult> process(ma_img_t* img, OcrTimings& timings);

private:
    // Crop and perspective-correct text region from image
    bool cropTextRegion(const ma_img_t* img, const TextBox& box,
                        std::vector<uint8_t>& output, int& out_w, int& out_h);

    // Sort boxes top-to-bottom, left-to-right
    void sortBoxes(std::vector<TextBox>& boxes);

    TextDetector detector_;
    TextRecognizer recognizer_;
    std::vector<uint8_t> crop_buffer_;

    bool initialized_;
    bool rec_available_;
};

}  // namespace ppocr

#endif  // _OCR_PIPELINE_H_
