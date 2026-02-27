#ifndef _TEXT_RECOGNIZER_H_
#define _TEXT_RECOGNIZER_H_

#include <vector>
#include <string>
#include <memory>

#include <sscma.h>

namespace ppocr {

struct RecognitionResult {
    std::string text;
    float confidence;
};

class TextRecognizer {
public:
    TextRecognizer();
    ~TextRecognizer();

    bool init(const std::string& model_path, const std::string& dict_path);

    // Recognize text from a pre-cropped and perspective-corrected text image
    RecognitionResult recognize(const uint8_t* rgb_data, int width, int height);

    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

private:
    bool loadDictionary(const std::string& dict_path);
    void preprocess(const uint8_t* rgb_data, int width, int height);
    RecognitionResult ctcDecode(const ma_tensor_t& output);

    std::unique_ptr<ma::engine::EngineCVI> engine_;
    ma_tensor_t input_tensor_;
    std::vector<uint8_t> resize_buffer_;

    std::vector<std::string> dictionary_;  // Character dictionary
    int input_width_;
    int input_height_;
    int dict_size_;  // Number of classes (dict + blank)
    size_t tensor_stride_;  // Row stride in tensor (may differ from width*3 if aligned_input)

    bool initialized_;
};

}  // namespace ppocr

#endif  // _TEXT_RECOGNIZER_H_
