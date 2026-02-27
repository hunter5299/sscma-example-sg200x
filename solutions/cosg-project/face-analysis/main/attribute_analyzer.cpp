#include "attribute_analyzer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#define TAG "AttributeAnalyzer"

namespace face_analysis {

bool AttributeAnalyzer::init(const std::string& genderage_model,
                              const std::string& emotion_model) {
    // Initialize AGR runner
    if (!genderage_model.empty()) {
        MA_LOGI(TAG, "Loading AGR model: %s", genderage_model.c_str());

        // ImageNet normalization: mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)
        agr_runner_.setPreprocess(123.675f, 116.28f, 103.53f,
                                  1.0f / (255.0f * 0.229f),
                                  1.0f / (255.0f * 0.224f),
                                  1.0f / (255.0f * 0.225f));

        if (!agr_runner_.init(genderage_model)) {
            MA_LOGE(TAG, "Failed to init AGR runner");
            return false;
        }

        MA_LOGI(TAG, "AGR model loaded, input size: %d", agr_runner_.inputSize());
        genderage_ready_ = true;
    }

    // Initialize Emotion runner
    if (!emotion_model.empty()) {
        MA_LOGI(TAG, "Loading Emotion model: %s", emotion_model.c_str());

        // Simple [0,1] normalization: mean=(0,0,0), scale=(1/255, 1/255, 1/255)
        emotion_runner_.setPreprocess(0.0f, 0.0f, 0.0f,
                                      1.0f / 255.0f,
                                      1.0f / 255.0f,
                                      1.0f / 255.0f);

        if (!emotion_runner_.init(emotion_model)) {
            MA_LOGE(TAG, "Failed to init Emotion runner");
        } else {
            MA_LOGI(TAG, "Emotion model loaded, input size: %d", emotion_runner_.inputSize());
            emotion_ready_ = true;
        }
    }

    return genderage_ready_;
}

std::vector<AnalyzedFace> AttributeAnalyzer::analyzeAll(
    ma_img_t* full_frame,
    const std::vector<FaceInfo>& faces) {

    std::vector<AnalyzedFace> results;
    results.reserve(faces.size());

    if (!full_frame || !full_frame->data || full_frame->width <= 0 || full_frame->height <= 0) {
        return results;
    }

    const uint8_t* frame_ptr = static_cast<const uint8_t*>(full_frame->data);
    const int fw = full_frame->width;
    const int fh = full_frame->height;

    for (const auto& face : faces) {
        AnalyzedFace analyzed;
        analyzed.face = face;

        // Convert normalized bbox to pixel coordinates
        float x1 = face.x * fw;
        float y1 = face.y * fh;
        float x2 = (face.x + face.w) * fw;
        float y2 = (face.y + face.h) * fh;

        // AGR inference
        if (genderage_ready_) {
            AgeGenderRaceResult agr;
            if (agr_runner_.infer(frame_ptr, fw, fh, x1, y1, x2, y2, agr) && agr.ok) {
                analyzed.attributes.is_fairface = agr.is_fairface;
                analyzed.attributes.gender = (agr.gender == 1) ? "male" : "female";
                analyzed.attributes.gender_confidence = agr.gender_score;

                if (agr.is_fairface) {
                    // FairFace: age bins + race
                    analyzed.attributes.age_bin = agr.age;
                    analyzed.attributes.age_label = getAgeBinLabel(agr.age);
                    analyzed.attributes.age_confidence = agr.age_score;
                    analyzed.attributes.race_bin = agr.race;
                    analyzed.attributes.race_label = getRaceLabel(agr.race);
                    analyzed.attributes.race_confidence = agr.race_score;
                } else {
                    // InsightFace: continuous age in years, no race
                    analyzed.attributes.age_continuous = agr.age;
                    analyzed.attributes.age_label = std::to_string(agr.age);
                    analyzed.attributes.age_confidence = agr.age_score;
                }
            }
        }

        // Emotion inference
        if (emotion_ready_) {
            EmotionResult emo;
            if (emotion_runner_.infer(frame_ptr, fw, fh, x1, y1, x2, y2, emo) && emo.ok) {
                analyzed.attributes.emotion = static_cast<Emotion>(emo.emotion);
                analyzed.attributes.emotion_confidence = emo.score;

                // Compute all probabilities for MQTT output
                // (runners only return argmax; recompute softmax from raw logits isn't available,
                //  so set dominant class = score, rest = 0 as approximation)
                analyzed.attributes.emotion_probs.fill(0.f);
                if (emo.emotion >= 0 && emo.emotion < 7) {
                    analyzed.attributes.emotion_probs[emo.emotion] = emo.score;
                }
            }
        }

        results.push_back(analyzed);
    }

    return results;
}

}  // namespace face_analysis
