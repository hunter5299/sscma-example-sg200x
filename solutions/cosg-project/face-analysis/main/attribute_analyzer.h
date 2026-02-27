#ifndef _ATTRIBUTE_ANALYZER_H_
#define _ATTRIBUTE_ANALYZER_H_

#include <vector>
#include <string>
#include <array>

#include "face_detector.h"
#include "age_gender_race_runner.h"
#include "emotion_runner.h"

namespace face_analysis {

// FairFace 7-class emotion
enum class Emotion {
    ANGRY = 0,
    DISGUST,
    FEAR,
    HAPPY,
    SAD,
    SURPRISE,
    NEUTRAL,
    COUNT
};

inline const char* getEmotionName(Emotion emotion) {
    static const char* names[] = {
        "angry", "disgust", "fear", "happy",
        "sad", "surprise", "neutral"
    };
    int idx = static_cast<int>(emotion);
    if (idx >= 0 && idx < static_cast<int>(Emotion::COUNT)) {
        return names[idx];
    }
    return "unknown";
}

// FairFace age bin labels
inline const char* getAgeBinLabel(int age_bin) {
    static const char* labels[] = {
        "0-2", "3-9", "10-19", "20-29", "30-39",
        "40-49", "50-59", "60-69", "70+"
    };
    if (age_bin >= 0 && age_bin < 9) return labels[age_bin];
    return "unknown";
}

// FairFace race labels
inline const char* getRaceLabel(int race_bin) {
    static const char* labels[] = {
        "White", "Black", "Latino_Hispanic", "East_Asian",
        "Southeast_Asian", "Indian", "Middle_Eastern"
    };
    if (race_bin >= 0 && race_bin < 7) return labels[race_bin];
    return "unknown";
}

struct FaceAttributes {
    // Model format indicator
    bool is_fairface = false;  // true=FairFace (age bins+race), false=InsightFace (continuous age)

    // Age prediction
    // FairFace: age_bin=0..8 (bin index), age_label="0-2","3-9",...,"70+"
    // InsightFace: age_bin=-1, age_continuous=0..100 (years), age_label="28"
    int age_bin = -1;
    int age_continuous = -1;   // InsightFace: age in years (0-100), FairFace: -1
    std::string age_label;
    float age_confidence = 0.f;

    // Gender prediction
    std::string gender;  // "male" or "female"
    float gender_confidence = 0.f;

    // Race prediction (FairFace 7-class only, empty for InsightFace)
    int race_bin = -1;
    std::string race_label;
    float race_confidence = 0.f;

    // Emotion prediction (FairFace 7-class)
    Emotion emotion = Emotion::NEUTRAL;
    float emotion_confidence = 0.f;
    std::array<float, 7> emotion_probs = {};
};

struct AnalyzedFace {
    FaceInfo face;
    FaceAttributes attributes;
};

class AttributeAnalyzer {
public:
    AttributeAnalyzer() = default;
    ~AttributeAnalyzer() = default;

    // Initialize with model paths
    bool init(const std::string& genderage_model, const std::string& emotion_model = "");

    // Analyze attributes for multiple faces from full frame
    std::vector<AnalyzedFace> analyzeAll(ma_img_t* full_frame,
                                          const std::vector<FaceInfo>& faces);

    bool isGenderAgeReady() const { return genderage_ready_; }
    bool isEmotionReady() const { return emotion_ready_; }

private:
    AgeGenderRaceRunner agr_runner_;
    EmotionRunner emotion_runner_;
    bool genderage_ready_ = false;
    bool emotion_ready_ = false;
};

}  // namespace face_analysis

#endif  // _ATTRIBUTE_ANALYZER_H_
