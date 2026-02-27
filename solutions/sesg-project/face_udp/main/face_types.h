#pragma once

#include <cstdint>
#include <type_traits>

namespace udp_face {

// 单帧最多发送的人脸数量（与 UDP 协议/发送端缓冲一致）
static constexpr int MAX_DETECTIONS = 10;

// UDP 发送的人脸结果（与 Python 端 '<fffffiiiiiffff' 对齐）：
// 5x float + 5x int32 + 4x float = 56 bytes
struct FaceResult {
    float x;            // bbox center_x (normalized 0..1)
    float y;            // bbox center_y (normalized 0..1)
    float w;            // bbox width    (normalized 0..1)
    float h;            // bbox height   (normalized 0..1)
    float score;        // detection score
    int32_t target;     // class id (face=0)
    int32_t gender;     // 1=Male, 0=Female, -1=Unknown
    int32_t age;        // age group id (0..8), -1=Unknown
    int32_t race;       // race id (0..6), -1=Unknown
    int32_t emotion;    // emotion id (0..6), -1=Unknown
    float gender_score; // gender confidence (softmax max prob)
    float age_score;    // age confidence (softmax max prob)
    float race_score;   // race confidence (softmax max prob)
    float emotion_score;// emotion confidence (softmax max prob)
};

static_assert(std::is_standard_layout<FaceResult>::value, "FaceResult must be standard layout");
static_assert(sizeof(FaceResult) == 56, "FaceResult must be 56 bytes for UDP protocol");

}  // namespace udp_face
