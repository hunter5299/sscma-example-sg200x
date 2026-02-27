#include "ocr_pipeline.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>

#define TAG "OcrPipeline"

namespace ppocr {

OcrPipeline::OcrPipeline() : initialized_(false), rec_available_(false) {}

OcrPipeline::~OcrPipeline() {}

bool OcrPipeline::init(const std::string& det_model_path,
                        const std::string& rec_model_path,
                        const std::string& dict_path) {
    if (!detector_.init(det_model_path)) {
        MA_LOGE(TAG, "Failed to initialize text detector");
        return false;
    }

    if (!recognizer_.init(rec_model_path, dict_path)) {
        MA_LOGW(TAG, "Text recognizer init failed - detection only mode");
        rec_available_ = false;
    } else {
        rec_available_ = true;
    }

    // Pre-allocate crop buffer (max reasonable text region at 640x480 input)
    crop_buffer_.resize(640 * 480 * 3);

    initialized_ = true;
    MA_LOGI(TAG, "OCR pipeline initialized (recognition: %s)",
            rec_available_ ? "enabled" : "disabled");
    return true;
}

void OcrPipeline::sortBoxes(std::vector<TextBox>& boxes) {
    // Sort boxes top-to-bottom, left-to-right
    // Group by approximate row (within 20px vertical tolerance)
    std::sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b) {
        float ay = std::min({a.points[0][1], a.points[1][1], a.points[2][1], a.points[3][1]});
        float by = std::min({b.points[0][1], b.points[1][1], b.points[2][1], b.points[3][1]});

        if (std::abs(ay - by) < 20.0f) {
            float ax = std::min({a.points[0][0], a.points[1][0], a.points[2][0], a.points[3][0]});
            float bx = std::min({b.points[0][0], b.points[1][0], b.points[2][0], b.points[3][0]});
            return ax < bx;
        }
        return ay < by;
    });
}

bool OcrPipeline::cropTextRegion(const ma_img_t* img, const TextBox& box,
                                  std::vector<uint8_t>& output, int& out_w, int& out_h) {
    // Source points (the detected text polygon)
    // Points are ordered: [0]=TL, [1]=TR, [2]=BR, [3]=BL
    cv::Point2f src_pts[4];
    for (int i = 0; i < 4; ++i) {
        src_pts[i] = cv::Point2f(box.points[i][0], box.points[i][1]);
    }

    // Calculate destination size from polygon dimensions
    float width_top = std::sqrt(
        std::pow(src_pts[1].x - src_pts[0].x, 2) + std::pow(src_pts[1].y - src_pts[0].y, 2));
    float width_bot = std::sqrt(
        std::pow(src_pts[2].x - src_pts[3].x, 2) + std::pow(src_pts[2].y - src_pts[3].y, 2));
    float height_left = std::sqrt(
        std::pow(src_pts[3].x - src_pts[0].x, 2) + std::pow(src_pts[3].y - src_pts[0].y, 2));
    float height_right = std::sqrt(
        std::pow(src_pts[2].x - src_pts[1].x, 2) + std::pow(src_pts[2].y - src_pts[1].y, 2));

    out_w = static_cast<int>(std::max(width_top, width_bot));
    out_h = static_cast<int>(std::max(height_left, height_right));

    if (out_w < 2 || out_h < 2) return false;

    // Add vertical padding (15% of height on each side) to ensure
    // the recognition model has enough context around text edges.
    float pad_ratio = 0.15f;
    float pad_h = std::max(height_left, height_right) * pad_ratio;

    // Compute perpendicular direction to the top edge
    cv::Point2f top_dir(src_pts[1].x - src_pts[0].x, src_pts[1].y - src_pts[0].y);
    float top_len = std::sqrt(top_dir.x * top_dir.x + top_dir.y * top_dir.y);
    if (top_len > 1e-6f) {
        top_dir.x /= top_len;
        top_dir.y /= top_len;
    }
    cv::Point2f up_normal(-top_dir.y, top_dir.x);

    // Expand: move top points up, bottom points down
    src_pts[0].x += up_normal.x * pad_h;
    src_pts[0].y += up_normal.y * pad_h;
    src_pts[1].x += up_normal.x * pad_h;
    src_pts[1].y += up_normal.y * pad_h;
    src_pts[2].x -= up_normal.x * pad_h;
    src_pts[2].y -= up_normal.y * pad_h;
    src_pts[3].x -= up_normal.x * pad_h;
    src_pts[3].y -= up_normal.y * pad_h;

    // Clip source points to image bounds
    for (int i = 0; i < 4; ++i) {
        src_pts[i].x = std::max(0.0f, std::min(static_cast<float>(img->width - 1), src_pts[i].x));
        src_pts[i].y = std::max(0.0f, std::min(static_cast<float>(img->height - 1), src_pts[i].y));
    }

    // Update output height to include padding
    out_h = static_cast<int>(out_h + 2 * pad_h);

    // Destination points (rectangle)
    cv::Point2f dst_pts[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(out_w), 0),
        cv::Point2f(static_cast<float>(out_w), static_cast<float>(out_h)),
        cv::Point2f(0, static_cast<float>(out_h))
    };

    // Optimization: crop a bounding-box ROI from the full image first,
    // then run warpPerspective on the smaller ROI (much faster on RISC-V).
    float min_x = src_pts[0].x, max_x = src_pts[0].x;
    float min_y = src_pts[0].y, max_y = src_pts[0].y;
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, src_pts[i].x);
        max_x = std::max(max_x, src_pts[i].x);
        min_y = std::min(min_y, src_pts[i].y);
        max_y = std::max(max_y, src_pts[i].y);
    }

    int roi_x = static_cast<int>(std::max(0.0f, std::floor(min_x)));
    int roi_y = static_cast<int>(std::max(0.0f, std::floor(min_y)));
    int roi_x2 = static_cast<int>(std::min(static_cast<float>(img->width), std::ceil(max_x)));
    int roi_y2 = static_cast<int>(std::min(static_cast<float>(img->height), std::ceil(max_y)));
    int roi_w = roi_x2 - roi_x;
    int roi_h = roi_y2 - roi_y;

    if (roi_w < 2 || roi_h < 2) return false;

    // Adjust source points to be relative to ROI
    cv::Point2f roi_pts[4];
    for (int i = 0; i < 4; ++i) {
        roi_pts[i] = cv::Point2f(src_pts[i].x - roi_x, src_pts[i].y - roi_y);
    }

    cv::Mat full_mat(img->height, img->width, CV_8UC3, img->data);
    cv::Mat roi_mat = full_mat(cv::Rect(roi_x, roi_y, roi_w, roi_h));

    cv::Mat M = cv::getPerspectiveTransform(roi_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(roi_mat, warped, M, cv::Size(out_w, out_h));

    // If text is taller than wide, it's likely vertical - rotate
    if (static_cast<float>(out_h) > static_cast<float>(out_w) * 1.5f) {
        cv::Mat rotated;
        cv::rotate(warped, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        warped = rotated;
        std::swap(out_w, out_h);
    }

    // Mild unsharp mask to sharpen text edges
    cv::Mat blurred;
    cv::GaussianBlur(warped, blurred, cv::Size(0, 0), 2.0);
    cv::addWeighted(warped, 1.5, blurred, -0.5, 0, warped);

    // Copy to output buffer
    size_t required = out_w * out_h * 3;
    if (output.size() < required) {
        output.resize(required);
    }
    std::memcpy(output.data(), warped.data, required);

    return true;
}

std::vector<OcrResult> OcrPipeline::process(ma_img_t* img, OcrTimings& timings) {
    std::vector<OcrResult> results;

    if (!initialized_ || !img) return results;

    // Detection
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<TextBox> boxes = detector_.detect(img);

    auto t1 = std::chrono::high_resolution_clock::now();
    timings.detection_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (boxes.empty()) {
        timings.recognition_ms = 0;
        timings.total_ms = timings.detection_ms;
        return results;
    }

    // Limit to top-N boxes by detection confidence to cap recognition time
    static constexpr size_t kMaxBoxes = 5;
    if (boxes.size() > kMaxBoxes) {
        std::partial_sort(boxes.begin(), boxes.begin() + kMaxBoxes, boxes.end(),
                          [](const TextBox& a, const TextBox& b) { return a.score > b.score; });
        boxes.resize(kMaxBoxes);
    }

    // Sort boxes for reading order
    sortBoxes(boxes);

    // Recognition for each detected text region
    auto t2 = std::chrono::high_resolution_clock::now();

    for (size_t bi = 0; bi < boxes.size(); ++bi) {
        const auto& box = boxes[bi];

        if (!rec_available_) {
            // Detection-only mode: return boxes without text
            OcrResult ocr;
            ocr.box = box;
            ocr.det_confidence = box.score;
            ocr.rec_confidence = 0.0f;
            results.push_back(std::move(ocr));
            continue;
        }

        int crop_w = 0, crop_h = 0;
        if (!cropTextRegion(img, box, crop_buffer_, crop_w, crop_h)) {
            continue;
        }

        RecognitionResult rec = recognizer_.recognize(crop_buffer_.data(), crop_w, crop_h);

        OcrResult ocr;
        ocr.box = box;
        ocr.det_confidence = box.score;

        // Always include the detection box; attach text only if confidence is sufficient
        static constexpr float kMinRecConfidence = 0.3f;
        if (!rec.text.empty() && rec.confidence >= kMinRecConfidence) {
            ocr.text = rec.text;
            ocr.rec_confidence = rec.confidence;
        } else {
            MA_LOGD(TAG, "Box[%zu] rec dropped: text='%s' conf=%.4f (threshold=%.2f)",
                    bi, rec.text.c_str(), rec.confidence, kMinRecConfidence);
            ocr.rec_confidence = rec.confidence;
        }
        results.push_back(std::move(ocr));
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    timings.recognition_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();
    timings.total_ms = std::chrono::duration<float, std::milli>(t3 - t0).count();

    return results;
}

}  // namespace ppocr
