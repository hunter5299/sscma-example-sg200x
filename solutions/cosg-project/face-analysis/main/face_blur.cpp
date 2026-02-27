#include "face_blur.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#include <sscma.h>

#define TAG "FaceBlur"

namespace face_analysis {

// ============ KalmanFilter1D ============

void KalmanFilter1D::init(float x0, float pos_var, float vel_var) {
    x   = x0;
    v   = 0.0f;
    p00 = pos_var;
    p01 = 0.0f;
    p11 = vel_var;
}

void KalmanFilter1D::predict(float dt, float q) {
    x += v * dt;

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt2 * dt2;

    p00 = p00 + 2.0f * p01 * dt + p11 * dt2 + q * dt4 * 0.25f;
    p01 = p01 + p11 * dt + q * dt3 * 0.5f;
    p11 = p11 + q * dt2;
}

void KalmanFilter1D::update(float z, float r) {
    float y  = z - x;
    float s  = p00 + r;
    float k0 = p00 / s;
    float k1 = p01 / s;

    x += k0 * y;
    v += k1 * y;

    float new_p00 = p00 - k0 * p00;
    float new_p01 = p01 - k0 * p01;
    float new_p11 = p11 - k1 * p01;

    p00 = new_p00;
    p01 = new_p01;
    p11 = new_p11;
}

// ============ TrackedRegion ============

void TrackedRegion::init(const FaceInfo& face) {
    kf[0].init(face.x);
    kf[1].init(face.y);
    kf[2].init(face.w, 0.01f, 0.1f);
    kf[3].init(face.h, 0.01f, 0.1f);
    score      = face.score;
    miss_count = 0;
}

void TrackedRegion::predict(float dt, float q) {
    for (int i = 0; i < 4; i++) {
        kf[i].predict(dt, q);
    }
}

void TrackedRegion::update(const FaceInfo& face, float r) {
    kf[0].update(face.x, r);
    kf[1].update(face.y, r);
    kf[2].update(face.w, r * 2.0f);
    kf[3].update(face.h, r * 2.0f);
    score      = face.score;
    miss_count = 0;
}

FaceInfo TrackedRegion::getBox() const {
    FaceInfo f;
    f.x     = kf[0].x;
    f.y     = kf[1].x;
    f.w     = std::max(0.01f, kf[2].x);
    f.h     = std::max(0.01f, kf[3].x);
    f.score = score;
    f.id    = 0;
    return f;
}

// ============ FaceBlur ============

FaceBlur::FaceBlur()
    : max_regions_(kDefaultMaxRegions),
      vpss_grp_(0),
      vpss_chn_(2),
      regions_inited_(false),
      stream_width_(0),
      stream_height_(0),
      predicting_(false),
      process_noise_(5.0f),
      measurement_noise_(0.001f),
      max_miss_(15),
      predict_interval_ms_(33),
      iou_threshold_(0.2f),
      initialized_(false) {}

FaceBlur::~FaceBlur() {
    deinit();
}

void FaceBlur::setMaxRegions(int max_regions) {
    if (max_regions < 1) max_regions = 1;
    if (max_regions > kMaxRegionsLimit) max_regions = kMaxRegionsLimit;
    max_regions_ = max_regions;
}


bool FaceBlur::init(int stream_width, int stream_height, int vpss_grp, int vpss_chn) {
    if (initialized_) return true;

    stream_width_  = stream_width;
    stream_height_ = stream_height;
    vpss_grp_      = vpss_grp;
    vpss_chn_      = vpss_chn;

    if (stream_width_ <= 0 || stream_height_ <= 0) {
        MA_LOGE(TAG, "Invalid stream resolution: %dx%d", stream_width_, stream_height_);
        return false;
    }

    MA_LOGI(TAG, "Initializing face blur: stream %dx%d, vpss(%d,%d), max_regions=%d",
            stream_width_, stream_height_, vpss_grp_, vpss_chn_, max_regions_);

    initRegions();

    if (!regions_inited_) {
        MA_LOGW(TAG, "No RGN regions available, face blur disabled");
        return false;
    }

    predicting_.store(true);
    predict_thread_ = std::thread(&FaceBlur::predictThreadEntry, this);

    initialized_ = true;
    MA_LOGI(TAG, "Face blur initialized with %d regions", (int)handles_.size());
    return true;
}

void FaceBlur::deinit() {
    if (!initialized_) return;

    predicting_.store(false);
    if (predict_thread_.joinable()) {
        predict_thread_.join();
    }

    {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        trackers_.clear();
    }

    deinitRegions();

    initialized_ = false;
    MA_LOGI(TAG, "Face blur deinitialized");
}

// ============ IoU computation ============

float FaceBlur::computeIoU(const FaceInfo& a, const FaceInfo& b) {
    float a_l = a.x - a.w * 0.5f, a_r = a.x + a.w * 0.5f;
    float a_t = a.y - a.h * 0.5f, a_b = a.y + a.h * 0.5f;
    float b_l = b.x - b.w * 0.5f, b_r = b.x + b.w * 0.5f;
    float b_t = b.y - b.h * 0.5f, b_b = b.y + b.h * 0.5f;

    float inter_w = std::max(0.0f, std::min(a_r, b_r) - std::max(a_l, b_l));
    float inter_h = std::max(0.0f, std::min(a_b, b_b) - std::max(a_t, b_t));
    float inter   = inter_w * inter_h;

    float uni = a.w * a.h + b.w * b.h - inter;
    return (uni > 1e-6f) ? inter / uni : 0.0f;
}

// ============ Data association + Kalman update ============

void FaceBlur::associateAndUpdate(const std::vector<FaceInfo>& faces) {
    std::vector<bool> det_matched(faces.size(), false);

    for (auto& tracker : trackers_) {
        FaceInfo predicted = tracker.getBox();
        float best_iou = iou_threshold_;
        int best_idx   = -1;

        for (int d = 0; d < (int)faces.size(); d++) {
            if (det_matched[d]) continue;
            float iou = computeIoU(predicted, faces[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = d;
            }
        }

        if (best_idx >= 0) {
            tracker.update(faces[best_idx], measurement_noise_);
            det_matched[best_idx] = true;
        } else {
            tracker.miss_count++;
        }
    }

    // Remove dead trackers
    trackers_.erase(
        std::remove_if(trackers_.begin(), trackers_.end(),
            [this](const TrackedRegion& t) { return t.miss_count > max_miss_; }),
        trackers_.end());

    // Create new trackers for unmatched detections
    for (int d = 0; d < (int)faces.size(); d++) {
        if (det_matched[d]) continue;

        if ((int)trackers_.size() < max_regions_) {
            TrackedRegion tr;
            tr.init(faces[d]);
            trackers_.push_back(tr);
        } else {
            int worst_idx = -1;
            int worst_miss = 0;
            for (int t = 0; t < (int)trackers_.size(); t++) {
                if (trackers_[t].miss_count > worst_miss) {
                    worst_miss = trackers_[t].miss_count;
                    worst_idx  = t;
                }
            }
            if (worst_idx >= 0 && worst_miss > 0) {
                trackers_[worst_idx].init(faces[d]);
            }
        }
    }
}

// ============ Prediction thread ============

void FaceBlur::predictThreadEntry() {
    MA_LOGI(TAG, "Prediction thread started (%d ms interval)", predict_interval_ms_);

    float dt = (float)predict_interval_ms_ / 1000.0f;

    while (predicting_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(predict_interval_ms_));

        if (!predicting_.load()) break;

        if (!regions_inited_) continue;

        std::vector<FaceInfo> predicted_boxes;
        {
            std::lock_guard<std::mutex> lock(tracker_mutex_);

            for (auto& tracker : trackers_) {
                tracker.predict(dt, process_noise_);
            }

            for (const auto& tracker : trackers_) {
                if (tracker.miss_count <= max_miss_) {
                    predicted_boxes.push_back(tracker.getBox());
                }
            }

            std::sort(predicted_boxes.begin(), predicted_boxes.end(),
                [](const FaceInfo& a, const FaceInfo& b) {
                    return a.score > b.score;
                });
        }

        applyRegions(predicted_boxes);
    }

    MA_LOGI(TAG, "Prediction thread stopped");
}

// ============ Detection callback ============

void FaceBlur::onDetection(const std::vector<FaceInfo>& faces) {
    if (!initialized_ || !regions_inited_) return;

    std::lock_guard<std::mutex> lock(tracker_mutex_);
    associateAndUpdate(faces);
}

// ============ RGN hardware overlay management ============

void FaceBlur::initRegions() {
    if (regions_inited_) return;

    handles_.clear();

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (int i = 0; i < max_regions_; i++) {
        RGN_HANDLE hRgn = kRgnHandleBase + i;

        RGN_ATTR_S stRgnAttr;
        memset(&stRgnAttr, 0, sizeof(stRgnAttr));
        stRgnAttr.enType = MOSAIC_RGN;

        CVI_S32 ret = CVI_RGN_Create(hRgn, &stRgnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_Create(%d) failed: 0x%x", hRgn, ret);
            continue;
        }

        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.bShow  = CVI_FALSE;
        stChnAttr.enType = MOSAIC_RGN;
        stChnAttr.unChnAttr.stMosaicChn.stRect.s32X      = 0;
        stChnAttr.unChnAttr.stMosaicChn.stRect.s32Y      = 0;
        stChnAttr.unChnAttr.stMosaicChn.stRect.u32Width   = 64;
        stChnAttr.unChnAttr.stMosaicChn.stRect.u32Height  = 64;
        stChnAttr.unChnAttr.stMosaicChn.enBlkSize         = MOSAIC_BLK_SIZE_16;
        stChnAttr.unChnAttr.stMosaicChn.u32Layer          = i;

        ret = CVI_RGN_AttachToChn(hRgn, &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGE(TAG, "CVI_RGN_AttachToChn(%d) failed: 0x%x", hRgn, ret);
            CVI_RGN_Destroy(hRgn);
            continue;
        }

        handles_.push_back(hRgn);
    }

    if (handles_.empty()) {
        MA_LOGE(TAG, "Failed to create any blur regions, disabling face blur");
        return;
    }

    regions_inited_ = true;
    MA_LOGI(TAG, "Initialized %d/%d mosaic regions on VPSS(%d,%d)",
            (int)handles_.size(), max_regions_, vpss_grp_, vpss_chn_);
}

void FaceBlur::deinitRegions() {
    if (!regions_inited_ && handles_.empty()) return;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    for (auto hRgn : handles_) {
        CVI_RGN_DetachFromChn(hRgn, &stChn);
        CVI_RGN_Destroy(hRgn);
    }

    handles_.clear();
    regions_inited_ = false;
    MA_LOGI(TAG, "Deinitialized blur regions");
}

void FaceBlur::applyRegions(const std::vector<FaceInfo>& boxes) {
    if (!regions_inited_ || stream_width_ <= 0 || stream_height_ <= 0) return;

    // Coordinate mapping: normalized [0,1] -> H264 stream pixel coords
    // Handles letterbox padding for non-square aspect ratios
    float scale_h    = 1.0f;
    float scale_w    = 1.0f;
    int32_t offset_x = 0;
    int32_t offset_y = 0;

    if (stream_width_ > stream_height_) {
        scale_h  = (float)stream_width_ / (float)stream_height_;
        offset_y = (stream_height_ - stream_width_) / 2;
    } else {
        scale_w  = (float)stream_height_ / (float)stream_width_;
        offset_x = (stream_width_ - stream_height_) / 2;
    }

    int target_w = stream_width_ * scale_w;
    int target_h = stream_height_ * scale_h;

    MMF_CHN_S stChn;
    stChn.enModId  = CVI_ID_VPSS;
    stChn.s32DevId = vpss_grp_;
    stChn.s32ChnId = vpss_chn_;

    int num_handles = (int)handles_.size();
    int active_count = std::min((int)boxes.size(), num_handles);

    for (int i = 0; i < num_handles; i++) {
        RGN_CHN_ATTR_S stChnAttr;
        memset(&stChnAttr, 0, sizeof(stChnAttr));
        stChnAttr.enType = MOSAIC_RGN;

        if (i < active_count) {
            const auto& box = boxes[i];

            // Skip near-full-frame detections (noise frames during ISP init can crash VPSS)
            if (box.w > 0.7f && box.h > 0.7f) {
                stChnAttr.bShow = CVI_FALSE;
                stChnAttr.unChnAttr.stMosaicChn.stRect.s32X      = 0;
                stChnAttr.unChnAttr.stMosaicChn.stRect.s32Y      = 0;
                stChnAttr.unChnAttr.stMosaicChn.stRect.u32Width   = 64;
                stChnAttr.unChnAttr.stMosaicChn.stRect.u32Height  = 64;
                stChnAttr.unChnAttr.stMosaicChn.enBlkSize         = MOSAIC_BLK_SIZE_16;
                stChnAttr.unChnAttr.stMosaicChn.u32Layer          = i;
                CVI_RGN_SetDisplayAttr(handles_[i], &stChn, &stChnAttr);
                continue;
            }

            int left = (int)((box.x - box.w / 2.0f) * target_w + offset_x);
            int top  = (int)((box.y - box.h / 2.0f) * target_h + offset_y);
            int w    = (int)(box.w * target_w);
            int h    = (int)(box.h * target_h);

            // Clamp to frame bounds
            left = std::max(0, left);
            top  = std::max(0, top);
            w    = std::min(w, stream_width_ - left);
            h    = std::min(h, stream_height_ - top);

            // Align to 8 pixels (MOSAIC hardware requirement)
            left = left & ~7;
            top  = top & ~7;
            w    = std::max(8, (w + 7) & ~7);
            h    = std::max(8, (h + 7) & ~7);

            // Re-check bounds after alignment
            if (left + w > stream_width_) w = (stream_width_ - left) & ~7;
            if (top + h > stream_height_) h = (stream_height_ - top) & ~7;
            w = std::max(8, w);
            h = std::max(8, h);

            stChnAttr.bShow = CVI_TRUE;
            stChnAttr.unChnAttr.stMosaicChn.stRect.s32X      = left;
            stChnAttr.unChnAttr.stMosaicChn.stRect.s32Y      = top;
            stChnAttr.unChnAttr.stMosaicChn.stRect.u32Width   = w;
            stChnAttr.unChnAttr.stMosaicChn.stRect.u32Height  = h;
            stChnAttr.unChnAttr.stMosaicChn.enBlkSize         = MOSAIC_BLK_SIZE_16;
            stChnAttr.unChnAttr.stMosaicChn.u32Layer          = i;
        } else {
            stChnAttr.bShow = CVI_FALSE;
            stChnAttr.unChnAttr.stMosaicChn.stRect.s32X      = 0;
            stChnAttr.unChnAttr.stMosaicChn.stRect.s32Y      = 0;
            stChnAttr.unChnAttr.stMosaicChn.stRect.u32Width   = 64;
            stChnAttr.unChnAttr.stMosaicChn.stRect.u32Height  = 64;
            stChnAttr.unChnAttr.stMosaicChn.enBlkSize         = MOSAIC_BLK_SIZE_16;
            stChnAttr.unChnAttr.stMosaicChn.u32Layer          = i;
        }

        CVI_S32 ret = CVI_RGN_SetDisplayAttr(handles_[i], &stChn, &stChnAttr);
        if (ret != CVI_SUCCESS) {
            MA_LOGW(TAG, "CVI_RGN_SetDisplayAttr(%d) failed: 0x%x", handles_[i], ret);
        }
    }
}

}  // namespace face_analysis
