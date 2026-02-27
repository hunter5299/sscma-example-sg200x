#ifndef _FACE_BLUR_H_
#define _FACE_BLUR_H_

#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

#include <cvi_region.h>

#include "face_detector.h"

namespace face_analysis {

// 1D Kalman filter with constant-velocity model
// State: [position, velocity], Observation: [position]
struct KalmanFilter1D {
    float x;    // position estimate
    float v;    // velocity estimate
    float p00;  // var(position)
    float p01;  // cov(position, velocity)
    float p11;  // var(velocity)

    void init(float x0, float pos_var = 0.01f, float vel_var = 1.0f);
    void predict(float dt, float q);
    void update(float z, float r);
};

// Tracked bounding box with 4 independent Kalman filters (x, y, w, h)
struct TrackedRegion {
    KalmanFilter1D kf[4];  // center_x, center_y, width, height
    float score;
    int miss_count;

    void init(const FaceInfo& face);
    void predict(float dt, float q);
    void update(const FaceInfo& face, float r);
    FaceInfo getBox() const;
};

class FaceBlur {
public:
    FaceBlur();
    ~FaceBlur();

    // Initialize with stream resolution and VPSS channel info
    bool init(int stream_width, int stream_height, int vpss_grp = 0, int vpss_chn = 2);

    // Deinitialize and release RGN resources
    void deinit();

    // Set max number of blur regions (1-16)
    void setMaxRegions(int max_regions);



    // Feed detection results to update tracking
    void onDetection(const std::vector<FaceInfo>& faces);

private:
    void initRegions();
    void deinitRegions();
    void applyRegions(const std::vector<FaceInfo>& boxes);
    void associateAndUpdate(const std::vector<FaceInfo>& faces);
    float computeIoU(const FaceInfo& a, const FaceInfo& b);
    void predictThreadEntry();

private:
    static constexpr int kRgnHandleBase = 100;
    static constexpr int kDefaultMaxRegions = 8;
    static constexpr int kMaxRegionsLimit = 8;  // RGN_MOSAIC_MAX_NUM per channel on CV181x

    // RGN hardware overlay config
    int max_regions_;
    int vpss_grp_;
    int vpss_chn_;
    std::vector<RGN_HANDLE> handles_;
    bool regions_inited_;

    // Stream resolution
    int stream_width_;
    int stream_height_;

    // Kalman prediction tracking
    std::vector<TrackedRegion> trackers_;
    std::mutex tracker_mutex_;
    std::thread predict_thread_;
    std::atomic<bool> predicting_;
    float process_noise_;
    float measurement_noise_;
    int max_miss_;
    int predict_interval_ms_;
    float iou_threshold_;

    bool initialized_;
};

}  // namespace face_analysis

#endif  // _FACE_BLUR_H_
