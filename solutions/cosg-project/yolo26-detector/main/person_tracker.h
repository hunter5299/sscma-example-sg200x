#ifndef _PERSON_TRACKER_H_
#define _PERSON_TRACKER_H_

#include <vector>
#include <map>
#include <cmath>

#include "yolo26_detector.h"

namespace yolo26 {

// Dwell state enum
enum class DwellState {
    TRANSIENT,      // Moving - just passing by
    DWELLING,       // Stopped < 1.5s - might be browsing
    ENGAGED,        // Stopped 1.5-20s - actively engaged
    ASSISTANCE      // Stopped > 20s - may need assistance
};

// Get string name for dwell state
inline const char* getDwellStateName(DwellState state) {
    switch (state) {
        case DwellState::TRANSIENT:  return "transient";
        case DwellState::DWELLING:   return "dwelling";
        case DwellState::ENGAGED:    return "engaged";
        case DwellState::ASSISTANCE: return "assistance";
        default: return "unknown";
    }
}

// Tracked person structure
struct TrackedPerson {
    int track_id;               // Persistent track ID
    Detection detection;        // Current detection

    // Velocity (pixels per second, based on 640x640 frame)
    float velocity_x = 0.0f;
    float velocity_y = 0.0f;
    float speed_px_s = 0.0f;            // Speed in pixels/sec
    float speed_normalized = 0.0f;      // Speed as % of body height/sec

    // Dwell state
    DwellState dwell_state = DwellState::TRANSIENT;
    float dwell_duration_sec = 0.0f;    // How long in current dwell position

    // Internal tracking data
    float last_seen_time = 0.0f;        // Last update time (seconds)
    float dwell_start_time = 0.0f;      // When dwell started
    int frames_tracked = 0;             // Total frames tracked
    int stationary_frames = 0;          // Consecutive stationary frames
    int lost_frames = 0;                // Frames since last detection
};

// Tracker configuration
struct TrackerConfig {
    // Matching
    float iou_threshold = 0.3f;         // Min IoU for matching
    int max_lost_frames = 30;           // Frames before track deletion (~1s at 30fps)

    // Velocity estimation (EMA smoothing)
    float vel_alpha = 0.08f;            // Normal EMA alpha
    float vel_alpha_sudden = 0.6f;      // Alpha for sudden stops
    float velocity_zero_threshold = 3.0f;  // Below this = zero velocity (px/s)

    // Dwell detection thresholds
    float dwell_speed_threshold = 10.0f;    // Below this = stationary (px/s)
    float dwell_min_duration = 1.5f;        // TRANSIENT->ENGAGED threshold (sec)
    float dwell_assistance_threshold = 20.0f; // ENGAGED->ASSISTANCE threshold (sec)
    int dwell_min_frames = 5;               // Min frames to confirm stationary

    // Frame dimensions (for pixel calculations)
    int frame_width = 640;
    int frame_height = 640;
};

// Zone occupancy counts
struct StateCount {
    int total = 0;
    int browsing = 0;       // TRANSIENT + DWELLING
    int engaged = 0;        // ENGAGED
    int assistance = 0;     // ASSISTANCE
};

/**
 * Simple ByteTrack-like person tracker with dwell state detection
 *
 * Features:
 * - IoU-based track matching
 * - Velocity estimation with EMA smoothing
 * - Dwell state machine (TRANSIENT->DWELLING->ENGAGED->ASSISTANCE)
 * - Track lifecycle management (create/update/delete)
 */
class PersonTracker {
public:
    PersonTracker();
    ~PersonTracker() = default;

    // Set tracker configuration
    void setConfig(const TrackerConfig& config);

    // Main update - takes person detections, returns tracked persons
    // current_time_sec: Time since application start in seconds
    std::vector<TrackedPerson> update(
        const std::vector<Detection>& detections,
        float current_time_sec
    );

    // Get current state counts
    StateCount getStateCounts() const;

    // Get active track count
    int getTrackCount() const { return static_cast<int>(tracks_.size()); }

private:
    // Compute IoU between two detections
    float computeIoU(const Detection& a, const Detection& b) const;

    // Match detections to existing tracks using IoU
    // Returns vector of (track_id, detection_index) pairs
    std::vector<std::pair<int, int>> matchDetections(
        const std::vector<Detection>& detections
    ) const;

    // Update velocity using EMA
    void updateVelocity(TrackedPerson& track,
                        const Detection& new_det,
                        float dt);

    // Update dwell state based on velocity and duration
    void updateDwellState(TrackedPerson& track, float current_time);

private:
    TrackerConfig config_;
    std::map<int, TrackedPerson> tracks_;
    int next_track_id_ = 0;
    float last_update_time_ = 0.0f;
};

}  // namespace yolo26

#endif  // _PERSON_TRACKER_H_
