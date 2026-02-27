#include "person_tracker.h"

#include <algorithm>
#include <cmath>

#define TAG "PersonTracker"

#include <sscma.h>

namespace yolo26 {

PersonTracker::PersonTracker() {
    // Default config is set in TrackerConfig struct
}

void PersonTracker::setConfig(const TrackerConfig& config) {
    config_ = config;
}

float PersonTracker::computeIoU(const Detection& a, const Detection& b) const {
    // Convert center format to corner format
    float ax1 = a.x - a.w / 2.0f;
    float ay1 = a.y - a.h / 2.0f;
    float ax2 = a.x + a.w / 2.0f;
    float ay2 = a.y + a.h / 2.0f;

    float bx1 = b.x - b.w / 2.0f;
    float by1 = b.y - b.h / 2.0f;
    float bx2 = b.x + b.w / 2.0f;
    float by2 = b.y + b.h / 2.0f;

    // Compute intersection
    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    // Compute union
    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;

    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

std::vector<std::pair<int, int>> PersonTracker::matchDetections(
    const std::vector<Detection>& detections
) const {
    std::vector<std::pair<int, int>> matches;

    if (tracks_.empty() || detections.empty()) {
        return matches;
    }

    // Build cost matrix (negative IoU for minimization)
    std::vector<int> track_ids;
    for (const auto& pair : tracks_) {
        track_ids.push_back(pair.first);
    }

    // Simple greedy matching: for each track, find best detection
    std::vector<bool> det_used(detections.size(), false);

    // Sort tracks by number of frames tracked (prefer older tracks)
    std::sort(track_ids.begin(), track_ids.end(),
        [this](int a, int b) {
            return tracks_.at(a).frames_tracked > tracks_.at(b).frames_tracked;
        });

    for (int track_id : track_ids) {
        const auto& track = tracks_.at(track_id);
        float best_iou = config_.iou_threshold;
        int best_det_idx = -1;

        for (size_t d = 0; d < detections.size(); d++) {
            if (det_used[d]) continue;

            float iou = computeIoU(track.detection, detections[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_det_idx = static_cast<int>(d);
            }
        }

        if (best_det_idx >= 0) {
            matches.push_back({track_id, best_det_idx});
            det_used[best_det_idx] = true;
        }
    }

    return matches;
}

void PersonTracker::updateVelocity(TrackedPerson& track,
                                    const Detection& new_det,
                                    float dt) {
    if (dt <= 0.001f) {
        return;  // Skip if time delta too small
    }

    // Calculate instantaneous velocity in pixels
    float dx = (new_det.x - track.detection.x) * config_.frame_width;
    float dy = (new_det.y - track.detection.y) * config_.frame_height;
    float ivx = dx / dt;
    float ivy = dy / dt;
    float instant_speed = std::hypot(ivx, ivy);

    // Detect sudden stop (speed drops by >50%)
    float prev_speed = track.speed_px_s;
    bool sudden_stop = (prev_speed > 10.0f) &&
                       ((prev_speed - instant_speed) / prev_speed > 0.5f);

    // Force velocity to zero if very slow
    if (instant_speed < config_.velocity_zero_threshold) {
        track.velocity_x = 0.0f;
        track.velocity_y = 0.0f;
        track.speed_px_s = 0.0f;
    } else {
        // Apply EMA smoothing with adaptive alpha
        float alpha = sudden_stop ? config_.vel_alpha_sudden : config_.vel_alpha;
        track.velocity_x = (1.0f - alpha) * track.velocity_x + alpha * ivx;
        track.velocity_y = (1.0f - alpha) * track.velocity_y + alpha * ivy;
        track.speed_px_s = std::hypot(track.velocity_x, track.velocity_y);
    }

    // Calculate normalized speed (% of body height per second)
    float bbox_h_px = new_det.h * config_.frame_height;
    if (bbox_h_px > 1.0f) {
        track.speed_normalized = (track.speed_px_s / bbox_h_px) * 100.0f;
    } else {
        track.speed_normalized = 0.0f;
    }
}

void PersonTracker::updateDwellState(TrackedPerson& track, float current_time) {
    bool is_stationary = track.speed_px_s < config_.dwell_speed_threshold;

    if (!is_stationary) {
        // Moving - reset to TRANSIENT
        track.dwell_state = DwellState::TRANSIENT;
        track.stationary_frames = 0;
        track.dwell_start_time = 0.0f;
        track.dwell_duration_sec = 0.0f;
        return;
    }

    // Stationary - increment frame counter
    track.stationary_frames++;

    // Need minimum frames for stability before starting dwell timer
    if (track.stationary_frames < config_.dwell_min_frames) {
        return;  // Stay in current state, don't start timer yet
    }

    // Start dwell timer if not already started
    if (track.dwell_start_time <= 0.0f) {
        track.dwell_start_time = current_time;
    }

    // Update dwell duration
    track.dwell_duration_sec = current_time - track.dwell_start_time;

    // State transitions based on duration
    if (track.dwell_duration_sec >= config_.dwell_assistance_threshold) {
        track.dwell_state = DwellState::ASSISTANCE;
    } else if (track.dwell_duration_sec >= config_.dwell_min_duration) {
        track.dwell_state = DwellState::ENGAGED;
    } else {
        track.dwell_state = DwellState::DWELLING;
    }
}

std::vector<TrackedPerson> PersonTracker::update(
    const std::vector<Detection>& detections,
    float current_time_sec
) {
    // Calculate time delta
    float dt = (last_update_time_ > 0) ? (current_time_sec - last_update_time_) : 0.0f;
    last_update_time_ = current_time_sec;

    // Filter only person detections (class_id == 0 in COCO)
    std::vector<Detection> person_detections;
    for (const auto& det : detections) {
        if (det.class_id == 0) {  // person class
            person_detections.push_back(det);
        }
    }

    // Match detections to existing tracks
    auto matches = matchDetections(person_detections);

    // Track which detections were matched
    std::vector<bool> det_matched(person_detections.size(), false);
    std::vector<int> matched_track_ids;

    // Update matched tracks
    for (const auto& match : matches) {
        int track_id = match.first;
        int det_idx = match.second;
        det_matched[det_idx] = true;
        matched_track_ids.push_back(track_id);

        TrackedPerson& track = tracks_[track_id];

        // Update velocity before updating detection
        updateVelocity(track, person_detections[det_idx], dt);

        // Update detection
        track.detection = person_detections[det_idx];
        track.last_seen_time = current_time_sec;
        track.frames_tracked++;
        track.lost_frames = 0;

        // Update dwell state
        updateDwellState(track, current_time_sec);
    }

    // Increment lost_frames for unmatched tracks
    std::vector<int> tracks_to_remove;
    for (auto& pair : tracks_) {
        int track_id = pair.first;
        if (std::find(matched_track_ids.begin(), matched_track_ids.end(), track_id)
            == matched_track_ids.end()) {
            pair.second.lost_frames++;

            // Remove track if lost too long
            if (pair.second.lost_frames > config_.max_lost_frames) {
                tracks_to_remove.push_back(track_id);
            }
        }
    }

    // Remove lost tracks
    for (int track_id : tracks_to_remove) {
        MA_LOGV(TAG, "Removing lost track %d", track_id);
        tracks_.erase(track_id);
    }

    // Create new tracks for unmatched detections
    for (size_t i = 0; i < person_detections.size(); i++) {
        if (!det_matched[i]) {
            TrackedPerson new_track;
            new_track.track_id = next_track_id_++;
            new_track.detection = person_detections[i];
            new_track.last_seen_time = current_time_sec;
            new_track.frames_tracked = 1;
            new_track.lost_frames = 0;
            new_track.dwell_state = DwellState::TRANSIENT;

            tracks_[new_track.track_id] = new_track;
            MA_LOGV(TAG, "Created new track %d", new_track.track_id);
        }
    }

    // Build result vector (only active tracks, not lost)
    std::vector<TrackedPerson> result;
    for (const auto& pair : tracks_) {
        if (pair.second.lost_frames == 0) {
            result.push_back(pair.second);
        }
    }

    return result;
}

StateCount PersonTracker::getStateCounts() const {
    StateCount counts;

    for (const auto& pair : tracks_) {
        const auto& track = pair.second;
        if (track.lost_frames > 0) continue;  // Skip lost tracks

        counts.total++;
        switch (track.dwell_state) {
            case DwellState::TRANSIENT:
            case DwellState::DWELLING:
                counts.browsing++;
                break;
            case DwellState::ENGAGED:
                counts.engaged++;
                break;
            case DwellState::ASSISTANCE:
                counts.assistance++;
                break;
        }
    }

    return counts;
}

}  // namespace yolo26
