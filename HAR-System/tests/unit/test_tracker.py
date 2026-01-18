"""
Unit tests for `har_system.core.tracker.TemporalActivityTracker`.

We feed synthetic keypoints/bboxes to exercise the activity classification and fall detection
logic without requiring video input or external dependencies.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from har_system.core.tracker import TemporalActivityTracker


def _frame_data(timestamp: float, bbox: dict, keypoints: dict, confidence: float = 0.95) -> dict:
    return {
        "timestamp": timestamp,
        "bbox": bbox,
        "keypoints": keypoints,
        "confidence": confidence,
    }


def _bbox_from_center(cx: float, cy: float, height: float = 250.0, width: float = 80.0) -> dict:
    # tracker expects bbox values in the same coordinate system as keypoints (arbitrary pixels OK)
    return {
        "xmin": cx - width / 2,
        "ymin": cy - height,
        "xmax": cx + width / 2,
        "ymax": cy,
    }


def _keypoints_basic(cx: float, cy: float, standing: bool = True) -> dict:
    """Generate a minimal COCO-17-like keypoint dict required by tracker heuristics."""
    # Use only keys used by _calculate_pose_height_normalized and _calculate_hip_ratio.
    if standing:
        # Standing: hip further from ankle => hip_ratio around 0.5
        hip_offset = 150
        height = 250
    else:
        # Sitting: hip closer to ankle => hip_ratio higher (>= 0.62 default)
        hip_offset = 50
        height = 180

    return {
        "nose": (cx, cy - height + 20, 0.9),
        "left_hip": (cx - 20, cy - height + hip_offset, 0.9),
        "right_hip": (cx + 20, cy - height + hip_offset, 0.9),
        "left_ankle": (cx - 30, cy - 10, 0.9),
        "right_ankle": (cx + 30, cy - 10, 0.9),
    }


@pytest.mark.unit
def test_tracker_creates_new_track_and_updates_global_stats():
    tracker = TemporalActivityTracker(history_seconds=3.0, fps_estimate=15)
    t0 = time.time()
    bbox = _bbox_from_center(300, 400)
    kp = _keypoints_basic(300, 400, standing=True)

    activity = tracker.update(1, _frame_data(t0, bbox, kp))

    assert activity in {"unknown", "stationary", "moving", "sitting"}
    stats = tracker.get_global_stats()
    assert stats["total_tracks_seen"] == 1
    assert stats["active_tracks"] >= 1


@pytest.mark.unit
def test_tracker_stationary_classification_with_small_motion():
    tracker = TemporalActivityTracker(history_seconds=5.0, fps_estimate=15)
    t0 = 1000.0
    cx, cy = 300.0, 400.0

    # Feed enough frames to pass classification window (>=10 positions).
    for i in range(20):
        # Very small jitter to remain below stationary threshold.
        jitter_x = 0.2 * np.sin(i)
        jitter_y = 0.2 * np.cos(i)
        ts = t0 + i * (1.0 / 15.0)
        bbox = _bbox_from_center(cx + jitter_x, cy + jitter_y)
        kp = _keypoints_basic(cx + jitter_x, cy + jitter_y, standing=True)
        tracker.update(1, _frame_data(ts, bbox, kp))

    summary = tracker.get_summary(1)
    assert summary is not None
    assert summary["current_activity"] == "stationary"


@pytest.mark.unit
def test_tracker_moving_classification_with_clear_motion():
    tracker = TemporalActivityTracker(history_seconds=5.0, fps_estimate=15)
    t0 = 2000.0
    cy = 400.0

    for i in range(25):
        # Move 5 px per frame -> should be moving.
        cx = 100.0 + i * 5.0
        ts = t0 + i * (1.0 / 15.0)
        bbox = _bbox_from_center(cx, cy)
        kp = _keypoints_basic(cx, cy, standing=True)
        tracker.update(1, _frame_data(ts, bbox, kp))

    summary = tracker.get_summary(1)
    assert summary is not None
    assert summary["current_activity"] == "moving"


@pytest.mark.unit
def test_tracker_sitting_classification_with_high_hip_ratio():
    tracker = TemporalActivityTracker(history_seconds=5.0, fps_estimate=15)
    t0 = 3000.0
    cx, cy = 300.0, 400.0

    for i in range(25):
        ts = t0 + i * (1.0 / 15.0)
        # Sitting: keep position stable.
        bbox = _bbox_from_center(cx, cy, height=180.0)
        kp = _keypoints_basic(cx, cy, standing=False)
        tracker.update(1, _frame_data(ts, bbox, kp))

    summary = tracker.get_summary(1)
    assert summary is not None
    assert summary["current_activity"] == "sitting"


@pytest.mark.unit
def test_tracker_detects_fall_on_rapid_pose_height_drop():
    tracker = TemporalActivityTracker(history_seconds=5.0, fps_estimate=30)
    t0 = 4000.0
    cx, cy = 300.0, 400.0

    # Build a window where pose height ratio drops quickly.
    # Use a constant bbox but change nose y to simulate a sudden drop.
    bbox = _bbox_from_center(cx, cy, height=250.0)

    # High pose height for first frames (nose far from ankles).
    for i in range(10):
        ts = t0 + i * 0.02  # 50 FPS simulated (short dt)
        kp = _keypoints_basic(cx, cy, standing=True)
        tracker.update(1, _frame_data(ts, bbox, kp))

    # Sudden drop: move nose close to ankles.
    for i in range(10, 20):
        ts = t0 + i * 0.02
        kp = _keypoints_basic(cx, cy, standing=True)
        # Overwrite nose y near ankle y to reduce pose height drastically.
        ankle_y = (kp["left_ankle"][1] + kp["right_ankle"][1]) / 2
        kp["nose"] = (cx, ankle_y - 5, 0.9)
        tracker.update(1, _frame_data(ts, bbox, kp))

    summary = tracker.get_summary(1)
    assert summary is not None
    assert summary["stats"]["fall_detected"] is True

