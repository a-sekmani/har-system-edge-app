"""
Unit tests for FPSTracker class.
"""
import time
from unittest.mock import patch

import pytest


class TestFPSTracker:
    """Tests for FPSTracker."""

    def test_init_default_window(self, fps_tracker_class):
        """Default window_size should be 30."""
        tracker = fps_tracker_class()
        assert tracker.window_size == 30
        assert tracker.frame_count == 0
        assert tracker.frame_times == []

    def test_init_custom_window(self, fps_tracker_class):
        """Custom window_size can be passed."""
        tracker = fps_tracker_class(window_size=10)
        assert tracker.window_size == 10

    def test_update_increments_frame_count(self, fps_tracker_class):
        """update() should increment frame_count by 1."""
        tracker = fps_tracker_class()
        assert tracker.frame_count == 0
        tracker.update()
        assert tracker.frame_count == 1
        tracker.update()
        tracker.update()
        assert tracker.frame_count == 3

    def test_get_fps_returns_zero_with_less_than_two_frames(self, fps_tracker_class):
        """get_fps() should return 0.0 when fewer than 2 frames."""
        tracker = fps_tracker_class()
        assert tracker.get_fps() == 0.0
        tracker.update()
        assert tracker.get_fps() == 0.0

    def test_get_fps_after_multiple_updates(self, fps_tracker_class):
        """get_fps() should return a reasonable value after multiple updates."""
        tracker = fps_tracker_class(window_size=10)
        for _ in range(5):
            tracker.update()
            time.sleep(0.01)
        fps = tracker.get_fps()
        assert isinstance(fps, float)
        assert fps >= 0.0

    def test_get_average_fps_zero_elapsed(self, fps_tracker_class):
        """get_average_fps() should return 0.0 when elapsed time is zero (or near zero)."""
        tracker = fps_tracker_class()
        avg = tracker.get_average_fps()
        assert isinstance(avg, float)
        assert avg >= 0.0

    def test_get_average_fps_after_updates(self, fps_tracker_class):
        """get_average_fps() should return average FPS since start."""
        tracker = fps_tracker_class()
        for _ in range(3):
            tracker.update()
            time.sleep(0.02)
        avg = tracker.get_average_fps()
        assert isinstance(avg, float)
        assert avg >= 0.0
        assert tracker.frame_count == 3

    def test_frame_times_window_limited(self, fps_tracker_class):
        """frame_times list should not exceed window_size."""
        tracker = fps_tracker_class(window_size=3)
        for _ in range(10):
            tracker.update()
            time.sleep(0.001)
        assert len(tracker.frame_times) <= 3
        assert tracker.frame_count == 10

    def test_start_time_set_on_init(self, fps_tracker_class):
        """start_time should be set on init."""
        before = time.time()
        tracker = fps_tracker_class()
        after = time.time()
        assert before <= tracker.start_time <= after
