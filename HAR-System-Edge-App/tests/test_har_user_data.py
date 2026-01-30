"""
Unit tests for HARUserData class.
"""
import pytest


class TestHARUserData:
    """Tests for HARUserData."""

    def test_inherits_from_app_callback_class(self, har_user_data_class, har_app_module):
        """HARUserData should inherit from app_callback_class."""
        assert issubclass(har_user_data_class, har_app_module.app_callback_class)

    def test_has_fps_tracker(self, har_user_data_class):
        """HARUserData should have fps_tracker of type FPSTracker."""
        user_data = har_user_data_class()
        assert hasattr(user_data, "fps_tracker")
        assert user_data.fps_tracker is not None
        assert hasattr(user_data.fps_tracker, "update")
        assert hasattr(user_data.fps_tracker, "get_fps")
        assert hasattr(user_data.fps_tracker, "get_average_fps")

    def test_has_last_fps_log_time(self, har_user_data_class):
        """HARUserData should have last_fps_log_time."""
        user_data = har_user_data_class()
        assert hasattr(user_data, "last_fps_log_time")
        assert user_data.last_fps_log_time is not None

    def test_has_fps_log_interval(self, har_user_data_class):
        """HARUserData should have fps_log_interval (default 5.0)."""
        user_data = har_user_data_class()
        assert hasattr(user_data, "fps_log_interval")
        assert user_data.fps_log_interval == 5.0

    def test_has_get_count_from_parent(self, har_user_data_class):
        """HARUserData should support get_count() from parent."""
        user_data = har_user_data_class()
        assert hasattr(user_data, "get_count")
        assert callable(user_data.get_count)
        assert user_data.get_count() == 0

    def test_fps_tracker_update_works(self, har_user_data_class):
        """Updating fps_tracker through user_data should work."""
        user_data = har_user_data_class()
        user_data.fps_tracker.update()
        user_data.fps_tracker.update()
        assert user_data.fps_tracker.frame_count == 2

    def test_has_phase1_counters_default_zero(self, har_user_data_class):
        """HARUserData should have Phase 1 counters defaulting to 0."""
        user_data = har_user_data_class()
        assert getattr(user_data, "invalid_caps_count", None) == 0
        assert getattr(user_data, "invalid_validate_count", None) == 0
        assert getattr(user_data, "frames_with_persons", None) == 0
        assert getattr(user_data, "frames_no_persons", None) == 0
        assert getattr(user_data, "persons_total", None) == 0
        assert getattr(user_data, "frames_with_landmarks", None) == 0
        assert getattr(user_data, "frames_keypoints_len_not_17", None) == 0
        assert getattr(user_data, "frame_events_count", None) == 0

    def test_optional_constructor_args(self, har_user_data_class):
        """HARUserData accepts optional log_pose_summary and dump_frames_path."""
        user_data = har_user_data_class(log_pose_summary=True, dump_frames_path="/tmp/out.json")
        assert user_data.log_pose_summary is True
        assert user_data.dump_frames_path == "/tmp/out.json"
        user_data2 = har_user_data_class()
        assert getattr(user_data2, "log_pose_summary", False) is False
        assert getattr(user_data2, "dump_frames_path", None) is None
