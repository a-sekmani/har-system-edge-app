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
