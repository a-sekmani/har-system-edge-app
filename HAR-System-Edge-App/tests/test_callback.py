"""
Unit tests for simple_callback function.
"""
from unittest.mock import MagicMock

import pytest


class TestSimpleCallback:
    """Tests for simple_callback."""

    def test_returns_none_when_buffer_is_none(self, simple_callback_func, har_user_data_class):
        """simple_callback should return without raising when buffer is None."""
        user_data = har_user_data_class()
        element = MagicMock()
        result = simple_callback_func(element, None, user_data)
        assert result is None
        assert user_data.fps_tracker.frame_count == 0

    def test_updates_fps_tracker_when_buffer_valid(self, simple_callback_func, har_user_data_class):
        """When buffer is valid, fps_tracker should be updated."""
        user_data = har_user_data_class()
        element = MagicMock()
        buffer = MagicMock()
        simple_callback_func(element, buffer, user_data)
        assert user_data.fps_tracker.frame_count == 1
        simple_callback_func(element, buffer, user_data)
        simple_callback_func(element, buffer, user_data)
        assert user_data.fps_tracker.frame_count == 3

    def test_does_not_raise_with_valid_args(self, simple_callback_func, har_user_data_class):
        """simple_callback should not raise with valid element, buffer, user_data."""
        user_data = har_user_data_class()
        element = MagicMock()
        buffer = MagicMock()
        result = simple_callback_func(element, buffer, user_data)
        assert result is None

    def test_user_data_must_have_fps_tracker(self, simple_callback_func):
        """user_data must have fps_tracker, last_fps_log_time, get_count."""
        user_data = MagicMock()
        user_data.fps_tracker = MagicMock()
        user_data.fps_tracker.frame_count = 0
        user_data.fps_tracker.update = MagicMock()
        user_data.fps_tracker.get_fps = MagicMock(return_value=10.0)
        user_data.fps_tracker.get_average_fps = MagicMock(return_value=10.0)
        user_data.last_fps_log_time = 0.0
        user_data.fps_log_interval = 5.0  # numeric so comparison in callback does not fail
        user_data.get_count = MagicMock(return_value=1)
        element = MagicMock()
        buffer = MagicMock()
        result = simple_callback_func(element, buffer, user_data)
        assert result is None
        user_data.fps_tracker.update.assert_called()
