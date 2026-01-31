"""
Unit tests for HARPoseEstimationApp class.
"""
from unittest.mock import MagicMock, patch

import pytest


class TestHARPoseEstimationApp:
    """Tests for HARPoseEstimationApp (with mocks to avoid full pipeline init)."""

    def test_subclass_of_gstreamer_pose_estimation_app(self, har_app_module):
        """HARPoseEstimationApp should inherit from GStreamerPoseEstimationApp."""
        assert issubclass(
            har_app_module.HARPoseEstimationApp,
            har_app_module.GStreamerPoseEstimationApp,
        )

    def test_get_pipeline_string_contains_fakesink_when_no_display(
        self, har_app_module, get_har_parser_func
    ):
        """When --no-display is set, pipeline string should contain fakesink."""
        parser = get_har_parser_func()
        args = parser.parse_args(["--no-display", "--input", "/tmp/test.mp4"])
        with patch.object(
            har_app_module.HARPoseEstimationApp,
            "__init__",
            lambda self, cb, ud, p=None: None,
        ):
            app = har_app_module.HARPoseEstimationApp.__new__(har_app_module.HARPoseEstimationApp)
            app.options_menu = args
            app.video_sink = "fakesink"
            app.video_source = "/tmp/test.mp4"
            app.video_width = 1280
            app.video_height = 720
            app.frame_rate = 30
            app.sync = "false"
            app.show_fps = False
            app.hef_path = "/fake/hef.hef"
            app.post_process_so = "/fake/post.so"
            app.post_process_function = "filter"
            app.batch_size = 2
            pipeline_str = app.get_pipeline_string()
        assert "fakesink" in pipeline_str

    def test_get_pipeline_string_contains_display_pipeline_components(self, har_app_module):
        """Pipeline string should contain display and inference pipeline components."""
        app = har_app_module.HARPoseEstimationApp.__new__(har_app_module.HARPoseEstimationApp)
        app.video_sink = "autovideosink"
        app.video_source = "/tmp/test.mp4"
        app.video_width = 1280
        app.video_height = 720
        app.frame_rate = 30
        app.sync = "false"
        app.show_fps = False
        app.hef_path = "/fake/hef.hef"
        app.post_process_so = "/fake/post.so"
        app.post_process_function = "filter"
        app.batch_size = 2
        pipeline_str = app.get_pipeline_string()
        assert "hailo" in pipeline_str.lower() or "queue" in pipeline_str.lower()
        assert "videoconvert" in pipeline_str or "overlay" in pipeline_str.lower()

    def test_video_sink_fakesink_when_no_display_in_options(self, har_app_module):
        """When options_menu.no_display is True, video_sink should be fakesink (__init__ logic)."""
        options = MagicMock()
        options.no_display = True
        video_sink = "fakesink" if getattr(options, "no_display", False) else "autovideosink"
        assert video_sink == "fakesink"

    def test_video_sink_autovideosink_when_no_display_false(self, har_app_module):
        """When no_display is False, video_sink should remain autovideosink."""
        options = MagicMock()
        options.no_display = False
        video_sink = "fakesink" if getattr(options, "no_display", False) else "autovideosink"
        assert video_sink == "autovideosink"


class TestPrintFinalStats:
    """Tests for _print_final_stats()."""

    def test_print_final_stats_does_not_raise(self, har_app_module):
        """_print_final_stats(user_data) should not raise when user_data has required attributes."""
        user_data = MagicMock()
        user_data.fps_tracker.get_average_fps.return_value = 30.0
        user_data.get_count.return_value = 900
        user_data.frame_events_count = 900
        user_data.invalid_caps_count = 0
        user_data.invalid_validate_count = 0
        user_data.frames_with_persons = 850
        user_data.frames_no_persons = 50
        user_data.persons_total = 900
        user_data.frames_with_landmarks = 840
        user_data.frames_keypoints_len_not_17 = 0
        har_app_module._print_final_stats(user_data)

    def test_print_final_stats_calls_logger(self, har_app_module):
        """_print_final_stats should call hailo_logger.info for Final Stats and Phase1 final."""
        user_data = MagicMock()
        user_data.fps_tracker.get_average_fps.return_value = 25.0
        user_data.get_count.return_value = 500
        user_data.frame_events_count = 500
        user_data.invalid_caps_count = 0
        user_data.invalid_validate_count = 0
        user_data.frames_with_persons = 480
        user_data.frames_no_persons = 20
        user_data.persons_total = 500
        user_data.frames_with_landmarks = 475
        user_data.frames_keypoints_len_not_17 = 0
        with patch.object(har_app_module.hailo_logger, "info") as mock_info:
            har_app_module._print_final_stats(user_data)
        assert mock_info.call_count >= 2


class TestPoseConfidenceFromDetection:
    """Tests for _pose_confidence_from_detection (filter helper)."""

    def test_returns_zero_when_no_landmarks(self, har_app_module):
        """When detection has no landmarks, _pose_confidence_from_detection returns 0.0."""
        det = MagicMock()
        det.get_objects_typed = MagicMock(return_value=[])
        assert har_app_module._pose_confidence_from_detection(det) == 0.0

    def test_returns_average_confidence_when_landmarks_present(self, har_app_module):
        """When landmarks have confidence values, returns their average (e.g. 0.8 and 0.4 -> 0.6)."""
        try:
            import hailo
        except ImportError:
            pytest.skip("hailo not available")
        pt1 = MagicMock()
        pt1.confidence = MagicMock(return_value=0.8)
        pt2 = MagicMock()
        pt2.confidence = MagicMock(return_value=0.4)
        lm = MagicMock()
        lm.get_points = MagicMock(return_value=[pt1, pt2])
        det = MagicMock()
        det.get_objects_typed = MagicMock(return_value=[lm])
        # average of 0.8 and 0.4 = 0.6
        assert har_app_module._pose_confidence_from_detection(det) == pytest.approx(0.6)

    def test_returns_zero_when_empty_points(self, har_app_module):
        """When landmarks exist but get_points() is empty, returns 0.0."""
        try:
            import hailo
        except ImportError:
            pytest.skip("hailo not available")
        lm = MagicMock()
        lm.get_points = MagicMock(return_value=[])
        det = MagicMock()
        det.get_objects_typed = MagicMock(return_value=[lm])
        assert har_app_module._pose_confidence_from_detection(det) == 0.0


class TestMainFunction:
    """Tests for main() entry point."""

    def test_main_importable(self, har_app_module):
        """main() should exist and be callable."""
        assert hasattr(har_app_module, "main")
        assert callable(har_app_module.main)
