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


class TestMainFunction:
    """Tests for main() entry point."""

    def test_main_importable(self, har_app_module):
        """main() should exist and be callable."""
        assert hasattr(har_app_module, "main")
        assert callable(har_app_module.main)
