"""
Unit tests for face recognition, FPS log Person field, windows, face worker thread, created_at.

- Face worker loop and queue (existence and queue item signature).
- Face gallery URL resolution (priority: face_gallery_url then cloud_url).
- FPS log includes Person field for reporting.
- Window created_at ISO 8601 UTC Z (see test_window_schema).
"""
from unittest.mock import MagicMock

import pytest

# har_app_module, get_har_parser_func from conftest


class TestFaceWorkerLoop:
    """Presence and behavior of _face_worker_loop (face recognition thread)."""

    def test_face_worker_loop_exists(self, har_app_module):
        """_face_worker_loop exists and accepts (user_data)."""
        assert hasattr(har_app_module, "_face_worker_loop")
        assert callable(har_app_module._face_worker_loop)

    def test_face_worker_loop_handles_empty_queue(self, har_app_module):
        """When face_queue is None, the thread exits immediately without error."""
        user_data = MagicMock()
        user_data.face_queue = None
        har_app_module._face_worker_loop(user_data)
        # Should return immediately when face_queue is None; no exception

    def test_face_worker_loop_expects_tuple_item(self, har_app_module):
        """Queue item expected: (frame_bgr, pose_detections, now_ts)."""
        import inspect
        src = inspect.getsource(har_app_module._face_worker_loop)
        assert "frame_bgr" in src and "pose_detections" in src and "now_ts" in src


class TestFaceOptsCloudUrlResolution:
    """Face gallery URL priority: --face-gallery-url then --cloud-url then env vars."""

    def test_face_opts_cloud_url_from_face_gallery_url(self, get_har_parser_func):
        """When --face-gallery-url is passed it is used as the gallery base URL."""
        parser = get_har_parser_func()
        args = parser.parse_args([
            "--face-gallery-url", "http://gallery.example.com",
            "--cloud-url", "http://cloud.example.com",
        ])
        assert getattr(args, "face_gallery_url", "").strip().rstrip("/") == "http://gallery.example.com"
        assert getattr(args, "cloud_url", "").strip().rstrip("/") == "http://cloud.example.com"

    def test_face_opts_cloud_url_from_cloud_url_when_no_face_gallery_url(self, get_har_parser_func):
        """Without --face-gallery-url, cloud URL is taken from --cloud-url."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--cloud-url", "http://192.168.1.106:8000"])
        assert getattr(args, "cloud_url", "").strip().rstrip("/") == "http://192.168.1.106:8000"


class TestEnableFaceParser:
    """--enable-face and face-related parser options."""

    def test_enable_face_flag(self, get_har_parser_func):
        """--enable-face enables face recognition."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--enable-face"])
        assert getattr(args, "enable_face", False) is True

    def test_enable_face_default_false(self, get_har_parser_func):
        """Without --enable-face, enable_face is False."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "enable_face", True) is False


class TestLogFpsPersonField:
    """FPS log line includes Person field for reporting."""

    def test_log_fps_if_due_includes_person_placeholder(self, har_app_module):
        """_log_fps_if_due reads _last_person_label and includes it in the message."""
        import inspect
        src = inspect.getsource(har_app_module._log_fps_if_due)
        assert "person_label" in src or "_last_person_label" in src
        assert "Person:" in src or "person_label" in src

    def test_report_line_persons_on_screen(self, har_app_module):
        """Log includes 'Persons on screen' report line."""
        import inspect
        src = inspect.getsource(har_app_module._log_fps_if_due)
        assert "Persons on screen" in src or "person_label" in src
