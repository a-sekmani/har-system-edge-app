"""
Unit tests for get_har_parser() and --no-display flag.
"""
import pytest


class TestGetHarParser:
    """Tests for get_har_parser()."""

    def test_returns_parser(self, get_har_parser_func):
        """get_har_parser() should return an ArgumentParser."""
        parser = get_har_parser_func()
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_has_no_display_flag(self, get_har_parser_func):
        """Parser should have --no-display option."""
        parser = get_har_parser_func()
        args = parser.parse_args(["--no-display"])
        assert hasattr(args, "no_display")
        assert args.no_display is True

    def test_no_display_default_false(self, get_har_parser_func):
        """Without --no-display, no_display should be False."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "no_display", False) is False

    def test_inherits_standard_flags(self, get_har_parser_func):
        """Parser should inherit standard options from pipeline parser (e.g. --show-fps, --input)."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--show-fps"])
        assert hasattr(args, "show_fps")
        assert args.show_fps is True

    def test_no_display_help_text(self, get_har_parser_func):
        """--no-display option should appear in help text."""
        parser = get_har_parser_func()
        help_str = parser.format_help()
        assert "--no-display" in help_str
        assert "no-display" in help_str or "display" in help_str.lower()

    def test_has_log_pose_summary_flag(self, get_har_parser_func):
        """Parser should have --log-pose-summary option."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--log-pose-summary"])
        assert hasattr(args, "log_pose_summary")
        assert args.log_pose_summary is True

    def test_log_pose_summary_default_false(self, get_har_parser_func):
        """Without --log-pose-summary, log_pose_summary should be False."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "log_pose_summary", False) is False

    def test_has_dump_frames_option(self, get_har_parser_func):
        """Parser should have --dump-frames option."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--dump-frames", "/path/to/file.json"])
        assert hasattr(args, "dump_frames")
        assert args.dump_frames == "/path/to/file.json"

    def test_dump_frames_default_none(self, get_har_parser_func):
        """Without --dump-frames, dump_frames should be None."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "dump_frames", None) is None

    def test_has_tracking_source_option(self, get_har_parser_func):
        """Parser should have --tracking-source option (metadata|fallback)."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--tracking-source", "fallback"])
        assert getattr(args, "tracking_source", None) == "fallback"

    def test_has_max_missing_frames_option(self, get_har_parser_func):
        """Parser should have --max-missing-frames option."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--max-missing-frames", "5"])
        assert getattr(args, "max_missing_frames", None) == 5

    def test_has_log_tracking_summary_flag(self, get_har_parser_func):
        """Parser should have --log-tracking-summary option."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--log-tracking-summary"])
        assert getattr(args, "log_tracking_summary", False) is True

    def test_has_min_bbox_height_option(self, get_har_parser_func):
        """Parser should have --min-bbox-height option (filter ghost detections)."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--min-bbox-height", "80"])
        assert getattr(args, "min_bbox_height", None) == 80.0

    def test_min_bbox_height_default_none(self, get_har_parser_func):
        """Without --min-bbox-height, min_bbox_height should be None (no filter)."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "min_bbox_height", "missing") is None

    def test_has_min_pose_confidence_option(self, get_har_parser_func):
        """Parser should have --min-pose-confidence option (filter low-confidence detections)."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--min-pose-confidence", "0.3"])
        assert getattr(args, "min_pose_confidence", None) == 0.3

    def test_min_pose_confidence_default_none(self, get_har_parser_func):
        """Without --min-pose-confidence, min_pose_confidence should be None (no filter)."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "min_pose_confidence", "missing") is None

    # --- Face recognition options (gallery URL, worker thread, FPS defaults) ---

    def test_face_skip_frames_default_10(self, get_har_parser_func):
        """Default --face-skip-frames should be 10 for better FPS."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "face_skip_frames", None) == 10

    def test_face_skip_frames_override(self, get_har_parser_func):
        """--face-skip-frames can override default."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--face-skip-frames", "5"])
        assert getattr(args, "face_skip_frames", None) == 5

    def test_face_det_size_default_256(self, get_har_parser_func):
        """Default --face-det-size should be 256 for better FPS."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "face_det_size", None) == 256

    def test_face_max_faces_default_1(self, get_har_parser_func):
        """Default --face-max-faces should be 1 for single-person FPS."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args([])
        assert getattr(args, "face_max_faces", None) == 1

    def test_face_gallery_url_option(self, get_har_parser_func):
        """Parser should have --face-gallery-url for gallery-only base URL."""
        parser = get_har_parser_func()
        args, _ = parser.parse_known_args(["--face-gallery-url", "http://192.168.1.106:8000"])
        assert getattr(args, "face_gallery_url", "").strip().rstrip("/") == "http://192.168.1.106:8000"
