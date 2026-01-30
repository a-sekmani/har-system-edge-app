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
