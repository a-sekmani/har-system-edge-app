"""
Unit tests for Phase 1 acceptance script logic: parse_counters and check_phase1_conditions.
Uses test_phase1 module from project root (HAR-System-Edge-App).
"""
import sys
from pathlib import Path

import pytest

# Ensure project root (parent of tests/) is on path so we can import test_phase1 script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _get_phase1_module():
    """Import test_phase1 module (script in project root)."""
    try:
        import test_phase1 as m
        return m
    except ImportError as e:
        pytest.skip(f"test_phase1 not importable: {e}")


class TestParseCounters:
    """Tests for parse_counters()."""

    def test_parse_counters_empty_returns_zeros(self):
        """parse_counters('') returns all counters as 0."""
        m = _get_phase1_module()
        out = m.parse_counters("")
        assert out["total_frames"] == 0
        assert out["frame_events"] == 0
        assert out["invalid_caps"] == 0
        assert out["invalid_validate"] == 0
        assert out["frames_with_persons"] == 0
        assert out["frames_with_landmarks"] == 0
        assert out["frames_keypoints_len_not_17"] == 0

    def test_parse_counters_single_fps_line(self):
        """parse_counters extracts total_frames, frame_events, invalid_caps, invalid_validate from FPS line."""
        m = _get_phase1_module()
        log = "INFO | __main__ | FPS Stats - Current: 30 FPS, Frames: 100, frame_events: 98, invalid_caps: 0, invalid_validate: 2\n"
        out = m.parse_counters(log)
        assert out["total_frames"] == 100
        assert out["frame_events"] == 98
        assert out["invalid_caps"] == 0
        assert out["invalid_validate"] == 2

    def test_parse_counters_uses_max_total_frames(self):
        """parse_counters uses snapshot where total_frames is highest (cumulative run)."""
        m = _get_phase1_module()
        log = (
            "FPS Stats ... Frames: 50, frame_events: 50, invalid_caps: 0, invalid_validate: 0\n"
            "FPS Stats ... Frames: 200, frame_events: 198, invalid_caps: 0, invalid_validate: 2\n"
        )
        out = m.parse_counters(log)
        assert out["total_frames"] == 200
        assert out["frame_events"] == 198
        assert out["invalid_validate"] == 2

    def test_parse_counters_phase1_summary(self):
        """parse_counters extracts frames_with_persons, frames_with_landmarks, frames_keypoints_len_not_17 from Phase1 summary."""
        m = _get_phase1_module()
        log = (
            "FPS Stats ... Frames: 100, frame_events: 100, invalid_caps: 0, invalid_validate: 0\n"
            "Phase1 summary: frames_with_persons=95, frames_no_persons=5, persons_total=100, "
            "frames_with_landmarks=95, frames_keypoints_len_not_17=0\n"
        )
        out = m.parse_counters(log)
        assert out["frames_with_persons"] == 95
        assert out["frames_with_landmarks"] == 95
        assert out["frames_keypoints_len_not_17"] == 0

    def test_parse_counters_phase1_final(self):
        """parse_counters extracts Phase1 final counters from Phase1 final line."""
        m = _get_phase1_module()
        log = "Phase1 final: frames_with_persons=80, frames_no_persons=20, persons_total=80, frames_with_landmarks=80, frames_keypoints_len_not_17=0\n"
        out = m.parse_counters(log)
        assert out["frames_with_persons"] == 80
        assert out["frames_with_landmarks"] == 80
        assert out["frames_keypoints_len_not_17"] == 0


class TestCheckPhase1Conditions:
    """Tests for check_phase1_conditions()."""

    def test_total_frames_zero_fails(self):
        """check_phase1_conditions fails when total_frames is 0."""
        m = _get_phase1_module()
        ok, reasons = m.check_phase1_conditions({"total_frames": 0})
        assert ok is False
        assert any("total_frames" in r for r in reasons)

    def test_all_conditions_pass(self):
        """check_phase1_conditions returns (True, []) when all counters meet criteria."""
        m = _get_phase1_module()
        counters = {
            "total_frames": 100,
            "frame_events": 98,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 95,
            "frames_with_landmarks": 90,
            "frames_keypoints_len_not_17": 0,
        }
        ok, reasons = m.check_phase1_conditions(counters)
        assert ok is True
        assert reasons == []

    def test_frame_events_below_95_percent_fails(self):
        """check_phase1_conditions fails when frame_events < 0.95 * total_frames."""
        m = _get_phase1_module()
        counters = {
            "total_frames": 100,
            "frame_events": 90,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 50,
            "frames_with_landmarks": 45,
            "frames_keypoints_len_not_17": 0,
        }
        ok, reasons = m.check_phase1_conditions(counters)
        assert ok is False
        assert any("frame_events" in r or "0.95" in r for r in reasons)

    def test_invalid_caps_nonzero_fails(self):
        """check_phase1_conditions fails when invalid_caps != 0."""
        m = _get_phase1_module()
        counters = {
            "total_frames": 100,
            "frame_events": 100,
            "invalid_caps": 1,
            "invalid_validate": 0,
            "frames_with_persons": 50,
            "frames_with_landmarks": 50,
            "frames_keypoints_len_not_17": 0,
        }
        ok, reasons = m.check_phase1_conditions(counters)
        assert ok is False
        assert any("invalid_caps" in r for r in reasons)

    def test_frames_keypoints_len_not_17_nonzero_fails(self):
        """check_phase1_conditions fails when frames_keypoints_len_not_17 != 0."""
        m = _get_phase1_module()
        counters = {
            "total_frames": 100,
            "frame_events": 100,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 50,
            "frames_with_landmarks": 50,
            "frames_keypoints_len_not_17": 1,
        }
        ok, reasons = m.check_phase1_conditions(counters)
        assert ok is False
        assert any("keypoints_len" in r for r in reasons)

    def test_frames_with_persons_below_min_fails(self):
        """check_phase1_conditions fails when frames_with_persons < min_person_frames."""
        m = _get_phase1_module()
        counters = {
            "total_frames": 100,
            "frame_events": 100,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 20,
            "frames_with_landmarks": 20,
            "frames_keypoints_len_not_17": 0,
        }
        ok, reasons = m.check_phase1_conditions(counters, min_person_frames=30)
        assert ok is False
        assert any("frames_with_persons" in r or "MIN_PERSON" in r for r in reasons)

    def test_frames_with_landmarks_below_80_percent_of_persons_fails(self):
        """check_phase1_conditions fails when frames_with_landmarks < 0.8 * frames_with_persons."""
        m = _get_phase1_module()
        counters = {
            "total_frames": 100,
            "frame_events": 100,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 100,
            "frames_with_landmarks": 70,
            "frames_keypoints_len_not_17": 0,
        }
        ok, reasons = m.check_phase1_conditions(counters)
        assert ok is False
        assert any("frames_with_landmarks" in r or "0.8" in r for r in reasons)
