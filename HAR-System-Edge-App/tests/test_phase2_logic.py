"""
Unit tests for Phase 2 acceptance script logic: parse_counters and check_phase2_conditions.
Uses test_phase2 module from acceptance_tests/.
"""
import sys
from pathlib import Path

import pytest

# Project root and acceptance_tests for importing test_phase2
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ACCEPTANCE_DIR = _REPO_ROOT / "acceptance_tests"
if str(_ACCEPTANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_ACCEPTANCE_DIR))


def _get_phase2_module():
    """Import test_phase2 module (acceptance_tests/test_phase2.py)."""
    try:
        import test_phase2 as m
        return m
    except ImportError as e:
        pytest.skip(f"test_phase2 not importable: {e}")


class TestParseCountersPhase2:
    """Tests for parse_counters() Phase 2 fields."""

    def test_parse_counters_empty_returns_zeros(self):
        """parse_counters('') returns Phase 2 counters as 0."""
        m = _get_phase2_module()
        out = m.parse_counters("")
        assert out.get("unique_track_ids", 0) == 0
        assert out.get("new_tracks_created", 0) == 0
        assert out.get("tracks_ended", 0) == 0
        assert out.get("id_switch_suspected", 0) == 0
        assert out.get("multi_person_frames", 0) == 0

    def test_parse_counters_phase2_summary(self):
        """parse_counters extracts Phase 2 fields from Phase2 summary line."""
        m = _get_phase2_module()
        log = (
            "FPS Stats ... Frames: 100, frame_events: 98, invalid_caps: 0, invalid_validate: 0\n"
            "Phase1 summary: frames_with_persons=95, frames_with_landmarks=90, frames_keypoints_len_not_17=0\n"
            "Phase2 summary: unique_track_ids=2, new_tracks_created=1, tracks_ended=0, "
            "id_switch_suspected=1, multi_person_frames=0\n"
        )
        out = m.parse_counters(log)
        assert out["unique_track_ids"] == 2
        assert out["new_tracks_created"] == 1
        assert out["tracks_ended"] == 0
        assert out["id_switch_suspected"] == 1
        assert out["multi_person_frames"] == 0

    def test_parse_counters_phase2_final(self):
        """parse_counters extracts Phase 2 counters from Phase2 final line."""
        m = _get_phase2_module()
        log = (
            "Phase2 final: unique_track_ids=3, new_tracks_created=2, tracks_ended=1, "
            "id_switch_suspected=2, multi_person_frames=10\n"
        )
        out = m.parse_counters(log)
        assert out["unique_track_ids"] == 3
        assert out["new_tracks_created"] == 2
        assert out["tracks_ended"] == 1
        assert out["id_switch_suspected"] == 2
        assert out["multi_person_frames"] == 10

    def test_parse_counters_phase2_with_detections_filter_counts(self):
        """Phase2 summary/final can include detections_total and filtered_detections_total."""
        m = _get_phase2_module()
        log = (
            "Phase2 summary: unique_track_ids=2, new_tracks_created=1, tracks_ended=0, "
            "id_switch_suspected=0, multi_person_frames=5, detections_total=1000, filtered_detections_total=120\n"
        )
        out = m.parse_counters(log)
        assert out["unique_track_ids"] == 2
        assert out["detections_total"] == 1000
        assert out["filtered_detections_total"] == 120


class TestCheckPhase2Conditions:
    """Tests for check_phase2_single_person_conditions()."""

    def test_total_frames_zero_fails(self):
        """check_phase2_single_person_conditions fails when total_frames is 0."""
        m = _get_phase2_module()
        ok, reasons = m.check_phase2_single_person_conditions({"total_frames": 0})
        assert ok is False
        assert any("total_frames" in r for r in reasons)

    def test_all_single_person_conditions_pass(self):
        m = _get_phase2_module()
        counters = {
            "total_frames": 100,
            "frame_events": 96,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 50,
            "unique_track_ids": 2,
            "id_switch_suspected": 0,
        }
        ok, reasons = m.check_phase2_single_person_conditions(counters)
        assert ok is True
        assert reasons == []

    def test_unique_track_ids_above_2_fails(self):
        """check_phase2_single_person_conditions fails when unique_track_ids > 2."""
        m = _get_phase2_module()
        counters = {
            "total_frames": 100,
            "frame_events": 98,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 50,
            "unique_track_ids": 3,
            "id_switch_suspected": 0,
        }
        ok, reasons = m.check_phase2_single_person_conditions(counters)
        assert ok is False
        assert any("unique_track_ids" in r for r in reasons)

    def test_id_switch_suspected_nonzero_fails(self):
        """check_phase2_single_person_conditions fails when id_switch_suspected != 0."""
        m = _get_phase2_module()
        counters = {
            "total_frames": 100,
            "frame_events": 98,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 50,
            "unique_track_ids": 2,
            "id_switch_suspected": 1,
        }
        ok, reasons = m.check_phase2_single_person_conditions(counters)
        assert ok is False
        assert any("id_switch_suspected" in r for r in reasons)

    def test_frames_with_persons_below_min_fails(self):
        """check_phase2_single_person_conditions fails when frames_with_persons < min_person_frames."""
        m = _get_phase2_module()
        counters = {
            "total_frames": 100,
            "frame_events": 98,
            "invalid_caps": 0,
            "invalid_validate": 0,
            "frames_with_persons": 20,
            "unique_track_ids": 1,
            "id_switch_suspected": 0,
        }
        ok, reasons = m.check_phase2_single_person_conditions(counters, min_person_frames=30)
        assert ok is False
        assert any("frames_with_persons" in r for r in reasons)


class TestCheckPhase2TwoPersonConditions:
    """Tests for check_phase2_two_person_conditions() (optional two-person mode)."""

    def test_two_person_conditions_pass(self):
        """check_phase2_two_person_conditions returns (True, []) when multi-person criteria met."""
        m = _get_phase2_module()
        counters = {
            "multi_person_frames": 50,
            "unique_track_ids": 3,
            "id_switch_suspected": 5,
        }
        ok, reasons = m.check_phase2_two_person_conditions(counters)
        assert ok is True
        assert reasons == []

    def test_multi_person_frames_below_min_fails(self):
        """check_phase2_two_person_conditions fails when multi_person_frames < min_multi."""
        m = _get_phase2_module()
        counters = {
            "multi_person_frames": 10,
            "unique_track_ids": 2,
            "id_switch_suspected": 0,
        }
        ok, reasons = m.check_phase2_two_person_conditions(counters, min_multi=30)
        assert ok is False
        assert any("multi_person_frames" in r for r in reasons)

    def test_unique_track_ids_out_of_range_fails(self):
        """check_phase2_two_person_conditions fails when unique_track_ids not in [2, 6]."""
        m = _get_phase2_module()
        counters = {
            "multi_person_frames": 50,
            "unique_track_ids": 10,
            "id_switch_suspected": 0,
        }
        ok, reasons = m.check_phase2_two_person_conditions(counters)
        assert ok is False
        assert any("unique_track_ids" in r for r in reasons)
