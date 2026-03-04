"""
Unit tests for Phase 4 parse_counters: windows_built, windows_sent, windows_failed, windows_dropped, windows_queue_depth_max.
Uses test_phase4 module from acceptance_tests/.
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ACCEPTANCE_DIR = _REPO_ROOT / "acceptance_tests"
if str(_ACCEPTANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_ACCEPTANCE_DIR))


def _get_phase4_module():
    """Import test_phase4 module (acceptance_tests/test_phase4.py)."""
    try:
        import test_phase4 as m
        return m
    except ImportError as e:
        pytest.skip(f"test_phase4 not importable: {e}")


class TestPhase4ParseCounters:
    """Tests for parse_counters() Phase4 fields in test_phase4."""

    def test_phase4_parse_empty_returns_zeros(self):
        """parse_counters('') returns Phase4 counters as 0."""
        m = _get_phase4_module()
        out = m.parse_counters("")
        assert out["windows_built"] == 0
        assert out["windows_sent"] == 0
        assert out["windows_failed"] == 0
        assert out["windows_dropped"] == 0
        assert out["windows_queue_depth_max"] == 0

    def test_phase4_parse_summary_line(self):
        """parse_counters extracts Phase4 summary: windows_built, windows_sent, windows_failed, windows_dropped, windows_queue_depth_max."""
        m = _get_phase4_module()
        log = (
            "INFO | __main__ | Phase4 summary: windows_built=50, windows_sent=48, windows_failed=0, "
            "windows_dropped=2, windows_queue_depth_max=3\n"
        )
        out = m.parse_counters(log)
        assert out["windows_built"] == 50
        assert out["windows_sent"] == 48
        assert out["windows_failed"] == 0
        assert out["windows_dropped"] == 2
        assert out["windows_queue_depth_max"] == 3

    def test_phase4_parse_final_line(self):
        """parse_counters extracts Phase4 final line (last one wins)."""
        m = _get_phase4_module()
        log = (
            "Phase4 summary: windows_built=10, windows_sent=10, windows_failed=0, windows_dropped=0, windows_queue_depth_max=1\n"
            "Phase4 final: windows_built=10, windows_sent=10, windows_failed=0, windows_dropped=0, windows_queue_depth_max=1\n"
        )
        out = m.parse_counters(log)
        assert out["windows_built"] == 10
        assert out["windows_sent"] == 10
        assert out["windows_failed"] == 0
        assert out["windows_dropped"] == 0
        assert out["windows_queue_depth_max"] == 1

    def test_phase4_parse_uses_last_match(self):
        """parse_counters uses last Phase4 match (final over summary)."""
        m = _get_phase4_module()
        log = (
            "Phase4 summary: windows_built=5, windows_sent=5, windows_failed=0, windows_dropped=0, windows_queue_depth_max=0\n"
            "Phase4 final: windows_built=20, windows_sent=18, windows_failed=1, windows_dropped=1, windows_queue_depth_max=2\n"
        )
        out = m.parse_counters(log)
        assert out["windows_built"] == 20
        assert out["windows_sent"] == 18
        assert out["windows_failed"] == 1
        assert out["windows_dropped"] == 1
        assert out["windows_queue_depth_max"] == 2
