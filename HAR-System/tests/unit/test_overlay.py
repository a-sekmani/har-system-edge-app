"""
Unit tests for `har_system.utils.overlay`.

These tests verify that drawing helpers are safe (do not crash) and preserve frame shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from har_system.utils.overlay import PersonOverlay, get_overlay


@pytest.mark.unit
def test_get_overlay_is_singleton():
    a = get_overlay()
    b = get_overlay()
    assert a is b


@pytest.mark.unit
def test_draw_person_info_does_not_crash_and_preserves_shape():
    overlay = PersonOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = {"xmin": 0.1, "ymin": 0.2, "xmax": 0.3, "ymax": 0.6}

    out = overlay.draw_person_info(
        frame=frame.copy(),
        bbox=bbox,
        track_id=1,
        name="Unknown",
        confidence=0.0,
        activity="stationary",
    )
    assert out.shape == frame.shape


@pytest.mark.unit
def test_draw_stats_does_not_crash_and_preserves_shape():
    overlay = PersonOverlay()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    stats = {"total_tracks_seen": 2, "total_falls_detected": 1, "total_activity_changes": 3}

    out = overlay.draw_stats(frame.copy(), stats)
    assert out.shape == frame.shape

