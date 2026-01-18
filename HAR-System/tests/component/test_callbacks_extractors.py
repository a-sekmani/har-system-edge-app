"""
Component tests for `har_system.core.callbacks` extractor helpers.

These tests validate the contract between our extractor logic and the Hailo detection-like
objects. They are skipped automatically if the optional Hailo dependencies are not installed.
"""

from __future__ import annotations

import pytest


# callbacks.py depends on hailo + hailo_apps buffer utilities.
hailo = pytest.importorskip("hailo")
callbacks = pytest.importorskip("har_system.core.callbacks")


class _FakeUniqueId:
    def __init__(self, track_id: int):
        self._id = track_id

    def get_id(self) -> int:
        return self._id


class _FakeBbox:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    def width(self) -> float:
        return self._xmax - self._xmin

    def height(self) -> float:
        return self._ymax - self._ymin

    def xmin(self) -> float:
        return self._xmin

    def ymin(self) -> float:
        return self._ymin

    def xmax(self) -> float:
        return self._xmax

    def ymax(self) -> float:
        return self._ymax


class _FakePoint:
    def __init__(self, x: float, y: float, conf: float):
        self._x = x
        self._y = y
        self._conf = conf

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y

    def confidence(self) -> float:
        return self._conf


class _FakeLandmarks:
    def __init__(self, points):
        self._points = points

    def get_points(self):
        return self._points


class _FakeDetection:
    def __init__(self, track_id: int, bbox: _FakeBbox, points, confidence: float = 0.9):
        self._track_id = track_id
        self._bbox = bbox
        self._points = points
        self._confidence = confidence

    def get_objects_typed(self, obj_type):
        if obj_type == hailo.HAILO_UNIQUE_ID:
            return [_FakeUniqueId(self._track_id)]
        if obj_type == hailo.HAILO_LANDMARKS:
            return [_FakeLandmarks(self._points)]
        return []

    def get_bbox(self):
        return self._bbox

    def get_confidence(self) -> float:
        return self._confidence


@pytest.mark.component
def test_extract_frame_data_happy_path():
    keypoint_map = callbacks.get_keypoint_mapping()

    # Provide 17 landmark points in the expected order.
    points = [_FakePoint(0.1, 0.2, 0.9) for _ in range(17)]
    det = _FakeDetection(
        track_id=7,
        bbox=_FakeBbox(0.1, 0.2, 0.5, 0.8),
        points=points,
        confidence=0.95,
    )

    frame_data = callbacks.extract_frame_data(det, keypoint_map)
    assert frame_data is not None
    assert frame_data["track_id"] == 7
    assert "timestamp" in frame_data
    assert frame_data["bbox"]["xmin"] == pytest.approx(0.1)
    assert "nose" in frame_data["keypoints"]


@pytest.mark.component
def test_extract_eye_positions_returns_pixels():
    keypoint_map = callbacks.get_keypoint_mapping()

    # left_eye=idx 1, right_eye=idx 2; values are relative to bbox (as used by callbacks).
    points = [_FakePoint(0.0, 0.0, 0.0) for _ in range(17)]
    points[keypoint_map["left_eye"]] = _FakePoint(0.2, 0.3, 0.9)
    points[keypoint_map["right_eye"]] = _FakePoint(0.7, 0.4, 0.8)

    det = _FakeDetection(
        track_id=3,
        bbox=_FakeBbox(0.1, 0.2, 0.5, 0.8),
        points=points,
        confidence=0.9,
    )

    frame_w, frame_h = 640, 480
    result = callbacks.extract_eye_positions(det, keypoint_map, frame_w, frame_h)
    assert result is not None

    person_id, lx, ly, rx, ry = result
    assert person_id == 3

    # Must be integers (pixel coordinates).
    assert isinstance(lx, int)
    assert isinstance(ly, int)
    assert isinstance(rx, int)
    assert isinstance(ry, int)

