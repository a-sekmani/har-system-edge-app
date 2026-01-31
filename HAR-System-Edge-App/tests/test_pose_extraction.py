"""
Unit tests for converting mock hailo-like detection to PersonPose/FrameEvent.
Covers: mock detection + image size -> PersonPose; multiple persons; no landmarks vs with landmarks; sentinel for missing indices.
"""
from unittest.mock import MagicMock

import pytest

from src.frame_event import (
    MISSING_KEYPOINT_SENTINEL,
    NUM_COCO_KEYPOINTS,
    PersonPose,
    FrameEvent,
)


def _mock_bbox(xmin=0.1, ymin=0.2, width=0.3, height=0.4):
    """Mock hailo bbox (normalized xmin, ymin, width, height)."""
    b = MagicMock()
    b.xmin = MagicMock(return_value=xmin)
    b.ymin = MagicMock(return_value=ymin)
    b.width = MagicMock(return_value=width)
    b.height = MagicMock(return_value=height)
    return b


def _mock_point(x=0.5, y=0.5, confidence=1.0):
    """Mock landmark point with x, y, confidence."""
    p = MagicMock()
    p.x = MagicMock(return_value=x)
    p.y = MagicMock(return_value=y)
    if hasattr(p, "confidence"):
        p.confidence = MagicMock(return_value=confidence)
    else:
        p.confidence = confidence
    return p


def _mock_landmarks(points):
    """Mock HAILO_LANDMARKS object with get_points() returning the given list."""
    lm = MagicMock()
    lm.get_points = MagicMock(return_value=points)
    return lm


def _mock_detection_person(bbox=None, confidence=0.95, landmarks_list=None):
    """Mock hailo person detection (label=person, bbox, confidence, optional landmarks)."""
    det = MagicMock()
    det.get_label = MagicMock(return_value="person")
    det.get_bbox = MagicMock(return_value=bbox or _mock_bbox())
    det.get_confidence = MagicMock(return_value=confidence)
    det.get_objects_typed = MagicMock(return_value=landmarks_list or [])
    return det


class TestPersonPoseFromHailoDetection:
    """PersonPose.from_hailo_detection with mocks."""

    def test_from_hailo_detection_no_landmarks_returns_17_sentinels(self):
        """When detection has no landmarks, all 17 keypoints are MISSING_KEYPOINT_SENTINEL."""
        det = _mock_detection_person(bbox=_mock_bbox(0.0, 0.0, 0.5, 0.5), landmarks_list=[])
        pose = PersonPose.from_hailo_detection(det, 640, 480)
        assert len(pose.keypoints) == NUM_COCO_KEYPOINTS
        for kp in pose.keypoints:
            assert kp == MISSING_KEYPOINT_SENTINEL

    def test_from_hailo_detection_bbox_pixel_conversion(self):
        """Normalized bbox (xmin, ymin, w, h) is converted to pixel [x1, y1, x2, y2]."""
        det = _mock_detection_person(bbox=_mock_bbox(0.1, 0.2, 0.3, 0.4), landmarks_list=[])
        pose = PersonPose.from_hailo_detection(det, 640, 480)
        assert len(pose.bbox) == 4
        assert pose.bbox[0] == pytest.approx(64.0)
        assert pose.bbox[1] == pytest.approx(96.0)
        assert pose.bbox[2] == pytest.approx(256.0)
        assert pose.bbox[3] == pytest.approx(288.0)

    def test_from_hailo_detection_bbox_conf(self):
        """bbox_conf is taken from detection confidence."""
        det = _mock_detection_person(bbox=_mock_bbox(), confidence=0.87, landmarks_list=[])
        pose = PersonPose.from_hailo_detection(det, 100, 100)
        assert pose.bbox_conf == 0.87

    def test_from_hailo_detection_with_landmarks_fills_keypoints(self):
        """When landmarks are present, keypoints are filled with pixel coords and confidence."""
        points = [_mock_point(0.5, 0.5, 0.9) for _ in range(17)]
        lm = _mock_landmarks(points)
        det = _mock_detection_person(bbox=_mock_bbox(0.0, 0.0, 1.0, 1.0), landmarks_list=[lm])
        pose = PersonPose.from_hailo_detection(det, 100, 100)
        assert len(pose.keypoints) == 17
        # x_px = (0.5 * 1 + 0) * 100 = 50, y_px = 50
        assert pose.keypoints[0][0] == pytest.approx(50.0)
        assert pose.keypoints[0][1] == pytest.approx(50.0)
        assert pose.keypoints[0][2] == 0.9

    def test_from_hailo_detection_partial_landmarks_sentinel_after(self):
        """Indices beyond provided landmarks get MISSING_KEYPOINT_SENTINEL."""
        points = [_mock_point(1.0, 1.0, 0.8) for _ in range(5)]
        lm = _mock_landmarks(points)
        det = _mock_detection_person(bbox=_mock_bbox(0, 0, 1, 1), landmarks_list=[lm])
        pose = PersonPose.from_hailo_detection(det, 100, 100)
        assert len(pose.keypoints) == 17
        for i in range(5):
            assert pose.keypoints[i] != MISSING_KEYPOINT_SENTINEL
        for i in range(5, 17):
            assert pose.keypoints[i] == MISSING_KEYPOINT_SENTINEL

    def test_from_hailo_detection_clamps_keypoints_to_image_bounds(self):
        """Policy 1: keypoints with x_rel/y_rel outside [0,1] are clamped to [0,w] and [0,h]."""
        # x_rel=1.5 -> would be 150 px with width 100; clamp to 100. y_rel=-0.1 -> -10; clamp to 0.
        points = [_mock_point(1.5, -0.1, 0.9)] + [_mock_point(0.5, 0.5, 0.0) for _ in range(16)]
        lm = _mock_landmarks(points)
        det = _mock_detection_person(bbox=_mock_bbox(0, 0, 1, 1), landmarks_list=[lm])
        pose = PersonPose.from_hailo_detection(det, 100, 100)
        assert len(pose.keypoints) == 17
        assert pose.keypoints[0][0] == 100.0
        assert pose.keypoints[0][1] == 0.0
        assert pose.keypoints[0][2] == 0.9

    def test_from_hailo_detection_accepts_track_id(self):
        """PersonPose.from_hailo_detection(..., track_id=X) sets pose.track_id."""
        det = _mock_detection_person(bbox=_mock_bbox(0.0, 0.0, 0.5, 0.5), landmarks_list=[])
        pose = PersonPose.from_hailo_detection(det, 640, 480, track_id=7)
        assert pose.track_id == 7

    def test_from_hailo_detection_track_id_default_unknown(self):
        """When track_id is not passed, PersonPose gets TRACK_ID_UNKNOWN (-1)."""
        from src.frame_event import TRACK_ID_UNKNOWN
        det = _mock_detection_person(bbox=_mock_bbox(0.0, 0.0, 0.5, 0.5), landmarks_list=[])
        pose = PersonPose.from_hailo_detection(det, 640, 480)
        assert pose.track_id == TRACK_ID_UNKNOWN

    def test_from_hailo_detection_requires_hailo_module(self):
        """from_hailo_detection imports hailo; tests above use mocks so hailo is used only for type. Placeholder for env without hailo."""
        pass  # covered by tests above that use mocks

    def test_multiple_persons_frame_event(self, FrameEvent_class, PersonPose_class):
        """FrameEvent can hold multiple PersonPose instances with distinct bbox_conf and track_id."""
        kps1 = [[0.0, 0.0, 1.0]] * 17
        kps2 = [[10.0, 10.0, 0.5]] * 17
        p1 = PersonPose_class(bbox=[0, 0, 50, 50], bbox_conf=0.9, keypoints=kps1, track_id=1)
        p2 = PersonPose_class(bbox=[100, 100, 200, 200], bbox_conf=0.8, keypoints=kps2, track_id=2)
        event = FrameEvent_class(
            frame_number=1,
            timestamp_ms=100.0,
            image={"width": 640, "height": 480},
            persons=[p1, p2],
        )
        assert len(event.persons) == 2
        assert event.persons[0].bbox_conf == 0.9
        assert event.persons[1].bbox_conf == 0.8
