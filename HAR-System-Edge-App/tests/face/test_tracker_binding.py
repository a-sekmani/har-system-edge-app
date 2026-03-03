"""Unit tests for face tracker_binding: IoU association, TTL, get_identity policy."""

import time
import pytest
from src.face.schemas import FaceDetection
from src.face.tracker_binding import TrackerBinding, _iou_bbox, PERSON_SOURCE_EDGE_FACE


def test_iou_bbox_same():
    a = (0.0, 0.0, 10.0, 10.0)
    assert abs(_iou_bbox(a, a) - 1.0) < 1e-6


def test_iou_bbox_no_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert _iou_bbox(a, b) == 0.0


def test_iou_bbox_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    inter = 5.0 * 10.0
    area_a = 100.0
    area_b = 100.0
    expected = inter / (area_a + area_b - inter)
    assert abs(_iou_bbox(a, b) - expected) < 1e-6


def test_tracker_binding_update_associates_face_to_pose():
    binding = TrackerBinding(iou_threshold=0.2, track_ttl_s=10.0, recheck_every_s=2.0, sim_threshold=0.3, min_votes_stable=1)
    pose_detections = [(1, (100.0, 100.0, 200.0, 200.0))]
    face_bbox = (110.0, 110.0, 190.0, 190.0)
    face_det = FaceDetection(bbox_xyxy=face_bbox, det_conf=0.9)
    match_result = ("person-uuid", "Alice", 0.85)
    binding.update(pose_detections, [(face_det, match_result)])
    att = binding.get_identity(1, attach_policy="auto")
    assert att is not None
    assert att.person_id == "person-uuid"
    assert att.name == "Alice"
    assert att.face_conf == 0.85
    assert att.source == PERSON_SOURCE_EDGE_FACE


def test_tracker_binding_ttl_expires():
    binding = TrackerBinding(iou_threshold=0.2, track_ttl_s=0.5, recheck_every_s=1.0, sim_threshold=0.3, min_votes_stable=1)
    pose_detections = [(1, (0.0, 0.0, 50.0, 50.0))]
    face_det = FaceDetection(bbox_xyxy=(5.0, 5.0, 45.0, 45.0), det_conf=0.9)
    binding.update(pose_detections, [(face_det, ("pid", "P", 0.8))])
    att = binding.get_identity(1, attach_policy="auto")
    assert att is not None
    time.sleep(0.6)
    att2 = binding.get_identity(1, attach_policy="auto")
    assert att2 is None
    att3 = binding.get_identity(1, attach_policy="always")
    assert att3 is not None
    assert att3.person_id is None
    assert att3.face_conf == 0.0


def test_get_identity_never_returns_none():
    binding = TrackerBinding()
    binding.update([(1, (0, 0, 10, 10))], [(FaceDetection(bbox_xyxy=(1, 1, 9, 9), det_conf=0.9), ("pid", "N", 0.9))])
    assert binding.get_identity(1, attach_policy="never") is None


def test_get_identity_always_includes_unknown():
    binding = TrackerBinding()
    att = binding.get_identity(999, attach_policy="always")
    assert att is not None
    assert att.person_id is None
    assert att.name is None
    assert att.face_conf == 0.0
    assert att.source == PERSON_SOURCE_EDGE_FACE
