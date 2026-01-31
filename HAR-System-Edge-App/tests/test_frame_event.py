"""
Unit tests for FrameEvent model, PersonPose, COCO-17 order, sentinel, bbox format.
"""
import pytest

from src.frame_event import (
    COCO_17_ORDER,
    MISSING_KEYPOINT_SENTINEL,
    NUM_COCO_KEYPOINTS,
    FrameEvent,
    PersonPose,
)


class TestCOCO17Order:
    """COCO-17 order and constants."""

    def test_coco_17_order_has_17_elements(self):
        """COCO_17_ORDER must contain exactly 17 keypoint names."""
        assert len(COCO_17_ORDER) == 17

    def test_coco_17_order_first_last(self):
        """First keypoint is nose, last is right_ankle."""
        assert COCO_17_ORDER[0] == "nose"
        assert COCO_17_ORDER[16] == "right_ankle"

    def test_num_coco_keypoints(self):
        """NUM_COCO_KEYPOINTS constant must be 17."""
        assert NUM_COCO_KEYPOINTS == 17


class TestMissingKeypointSentinel:
    """Missing keypoint representation."""

    def test_sentinel_value(self):
        """Missing keypoint sentinel is [-1, -1, 0] (x, y, confidence)."""
        assert MISSING_KEYPOINT_SENTINEL == [-1.0, -1.0, 0.0]

    def test_sentinel_length_three(self):
        """Sentinel is a 3-element list [x, y, c]."""
        assert len(MISSING_KEYPOINT_SENTINEL) == 3


class TestPersonPose:
    """PersonPose construction and keypoints length."""

    def test_construct_with_17_keypoints(self, PersonPose_class):
        """PersonPose accepts exactly 17 keypoints; track_id defaults to -1."""
        kps = [[0.0, 0.0, 1.0]] * 17
        pose = PersonPose_class(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps)  # track_id defaults -1
        assert len(pose.keypoints) == 17

    def test_construct_raises_if_keypoints_not_17(self, PersonPose_class):
        """PersonPose raises ValueError if keypoints length is not 17."""
        kps = [[0.0, 0.0, 1.0]] * 16
        with pytest.raises(ValueError, match="keypoints must have length 17"):
            PersonPose_class(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps)

    def test_construct_raises_if_bbox_not_4(self, PersonPose_class):
        """PersonPose raises ValueError if bbox length is not 4."""
        kps = [[0.0, 0.0, 1.0]] * 17
        with pytest.raises(ValueError, match="bbox must have length 4"):
            PersonPose_class(bbox=[0, 0, 100], bbox_conf=0.9, keypoints=kps)

    def test_to_dict(self, PersonPose_class):
        """to_dict() includes bbox, bbox_conf, keypoints, and track_id."""
        kps = [[1.0, 2.0, 0.5]] + [[-1.0, -1.0, 0.0]] * 16
        pose = PersonPose_class(bbox=[10.0, 20.0, 50.0, 60.0], bbox_conf=0.8, keypoints=kps, track_id=1)
        d = pose.to_dict()
        assert d["bbox"] == [10.0, 20.0, 50.0, 60.0]
        assert d["bbox_conf"] == 0.8
        assert len(d["keypoints"]) == 17
        assert d["keypoints"][0] == [1.0, 2.0, 0.5]
        assert d["track_id"] == 1

    def test_construct_with_track_id_default(self, PersonPose_class):
        """When track_id is omitted, it defaults to -1 (TRACK_ID_UNKNOWN)."""
        kps = [[0.0, 0.0, 1.0]] * 17
        pose = PersonPose_class(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps)
        assert pose.track_id == -1

    def test_construct_with_track_id_explicit(self, PersonPose_class):
        """PersonPose accepts explicit track_id and stores it."""
        kps = [[0.0, 0.0, 1.0]] * 17
        pose = PersonPose_class(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps, track_id=5)
        assert pose.track_id == 5


class TestFrameEvent:
    """FrameEvent construction and schema."""

    def test_construct_empty_persons(self, FrameEvent_class, PersonPose_class):
        """FrameEvent can have an empty persons list."""
        event = FrameEvent_class(
            frame_number=1,
            timestamp_ms=1000.0,
            image={"width": 640, "height": 480},
            persons=[],
        )
        assert event.frame_number == 1
        assert event.timestamp_ms == 1000.0
        assert event.image["width"] == 640
        assert event.image["height"] == 480
        assert event.persons == []

    def test_construct_with_one_person(self, FrameEvent_class, PersonPose_class):
        """FrameEvent with one PersonPose stores it correctly."""
        kps = [MISSING_KEYPOINT_SENTINEL] * 17
        pose = PersonPose_class(bbox=[0, 0, 100, 200], bbox_conf=0.9, keypoints=kps)
        event = FrameEvent_class(
            frame_number=2,
            timestamp_ms=2000.0,
            image={"width": 640, "height": 480},
            persons=[pose],
        )
        assert len(event.persons) == 1
        assert event.persons[0].bbox == [0, 0, 100, 200]

    def test_to_dict_json_serializable(self, FrameEvent_class, PersonPose_class):
        """to_dict() returns a dict that can be serialized to JSON."""
        kps = [[0.0, 0.0, 1.0]] * 17
        pose = PersonPose_class(bbox=[0, 0, 10, 10], bbox_conf=1.0, keypoints=kps)
        event = FrameEvent_class(
            frame_number=0,
            timestamp_ms=0.0,
            image={"width": 320, "height": 240},
            persons=[pose],
        )
        d = event.to_dict()
        assert d["frame_number"] == 0
        assert d["timestamp_ms"] == 0.0
        assert d["image"] == {"width": 320, "height": 240}
        assert len(d["persons"]) == 1
        import json
        json.dumps(d)
