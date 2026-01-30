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
        assert len(COCO_17_ORDER) == 17

    def test_coco_17_order_first_last(self):
        assert COCO_17_ORDER[0] == "nose"
        assert COCO_17_ORDER[16] == "right_ankle"

    def test_num_coco_keypoints(self):
        assert NUM_COCO_KEYPOINTS == 17


class TestMissingKeypointSentinel:
    """Missing keypoint representation."""

    def test_sentinel_value(self):
        assert MISSING_KEYPOINT_SENTINEL == [-1.0, -1.0, 0.0]

    def test_sentinel_length_three(self):
        assert len(MISSING_KEYPOINT_SENTINEL) == 3


class TestPersonPose:
    """PersonPose construction and keypoints length."""

    def test_construct_with_17_keypoints(self, PersonPose_class):
        kps = [[0.0, 0.0, 1.0]] * 17
        pose = PersonPose_class(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps)
        assert len(pose.keypoints) == 17

    def test_construct_raises_if_keypoints_not_17(self, PersonPose_class):
        kps = [[0.0, 0.0, 1.0]] * 16
        with pytest.raises(ValueError, match="keypoints must have length 17"):
            PersonPose_class(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps)

    def test_construct_raises_if_bbox_not_4(self, PersonPose_class):
        kps = [[0.0, 0.0, 1.0]] * 17
        with pytest.raises(ValueError, match="bbox must have length 4"):
            PersonPose_class(bbox=[0, 0, 100], bbox_conf=0.9, keypoints=kps)

    def test_to_dict(self, PersonPose_class):
        kps = [[1.0, 2.0, 0.5]] + [[-1.0, -1.0, 0.0]] * 16
        pose = PersonPose_class(bbox=[10.0, 20.0, 50.0, 60.0], bbox_conf=0.8, keypoints=kps)
        d = pose.to_dict()
        assert d["bbox"] == [10.0, 20.0, 50.0, 60.0]
        assert d["bbox_conf"] == 0.8
        assert len(d["keypoints"]) == 17
        assert d["keypoints"][0] == [1.0, 2.0, 0.5]


class TestFrameEvent:
    """FrameEvent construction and schema."""

    def test_construct_empty_persons(self, FrameEvent_class, PersonPose_class):
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
