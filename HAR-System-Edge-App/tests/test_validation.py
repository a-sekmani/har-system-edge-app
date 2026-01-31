"""
Unit tests for FrameEvent validation: valid passes; invalid (wrong keypoints length, out-of-bounds x/y, malformed bbox) fails; skip policy (no exception).
"""
import pytest

from src.frame_event import (
    FrameEvent,
    PersonPose,
    MISSING_KEYPOINT_SENTINEL,
    NUM_COCO_KEYPOINTS,
    validate_frame_event,
)


def _valid_person():
    """Return a PersonPose that passes validation (17 keypoints, valid bbox)."""
    kps = [[0.0, 0.0, 1.0]] * NUM_COCO_KEYPOINTS
    return PersonPose(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps)


def _valid_event(persons=None):
    """Return a FrameEvent that passes validation (default: one valid person)."""
    return FrameEvent(
        frame_number=1,
        timestamp_ms=1000.0,
        image={"width": 640, "height": 480},
        persons=persons or [_valid_person()],
    )


class TestValidateFrameEvent:
    """Validation pass/fail and error messages."""

    def test_valid_event_passes(self):
        """A valid FrameEvent (17 keypoints, in-bounds, valid bbox) passes validation."""
        event = _valid_event()
        valid, errors = validate_frame_event(event)
        assert valid is True
        assert errors == []

    def test_valid_event_with_track_id_still_passes(self):
        """validate_frame_event does not depend on track_id; persons with track_id still pass."""
        kps = [[0.0, 0.0, 1.0]] * NUM_COCO_KEYPOINTS
        p = PersonPose(bbox=[0, 0, 100, 100], bbox_conf=0.9, keypoints=kps, track_id=42)
        event = FrameEvent(
            frame_number=1,
            timestamp_ms=1000.0,
            image={"width": 640, "height": 480},
            persons=[p],
        )
        valid, errors = validate_frame_event(event)
        assert valid is True
        assert errors == []

    def test_valid_event_empty_persons_passes(self):
        """FrameEvent with empty persons list is valid."""
        event = _valid_event(persons=[])
        valid, errors = validate_frame_event(event)
        assert valid is True
        assert errors == []

    def test_invalid_keypoints_length_fails(self):
        """Validation fails when person has fewer than 17 keypoints (e.g. mutated after creation)."""
        event = _valid_event()
        event.persons[0].keypoints.pop()
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("keypoints length" in e for e in errors)

    def test_invalid_keypoint_out_of_bounds_x_fails(self):
        """Keypoint x outside [0, image_width] with c>0 causes validation to fail."""
        event = _valid_event()
        event.persons[0].keypoints[0] = [1000.0, 0.0, 0.5]  # x > width 640
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("out of" in e or "keypoint" in e for e in errors)

    def test_invalid_keypoint_out_of_bounds_y_fails(self):
        """Keypoint y outside [0, image_height] with c>0 causes validation to fail."""
        event = _valid_event()
        event.persons[0].keypoints[0] = [0.0, 500.0, 0.5]  # y > height 480
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("out of" in e or "keypoint" in e for e in errors)

    def test_sentinel_keypoint_ignored_for_bounds(self):
        """Keypoints with confidence 0 (sentinel) are not checked for bounds."""
        event = _valid_event()
        event.persons[0].keypoints[0] = list(MISSING_KEYPOINT_SENTINEL)  # c=0, should not check x,y
        valid, errors = validate_frame_event(event)
        assert valid is True

    def test_invalid_bbox_x1_gt_x2_fails(self):
        """Validation fails when bbox x1 > x2."""
        event = _valid_event()
        event.persons[0].bbox = [100, 0, 50, 100]
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("x1" in e and "x2" in e for e in errors)

    def test_invalid_bbox_y1_gt_y2_fails(self):
        """Validation fails when bbox y1 > y2."""
        event = _valid_event()
        event.persons[0].bbox = [0, 100, 100, 50]
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("y1" in e and "y2" in e for e in errors)

    def test_invalid_bbox_negative_fails(self):
        """Validation fails when bbox has negative x1 or y1."""
        event = _valid_event()
        event.persons[0].bbox = [-1, 0, 100, 100]
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("non-negative" in e or "bbox" in e for e in errors)

    def test_invalid_image_missing_width_height_fails(self):
        """Validation fails when image has no width/height."""
        event = _valid_event()
        event.image = {}
        valid, errors = validate_frame_event(event)
        assert valid is False
        assert any("width" in e or "height" in e for e in errors)

    def test_skip_policy_no_exception_on_invalid(self):
        """Validation returns (False, errors) without raising; caller skips invalid frames."""
        event = _valid_event()
        event.persons[0].keypoints.pop()
        valid, errors = validate_frame_event(event)
        assert valid is False
        # Caller is responsible to skip (increment counter, not use event); validation just returns False
        assert isinstance(errors, list)


class TestValidationKeypointLength:
    """Keypoints length 17 always in valid construction."""

    def test_valid_person_has_17_keypoints(self):
        """A valid PersonPose has exactly 17 keypoints (COCO)."""
        p = _valid_person()
        assert len(p.keypoints) == NUM_COCO_KEYPOINTS
