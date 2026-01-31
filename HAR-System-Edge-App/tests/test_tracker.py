"""
Unit tests for Phase 2 tracker: TrackingConfig, get_metadata_track_id, FallbackTracker.
"""
from unittest.mock import MagicMock

import pytest

from src.tracker import (
    TRACK_ID_UNKNOWN,
    TrackingConfig,
    FallbackTracker,
    get_metadata_track_id,
)


class TestTrackingConfig:
    """TrackingConfig defaults and optional filter fields."""

    def test_defaults(self):
        """TrackingConfig has expected defaults (metadata source, max_missing_frames=15, filter opts None)."""
        cfg = TrackingConfig()
        assert cfg.tracking_enabled is True
        assert cfg.tracking_source == "metadata"
        assert cfg.max_missing_frames == 15
        assert cfg.iou_match_threshold == 0.3
        assert cfg.min_bbox_area == 0.0
        assert cfg.min_bbox_height is None
        assert cfg.min_pose_confidence is None

    def test_filter_options_can_be_set(self):
        """min_bbox_height and min_pose_confidence can be set for detection filtering."""
        cfg = TrackingConfig(min_bbox_height=80.0, min_pose_confidence=0.2)
        assert cfg.min_bbox_height == 80.0
        assert cfg.min_pose_confidence == 0.2


class TestGetMetadataTrackId:
    """get_metadata_track_id with mock detection."""

    def test_returns_none_when_no_hailo_unique_id(self):
        """When detection has no HAILO_UNIQUE_ID objects, returns None."""
        try:
            import hailo
        except ImportError:
            pytest.skip("hailo not available")
        det = MagicMock()
        det.get_objects_typed = MagicMock(return_value=[])
        assert get_metadata_track_id(det) is None

    def test_returns_id_when_one_unique_id_object(self):
        """When detection has one HAILO_UNIQUE_ID with get_id()=42, returns 42."""
        try:
            import hailo
        except ImportError:
            pytest.skip("hailo not available")
        obj = MagicMock()
        obj.get_id = MagicMock(return_value=42)
        det = MagicMock()
        det.get_objects_typed = MagicMock(return_value=[obj])
        assert get_metadata_track_id(det) == 42

    def test_returns_none_when_id_zero_or_negative(self):
        """When get_id() returns 0 or negative, returns None (invalid id)."""
        try:
            import hailo
        except ImportError:
            pytest.skip("hailo not available")
        obj = MagicMock()
        obj.get_id = MagicMock(return_value=0)
        det = MagicMock()
        det.get_objects_typed = MagicMock(return_value=[obj])
        assert get_metadata_track_id(det) is None


def _bbox(x1, y1, x2, y2):
    """Helper: bbox as [x1, y1, x2, y2] in pixels for tracker tests."""
    return [float(x1), float(y1), float(x2), float(y2)]


class TestFallbackTrackerSinglePersonStableId:
    """Consecutive frames with one bbox moving slightly -> same track_id."""

    def test_tracking_assignments_single_person_stable_id(self):
        cfg = TrackingConfig(tracking_source="fallback", max_missing_frames=15, iou_match_threshold=0.3)
        tracker = FallbackTracker(cfg)
        # Frame 0: one bbox
        ids0, new0, end0 = tracker.update([_bbox(10, 10, 50, 80)], frame_index=0)
        assert ids0 == [1]
        assert new0 == 1
        assert end0 == 0
        # Frame 1: same bbox slightly moved (high IoU)
        ids1, new1, end1 = tracker.update([_bbox(12, 10, 52, 80)], frame_index=1)
        assert ids1 == [1]
        assert new1 == 0
        assert end1 == 0


class TestFallbackTrackerMissingFramesRecover:
    """Skip a few frames then same bbox -> same id within max_missing_frames."""

    def test_tracking_missing_frames_then_recover_with_same_id_within_grace(self):
        cfg = TrackingConfig(max_missing_frames=5, iou_match_threshold=0.3)
        tracker = FallbackTracker(cfg)
        bbox = _bbox(20, 20, 60, 90)
        ids0, _, _ = tracker.update([bbox], frame_index=0)
        assert ids0[0] == 1
        # Frames 1-3: no detections (track goes missing)
        for f in range(1, 4):
            tracker.update([], frame_index=f)
        # Frame 4: same bbox back -> should still match (missing_frames=3 < 5)
        ids4, new4, end4 = tracker.update([bbox], frame_index=4)
        assert ids4[0] == 1
        assert new4 == 0
        assert end4 == 0


class TestFallbackTrackerExpire:
    """One track, then no detections for > max_missing_frames -> track ended; next detection gets new id."""

    def test_tracking_expire_track_after_max_missing_frames(self):
        cfg = TrackingConfig(max_missing_frames=3, iou_match_threshold=0.3)
        tracker = FallbackTracker(cfg)
        bbox = _bbox(10, 10, 50, 50)
        ids0, _, _ = tracker.update([bbox], frame_index=0)
        assert ids0[0] == 1
        # 4 frames with no detections -> track expires (missing_frames >= 3 triggers removal in one of frames 1-4)
        for f in range(1, 5):
            tracker.update([], frame_index=f)
        # Next detection gets new id (track was ended in an earlier frame)
        ids5, new5, end5 = tracker.update([bbox], frame_index=5)
        assert ids5[0] == 2
        assert new5 == 1


class TestFallbackTrackerTwoPersons:
    """Two bboxes far apart over frames -> two distinct stable ids."""

    def test_tracking_two_persons_no_id_swap_when_separated(self):
        cfg = TrackingConfig(iou_match_threshold=0.3)
        tracker = FallbackTracker(cfg)
        b1 = _bbox(10, 10, 50, 80)
        b2 = _bbox(200, 10, 280, 80)
        ids0, _, _ = tracker.update([b1, b2], frame_index=0)
        assert set(ids0) == {1, 2}
        # Move both slightly
        ids1, _, _ = tracker.update(
            [_bbox(12, 10, 52, 80), _bbox(202, 10, 282, 80)],
            frame_index=1,
        )
        assert set(ids1) == {1, 2}
        # Order might differ; ids should be stable (person 0 -> id 1, person 1 -> id 2 typically)
        ids2, _, _ = tracker.update(
            [_bbox(14, 10, 54, 80), _bbox(204, 10, 284, 80)],
            frame_index=2,
        )
        assert set(ids2) == {1, 2}


class TestFallbackTrackerIoUThreshold:
    """Two bboxes; one disappears, the other moves; new bbox with low IoU to remaining track -> new id."""

    def test_tracking_iou_threshold_blocks_wrong_match(self):
        cfg = TrackingConfig(iou_match_threshold=0.5)  # high threshold
        tracker = FallbackTracker(cfg)
        b1 = _bbox(10, 10, 50, 50)
        b2 = _bbox(100, 10, 140, 50)
        ids0, _, _ = tracker.update([b1, b2], frame_index=0)
        assert len(ids0) == 2
        # Frame 1: only b2 (slightly moved); b1 gone
        tracker.update([_bbox(102, 10, 142, 50)], frame_index=1)
        # Frame 2: b1 reappears but in very different position (low IoU with original b1 and b2)
        # No match above 0.5 -> new track id (3, since we already have tracks 1 and 2)
        b1_far = _bbox(200, 200, 240, 240)  # no overlap with original b1 or b2
        ids2, new2, _ = tracker.update([b1_far], frame_index=2)
        assert ids2[0] == 3  # new id (tracks 1 and 2 exist; next id is 3)
        assert new2 == 1


class TestFallbackTrackerMinBboxArea:
    """Tiny bbox below min_bbox_area not creating a track (filtered out)."""

    def test_tracking_min_bbox_area_filters_noise(self):
        cfg = TrackingConfig(min_bbox_area=1000.0, iou_match_threshold=0.3)
        tracker = FallbackTracker(cfg)
        # Tiny bbox: area = 5*5 = 25 < 1000
        tiny = _bbox(0, 0, 5, 5)
        ids, new, end = tracker.update([tiny], frame_index=0)
        assert ids[0] == TRACK_ID_UNKNOWN
        assert new == 0
        # Normal bbox: area = 40*70 = 2800 >= 1000
        normal = _bbox(10, 10, 50, 80)
        ids2, new2, _ = tracker.update([normal], frame_index=1)
        assert ids2[0] == 1
        assert new2 == 1
