"""
Unit tests for Phase 3 Cloud Event schema: build_cloud_payload from FrameEvent.
Includes schema sanity: device_id, session_id, frame_index, timestamp, persons list;
per-person: track_id, 17 keypoints, x,y within image bounds, coords, keypoints_format.
"""
import pytest

from src.frame_event import FrameEvent, PersonPose, NUM_COCO_KEYPOINTS
from src.cloud_schema import (
    build_cloud_payload,
    EVENT_TYPE_FRAME,
    KEYPOINTS_FORMAT,
    COORDS_PIXEL,
)


def assert_payload_schema_sanity(payload: dict, image_w: int, image_h: int) -> None:
    """
    Assert Cloud Event payload has required fields and valid structure.
    - event has device_id, session_id, frame_index, timestamp (ts_unix_ms)
    - persons is a list
    - each person: track_id, keypoints length 17, x,y within image bounds or sentinel, coords, keypoints_format
    """
    assert "source" in payload
    assert "device_id" in payload["source"], "payload.source must have device_id"
    assert "session_id" in payload["source"], "payload.source must have session_id"
    assert "frame" in payload
    assert "frame_index" in payload["frame"], "payload.frame must have frame_index"
    assert "ts_unix_ms" in payload["frame"], "payload.frame must have timestamp (ts_unix_ms)"
    assert "persons" in payload
    assert isinstance(payload["persons"], list), "persons must be a list"
    for p in payload["persons"]:
        assert "track_id" in p, "each person must have track_id"
        assert "keypoints" in p
        assert len(p["keypoints"]) == 17, "each person must have 17 keypoints"
        assert p.get("keypoints_format") == KEYPOINTS_FORMAT, "keypoints_format must be coco17"
        assert p.get("coords") == COORDS_PIXEL, "coords must be pixel"
        for kp in p["keypoints"]:
            assert "name" in kp and "x" in kp and "y" in kp and "c" in kp
            x, y = kp["x"], kp["y"]
            # Sentinel or within image bounds
            if (x, y) == (-1.0, -1.0):
                continue
            assert 0 <= x <= image_w, f"keypoint x={x} must be in [0, image_w={image_w}]"
            assert 0 <= y <= image_h, f"keypoint y={y} must be in [0, image_h={image_h}]"


def _valid_person(track_id=1):
    """Return a PersonPose with 17 keypoints and valid bbox."""
    kps = [[10.0 + i, 20.0 + i, 0.9] for i in range(NUM_COCO_KEYPOINTS)]
    return PersonPose(bbox=[0, 0, 100, 200], bbox_conf=0.95, keypoints=kps, track_id=track_id)


def _valid_event(persons=None):
    """Return a FrameEvent with one valid person by default."""
    return FrameEvent(
        frame_number=1,
        timestamp_ms=1000.0,
        image={"width": 640, "height": 480},
        persons=persons or [_valid_person()],
    )


class TestBuildCloudPayload:
    """build_cloud_payload produces schema with event_type, source, frame, persons; keypoints length 17."""

    def test_event_type_is_frame_event(self):
        """Payload has event_type 'frame_event'."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="dev1",
            session_id="sess-uuid",
            model="yolov8m_pose",
            tracking_source="fallback",
        )
        assert payload["event_type"] == EVENT_TYPE_FRAME

    def test_source_has_device_id_session_id_model_tracking_source(self):
        """Payload source contains device_id, session_id, model, tracking_source."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="my-device",
            session_id="uuid-123",
            model="yolov8m_pose",
            tracking_source="metadata",
        )
        assert payload["source"]["device_id"] == "my-device"
        assert payload["source"]["session_id"] == "uuid-123"
        assert payload["source"]["model"] == "yolov8m_pose"
        assert payload["source"]["tracking_source"] == "metadata"

    def test_frame_has_frame_index_ts_unix_ms_image_w_h(self):
        """Payload frame has frame_index, ts_unix_ms, image_w, image_h."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
        )
        assert payload["frame"]["frame_index"] == 1
        assert payload["frame"]["ts_unix_ms"] == 1000.0
        assert payload["frame"]["image_w"] == 640
        assert payload["frame"]["image_h"] == 480

    def test_frame_optional_fps_current_fps_avg(self):
        """Payload frame can include fps_current and fps_avg when provided."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
            fps_current=30.0,
            fps_avg=28.5,
        )
        assert payload["frame"]["fps_current"] == 30.0
        assert payload["frame"]["fps_avg"] == 28.5

    def test_persons_have_track_id_bbox_xyxy_score_keypoints_17(self):
        """Each person has track_id, bbox_xyxy, score, keypoints (length 17), keypoints_format, coords."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
        )
        assert len(payload["persons"]) == 1
        p = payload["persons"][0]
        assert p["track_id"] == 1
        assert p["bbox_xyxy"] == [0, 0, 100, 200]
        assert p["score"] == 0.95
        assert len(p["keypoints"]) == NUM_COCO_KEYPOINTS
        assert p["keypoints_format"] == KEYPOINTS_FORMAT
        assert p["coords"] == COORDS_PIXEL

    def test_keypoints_have_name_x_y_c(self):
        """Each keypoint is {name, x, y, c} with COCO-17 names."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
        )
        kps = payload["persons"][0]["keypoints"]
        assert len(kps) == 17
        assert kps[0]["name"] == "nose"
        assert "x" in kps[0] and "y" in kps[0] and "c" in kps[0]
        assert kps[16]["name"] == "right_ankle"

    def test_multiple_persons(self):
        """Payload can have multiple persons with distinct track_ids."""
        p1 = _valid_person(track_id=1)
        p2 = _valid_person(track_id=2)
        event = _valid_event(persons=[p1, p2])
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
        )
        assert len(payload["persons"]) == 2
        assert payload["persons"][0]["track_id"] == 1
        assert payload["persons"][1]["track_id"] == 2
        assert len(payload["persons"][0]["keypoints"]) == 17
        assert len(payload["persons"][1]["keypoints"]) == 17


class TestPayloadSchemaSanity:
    """Schema sanity: every event has device_id, session_id, frame_index, timestamp; persons list; per-person checks."""

    def test_sanity_required_top_level_fields(self):
        """Payload has device_id, session_id, frame_index, timestamp (ts_unix_ms)."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="dev-001",
            session_id="sess-uuid",
            model="yolov8",
            tracking_source="fallback",
        )
        assert payload["source"]["device_id"] == "dev-001"
        assert payload["source"]["session_id"] == "sess-uuid"
        assert payload["frame"]["frame_index"] == event.frame_number
        assert "ts_unix_ms" in payload["frame"]
        assert payload["frame"]["ts_unix_ms"] == event.timestamp_ms
        assert isinstance(payload["persons"], list)

    def test_sanity_each_person_has_track_id_keypoints_17_coords_format(self):
        """Each person has track_id, keypoints length 17, coords='pixel', keypoints_format='coco17'."""
        event = _valid_event()
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="metadata",
        )
        for p in payload["persons"]:
            assert "track_id" in p
            assert len(p["keypoints"]) == 17
            assert p["coords"] == COORDS_PIXEL
            assert p["keypoints_format"] == KEYPOINTS_FORMAT

    def test_sanity_keypoints_within_image_bounds(self):
        """Keypoint x,y are within [0, image_w] and [0, image_h], or sentinel (-1, -1)."""
        # Keypoints inside 100x100 image
        kps = [[float(i % 100), float(i % 100), 0.9] for i in range(NUM_COCO_KEYPOINTS)]
        person = PersonPose(bbox=[0, 0, 50, 50], bbox_conf=0.9, keypoints=kps, track_id=1)
        event = FrameEvent(
            frame_number=0,
            timestamp_ms=0.0,
            image={"width": 100, "height": 100},
            persons=[person],
        )
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
        )
        assert_payload_schema_sanity(payload, image_w=100, image_h=100)

    def test_sanity_keypoints_sentinel_allowed(self):
        """Sentinel keypoints (-1, -1, 0) are allowed and do not fail bounds check."""
        from src.frame_event import MISSING_KEYPOINT_SENTINEL
        kps = [list(MISSING_KEYPOINT_SENTINEL) for _ in range(NUM_COCO_KEYPOINTS)]
        person = PersonPose(bbox=[0, 0, 10, 10], bbox_conf=0.8, keypoints=kps, track_id=0)
        event = FrameEvent(
            frame_number=1,
            timestamp_ms=100.0,
            image={"width": 640, "height": 480},
            persons=[person],
        )
        payload = build_cloud_payload(
            event,
            device_id="d",
            session_id="s",
            model="m",
            tracking_source="fallback",
        )
        assert_payload_schema_sanity(payload, image_w=640, image_h=480)
        assert all(kp["x"] == -1.0 and kp["y"] == -1.0 for kp in payload["persons"][0]["keypoints"])
