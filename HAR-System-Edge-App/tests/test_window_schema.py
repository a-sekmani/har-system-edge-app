"""Unit tests for Phase 4 window_schema: keypoints_to_17x3_normalized, WindowPayload, build_window_payload, created_at ISO 8601."""

import re

from src.frame_event import NUM_COCO_KEYPOINTS
from src.window_schema import (
    MISSING_KEYPOINT_WINDOW,
    FPS_CLAMP_MIN,
    FPS_CLAMP_MAX,
    format_created_at_iso8601_utc,
    keypoints_to_17x3_normalized,
    build_window_payload,
    WindowPayload,
)

# ISO 8601 with Z and 3 decimal places: YYYY-MM-DDTHH:mm:ss.sssZ
CREATED_AT_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)


class TestKeypointsTo17x3Normalized:
    def test_length_17(self):
        kp = [[100.0, 200.0, 0.9] for _ in range(17)]
        out = keypoints_to_17x3_normalized(kp, 640, 480)
        assert len(out) == NUM_COCO_KEYPOINTS

    def test_missing_sentinel(self):
        kp = [[-1.0, -1.0, 0.0] for _ in range(17)]
        out = keypoints_to_17x3_normalized(kp, 640, 480)
        for pt in out:
            assert pt == [0.0, 0.0, 0.0]

    def test_normalization_bounds(self):
        kp = [[320.0, 240.0, 0.8] for _ in range(17)]
        out = keypoints_to_17x3_normalized(kp, 640, 480)
        for pt in out:
            assert 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1 and 0 <= pt[2] <= 1
        assert abs(out[0][0] - 0.5) < 1e-6
        assert abs(out[0][1] - 0.5) < 1e-6

    def test_zero_dimensions(self):
        kp = [[100.0, 200.0, 0.9] for _ in range(17)]
        out = keypoints_to_17x3_normalized(kp, 0, 480)
        assert len(out) == 17
        assert all(pt == [0.0, 0.0, 0.0] for pt in out)

    def test_fewer_than_17_keypoints_pads_with_sentinel(self):
        kp = [[100.0, 200.0, 0.9] for _ in range(10)]
        out = keypoints_to_17x3_normalized(kp, 640, 480)
        assert len(out) == NUM_COCO_KEYPOINTS
        for i in range(10):
            assert out[i] != list(MISSING_KEYPOINT_WINDOW)
        for i in range(10, 17):
            assert out[i] == [0.0, 0.0, 0.0]

    def test_confidence_zero_yields_sentinel(self):
        kp = [[100.0, 200.0, 0.0] for _ in range(17)]
        out = keypoints_to_17x3_normalized(kp, 640, 480)
        assert len(out) == 17
        assert all(pt == [0.0, 0.0, 0.0] for pt in out)

    def test_negative_xy_yields_sentinel(self):
        kp = [[-10.0, 200.0, 0.9] for _ in range(17)]
        out = keypoints_to_17x3_normalized(kp, 640, 480)
        assert all(pt == [0.0, 0.0, 0.0] for pt in out)


class TestFormatCreatedAtIso8601Utc:
    """created_at: ISO 8601 with timezone (UTC Z, 3 decimal places) for edge → cloud."""

    def test_format_ends_with_z(self):
        s = format_created_at_iso8601_utc()
        assert s.endswith("Z")

    def test_format_has_date_time_and_three_decimals(self):
        s = format_created_at_iso8601_utc()
        assert CREATED_AT_PATTERN.match(s), f"expected YYYY-MM-DDTHH:mm:ss.sssZ, got {s!r}"

    def test_format_no_plus_00_00(self):
        s = format_created_at_iso8601_utc()
        assert "+00:00" not in s

    def test_format_parseable_as_iso8601(self):
        from datetime import datetime
        s = format_created_at_iso8601_utc()
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
        assert parsed is not None


class TestBuildWindowPayload:
    def test_shape(self):
        frames = [[[0.5, 0.5, 0.8] for _ in range(17)] for _ in range(30)]
        w = build_window_payload(
            "dev", "cam", "sess", 1, 1000.0, 2000.0, 30, frames
        )
        assert w.window_size == 30
        assert len(w.keypoints) == 30
        assert len(w.keypoints[0]) == 17
        assert len(w.keypoints[0][0]) == 3

    def test_fps_from_timestamps(self):
        frames = [[[0.0, 0.0, 0.0] for _ in range(17)] for _ in range(30)]
        # ts_end - ts_start = 1000 ms, 29 intervals -> ~29 fps
        w = build_window_payload(
            "d", "c", "s", 1, 0.0, 1000.0, 30, frames
        )
        assert FPS_CLAMP_MIN <= w.fps <= FPS_CLAMP_MAX

    def test_to_dict_serializable(self):
        frames = [[[0.1, 0.2, 0.3] for _ in range(17)] for _ in range(30)]
        w = build_window_payload(
            "d", "c", "s", 1, 0.0, 100.0, 30, frames
        )
        d = w.to_dict()
        assert d["device_id"] == "d"
        assert d["track_id"] == 1
        assert d["window_size"] == 30
        assert len(d["keypoints"]) == 30
        assert "person" not in d

    def test_to_dict_with_person_optional(self):
        """WindowPayload.person is optional; when set it appears in to_dict()."""
        frames = [[[0.0, 0.0, 0.0] for _ in range(17)] for _ in range(30)]
        w = build_window_payload("d", "c", "s", 1, 0.0, 100.0, 30, frames)
        assert w.person is None
        d = w.to_dict()
        assert "person" not in d
        w2 = WindowPayload(
            id=w.id, created_at=w.created_at, device_id=w.device_id, camera_id=w.camera_id,
            session_id=w.session_id, track_id=w.track_id, ts_start_ms=w.ts_start_ms, ts_end_ms=w.ts_end_ms,
            fps=w.fps, window_size=w.window_size, keypoints=w.keypoints,
            person={"person_id": "uuid-1", "name": "Ahmad", "face_conf": 0.78, "source": "edge_face", "verified_at_ms": 1234567890},
        )
        d2 = w2.to_dict()
        assert "person" in d2
        assert d2["person"]["person_id"] == "uuid-1"
        assert d2["person"]["name"] == "Ahmad"
        assert d2["person"]["face_conf"] == 0.78
        assert d2["person"]["source"] == "edge_face"
        assert d2["person"]["verified_at_ms"] == 1234567890

    def test_created_at_iso8601_utc_z(self):
        """build_window_payload sets created_at via format_created_at_iso8601_utc (YYYY-MM-DDTHH:mm:ss.sssZ)."""
        frames = [[[0.5, 0.5, 0.9] for _ in range(17)] for _ in range(30)]
        w = build_window_payload("d", "c", "s", 1, 0.0, 1000.0, 30, frames)
        assert CREATED_AT_PATTERN.match(w.created_at), w.created_at
        assert w.created_at.endswith("Z")
        assert "+00:00" not in w.created_at
        parts = w.created_at[:-1].split(".")
        assert len(parts) == 2 and len(parts[1]) == 3 and parts[0].count("T") == 1
        d = w.to_dict()
        assert d["created_at"] == w.created_at

    def test_fps_clamped_when_delta_zero_or_negative(self):
        frames = [[[0.0, 0.0, 0.0] for _ in range(17)] for _ in range(30)]
        w = build_window_payload("d", "c", "s", 1, 1000.0, 1000.0, 30, frames)
        assert w.fps == FPS_CLAMP_MIN

    def test_fps_clamped_to_max(self):
        frames = [[[0.0, 0.0, 0.0] for _ in range(17)] for _ in range(30)]
        w = build_window_payload("d", "c", "s", 1, 0.0, 50.0, 30, frames)
        assert w.fps <= FPS_CLAMP_MAX and w.fps >= FPS_CLAMP_MIN

    def test_to_dict_rounds_ts_to_int(self):
        frames = [[[0.0, 0.0, 0.0] for _ in range(17)] for _ in range(30)]
        w = build_window_payload("d", "c", "s", 1, 1000.4, 2000.6, 30, frames)
        d = w.to_dict()
        assert isinstance(d["ts_start_ms"], int)
        assert isinstance(d["ts_end_ms"], int)
        assert d["ts_start_ms"] == 1000
        assert d["ts_end_ms"] == 2001
