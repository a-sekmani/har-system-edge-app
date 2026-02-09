"""Unit tests for Phase 4 window_schema: keypoints_to_17x3_normalized, WindowPayload, build_window_payload."""

from src.frame_event import NUM_COCO_KEYPOINTS
from src.window_schema import (
    MISSING_KEYPOINT_WINDOW,
    FPS_CLAMP_MIN,
    FPS_CLAMP_MAX,
    keypoints_to_17x3_normalized,
    build_window_payload,
    WindowPayload,
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
