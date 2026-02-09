"""Unit tests for Phase 4 WindowAssembler."""

from src.frame_event import FrameEvent, PersonPose
from src.window_assembler import WindowAssembler


def _kp17(x, y, c=1.0):
    return [[float(x), float(y), float(c)] for _ in range(17)]


def _event(frame_num, ts_ms, track_id, x=100, y=200):
    return FrameEvent(
        frame_number=frame_num,
        timestamp_ms=ts_ms,
        image={"width": 640, "height": 480},
        persons=[
            PersonPose(
                bbox=[0, 0, 100, 100],
                bbox_conf=0.9,
                keypoints=_kp17(x, y),
                track_id=track_id,
            )
        ],
    )


class TestWindowAssembler:
    def test_non_overlap_completes_after_window_size(self):
        wa = WindowAssembler(window_size=3, window_stride=3, window_max_buffers=10)
        assert wa.push_frame(_event(1, 1000.0, 1), "d", "c", "s") == []
        assert wa.push_frame(_event(2, 1100.0, 1), "d", "c", "s") == []
        completed = wa.push_frame(_event(3, 1200.0, 1), "d", "c", "s")
        assert len(completed) == 1
        assert completed[0].window_size == 3
        assert len(completed[0].keypoints) == 3
        assert completed[0].track_id == 1
        assert completed[0].ts_start_ms == 1000.0
        assert completed[0].ts_end_ms == 1200.0

    def test_per_track_buffers(self):
        wa = WindowAssembler(window_size=2, window_stride=2, window_max_buffers=10)
        wa.push_frame(_event(1, 1000.0, 1), "d", "c", "s")
        completed = wa.push_frame(_event(2, 1100.0, 1), "d", "c", "s")
        assert len(completed) == 1
        assert completed[0].track_id == 1
        wa.push_frame(_event(3, 1200.0, 2), "d", "c", "s")
        completed2 = wa.push_frame(_event(4, 1300.0, 2), "d", "c", "s")
        assert len(completed2) == 1
        assert completed2[0].track_id == 2

    def test_max_buffers_eviction(self):
        wa = WindowAssembler(window_size=10, window_stride=10, window_max_buffers=2)
        for i in range(5):
            wa.push_frame(_event(i, 1000.0 + i, i), "d", "c", "s")
        assert len(wa._buffers) <= 2
