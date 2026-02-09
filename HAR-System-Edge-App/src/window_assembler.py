"""
Phase 4: Window assembler — buffers per (device_id, camera_id, session_id, track_id),
non-overlap policy: when buffer reaches window_size, build WindowPayload and clear.
"""

from collections import deque
from typing import List, Tuple

from src.frame_event import FrameEvent
from src.window_schema import (
    WindowPayload,
    build_window_payload,
    keypoints_to_17x3_normalized,
)


def _buffer_key(device_id: str, camera_id: str, session_id: str, track_id: int) -> Tuple[str, str, str, int]:
    return (device_id, camera_id, session_id, track_id)


class WindowAssembler:
    """
    Manages one buffer per (device_id, camera_id, session_id, track_id).
    Each buffer is a deque of (ts_unix_ms, kp_17x3_normalized).
    When length reaches window_size, build WindowPayload and clear buffer (non-overlap).
    Cap total buffers at window_max_buffers; drop oldest buffer when over.
    """

    def __init__(
        self,
        window_size: int = 30,
        window_stride: int = 30,
        window_max_buffers: int = 50,
    ):
        self.window_size = max(1, window_size)
        self.window_stride = max(1, window_stride)
        self.window_max_buffers = max(1, window_max_buffers)
        # key -> deque of (ts_unix_ms, kp_17x3: list[list[float]])
        self._buffers: dict = {}
        # insertion order for eviction (oldest first)
        self._key_order: List[Tuple[str, str, str, int]] = []

    def _evict_one_if_needed(self) -> None:
        if len(self._buffers) < self.window_max_buffers:
            return
        if not self._key_order:
            return
        oldest_key = self._key_order.pop(0)
        self._buffers.pop(oldest_key, None)

    def _get_or_create_buffer(self, key: Tuple[str, str, str, int]):
        if key in self._buffers:
            return self._buffers[key]
        self._evict_one_if_needed()
        self._buffers[key] = deque(maxlen=self.window_size * 2)
        self._key_order.append(key)
        return self._buffers[key]

    def push_frame(
        self,
        event: FrameEvent,
        device_id: str,
        camera_id: str,
        session_id: str,
    ) -> List[WindowPayload]:
        """
        For each person in event, push (timestamp_ms, kp_17x3_normalized) to that track's buffer.
        When any buffer reaches window_size, build WindowPayload and clear it (non-overlap).
        Returns list of completed windows (0 or more).
        """
        image_w = event.image.get("width", 0) or 0
        image_h = event.image.get("height", 0) or 0
        ts_ms = event.timestamp_ms
        completed: List[WindowPayload] = []

        for person in event.persons:
            key = _buffer_key(device_id, camera_id, session_id, person.track_id)
            kp_17x3 = keypoints_to_17x3_normalized(person.keypoints, image_w, image_h)
            buf = self._get_or_create_buffer(key)
            buf.append((ts_ms, kp_17x3))

            if len(buf) >= self.window_size:
                # Non-overlap: take window_size elements, build payload, clear buffer
                frames_ts_kp = list(buf)[: self.window_size]
                for _ in range(self.window_size):
                    if buf:
                        buf.popleft()
                ts_start = frames_ts_kp[0][0]
                ts_end = frames_ts_kp[-1][0]
                keypoints_frames = [kp for _, kp in frames_ts_kp]
                payload = build_window_payload(
                    device_id=device_id,
                    camera_id=camera_id,
                    session_id=session_id,
                    track_id=person.track_id,
                    ts_start_ms=ts_start,
                    ts_end_ms=ts_end,
                    window_size=self.window_size,
                    keypoints_frames=keypoints_frames,
                )
                completed.append(payload)

        return completed
