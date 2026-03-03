"""
Phase 4: Window payload schema for cloud ingest.

- WindowPayload: JSON-serializable window (id, created_at, device_id, camera_id, session_id,
  track_id, ts_start_ms, ts_end_ms, fps, window_size, keypoints [T][17][3] normalized 0..1).
- created_at: ISO 8601 with timezone (recommended: UTC with Z and 3 decimal places, e.g. 2026-02-24T11:32:05.123Z).
- Missing or invalid keypoints sentinel: [0.0, 0.0, 0.0].
- keypoints_to_17x3_normalized(): convert PersonPose.keypoints (pixel [x,y,c]) to [17][3] normalized.
"""

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.frame_event import NUM_COCO_KEYPOINTS, PersonPose

# Missing/invalid keypoint in window format (Phase 4: use 0,0,0 not -1).
MISSING_KEYPOINT_WINDOW: List[float] = [0.0, 0.0, 0.0]

FPS_CLAMP_MIN = 1.0
FPS_CLAMP_MAX = 120.0

# ISO 8601: YYYY-MM-DDTHH:mm:ss[.sss](Z|+00:00|±HH:mm). Edge must send one date-time with timezone.
CREATED_AT_FORMAT_UTC_Z = "%Y-%m-%dT%H:%M:%S.%f"  # trim to 3 decimals and append "Z"


def format_created_at_iso8601_utc() -> str:
    """
    Return current UTC time as ISO 8601 string with Z and 3 decimal places (milliseconds).
    Format: YYYY-MM-DDTHH:mm:ss.sssZ (e.g. 2026-02-24T11:32:05.123Z).
    Use when creating a window or at send time so the cloud shows correct Date/Time in Recent Windows.
    """
    now = datetime.now(timezone.utc)
    return now.strftime(CREATED_AT_FORMAT_UTC_Z)[:-3] + "Z"


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Return float or default if NaN/Inf/invalid."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return default


def keypoints_to_17x3_normalized(
    keypoints: List[List[float]],
    image_w: int,
    image_h: int,
) -> List[List[float]]:
    """
    Convert PersonPose keypoints (17 x [x, y, c] in pixels) to [17][3] normalized.
    - x_norm = clamp(x / image_w, 0, 1), same for y; c = clamp(c, 0, 1).
    - If x < 0 or y < 0 or c == 0 or NaN/Inf -> [0.0, 0.0, 0.0].
    - Returns exactly 17 keypoints in COCO-17 order (input is already ordered).
    """
    if image_w <= 0 or image_h <= 0:
        return [list(MISSING_KEYPOINT_WINDOW) for _ in range(NUM_COCO_KEYPOINTS)]
    out: List[List[float]] = []
    for i in range(NUM_COCO_KEYPOINTS):
        if i >= len(keypoints):
            out.append(list(MISSING_KEYPOINT_WINDOW))
            continue
        kp = keypoints[i]
        x = _safe_float(kp[0] if len(kp) > 0 else -1.0, -1.0)
        y = _safe_float(kp[1] if len(kp) > 1 else -1.0, -1.0)
        c = _safe_float(kp[2] if len(kp) > 2 else 0.0, 0.0)
        if x < 0 or y < 0 or c <= 0:
            out.append(list(MISSING_KEYPOINT_WINDOW))
            continue
        x_norm = max(0.0, min(1.0, x / image_w))
        y_norm = max(0.0, min(1.0, y / image_h))
        c_clamp = max(0.0, min(1.0, c))
        out.append([x_norm, y_norm, c_clamp])
    return out


@dataclass
class WindowPayload:
    """
    One window ready for POST /v1/windows/ingest.
    keypoints: [T][17][3] with T = window_size (e.g. 30), normalized 0..1.
    person: optional attachment from face recognition (person_id, name, face_conf, source, verified_at_ms).
    """

    id: str  # UUID
    created_at: str  # ISO UTC
    device_id: str
    camera_id: str
    session_id: str
    track_id: int
    ts_start_ms: float
    ts_end_ms: float
    fps: float
    window_size: int
    keypoints: List[List[List[float]]]  # [T][17][3]
    person: Optional[Dict[str, Any]] = None  # from face recognition when available

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict for POST body. ts_start_ms/ts_end_ms as int for cloud schema."""
        out = {
            "id": self.id,
            "created_at": self.created_at,
            "device_id": self.device_id,
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "track_id": int(self.track_id),
            "ts_start_ms": int(round(self.ts_start_ms)),
            "ts_end_ms": int(round(self.ts_end_ms)),
            "fps": float(self.fps),
            "window_size": int(self.window_size),
            "keypoints": [[list(pt) for pt in frame] for frame in self.keypoints],
        }
        if self.person is not None:
            out["person"] = self.person
        return out


def build_window_payload(
    device_id: str,
    camera_id: str,
    session_id: str,
    track_id: int,
    ts_start_ms: float,
    ts_end_ms: float,
    window_size: int,
    keypoints_frames: List[List[List[float]]],  # [T][17][3] already normalized
) -> WindowPayload:
    """
    Build WindowPayload from assembled keypoints.
    fps = (window_size - 1) * 1000 / (ts_end_ms - ts_start_ms) when delta > 0, else clamped.
    """
    delta_ms = ts_end_ms - ts_start_ms
    if delta_ms > 0:
        fps = (window_size - 1) * 1000.0 / delta_ms
        fps = max(FPS_CLAMP_MIN, min(FPS_CLAMP_MAX, fps))
    else:
        fps = FPS_CLAMP_MIN
    created_at_str = format_created_at_iso8601_utc()
    return WindowPayload(
        id=str(uuid.uuid4()),
        created_at=created_at_str,
        device_id=device_id,
        camera_id=camera_id,
        session_id=session_id,
        track_id=track_id,
        ts_start_ms=ts_start_ms,
        ts_end_ms=ts_end_ms,
        fps=fps,
        window_size=window_size,
        keypoints=keypoints_frames,
    )
