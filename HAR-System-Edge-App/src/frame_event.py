"""
Frame Event model: raw pose data per frame (Phase 1 schema + Phase 2 track_id).

Schema:
- COCO-17 keypoint order (single source of truth).
- Missing keypoint sentinel: [-1, -1, 0.0] (same type for all keypoints; no null).
- PersonPose: bbox [x1, y1, x2, y2] pixels, bbox_conf, 17 keypoints [x, y, c], track_id (int).
- FrameEvent: frame_number, timestamp_ms, image {width, height}, persons.
All structures are JSON-serializable. Validation does not depend on track_id.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# COCO-17 keypoint order (single source of truth)
# Indices 0-16: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
# right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip,
# right_hip, left_knee, right_knee, left_ankle, right_ankle.
# ---------------------------------------------------------------------------
COCO_17_ORDER = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NUM_COCO_KEYPOINTS = 17

# Missing or invalid keypoint: [x, y, confidence] with no null; same type for all.
MISSING_KEYPOINT_SENTINEL: List[float] = [-1.0, -1.0, 0.0]

# Task B: debug sample (raw rel + pixel) for first person when store_raw_sample=True
_debug_raw_keypoints_sample: Optional[List[tuple]] = None


def _keypoint_sentinel() -> List[float]:
    """Return a copy of the sentinel (so mutations don't affect constant)."""
    return list(MISSING_KEYPOINT_SENTINEL)


# Sentinel for unknown track ID when metadata/fallback fails. Valid output should use real id when tracking is enabled.
TRACK_ID_UNKNOWN = -1

# ---------------------------------------------------------------------------
# PersonPose
# ---------------------------------------------------------------------------
@dataclass
class PersonPose:
    """One person's pose: bbox in pixels, detection confidence, 17 keypoints [x, y, c], track_id.
    track_id: int; use TRACK_ID_UNKNOWN (-1) only when tracking cannot assign an id. Valid output should have real id when tracking is enabled."""

    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    bbox_conf: float
    keypoints: List[List[float]]  # 17 elements, each [x, y, c]; use MISSING_KEYPOINT_SENTINEL for missing
    track_id: int = TRACK_ID_UNKNOWN  # required in output; -1 = unknown when fallback fails

    def __post_init__(self) -> None:
        if len(self.keypoints) != NUM_COCO_KEYPOINTS:
            raise ValueError(f"keypoints must have length {NUM_COCO_KEYPOINTS}, got {len(self.keypoints)}")
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have length 4 [x1, y1, x2, y2], got {len(self.bbox)}")

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return {
            "bbox": list(self.bbox),
            "bbox_conf": self.bbox_conf,
            "keypoints": [list(kp) for kp in self.keypoints],
            "track_id": self.track_id,
        }

    @classmethod
    def from_hailo_detection(
        cls,
        detection: Any,
        image_width: int,
        image_height: int,
        store_raw_sample: bool = False,
        track_id: Optional[int] = None,
    ) -> "PersonPose":
        """
        Build PersonPose from a hailo detection (label "person") and image dimensions.
        Bbox from detection is normalized (xmin, ymin, width, height); converted to pixel [x1,y1,x2,y2].
        Keypoints from detection.get_objects_typed(hailo.HAILO_LANDMARKS); points relative to bbox:
          x_px = (point.x() * bbox.width() + bbox.xmin()) * image_width
        If Python point API exposes confidence (e.g. point.confidence()), use it; else 1.0 for present, 0.0 for sentinel.
        """
        try:
            import hailo
        except ImportError:
            raise RuntimeError("hailo is required for from_hailo_detection") from None

        bbox_norm = detection.get_bbox()
        # Normalized: xmin, ymin, width, height (0-1)
        xmin = max(0.0, min(1.0, bbox_norm.xmin()))
        ymin = max(0.0, min(1.0, bbox_norm.ymin()))
        w = max(0.0, min(1.0 - xmin, bbox_norm.width()))
        h = max(0.0, min(1.0 - ymin, bbox_norm.height()))
        x1 = xmin * image_width
        y1 = ymin * image_height
        x2 = (xmin + w) * image_width
        y2 = (ymin + h) * image_height
        x1 = max(0, min(image_width, x1))
        y1 = max(0, min(image_height, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        bbox_px = [float(x1), float(y1), float(x2), float(y2)]
        bbox_conf = float(detection.get_confidence())

        keypoints: List[List[float]] = []
        landmarks_list = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        points_list: List[Any] = []
        if landmarks_list:
            points_list = landmarks_list[0].get_points()

        def point_confidence(pt: Any) -> float:
            if hasattr(pt, "confidence") and callable(getattr(pt, "confidence")):
                return float(pt.confidence())
            if hasattr(pt, "confidence") and not callable(getattr(pt, "confidence")):
                return float(pt.confidence)
            return 1.0

        raw_sample: List[tuple] = []  # (x_rel, y_rel, c, x_px, y_px) for Task B
        for i in range(NUM_COCO_KEYPOINTS):
            if i < len(points_list):
                pt = points_list[i]
                try:
                    x_rel = float(pt.x())
                    y_rel = float(pt.y())
                except (AttributeError, TypeError):
                    keypoints.append(_keypoint_sentinel())
                    continue
                x_px = (x_rel * w + xmin) * image_width
                y_px = (y_rel * h + ymin) * image_height
                # Task C (Policy 1): clamp to image bounds so validation does not reject
                x_px = max(0.0, min(float(image_width), x_px))
                y_px = max(0.0, min(float(image_height), y_px))
                conf = point_confidence(pt)
                keypoints.append([x_px, y_px, conf])
                if store_raw_sample and conf > 0 and len(raw_sample) < 3:
                    raw_sample.append((x_rel, y_rel, conf, x_px, y_px))
            else:
                keypoints.append(_keypoint_sentinel())

        if store_raw_sample and raw_sample:
            global _debug_raw_keypoints_sample
            _debug_raw_keypoints_sample = raw_sample

        tid = track_id if track_id is not None else TRACK_ID_UNKNOWN
        return cls(bbox=bbox_px, bbox_conf=bbox_conf, keypoints=keypoints, track_id=tid)


# ---------------------------------------------------------------------------
# FrameEvent
# ---------------------------------------------------------------------------
@dataclass
class FrameEvent:
    """One frame: frame number, timestamp, image size, list of person poses."""

    frame_number: int
    timestamp_ms: float
    image: dict  # {"width": int, "height": int}
    persons: List[PersonPose] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "image": dict(self.image),
            "persons": [p.to_dict() for p in self.persons],
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_frame_event(event: FrameEvent) -> Tuple[bool, List[str]]:
    """
    Validate a FrameEvent. Returns (valid, list of error messages).
    - Each person: len(keypoints) == 17.
    - For each keypoint with c > 0: x in [0, image_width], y in [0, image_height] (tolerance 1e-6).
    - Each bbox: length 4; x1 <= x2, y1 <= y2; non-negative.
    """
    errors: List[str] = []
    w = event.image.get("width")
    h = event.image.get("height")
    if w is None or h is None:
        errors.append("image must have 'width' and 'height'")
    else:
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            errors.append("image width and height must be positive")

    for i, person in enumerate(event.persons):
        if len(person.keypoints) != NUM_COCO_KEYPOINTS:
            errors.append(f"person[{i}]: keypoints length must be {NUM_COCO_KEYPOINTS}, got {len(person.keypoints)}")
        for j, kp in enumerate(person.keypoints):
            if len(kp) != 3:
                errors.append(f"person[{i}] keypoint[{j}]: must be [x, y, c], got length {len(kp)}")
                continue
            x, y, c = kp[0], kp[1], kp[2]
            if c > 0 and w is not None and h is not None:
                if not (0 <= x <= w):
                    errors.append(f"person[{i}] keypoint[{j}]: x={x} out of [0, {w}]")
                if not (0 <= y <= h):
                    errors.append(f"person[{i}] keypoint[{j}]: y={y} out of [0, {h}]")
        if len(person.bbox) != 4:
            errors.append(f"person[{i}]: bbox must have length 4, got {len(person.bbox)}")
        else:
            x1, y1, x2, y2 = person.bbox[0], person.bbox[1], person.bbox[2], person.bbox[3]
            if x1 < 0 or y1 < 0:
                errors.append(f"person[{i}]: bbox x1,y1 must be non-negative")
            if x1 > x2:
                errors.append(f"person[{i}]: bbox x1 must be <= x2")
            if y1 > y2:
                errors.append(f"person[{i}]: bbox y1 must be <= y2")

    return (len(errors) == 0, errors)
