"""
Phase 2 tracking: config, metadata track_id extraction, fallback IoU-based tracker.

- Config: tracking_enabled, tracking_source (metadata | fallback), max_missing_frames,
  iou_match_threshold, max_track_age_seconds, min_bbox_area, debug_first_n_switches.
- Metadata: from detection.get_objects_typed(hailo.HAILO_UNIQUE_ID); if len==1 use track[0].get_id().
- Fallback: match detections to active tracks by IoU; greedy assignment; new id on no match;
  expire tracks after max_missing_frames (or max_track_age_seconds); cleanup to bound track count.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import time

# Same as frame_event.TRACK_ID_UNKNOWN; used for filtered/small bboxes
TRACK_ID_UNKNOWN = -1


@dataclass
class TrackingConfig:
    """Tracking configuration."""

    tracking_enabled: bool = True
    tracking_source: str = "metadata"  # "metadata" | "fallback"
    max_missing_frames: int = 15
    iou_match_threshold: float = 0.3
    max_track_age_seconds: Optional[float] = None
    min_bbox_area: float = 0.0
    min_bbox_height: Optional[float] = None  # pixels; exclude detections with bbox height below this
    min_pose_confidence: Optional[float] = None  # 0-1; exclude detections with avg keypoint conf below this
    debug_first_n_switches: int = 0
    debug_first_n_created: int = 0
    debug_first_n_ended: int = 0


def get_metadata_track_id(detection: Any) -> Optional[int]:
    """
    Get track ID from hailo detection metadata if available.
    From detection.get_objects_typed(hailo.HAILO_UNIQUE_ID); if len==1 use track[0].get_id().
    Returns None if not available or invalid (e.g. id <= 0).
    """
    try:
        import hailo
    except ImportError:
        return None
    objs = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
    if not objs or len(objs) != 1:
        return None
    tid = objs[0].get_id()
    if tid is None or (isinstance(tid, (int, float)) and int(tid) <= 0):
        return None
    return int(tid)


def _bbox_area(bbox: List[float]) -> float:
    """bbox [x1,y1,x2,y2]; return (x2-x1)*(y2-y1)."""
    if len(bbox) != 4:
        return 0.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return max(0.0, w * h)


def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    """Intersection over union of two bboxes [x1,y1,x2,y2]."""
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return 0.0
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _center(bbox: List[float]) -> Tuple[float, float]:
    """Return (cx, cy) center of bbox [x1, y1, x2, y2]; (0, 0) if invalid."""
    if len(bbox) != 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


@dataclass
class _Track:
    """Single track state."""

    track_id: int
    last_bbox: List[float]
    last_frame_index: int
    last_seen_time: float
    missing_frames: int = 0


class FallbackTracker:
    """
    IoU-based fallback tracker. Match detections to active tracks by IoU (greedy assignment).
    New detection with no match -> new track. Tracks not updated -> missing; after max_missing_frames
    (or max_track_age_seconds) remove and count as tracks_ended. Re-appearing detection gets new id
    (no reuse of expired id in this implementation).
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._next_id = 1
        self._tracks: List[_Track] = []
        self._tracks_ended_this_frame = 0
        self._new_tracks_this_frame = 0

    def update(
        self,
        bboxes: List[List[float]],
        frame_index: int,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[int], int, int]:
        """
        Assign track_id to each bbox. Bboxes below min_bbox_area get track_id -1 (unknown).
        Returns (list of track_id per bbox, same order as input; -1 for filtered),
                new_tracks_created_this_frame,
                tracks_ended_this_frame.
        """
        now = timestamp if timestamp is not None else time.time()
        self._tracks_ended_this_frame = 0
        self._new_tracks_this_frame = 0

        # Filter by min_bbox_area: kept get real ids, others get TRACK_ID_UNKNOWN
        kept: List[Tuple[int, List[float]]] = []
        for i, bbox in enumerate(bboxes):
            if _bbox_area(bbox) >= self.config.min_bbox_area:
                kept.append((i, bbox))

        # Output: one id per input bbox; TRACK_ID_UNKNOWN for not kept (below min_bbox_area)
        result_ids: List[int] = [TRACK_ID_UNKNOWN] * len(bboxes)
        if not kept:
            for t in self._tracks:
                t.missing_frames += 1
            self._mark_missing_and_expire(frame_index, now)
            return result_ids, 0, self._tracks_ended_this_frame

        det_bboxes = [bbox for _, bbox in kept]
        kept_indices = [i for i, _ in kept]

        # Greedy assignment
        assigned_track: List[Optional[int]] = [None] * len(det_bboxes)
        used_track_idx: set = set()

        for det_idx, dbox in enumerate(det_bboxes):
            best_iou = self.config.iou_match_threshold
            best_track_idx = -1
            for ti, t in enumerate(self._tracks):
                if ti in used_track_idx:
                    continue
                iou = _iou(t.last_bbox, dbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = ti
            if best_track_idx >= 0:
                used_track_idx.add(best_track_idx)
                t = self._tracks[best_track_idx]
                assigned_track[det_idx] = t.track_id
                t.last_bbox = list(dbox)
                t.last_frame_index = frame_index
                t.last_seen_time = now
                t.missing_frames = 0

        for det_idx, dbox in enumerate(det_bboxes):
            out_idx = kept_indices[det_idx]
            if assigned_track[det_idx] is not None:
                result_ids[out_idx] = assigned_track[det_idx]
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracks.append(
                    _Track(
                        track_id=new_id,
                        last_bbox=list(dbox),
                        last_frame_index=frame_index,
                        last_seen_time=now,
                        missing_frames=0,
                    )
                )
                self._new_tracks_this_frame += 1
                result_ids[out_idx] = new_id

        for ti in range(len(self._tracks)):
            if ti not in used_track_idx:
                self._tracks[ti].missing_frames += 1
        self._mark_missing_and_expire(frame_index, now)

        return result_ids, self._new_tracks_this_frame, self._tracks_ended_this_frame

    def _mark_missing_and_expire(self, frame_index: int, now: float) -> None:
        """Increment missing_frames for tracks not updated this frame; remove if over limit."""
        max_missing = self.config.max_missing_frames
        max_age = self.config.max_track_age_seconds
        to_remove = []
        for ti, t in enumerate(self._tracks):
            if t.missing_frames > 0:
                if t.missing_frames >= max_missing:
                    to_remove.append(ti)
                    self._tracks_ended_this_frame += 1
                elif max_age is not None and (now - t.last_seen_time) >= max_age:
                    to_remove.append(ti)
                    self._tracks_ended_this_frame += 1
        for ti in reversed(to_remove):
            self._tracks.pop(ti)

    def get_tracks_ended_count(self) -> int:
        """Return count of tracks ended in last update call."""
        return self._tracks_ended_this_frame

    def get_new_tracks_count(self) -> int:
        """Return count of new tracks created in last update call."""
        return self._new_tracks_this_frame

    def unique_ids_count(self) -> int:
        """Total distinct track ids ever created (upper bound; includes ended)."""
        return self._next_id - 1
