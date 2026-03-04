"""
Bind face identity to pose track_id: IoU association, voting, TTL, recheck interval.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from src.face.schemas import FaceDetection, FaceIdentity, PersonAttachment

_LOG = logging.getLogger(__name__)

DEFAULT_IOU_THRESHOLD = 0.2
PERSON_SOURCE_EDGE_FACE = "edge_face"


def _iou_bbox(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU of two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _face_center_in_pose(face_bbox: Tuple[float, float, float, float], pose_bbox: Tuple[float, float, float, float]) -> bool:
    """True if the center of the face bbox lies inside the pose bbox (body contains face)."""
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = pose_bbox
    cx = (fx1 + fx2) * 0.5
    cy = (fy1 + fy2) * 0.5
    return px1 <= cx <= px2 and py1 <= cy <= py2


class TrackerBinding:
    """
    Maps pose track_id to FaceIdentity. Updated each frame from pose bboxes and face detections + matches.
    TTL and recheck interval applied when reading identity.
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        track_ttl_s: float = 10.0,
        recheck_every_s: float = 2.0,
        sim_threshold: float = 0.45,
        min_votes_stable: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.track_ttl_s = track_ttl_s
        self.recheck_every_s = recheck_every_s
        self.sim_threshold = sim_threshold
        self.min_votes_stable = min_votes_stable
        # track_id -> FaceIdentity
        self._identities: Dict[int, FaceIdentity] = {}

    def update(
        self,
        pose_detections: List[Tuple[int, Tuple[float, float, float, float]]],  # (track_id, bbox_xyxy)
        face_detections: List[Tuple[FaceDetection, Optional[Tuple[str, str, float]]]],  # (face, (person_id, name, score) or None)
        now_ts: Optional[float] = None,
    ) -> None:
        """
        Associate each face to best matching pose by IoU; update that track's identity with match result.
        pose_detections: from current frame (track_id, bbox_xyxy in pixels).
        face_detections: list of (FaceDetection, match_result). match_result is (person_id, name, score) or None.
        """
        if now_ts is None:
            now_ts = time.time()
        # For each face, find best pose: first by IoU; if none (face small inside body), by face center inside pose bbox
        for face_det, match_result in face_detections:
            face_bbox = face_det.bbox_xyxy
            best_track_id: Optional[int] = None
            best_iou = self.iou_threshold
            for track_id, pose_bbox in pose_detections:
                iou = _iou_bbox(face_bbox, pose_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is None:
                for track_id, pose_bbox in pose_detections:
                    if _face_center_in_pose(face_bbox, pose_bbox):
                        best_track_id = track_id
                        break
            if best_track_id is None:
                continue
            # Update identity for this track
            if best_track_id not in self._identities:
                self._identities[best_track_id] = FaceIdentity(
                    person_id=None,
                    person_name=None,
                    score=0.0,
                    last_seen_ts=now_ts,
                    last_verified_ts=now_ts,
                    votes={},
                )
            ident = self._identities[best_track_id]
            ident.last_seen_ts = now_ts
            if match_result is not None:
                person_id, name, score = match_result
                ident.last_verified_ts = now_ts
                ident.votes[person_id] = ident.votes.get(person_id, 0) + 1
                # Promote to stable if this person_id has enough votes and best score
                if ident.votes.get(person_id, 0) >= self.min_votes_stable and score >= self.sim_threshold:
                    ident.person_id = person_id
                    ident.person_name = name
                    ident.score = score
        # TTL cleanup: remove identities not seen for track_ttl_s
        to_remove = [
            tid for tid, ident in self._identities.items()
            if (now_ts - ident.last_seen_ts) > self.track_ttl_s
        ]
        for tid in to_remove:
            del self._identities[tid]

    def get_identity(
        self,
        track_id: int,
        now_ts: Optional[float] = None,
        attach_policy: str = "auto",
    ) -> Optional[PersonAttachment]:
        """
        Get person attachment for this track_id when building a window.
        attach_policy: auto = only if known and score >= threshold; never = None; always = include even unknown.
        Returns None if policy is 'never' or (auto and no valid identity).
        """
        if now_ts is None:
            now_ts = time.time()
        ident = self._identities.get(track_id)
        if ident is None:
            if attach_policy == "always":
                return PersonAttachment(
                    person_id=None,
                    name=None,
                    face_conf=0.0,
                    source=PERSON_SOURCE_EDGE_FACE,
                    verified_at_ms=None,
                )
            return None
        # TTL: treat as unknown if too old
        if (now_ts - ident.last_seen_ts) > self.track_ttl_s:
            if attach_policy == "always":
                return PersonAttachment(
                    person_id=None,
                    name=None,
                    face_conf=0.0,
                    source=PERSON_SOURCE_EDGE_FACE,
                    verified_at_ms=None,
                )
            return None
        if attach_policy == "never":
            return None
        if attach_policy == "auto":
            if ident.person_id is None or ident.score < self.sim_threshold:
                return None
        verified_ms = int(ident.last_verified_ts * 1000) if ident.last_verified_ts else None
        return PersonAttachment(
            person_id=ident.person_id,
            name=ident.person_name,
            face_conf=ident.score,
            source=PERSON_SOURCE_EDGE_FACE,
            verified_at_ms=verified_ms,
        )
