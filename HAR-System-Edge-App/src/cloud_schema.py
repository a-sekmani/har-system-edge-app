"""
Cloud Event schema for Phase 3: unified JSON payload sent to the cloud.

Schema (no images in Phase 3):
- event_type: "frame_event"
- source: device_id, session_id, model, tracking_source
- frame: frame_index, ts_unix_ms, image_w, image_h, fps_current, fps_avg (optional)
- persons: list of { track_id, bbox_xyxy, score, keypoints [ {name, x, y, c} x 17 ], keypoints_format, coords }
- keypoints_format: "coco17"
- coords: "pixel"
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.frame_event import COCO_17_ORDER, FrameEvent, NUM_COCO_KEYPOINTS

if TYPE_CHECKING:
    pass

EVENT_TYPE_FRAME = "frame_event"
KEYPOINTS_FORMAT = "coco17"
COORDS_PIXEL = "pixel"


def build_cloud_payload(
    event: FrameEvent,
    *,
    device_id: str,
    session_id: str,
    model: str,
    tracking_source: str,
    fps_current: Optional[float] = None,
    fps_avg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable Cloud Event payload from a valid FrameEvent.

    Uses COCO_17_ORDER for keypoint names. Each person has track_id, bbox_xyxy [x1,y1,x2,y2],
    score (bbox_conf), and 17 keypoints as {name, x, y, c}. No images are included.
    """
    image_w = event.image.get("width", 0)
    image_h = event.image.get("height", 0)

    frame_block: Dict[str, Any] = {
        "frame_index": event.frame_number,
        "ts_unix_ms": event.timestamp_ms,
        "image_w": int(image_w),
        "image_h": int(image_h),
    }
    if fps_current is not None:
        frame_block["fps_current"] = fps_current
    if fps_avg is not None:
        frame_block["fps_avg"] = fps_avg

    persons_list: List[Dict[str, Any]] = []
    for p in event.persons:
        keypoints_with_names: List[Dict[str, Any]] = []
        for i, kp in enumerate(p.keypoints):
            name = COCO_17_ORDER[i] if i < len(COCO_17_ORDER) else f"keypoint_{i}"
            x = float(kp[0]) if len(kp) > 0 else -1.0
            y = float(kp[1]) if len(kp) > 1 else -1.0
            c = float(kp[2]) if len(kp) > 2 else 0.0
            keypoints_with_names.append({"name": name, "x": x, "y": y, "c": c})

        person_block: Dict[str, Any] = {
            "track_id": p.track_id,
            "bbox_xyxy": list(p.bbox),
            "score": p.bbox_conf,
            "keypoints": keypoints_with_names,
            "keypoints_format": KEYPOINTS_FORMAT,
            "coords": COORDS_PIXEL,
        }
        persons_list.append(person_block)

    payload: Dict[str, Any] = {
        "event_type": EVENT_TYPE_FRAME,
        "source": {
            "device_id": device_id,
            "session_id": session_id,
            "model": model,
            "tracking_source": tracking_source,
        },
        "frame": frame_block,
        "persons": persons_list,
    }
    return payload
