"""
Skeleton Exporter: Export COCO-17 keypoints from videos to JSONL files.

This module provides functionality to extract pose skeletons from NTU RGB+D videos
using the same Hailo pipeline as the edge app, saving them in a format suitable
for training HAR models.

Output format (JSONL):
- Line 1: meta object with video info
- Lines 2+: frame objects with keypoints

Keypoints are normalized to [0,1] and follow COCO-17 order.

=== CONVENTIONS (FIXED - DO NOT CHANGE) ===

1. frame_index: starts from 0 (standard convention)

2. ts_unix_ms: integer milliseconds (not float)

3. Missing/undetected keypoints: [0.0, 0.0, 0.0]
   - This is consistent with window_schema.MISSING_KEYPOINT_WINDOW
   - DO NOT use [-1, -1, 0] or any other sentinel

4. Normalization formula:
   - x_norm = x_pixel / image_w (clamped to [0, 1])
   - y_norm = y_pixel / image_h (clamped to [0, 1])
   - Implemented in window_schema.keypoints_to_17x3_normalized()

5. Field names (unified with HAR-WindowNet and cloud):
   - ts_unix_ms (not ts_ms)
   - persons (not people)
   - skeleton_format: "coco17"
   - coords: "normalized"

6. schema_version: integer version for future compatibility
   - Current version: 1

7. Each frame includes coords and skeleton_format for context
   (allows partial file reading without losing context)

8. mean_conf per frame for quick quality filtering
"""

# Schema version - increment when format changes
SKELETON_SCHEMA_VERSION = 1

import json
import os
import re
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from src.frame_event import NUM_COCO_KEYPOINTS, COCO_17_ORDER
from src.window_schema import keypoints_to_17x3_normalized


def extract_action_from_filename(filename: str) -> str:
    """
    Extract action ID from NTU RGB+D filename.
    
    NTU format: S001C001P001R001A009_rgb.avi
    Returns: "A009" or "unknown" if not found.
    """
    match = re.search(r'A(\d{3})', filename)
    if match:
        return f"A{match.group(1)}"
    return "unknown"


@dataclass
class ExportStats:
    """Statistics for a single video export."""
    video_name: str
    action_id: str
    frames_total: int
    frames_with_people: int
    mean_conf: float
    output_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "action_id": self.action_id,
            "frames_total": self.frames_total,
            "frames_with_people": self.frames_with_people,
            "mean_conf": round(self.mean_conf, 4),
            "output_path": self.output_path,
        }


class SkeletonExporter:
    """
    Exports skeleton keypoints from video frames to JSONL files.
    
    Usage:
        exporter = SkeletonExporter(output_dir="/data/skeletons", format="jsonl")
        exporter.start_video("video.avi", "A009", 30.0, 1920, 1080)
        for frame in frames:
            exporter.write_frame(frame_index, ts_unix_ms, persons, image_w, image_h)
        stats = exporter.finish_video()
    
    Output format:
        - meta line: video info, device_id, camera_id, session_id
        - frame lines: frame_index, ts_unix_ms (int), persons list
        - keypoints: normalized [0,1], COCO-17 order
    """
    
    def __init__(self, output_dir: str, format: str = "jsonl",
                 device_id: Optional[str] = None,
                 camera_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.format = format
        self._device_id = device_id or os.environ.get("DEVICE_ID") or self._get_hostname()
        self._camera_id = camera_id or os.environ.get("CAMERA_ID") or "default"
        self._session_id = session_id or str(uuid.uuid4())
        self._file: Optional[TextIO] = None
        self._video_name: str = ""
        self._action_id: str = ""
        self._output_path: Optional[Path] = None
        self._frame_count: int = 0
        self._frames_with_people: int = 0
        self._conf_sum: float = 0.0
        self._conf_count: int = 0
        self._fps: float = 30.0
        self._image_w: int = 0
        self._image_h: int = 0
    
    @staticmethod
    def _get_hostname() -> str:
        """Get hostname for device_id fallback."""
        try:
            return socket.gethostname() or "edge-device"
        except Exception:
            return "edge-device"
    
    def start_video(
        self,
        video_name: str,
        action_id: str,
        fps: float,
        image_w: int,
        image_h: int,
    ) -> Path:
        """
        Start exporting a new video. Creates output directory and writes meta line.
        
        Returns the output file path.
        """
        self._video_name = video_name
        self._action_id = action_id
        self._fps = fps
        self._image_w = image_w
        self._image_h = image_h
        self._frame_count = 0
        self._frames_with_people = 0
        self._conf_sum = 0.0
        self._conf_count = 0
        
        out_dir = self.output_dir / action_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(video_name).stem
        ext = "jsonl" if self.format == "jsonl" else "json"
        self._output_path = out_dir / f"{stem}.skeleton.{ext}"
        
        self._file = open(self._output_path, "w", encoding="utf-8")
        
        meta = {
            "type": "meta",
            "schema_version": SKELETON_SCHEMA_VERSION,
            "source_video": video_name,
            "video_name": video_name,
            "action_id": action_id,
            "fps": float(fps),
            "frame_count": 0,
            "image_w": int(image_w),
            "image_h": int(image_h),
            "pose_model": "yolov8m_pose",
            "skeleton_format": "coco17",
            "coords": "normalized",
            "device_id": self._device_id,
            "camera_id": self._camera_id,
            "session_id": self._session_id,
        }
        self._file.write(json.dumps(meta, ensure_ascii=False) + "\n")
        
        return self._output_path
    
    def write_frame(
        self,
        frame_index: int,
        ts_unix_ms: float,
        persons: List[Any],
        image_w: int,
        image_h: int,
    ) -> None:
        """
        Write a single frame to the output file.
        
        Args:
            frame_index: Frame number (0-indexed)
            ts_unix_ms: Timestamp in milliseconds (will be converted to int)
            persons: List of PersonPose objects (or empty if no person detected)
            image_w: Image width in pixels
            image_h: Image height in pixels
        """
        if self._file is None:
            raise RuntimeError("start_video() must be called before write_frame()")
        
        self._frame_count += 1
        
        persons_list = []
        frame_mean_conf = 0.0
        
        if persons:
            self._frames_with_people += 1
            best_person = max(persons, key=lambda p: p.bbox_conf)
            kp_normalized = keypoints_to_17x3_normalized(
                best_person.keypoints, image_w, image_h
            )
            
            conf_values = []
            for kp in kp_normalized:
                if len(kp) >= 3 and kp[2] > 0:
                    self._conf_sum += kp[2]
                    self._conf_count += 1
                    conf_values.append(kp[2])
            
            if conf_values:
                frame_mean_conf = sum(conf_values) / len(conf_values)
            
            persons_list.append({
                "track_id": 1,
                "keypoints": kp_normalized,
            })
        
        frame_obj = {
            "type": "frame",
            "frame_index": int(frame_index),
            "ts_unix_ms": int(ts_unix_ms),
            "coords": "normalized",
            "skeleton_format": "coco17",
            "mean_conf": round(frame_mean_conf, 4),
            "persons": persons_list,
        }
        self._file.write(json.dumps(frame_obj, ensure_ascii=False) + "\n")
    
    def finish_video(self) -> ExportStats:
        """
        Finish exporting the current video.
        
        Updates the meta line with actual frame_count and closes the file.
        Returns export statistics.
        """
        if self._file is None:
            raise RuntimeError("No video in progress")
        
        self._file.close()
        self._file = None
        
        if self._output_path and self._output_path.exists():
            lines = self._output_path.read_text(encoding="utf-8").split("\n")
            if lines and lines[0]:
                meta = json.loads(lines[0])
                meta["frame_count"] = self._frame_count
                lines[0] = json.dumps(meta, ensure_ascii=False)
                self._output_path.write_text("\n".join(lines), encoding="utf-8")
        
        mean_conf = self._conf_sum / self._conf_count if self._conf_count > 0 else 0.0
        
        stats = ExportStats(
            video_name=self._video_name,
            action_id=self._action_id,
            frames_total=self._frame_count,
            frames_with_people=self._frames_with_people,
            mean_conf=mean_conf,
            output_path=str(self._output_path) if self._output_path else "",
        )
        
        self._output_path = None
        self._video_name = ""
        self._action_id = ""
        
        return stats
    
    def close(self) -> None:
        """Close any open file handle (cleanup)."""
        if self._file is not None:
            self._file.close()
            self._file = None


def write_summary_csv(path: Path, stats_list: List[ExportStats]) -> None:
    """Write a summary CSV file with export statistics."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("video_name,action_id,frames_total,frames_with_people,mean_conf,output_path\n")
        for s in stats_list:
            f.write(f"{s.video_name},{s.action_id},{s.frames_total},{s.frames_with_people},{s.mean_conf:.4f},{s.output_path}\n")
