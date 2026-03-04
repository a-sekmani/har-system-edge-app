#!/usr/bin/env python3
"""
HAR System Edge App - Pose Estimation Application
Uses hailo-apps as a library for Pose analysis from Raspberry Pi camera.
"""

# region imports
import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure project root (parent of src) is on path for "from src.frame_event import ..."
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
# Persistent face gallery storage inside the project (load first; update from cloud only when cloud updated_at is newer)
DEFAULT_FACE_GALLERY_DIR = (_PROJECT_ROOT / "face_gallery").as_posix()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    resolve_hef_path,
)
from hailo_apps.python.core.common.defines import (
    POSE_ESTIMATION_PIPELINE,
    POSE_ESTIMATION_POSTPROCESS_FUNCTION,
    POSE_ESTIMATION_POSTPROCESS_SO_FILENAME,
    RESOURCES_SO_DIR_NAME,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    DISPLAY_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    SOURCE_PIPELINE,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
)

hailo_logger = get_logger(__name__)
import hailo
from src.frame_event import FrameEvent, PersonPose, validate_frame_event, TRACK_ID_UNKNOWN
from src.tracker import TrackingConfig, FallbackTracker, get_metadata_track_id
from src.cloud_schema import build_cloud_payload
from src.cloud_client import CloudConfig, CloudSender, CloudSendQueue
from src.window_assembler import WindowAssembler
from src.windows_client import WindowsConfig, WindowsSender, WindowsSendQueue
from src.skeleton_exporter import SkeletonExporter, extract_action_from_filename, write_summary_csv, ExportStats
# Face recognition (optional)
try:
    from src.face.gallery_client import fetch_face_gallery, fetch_gallery_updated_at
    from src.face.gallery_store import load_gallery as face_load_gallery, save_gallery as face_save_gallery
    from src.face.recognizer import FaceRecognizer
    from src.face.tracker_binding import TrackerBinding
    _FACE_AVAILABLE = True
except ImportError:
    _FACE_AVAILABLE = False
# endregion imports


def _pose_confidence_from_detection(detection: Any) -> float:
    """Average keypoint confidence for detection (0 if no landmarks). Used for min_pose_confidence filter."""
    try:
        landmarks_list = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks_list:
            return 0.0
        points_list = landmarks_list[0].get_points()
        if not points_list:
            return 0.0
        total = 0.0
        count = 0
        for pt in points_list:
            if hasattr(pt, "confidence") and callable(getattr(pt, "confidence")):
                total += float(pt.confidence())
                count += 1
            elif hasattr(pt, "confidence") and not callable(getattr(pt, "confidence")):
                total += float(pt.confidence)
                count += 1
            else:
                total += 1.0
                count += 1
        return total / count if count else 0.0
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------------------------
# Custom Parser with --no-display flag
# -----------------------------------------------------------------------------------------------
def get_har_parser():
    """Create a custom parser with --no-display flag."""
    parser = get_pipeline_parser()
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display (use fakesink instead of autovideosink) for better performance.",
    )
    parser.add_argument(
        "--log-pose-summary",
        action="store_true",
        help="Log pose summary every N seconds (persons count, sample bbox/keypoints).",
    )
    parser.add_argument(
        "--dump-frames",
        type=str,
        default=None,
        metavar="path",
        help="Write FrameEvents to JSON file (each frame or every K-th).",
    )
    # Phase 2 tracking
    parser.add_argument(
        "--tracking-source",
        type=str,
        choices=["metadata", "fallback"],
        default="metadata",
        help="Track ID source: metadata (hailo) or fallback (IoU tracker).",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=15,
        metavar="N",
        help="Expire track after N frames without detection (fallback).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        metavar="X",
        help="IoU threshold for fallback tracker matching.",
    )
    parser.add_argument(
        "--min-bbox-area",
        type=float,
        default=0.0,
        metavar="A",
        help="Filter detections with bbox area below this (pixels²).",
    )
    parser.add_argument(
        "--min-bbox-height",
        type=float,
        default=None,
        metavar="H",
        help="Filter detections with bbox height below H pixels (reduces ghost tracks).",
    )
    parser.add_argument(
        "--min-pose-confidence",
        type=float,
        default=None,
        metavar="C",
        help="Filter detections with avg keypoint confidence below C (0-1).",
    )
    parser.add_argument(
        "--log-tracking-summary",
        action="store_true",
        help="Log Phase 2 tracking summary periodically.",
    )
    # Phase 3 cloud streaming
    parser.add_argument(
        "--enable-cloud",
        action="store_true",
        help="Enable cloud event streaming (build and send frame events).",
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        default="",
        metavar="URL",
        help="Cloud base URL for ingest (e.g. https://api.example.com).",
    )
    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default="",
        metavar="KEY",
        help="API key for cloud auth (or set CLOUD_API_KEY env).",
    )
    parser.add_argument(
        "--cloud-ingest-path",
        type=str,
        default="/v1/edge/events",
        metavar="PATH",
        help="Path appended to cloud URL for POST.",
    )
    parser.add_argument(
        "--send-every-n-frames",
        type=int,
        default=1,
        metavar="N",
        help="Build and send cloud event every N valid frames.",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=1000,
        metavar="N",
        help="Max in-memory queue size for cloud events; drop when full.",
    )
    parser.add_argument(
        "--send-timeout-ms",
        type=int,
        default=5000,
        metavar="MS",
        help="HTTP timeout in milliseconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        metavar="N",
        help="Retries per event on send failure.",
    )
    parser.add_argument(
        "--drop-policy",
        type=str,
        choices=["oldest", "newest"],
        default="oldest",
        help="When queue is full: drop oldest or newest event.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and count cloud payloads but do not POST (events_sent=0).",
    )
    parser.add_argument(
        "--no-verify-tls",
        action="store_true",
        help="Disable TLS verification for cloud HTTPS.",
    )
    # Phase 4 windows ingest
    parser.add_argument(
        "--cloud-mode",
        type=str,
        choices=["frames", "windows"],
        default="frames",
        help="Cloud send mode: frames (per-frame) or windows (30-frame windows). Default: frames.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        metavar="N",
        help="Window size in frames (default: 30).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=30,
        metavar="N",
        help="Window stride; use same as window-size for non-overlap (default: 30).",
    )
    parser.add_argument(
        "--cloud-windows-path",
        type=str,
        default="/v1/windows/ingest",
        metavar="PATH",
        help="Path for windows ingest POST (default: /v1/windows/ingest).",
    )
    parser.add_argument(
        "--normalize-keypoints",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        metavar="BOOL",
        help="Normalize keypoints to 0..1 in windows mode (default: true).",
    )
    parser.add_argument(
        "--max-windows-queue-size",
        type=int,
        default=500,
        metavar="N",
        help="Max in-memory queue size for windows; drop when full (default: 500).",
    )
    parser.add_argument(
        "--window-max-buffers",
        type=int,
        default=50,
        metavar="N",
        help="Max track buffers to prevent memory growth (default: 50).",
    )
    parser.add_argument(
        "--windows-drop-policy",
        type=str,
        choices=["oldest", "newest"],
        default="oldest",
        help="When windows queue is full: drop oldest or newest (default: oldest).",
    )
    # Dataset Export Mode (skeleton extraction without cloud)
    parser.add_argument(
        "--export-skeleton",
        action="store_true",
        help="Enable skeleton export mode: extract COCO-17 keypoints to JSONL files (disables cloud).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory containing .avi videos for batch skeleton export.",
    )
    parser.add_argument(
        "--export-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Output directory for skeleton JSONL files.",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format for skeleton files (default: jsonl).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        metavar="N",
        help="Max videos to process in export mode (0=all).",
    )
    parser.add_argument(
        "--skip-existing",
        type=int,
        default=0,
        choices=[0, 1],
        metavar="0|1",
        help="Skip export if output file already exists (1=skip, 0=overwrite).",
    )
    # Face recognition (cloud gallery + CPU inference)
    parser.add_argument(
        "--enable-face",
        action="store_true",
        help="Enable face recognition: sync gallery from cloud, match faces to pose tracks, attach person to windows.",
    )
    parser.add_argument(
        "--face-gallery-url",
        type=str,
        default="",
        metavar="URL",
        help="Base URL for face gallery (overrides --cloud-url for gallery). Can also set FACE_GALLERY_URL or CLOUD_URL.",
    )
    parser.add_argument(
        "--cloud-face-gallery-path",
        type=str,
        default="/v1/face-gallery",
        metavar="PATH",
        help="Path for face gallery GET (default: /v1/face-gallery).",
    )
    parser.add_argument(
        "--cloud-face-gallery-version-path",
        type=str,
        default="/v1/face-gallery/version",
        metavar="PATH",
        help="Path for face gallery version GET (default: /v1/face-gallery/version).",
    )
    parser.add_argument(
        "--face-gallery-cache",
        type=str,
        default=DEFAULT_FACE_GALLERY_DIR,
        metavar="PATH",
        help="Persistent face gallery directory inside the project (default: <app_dir>/face_gallery). Loaded first; updated from cloud only when cloud updated_at is newer.",
    )
    parser.add_argument(
        "--face-gallery-refresh-s",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Refresh face gallery from cloud every N seconds (default: 60).",
    )
    parser.add_argument(
        "--face-gallery-timeout-s",
        type=float,
        default=5.0,
        metavar="SEC",
        help="HTTP timeout for face gallery requests in seconds (default: 5).",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default="insightface",
        help="Face model backend (default: insightface).",
    )
    parser.add_argument(
        "--face-det-size",
        type=int,
        default=256,
        metavar="N",
        help="Face detection input size (default: 256, use 320 for higher accuracy).",
    )
    parser.add_argument(
        "--face-max-faces",
        type=int,
        default=1,
        metavar="N",
        help="Max faces to detect per frame (default: 1 for better FPS).",
    )
    parser.add_argument(
        "--face-sim-threshold",
        type=float,
        default=0.45,
        metavar="X",
        help="Min similarity (or 1 - distance) to accept a match (default: 0.45).",
    )
    parser.add_argument(
        "--face-min-det-conf",
        type=float,
        default=0.6,
        metavar="X",
        help="Min face detection confidence (default: 0.6).",
    )
    parser.add_argument(
        "--face-skip-frames",
        type=int,
        default=10,
        metavar="N",
        help="Run face inference every N frames to reduce CPU (default: 10).",
    )
    parser.add_argument(
        "--face-recheck-every-s",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Re-verify identity per track every N seconds (default: 2.0).",
    )
    parser.add_argument(
        "--face-track-ttl-s",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Identity TTL per track without new face (default: 10).",
    )
    parser.add_argument(
        "--window-attach-person",
        type=str,
        choices=["auto", "never", "always"],
        default="auto",
        help="Attach person to window: auto (when identity known), never, always (include null for unknown).",
    )
    parser.add_argument(
        "--log-face-summary",
        action="store_true",
        help="Log face recognition summary periodically.",
    )
    return parser


# -----------------------------------------------------------------------------------------------
# HAR Pose Estimation App with --no-display support
# -----------------------------------------------------------------------------------------------
class HARPoseEstimationApp(GStreamerPoseEstimationApp):
    """HAR System Pose Estimation app with --no-display support."""
    
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_har_parser()
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, POSE_ESTIMATION_PIPELINE)
        
        hailo_logger.info("Initializing HAR Pose Estimation App...")
        
        super().__init__(app_callback, user_data, parser)
        
        # Check if --no-display is enabled
        self.no_display = getattr(self.options_menu, 'no_display', False)
        if self.no_display:
            hailo_logger.info("--no-display enabled: Using fakesink (no video display)")
            # Override video_sink to use fakesink
            self.video_sink = "fakesink"
        else:
            hailo_logger.info("Display enabled: Using autovideosink")
        
        # Rebuild pipeline with new video_sink if needed
        if self.no_display and self.pipeline:
            hailo_logger.debug("Rebuilding pipeline with fakesink...")
            self.create_pipeline()
    
    def get_pipeline_string(self):
        """Override pipeline string to support --no-display"""
        hailo_logger.debug("Building pipeline string...")
        
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        infer_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_process_function,
            batch_size=self.batch_size,
        )
        infer_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(infer_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=0)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # Use fakesink if --no-display is enabled
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
        )
        
        pipeline_string = (
            f"{source_pipeline} ! "
            f"{infer_pipeline_wrapper} ! "
            f"{tracker_pipeline} ! "
            f"{user_callback_pipeline} ! "
            f"{display_pipeline}"
        )
        hailo_logger.debug("Pipeline string: %s", pipeline_string)
        return pipeline_string

    def on_eos(self):
        """Handle end-of-stream: shutdown instead of looping for file sources."""
        if self.source_type == "file":
            hailo_logger.info("Video file finished (EOS). Shutting down...")
            self.shutdown()
        else:
            super().on_eos()


# -----------------------------------------------------------------------------------------------
# FPS Tracker for callback
# -----------------------------------------------------------------------------------------------
class FPSTracker:
    """Simple FPS tracker."""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.start_time = time.time()
        self.frame_count = 0
    
    def update(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last window_size frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Compute current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span == 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_span
    
    def get_average_fps(self):
        """Compute average FPS since start."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.frame_count / elapsed


# -----------------------------------------------------------------------------------------------
# User callback class with FPS tracking
# -----------------------------------------------------------------------------------------------
class HARUserData(app_callback_class):
    """User data class with FPS tracking, Phase 1 pose extraction, and Phase 2 tracking state."""
    
    def __init__(self, log_pose_summary=False, dump_frames_path=None, dump_frames_every_k=1):
        super().__init__()
        self.fps_tracker = FPSTracker()
        self.last_fps_log_time = time.time()
        self.fps_log_interval = 5.0  # Log FPS every 5 seconds
        self.invalid_frame_count = 0
        self.invalid_caps_count = 0
        self.invalid_validate_count = 0
        self._validation_errors_printed = False  # Task A: print first 5 errors only once
        self._first_invalid_sample_logged = False  # Task B: print raw sample for first invalid only
        self.frame_events_count = 0
        # Phase 1 acceptance counters (only for valid frame_events)
        self.frames_with_persons = 0
        self.frames_no_persons = 0
        self.persons_total = 0
        self.frames_with_landmarks = 0
        self.frames_keypoints_len_not_17 = 0
        self.log_pose_summary = log_pose_summary
        self.dump_frames_path = dump_frames_path
        self.dump_frames_every_k = max(1, int(dump_frames_every_k))
        self.last_pose_summary_time = time.time()
        self.pose_summary_interval = 5.0
        self._dump_file_handle = None
        # Phase 2 tracking (set from main after parsing)
        self.tracking_config = None  # TrackingConfig
        self.fallback_tracker = None  # FallbackTracker when tracking_source == "fallback"
        self.unique_track_ids = set()  # distinct track ids ever seen
        self.new_tracks_created = 0
        self.tracks_ended = 0
        self.id_switch_suspected = 0
        self.multi_person_frames = 0
        self.log_tracking_summary = False
        self.last_tracking_summary_time = time.time()
        self.tracking_summary_interval = 5.0
        self._debug_created_logged = 0
        self._debug_ended_logged = 0
        self._debug_switch_logged = 0
        self._prev_frame_track_ids = []  # for id_switch heuristic
        self._prev_frame_bboxes = []  # for id_switch heuristic
        # Detection filter counters (before vs after filter)
        self.detections_total = 0  # raw person detections from ROI (before filter)
        self.filtered_detections_total = 0  # detections excluded by min_bbox_area / min_bbox_height / min_pose_confidence
        # Phase 3 cloud (set from main when enable_cloud)
        self.enable_cloud = False
        self.dry_run = False
        self.events_built = 0
        self.events_sent = 0
        self.events_failed = 0
        self.events_dropped = 0
        self.queue_depth_max = 0
        self.cloud_queue = None  # CloudSendQueue when enable_cloud and not dry_run
        self.cloud_sender = None
        self.cloud_config = None  # CloudConfig; device_id, session_id, model for build_cloud_payload
        self.send_every_n_frames = 1
        self.device_id = ""
        self.session_id = ""
        self.model_name = "yolov8m_pose"
        self.tracking_source_str = "metadata"
        # Phase 4 windows
        self.cloud_mode = "frames"
        self.camera_id = "default"
        self.window_assembler = None
        self.windows_sender = None  # WindowsSendQueue when cloud_mode=windows and not dry_run
        self.windows_built = 0
        self.windows_sent = 0
        self.windows_failed = 0
        self.windows_dropped = 0
        self.windows_queue_depth_max = 0
        # Face recognition (set from main when enable_face)
        self.enable_face = False
        self.face_gallery = None  # current in-memory gallery (FaceGallery or None)
        self.face_recognizer = None  # FaceRecognizer instance
        self.face_tracker_binding = None  # TrackerBinding instance
        self.face_gallery_next_refresh_ts = 0.0  # next refresh timestamp
        self.log_face_summary = False
        self.last_face_summary_time = time.time()
        self.face_summary_interval = 5.0


# -----------------------------------------------------------------------------------------------
# Face gallery sync and refresh
# -----------------------------------------------------------------------------------------------
def _init_face_recognition(user_data: HARUserData) -> None:
    """Initialize face recognizer, tracker binding, and load/sync gallery. No-op if face not available."""
    if not _FACE_AVAILABLE or not getattr(user_data, "enable_face", False):
        return
    opts = getattr(user_data, "_face_opts", None)
    if not opts:
        return
    try:
        user_data.face_recognizer = FaceRecognizer(
            det_size=opts["face_det_size"],
            max_faces=opts["face_max_faces"],
            min_det_conf=opts["face_min_det_conf"],
            sim_threshold=opts["face_sim_threshold"],
        )
        user_data.face_tracker_binding = TrackerBinding(
            iou_threshold=0.2,
            track_ttl_s=opts["face_track_ttl_s"],
            recheck_every_s=opts["face_recheck_every_s"],
            sim_threshold=opts["face_sim_threshold"],
            min_votes_stable=2,
        )
    except Exception as e:
        hailo_logger.warning("Face recognizer/tracker init failed: %s", e)
        user_data.face_recognizer = None
        user_data.face_tracker_binding = None
        return
    cache_dir = opts["face_gallery_cache"]
    timeout_s = opts["face_gallery_timeout_s"]
    refresh_s = opts["face_gallery_refresh_s"]
    base_url = opts.get("cloud_url", "")
    api_key = opts.get("cloud_api_key", "")
    gallery_path = opts.get("cloud_face_gallery_path", "/v1/face-gallery")
    version_path = opts.get("cloud_face_gallery_version_path", "/v1/face-gallery/version")
    # Always load local copy first; we never overwrite it unless we successfully fetch a newer updated_at from cloud.
    user_data.face_gallery = face_load_gallery(cache_dir)
    if user_data.face_gallery:
        hailo_logger.info(
            "face gallery loaded from cache updated_at=%s persons=%s embeddings=%s",
            user_data.face_gallery.updated_at or "(none)",
            len(user_data.face_gallery.persons),
            user_data.face_gallery.total_embeddings(),
        )
    if base_url:
        try:
            remote_updated_at = fetch_gallery_updated_at(base_url, version_path, api_key, timeout_s)
            if remote_updated_at is None:
                hailo_logger.warning(
                    "face gallery updated_at fetch failed; keeping existing gallery (local copy unchanged)"
                )
            else:
                local_updated_at = (user_data.face_gallery.updated_at or "").strip() if user_data.face_gallery else ""
                # Update only when cloud has a newer date (ISO 8601 string comparison).
                if local_updated_at and remote_updated_at <= local_updated_at:
                    hailo_logger.info(
                        "face gallery updated_at not newer than local (%s); no update needed",
                        remote_updated_at,
                    )
                else:
                    gallery = fetch_face_gallery(base_url, gallery_path, api_key, timeout_s)
                    if gallery is not None:
                        user_data.face_gallery = gallery
                        face_save_gallery(cache_dir, gallery)
                        hailo_logger.info(
                            "face gallery synced from cloud updated_at=%s persons=%s embeddings=%s (saved to cache)",
                            gallery.updated_at or "(none)", len(gallery.persons), gallery.total_embeddings(),
                        )
                    else:
                        hailo_logger.warning(
                            "face gallery fetch failed; keeping existing gallery (local copy unchanged)"
                        )
        except Exception as e:
            hailo_logger.warning("face gallery sync failed: %s; keeping existing gallery (local copy unchanged)", e)
    else:
        hailo_logger.info(
            "Face recognition enabled but no gallery URL (set FACE_GALLERY_URL or CLOUD_URL or use --cloud-url); using cache only."
        )
    if user_data.face_gallery is None:
        hailo_logger.info("face gallery empty; recognition will run but all matches unknown")
    user_data.face_gallery_next_refresh_ts = time.time() + refresh_s
    # Log gallery path and cwd so operator can confirm (gallery path is absolute, does not depend on cwd)
    hailo_logger.info("face gallery dir=%s cwd=%s", cache_dir, os.getcwd())


def _refresh_face_gallery_if_due(user_data: HARUserData) -> None:
    """
    If refresh interval (e.g. 1 minute) elapsed, check cloud updated_at.
    Only update local gallery when we successfully fetch and remote updated_at is newer than local.
    On any fetch failure, keep existing gallery unchanged.
    """
    if not _FACE_AVAILABLE or not getattr(user_data, "enable_face", False):
        return
    opts = getattr(user_data, "_face_opts", None)
    if not opts or not opts.get("cloud_url"):
        return
    now = time.time()
    if now < getattr(user_data, "face_gallery_next_refresh_ts", 0):
        return
    cache_dir = opts["face_gallery_cache"]
    timeout_s = opts["face_gallery_timeout_s"]
    refresh_s = opts["face_gallery_refresh_s"]
    base_url = opts["cloud_url"]
    api_key = opts.get("cloud_api_key", "")
    gallery_path = opts.get("cloud_face_gallery_path", "/v1/face-gallery")
    version_path = opts.get("cloud_face_gallery_version_path", "/v1/face-gallery/version")
    try:
        remote_updated_at = fetch_gallery_updated_at(base_url, version_path, api_key, timeout_s)
        if remote_updated_at is None:
            hailo_logger.debug("face gallery refresh: updated_at fetch failed; keeping existing gallery")
            user_data.face_gallery_next_refresh_ts = now + refresh_s
            return
        local_updated_at = (user_data.face_gallery.updated_at or "").strip() if user_data.face_gallery else ""
        if local_updated_at and remote_updated_at <= local_updated_at:
            user_data.face_gallery_next_refresh_ts = now + refresh_s
            return
        gallery = fetch_face_gallery(base_url, gallery_path, api_key, timeout_s)
        if gallery is not None:
            user_data.face_gallery = gallery
            face_save_gallery(cache_dir, gallery)
            hailo_logger.info(
                "face gallery refreshed from cloud updated_at=%s persons=%s (saved to cache)",
                gallery.updated_at or "(none)", len(gallery.persons),
            )
        else:
            hailo_logger.debug("face gallery refresh: gallery fetch failed; keeping existing gallery")
    except Exception as e:
        hailo_logger.debug("face gallery refresh failed: %s; keeping existing gallery", e)
    user_data.face_gallery_next_refresh_ts = now + refresh_s


def _face_worker_loop(user_data: "HARUserData") -> None:
    """
    Background thread for face recognition so it does not slow down the pipeline.
    Reads (frame_bgr, pose_detections, now_ts) from the queue and runs detect + embed + match + binding.update.
    """
    face_queue = getattr(user_data, "face_queue", None)
    if face_queue is None:
        return
    while True:
        try:
            item = face_queue.get(timeout=0.5)
            if item is None:
                break
            frame_bgr, pose_detections, now_ts = item
            rec = getattr(user_data, "face_recognizer", None)
            binding = getattr(user_data, "face_tracker_binding", None)
            gallery = getattr(user_data, "face_gallery", None)
            if rec is None or binding is None:
                continue
            try:
                face_dets = rec.detect_faces(frame_bgr)
                face_detections_with_match = []
                for fd in face_dets:
                    emb = rec.get_embedding(frame_bgr, fd)
                    match_result = rec.match(emb, gallery) if emb else None
                    face_detections_with_match.append((fd, match_result))
                lock = getattr(user_data, "face_binding_lock", None)
                if lock is not None:
                    with lock:
                        binding.update(pose_detections, face_detections_with_match, now_ts=now_ts)
                else:
                    binding.update(pose_detections, face_detections_with_match, now_ts=now_ts)
            except Exception as e:
                hailo_logger.debug("face worker step failed: %s", e)
        except queue.Empty:
            continue
        except Exception as e:
            hailo_logger.debug("face worker: %s", e)


# -----------------------------------------------------------------------------------------------
# Simple callback function for Phase 0 (FPS only)
# -----------------------------------------------------------------------------------------------
def simple_callback(element, buffer, user_data):
    """Simple callback for Phase 0 - FPS tracking only."""
    if buffer is None:
        return
    
    # Update FPS tracker
    user_data.fps_tracker.update()
    
    # Log FPS every interval
    current_time = time.time()
    if current_time - user_data.last_fps_log_time >= user_data.fps_log_interval:
        current_fps = user_data.fps_tracker.get_fps()
        average_fps = user_data.fps_tracker.get_average_fps()
        frame_count = user_data.get_count()
        
        hailo_logger.info(
            f"FPS Stats - Current: {current_fps:.2f} FPS, "
            f"Average: {average_fps:.2f} FPS, "
            f"Frames: {frame_count}"
        )
        user_data.last_fps_log_time = current_time
    
    return


# -----------------------------------------------------------------------------------------------
# Pose extraction callback (Phase 1 + Phase 2): FPS + FrameEvent + track_id + validation
# -----------------------------------------------------------------------------------------------
def pose_extraction_callback(element, buffer, user_data):
    """Callback: FPS + extract pose, resolve track_id (metadata or fallback), build FrameEvent, validate."""
    if buffer is None:
        return

    user_data.fps_tracker.update()
    current_time = time.time()
    timestamp_ms = current_time * 1000.0
    frame_number = user_data.get_count()

    pad = element.get_static_pad("src")
    format_caps, width, height = get_caps_from_pad(pad)
    if width is None or height is None or width <= 0 or height <= 0:
        user_data.invalid_caps_count += 1
        user_data.invalid_frame_count += 1
        _log_fps_if_due(user_data, current_time)
        return

    width, height = int(width), int(height)
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    cfg = getattr(user_data, "tracking_config", None)
    min_bbox_area = getattr(cfg, "min_bbox_area", 0.0) if cfg else 0.0
    min_bbox_height = getattr(cfg, "min_bbox_height", None) if cfg else None
    min_pose_conf = getattr(cfg, "min_pose_confidence", None) if cfg else None

    # Build list of (detection, bbox_px) for person detections; apply filter to reduce false positives
    dets_and_bboxes = []
    for det in detections:
        if det.get_label() != "person":
            continue
        user_data.detections_total += 1
        try:
            bbox_norm = det.get_bbox()
            xmin = max(0.0, min(1.0, bbox_norm.xmin()))
            ymin = max(0.0, min(1.0, bbox_norm.ymin()))
            w = max(0.0, min(1.0 - xmin, bbox_norm.width()))
            h_norm = max(0.0, min(1.0 - ymin, bbox_norm.height()))
            x1 = xmin * width
            y1 = ymin * height
            x2 = (xmin + w) * width
            y2 = (ymin + h_norm) * height
            bbox_px = [float(x1), float(y1), float(x2), float(y2)]
            area_px = (x2 - x1) * (y2 - y1)
            height_px = y2 - y1

            # Filter: exclude small / low-confidence detections (reduces ghost tracks)
            if area_px < min_bbox_area:
                user_data.filtered_detections_total += 1
                continue
            if min_bbox_height is not None and height_px < min_bbox_height:
                user_data.filtered_detections_total += 1
                continue
            if min_pose_conf is not None:
                pose_conf = _pose_confidence_from_detection(det)
                if pose_conf < min_pose_conf:
                    user_data.filtered_detections_total += 1
                    continue
            dets_and_bboxes.append((det, bbox_px))
        except Exception as e:
            hailo_logger.debug("Skip detection bbox: %s", e)
            continue

    # Resolve track_id per detection (metadata or fallback)
    cfg = getattr(user_data, "tracking_config", None)
    track_ids = [TRACK_ID_UNKNOWN] * len(dets_and_bboxes)
    new_n, end_n = 0, 0
    if cfg and getattr(cfg, "tracking_enabled", True) and dets_and_bboxes:
        bboxes = [bbox for _, bbox in dets_and_bboxes]
        if getattr(cfg, "tracking_source", "metadata") == "metadata":
            metadata_ids = []
            need_fallback = False
            for det, _ in dets_and_bboxes:
                mid = get_metadata_track_id(det)
                metadata_ids.append(mid)
                if mid is None or mid <= 0:
                    need_fallback = True
            if need_fallback and user_data.fallback_tracker is not None:
                fallback_ids, new_n, end_n = user_data.fallback_tracker.update(
                    bboxes, frame_number, current_time
                )
                user_data.tracks_ended += end_n
                for i in range(len(dets_and_bboxes)):
                    track_ids[i] = metadata_ids[i] if (metadata_ids[i] is not None and metadata_ids[i] > 0) else fallback_ids[i]
            else:
                for i in range(len(dets_and_bboxes)):
                    track_ids[i] = metadata_ids[i] if (metadata_ids[i] is not None and metadata_ids[i] > 0) else TRACK_ID_UNKNOWN
            # Count first appearance of any track_id (metadata or fallback) as new_tracks_created
            new_this_frame = 0
            for tid in track_ids:
                if tid != TRACK_ID_UNKNOWN:
                    if tid not in user_data.unique_track_ids:
                        new_this_frame += 1
                        user_data.new_tracks_created += 1
                    user_data.unique_track_ids.add(tid)
        else:
            # fallback source
            if user_data.fallback_tracker is not None:
                fallback_ids, new_n, end_n = user_data.fallback_tracker.update(
                    bboxes, frame_number, current_time
                )
                track_ids = fallback_ids
                user_data.tracks_ended += end_n
            new_this_frame = 0
            for tid in track_ids:
                if tid != TRACK_ID_UNKNOWN:
                    if tid not in user_data.unique_track_ids:
                        new_this_frame += 1
                        user_data.new_tracks_created += 1
                    user_data.unique_track_ids.add(tid)

        # Diagnostics: first N track created / ended
        _log_tracking_diagnostics(user_data, new_this_frame, end_n, frame_number)

        # Id-switch heuristic: same frame count, assignment flip or continuity but id change (simplified: two persons, ids swap)
        if len(track_ids) >= 2 and hasattr(user_data, "_prev_frame_track_ids") and len(user_data._prev_frame_track_ids) >= 2:
            # Check if current ids are permutation of previous (possible swap)
            prev_set = set(user_data._prev_frame_track_ids)
            curr_set = set(track_ids)
            if prev_set == curr_set and user_data._prev_frame_track_ids != track_ids:
                # Same two ids but different order -> possible swap (heuristic)
                user_data.id_switch_suspected += 1
                if getattr(user_data, "_debug_switch_logged", 0) < getattr(cfg, "debug_first_n_switches", 0):
                    hailo_logger.info(
                        "[id_switch suspected] frame=%s prev_ids=%s current_ids=%s prev_bboxes=%s current_bboxes=%s",
                        frame_number, user_data._prev_frame_track_ids, track_ids,
                        getattr(user_data, "_prev_frame_bboxes", []), bboxes,
                    )
                    user_data._debug_switch_logged = getattr(user_data, "_debug_switch_logged", 0) + 1
        user_data._prev_frame_track_ids = list(track_ids)
        user_data._prev_frame_bboxes = [list(b) for _, b in dets_and_bboxes]

    # Build PersonPose list with track_id
    persons = []
    for (det, bbox_px), tid in zip(dets_and_bboxes, track_ids):
        try:
            pose = PersonPose.from_hailo_detection(
                det, width, height, store_raw_sample=(len(persons) == 0), track_id=tid
            )
            persons.append(pose)
        except Exception as e:
            hailo_logger.debug("Skip detection: %s", e)
            continue

    event = FrameEvent(
        frame_number=frame_number,
        timestamp_ms=timestamp_ms,
        image={"width": width, "height": height},
        persons=persons,
    )
    valid, errors = validate_frame_event(event)
    if not valid:
        user_data.invalid_validate_count += 1
        user_data.invalid_frame_count += 1
        if not getattr(user_data, "_validation_errors_printed", False):
            user_data._validation_errors_printed = True
            for err in errors[:5]:
                hailo_logger.info("[validation] %s", err)
        if not getattr(user_data, "_first_invalid_sample_logged", False) and event.persons:
            user_data._first_invalid_sample_logged = True
            _log_first_invalid_sample(event, width, height)
        _log_fps_if_due(user_data, current_time)
        return

    user_data.frame_events_count += 1
    if len(persons) > 0:
        user_data.frames_with_persons += 1
    else:
        user_data.frames_no_persons += 1
    user_data.persons_total += len(persons)
    if len(persons) >= 2:
        user_data.multi_person_frames += 1
    has_landmarks = any(
        any(len(kp) == 3 and kp[2] > 0 for kp in p.keypoints) for p in persons
    )
    if has_landmarks:
        user_data.frames_with_landmarks += 1
    if any(len(p.keypoints) != 17 for p in persons):
        user_data.frames_keypoints_len_not_17 += 1

    # Face recognition: refresh gallery if due; enqueue frame for face worker (no face work here to keep ~30 FPS)
    if _FACE_AVAILABLE and getattr(user_data, "enable_face", False):
        _refresh_face_gallery_if_due(user_data)
        opts_face = getattr(user_data, "_face_opts", None)
        skip_frames = int(opts_face.get("face_skip_frames", 10)) if opts_face else 10
        face_queue = getattr(user_data, "face_queue", None)
        if (
            frame_number % skip_frames == 0
            and face_queue is not None
            and getattr(user_data, "face_recognizer", None) is not None
            and getattr(user_data, "face_tracker_binding", None) is not None
        ):
            try:
                frame_np = get_numpy_from_buffer(buffer, format_caps, width, height)
                if hasattr(frame_np, "shape") and len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    import cv2
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR).copy()
                    pose_detections = [(p.track_id, (p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3])) for p in persons]
                    try:
                        face_queue.put_nowait((frame_bgr, pose_detections, current_time))
                    except queue.Full:
                        pass
            except Exception as e:
                hailo_logger.debug("face enqueue failed: %s", e)
        if getattr(user_data, "log_face_summary", False) and (current_time - getattr(user_data, "last_face_summary_time", 0)) >= getattr(user_data, "face_summary_interval", 5.0):
            user_data.last_face_summary_time = current_time
            binding = getattr(user_data, "face_tracker_binding", None)
            if binding and binding._identities:
                known = sum(1 for i in binding._identities.values() if i.person_id)
                hailo_logger.info("face summary: tracks_with_identity=%s/%s", known, len(binding._identities))

    # Phase 3: build and enqueue frame event (only when cloud_mode=frames)
    if getattr(user_data, "enable_cloud", False) and getattr(user_data, "cloud_mode", "frames") == "frames" and (
        frame_number % getattr(user_data, "send_every_n_frames", 1)
    ) == 0:
        user_data.events_built += 1
        if not getattr(user_data, "dry_run", True) and getattr(user_data, "cloud_queue", None) is not None:
            payload = build_cloud_payload(
                event,
                device_id=getattr(user_data, "device_id", ""),
                session_id=getattr(user_data, "session_id", ""),
                model=getattr(user_data, "model_name", "yolov8m_pose"),
                tracking_source=getattr(user_data, "tracking_source_str", "metadata"),
                fps_current=user_data.fps_tracker.get_fps(),
                fps_avg=user_data.fps_tracker.get_average_fps(),
            )
            user_data.cloud_queue.enqueue(payload)
            user_data.cloud_queue.drain_one()
            user_data.events_sent = user_data.cloud_queue.counters.get("events_sent", 0)
            user_data.events_failed = user_data.cloud_queue.counters.get("events_failed", 0)
            user_data.events_dropped = user_data.cloud_queue.counters.get("events_dropped", 0)
            user_data.queue_depth_max = user_data.cloud_queue.counters.get("queue_depth_max", 0)
    # Phase 4: windows — push frame to assembler; enqueue completed windows (non-blocking)
    if getattr(user_data, "enable_cloud", False) and getattr(user_data, "cloud_mode", "frames") == "windows" and getattr(user_data, "window_assembler", None) is not None:
        completed = user_data.window_assembler.push_frame(
            event,
            device_id=getattr(user_data, "device_id", ""),
            camera_id=getattr(user_data, "camera_id", "default"),
            session_id=getattr(user_data, "session_id", ""),
        )
        user_data.windows_built += len(completed)
        if not getattr(user_data, "dry_run", True) and getattr(user_data, "windows_sender", None) is not None:
            for w in completed:
                payload = w.to_dict()
                if getattr(user_data, "enable_face", False) and getattr(user_data, "face_tracker_binding", None) is not None:
                    opts_face = getattr(user_data, "_face_opts", None)
                    attach_policy = (opts_face or {}).get("window_attach_person", "auto")
                    lock = getattr(user_data, "face_binding_lock", None)
                    if lock is not None:
                        with lock:
                            person_att = user_data.face_tracker_binding.get_identity(
                                w.track_id, now_ts=current_time, attach_policy=attach_policy
                            )
                    else:
                        person_att = user_data.face_tracker_binding.get_identity(
                            w.track_id, now_ts=current_time, attach_policy=attach_policy
                        )
                    if person_att is not None:
                        payload["person"] = person_att.to_dict()
                user_data.windows_sender.enqueue(payload)
        if getattr(user_data, "windows_sender", None) is not None:
            user_data.windows_sent = user_data.windows_sender.counters.get("windows_sent", 0)
            user_data.windows_failed = user_data.windows_sender.counters.get("windows_failed", 0)
            user_data.windows_dropped = user_data.windows_sender.counters.get("windows_dropped", 0)
            user_data.windows_queue_depth_max = user_data.windows_sender.counters.get("windows_queue_depth_max", 0)
    if getattr(user_data, "log_pose_summary", False) and (
        current_time - getattr(user_data, "last_pose_summary_time", 0)
    ) >= getattr(user_data, "pose_summary_interval", 5.0):
        _log_pose_summary(user_data, event, current_time)
        user_data.last_pose_summary_time = current_time
    if getattr(user_data, "log_tracking_summary", False) and (
        current_time - getattr(user_data, "last_tracking_summary_time", 0)
    ) >= getattr(user_data, "tracking_summary_interval", 5.0):
        _log_tracking_summary(user_data, current_time)
        user_data.last_tracking_summary_time = current_time
    if getattr(user_data, "dump_frames_path", None):
        _dump_frame_event(user_data, event)
    # Resolve person label(s) for FPS log and "Persons on screen" report (from face binding or "Unknown")
    if not persons:
        user_data._last_person_label = "-"
    else:
        binding = getattr(user_data, "face_tracker_binding", None)
        opts_face = getattr(user_data, "_face_opts", None)
        attach_policy = (opts_face or {}).get("window_attach_person", "auto") if opts_face else "auto"
        now_ts = current_time
        labels = []
        lock = getattr(user_data, "face_binding_lock", None)
        for p in persons:
            if binding is not None:
                if lock is not None:
                    with lock:
                        att = binding.get_identity(p.track_id, now_ts=now_ts, attach_policy=attach_policy)
                else:
                    att = binding.get_identity(p.track_id, now_ts=now_ts, attach_policy=attach_policy)
                if att and (att.name or att.person_id):
                    labels.append(att.name or att.person_id)
                else:
                    labels.append("Unknown")
            else:
                labels.append("Unknown")
        user_data._last_person_label = ", ".join(labels)
    _log_fps_if_due(user_data, current_time)
    return


def _log_tracking_diagnostics(user_data, new_n, end_n, frame_number):
    """Log first N 'track created' and 'track ended' (config.debug_first_n_created/ended)."""
    cfg = getattr(user_data, "tracking_config", None)
    if not cfg:
        return
    if new_n > 0:
        n_logged = getattr(user_data, "_debug_created_logged", 0)
        max_log = getattr(cfg, "debug_first_n_created", 0)
        if max_log > 0 and n_logged < max_log:
            hailo_logger.info("[track created] frame=%s count=%s", frame_number, new_n)
            user_data._debug_created_logged = n_logged + 1
    if end_n > 0:
        n_logged = getattr(user_data, "_debug_ended_logged", 0)
        max_log = getattr(cfg, "debug_first_n_ended", 0)
        if max_log > 0 and n_logged < max_log:
            hailo_logger.info("[track ended] frame=%s count=%s", frame_number, end_n)
            user_data._debug_ended_logged = n_logged + 1


def _log_tracking_summary(user_data, current_time):
    """Log Phase 2 tracking summary (unique_track_ids, new_tracks_created, etc.)."""
    hailo_logger.info(
        "Phase2 summary: unique_track_ids=%s, new_tracks_created=%s, tracks_ended=%s, "
        "id_switch_suspected=%s, multi_person_frames=%s, detections_total=%s, filtered_detections_total=%s",
        len(getattr(user_data, "unique_track_ids", set())),
        getattr(user_data, "new_tracks_created", 0),
        getattr(user_data, "tracks_ended", 0),
        getattr(user_data, "id_switch_suspected", 0),
        getattr(user_data, "multi_person_frames", 0),
        getattr(user_data, "detections_total", 0),
        getattr(user_data, "filtered_detections_total", 0),
    )


def _log_fps_if_due(user_data, current_time):
    """Log FPS every interval (shared by callback). Includes person name when available."""
    if current_time - user_data.last_fps_log_time >= user_data.fps_log_interval:
        person_label = getattr(user_data, "_last_person_label", "-")
        hailo_logger.info(
            f"FPS Stats - Current: {user_data.fps_tracker.get_fps():.2f} FPS, "
            f"Average: {user_data.fps_tracker.get_average_fps():.2f} FPS, "
            f"Frames: {user_data.get_count()}, "
            f"Person: {person_label}, "
            f"frame_events: {getattr(user_data, 'frame_events_count', 0)}, "
            f"invalid_caps: {getattr(user_data, 'invalid_caps_count', 0)}, "
            f"invalid_validate: {getattr(user_data, 'invalid_validate_count', 0)}"
        )
        hailo_logger.info("Persons on screen: %s", person_label)
        hailo_logger.info(
            "Phase1 summary: frames_with_persons=%s, frames_no_persons=%s, persons_total=%s, "
            "frames_with_landmarks=%s, frames_keypoints_len_not_17=%s",
            getattr(user_data, "frames_with_persons", 0),
            getattr(user_data, "frames_no_persons", 0),
            getattr(user_data, "persons_total", 0),
            getattr(user_data, "frames_with_landmarks", 0),
            getattr(user_data, "frames_keypoints_len_not_17", 0),
        )
        hailo_logger.info(
            "Phase2 summary: unique_track_ids=%s, new_tracks_created=%s, tracks_ended=%s, "
            "id_switch_suspected=%s, multi_person_frames=%s, detections_total=%s, filtered_detections_total=%s",
            len(getattr(user_data, "unique_track_ids", set())),
            getattr(user_data, "new_tracks_created", 0),
            getattr(user_data, "tracks_ended", 0),
            getattr(user_data, "id_switch_suspected", 0),
            getattr(user_data, "multi_person_frames", 0),
            getattr(user_data, "detections_total", 0),
            getattr(user_data, "filtered_detections_total", 0),
        )
        if getattr(user_data, "enable_cloud", False) and getattr(user_data, "cloud_mode", "frames") == "frames":
            queue_depth = user_data.cloud_queue.queue_depth() if getattr(user_data, "cloud_queue", None) else 0
            hailo_logger.info(
                "Phase3 summary: events_built=%s, events_sent=%s, events_failed=%s, events_dropped=%s, "
                "queue_depth=%s, queue_depth_max=%s",
                getattr(user_data, "events_built", 0),
                getattr(user_data, "events_sent", 0),
                getattr(user_data, "events_failed", 0),
                getattr(user_data, "events_dropped", 0),
                queue_depth,
                getattr(user_data, "queue_depth_max", 0),
            )
        if getattr(user_data, "enable_cloud", False) and getattr(user_data, "cloud_mode", "frames") == "windows":
            wq_depth = user_data.windows_sender.queue_depth() if getattr(user_data, "windows_sender", None) else 0
            hailo_logger.info(
                "Phase4 summary: windows_built=%s, windows_sent=%s, windows_failed=%s, windows_dropped=%s, "
                "windows_queue_depth=%s, windows_queue_depth_max=%s",
                getattr(user_data, "windows_built", 0),
                getattr(user_data, "windows_sent", 0),
                getattr(user_data, "windows_failed", 0),
                getattr(user_data, "windows_dropped", 0),
                wq_depth,
                getattr(user_data, "windows_queue_depth_max", 0),
            )
        user_data.last_fps_log_time = current_time


def _log_first_invalid_sample(event, image_w, image_h):
    """Task B: log image size, first person bbox, first 3 keypoints with c>0 (raw + pixel)."""
    hailo_logger.info("[first invalid sample] image_w=%s image_h=%s", image_w, image_h)
    if not event.persons:
        return
    p = event.persons[0]
    hailo_logger.info("[first invalid sample] first person bbox: %s", p.bbox)
    kps_with_c = [(i, kp) for i, kp in enumerate(p.keypoints) if len(kp) == 3 and kp[2] > 0]
    for i, (idx, kp) in enumerate(kps_with_c[:3]):
        hailo_logger.info("[first invalid sample] keypoint[%s] after (pixels): [x=%s, y=%s, c=%s]", idx, kp[0], kp[1], kp[2])
    from src import frame_event as fe_mod
    raw = getattr(fe_mod, "_debug_raw_keypoints_sample", None)
    if raw:
        for i, r in enumerate(raw[:3]):
            hailo_logger.info(
                "[first invalid sample] raw[%s] before (rel): x_rel=%s y_rel=%s c=%s -> after (px): x_px=%s y_px=%s",
                i, r[0], r[1], r[2], r[3], r[4],
            )
    else:
        hailo_logger.info("[first invalid sample] no raw sample (store_raw_sample was not set for first person)")


def _log_pose_summary(user_data, event, current_time):
    """Log number of persons, sample bbox and first 2 keypoints for one person."""
    n = len(event.persons)
    hailo_logger.info("Pose summary: frame=%s persons=%s", event.frame_number, n)
    if n > 0:
        p = event.persons[0]
        hailo_logger.info("  sample bbox: %s", p.bbox)
        if len(p.keypoints) >= 2:
            hailo_logger.info("  first 2 keypoints: %s, %s", p.keypoints[0], p.keypoints[1])


def _dump_frame_event(user_data, event):
    """Append one FrameEvent as JSON line to dump file (or every K-th)."""
    path = user_data.dump_frames_path
    every_k = getattr(user_data, "dump_frames_every_k", 1)
    if event.frame_number % every_k != 0:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    except OSError as e:
        hailo_logger.warning("dump-frames write failed: %s", e)


def _print_final_stats(user_data):
    """Print final FPS, Phase1 and Phase2 counters (called on exit)."""
    try:
        final_fps = user_data.fps_tracker.get_average_fps()
        final_count = user_data.get_count()
        hailo_logger.info(
            f"Final Stats - Total Frames: {final_count}, "
            f"Average FPS: {final_fps:.2f}, "
            f"frame_events: {user_data.frame_events_count}, "
            f"invalid_caps: {user_data.invalid_caps_count}, "
            f"invalid_validate: {user_data.invalid_validate_count}"
        )
        hailo_logger.info(
            "Phase1 final: frames_with_persons=%s, frames_no_persons=%s, persons_total=%s, "
            "frames_with_landmarks=%s, frames_keypoints_len_not_17=%s",
            user_data.frames_with_persons,
            user_data.frames_no_persons,
            user_data.persons_total,
            user_data.frames_with_landmarks,
            user_data.frames_keypoints_len_not_17,
        )
        hailo_logger.info(
            "Phase2 final: unique_track_ids=%s, new_tracks_created=%s, tracks_ended=%s, "
            "id_switch_suspected=%s, multi_person_frames=%s, detections_total=%s, filtered_detections_total=%s",
            len(getattr(user_data, "unique_track_ids", set())),
            getattr(user_data, "new_tracks_created", 0),
            getattr(user_data, "tracks_ended", 0),
            getattr(user_data, "id_switch_suspected", 0),
            getattr(user_data, "multi_person_frames", 0),
            getattr(user_data, "detections_total", 0),
            getattr(user_data, "filtered_detections_total", 0),
        )
        if getattr(user_data, "enable_cloud", False) and getattr(user_data, "cloud_mode", "frames") == "frames":
            queue_depth = user_data.cloud_queue.queue_depth() if getattr(user_data, "cloud_queue", None) else 0
            hailo_logger.info(
                "Phase3 final: events_built=%s, events_sent=%s, events_failed=%s, events_dropped=%s, "
                "queue_depth=%s, queue_depth_max=%s",
                getattr(user_data, "events_built", 0),
                getattr(user_data, "events_sent", 0),
                getattr(user_data, "events_failed", 0),
                getattr(user_data, "events_dropped", 0),
                queue_depth,
                getattr(user_data, "queue_depth_max", 0),
            )
        if getattr(user_data, "enable_cloud", False) and getattr(user_data, "cloud_mode", "frames") == "windows":
            wq_depth = user_data.windows_sender.queue_depth() if getattr(user_data, "windows_sender", None) else 0
            hailo_logger.info(
                "Phase4 final: windows_built=%s, windows_sent=%s, windows_failed=%s, windows_dropped=%s, "
                "windows_queue_depth=%s, windows_queue_depth_max=%s",
                getattr(user_data, "windows_built", 0),
                getattr(user_data, "windows_sent", 0),
                getattr(user_data, "windows_failed", 0),
                getattr(user_data, "windows_dropped", 0),
                wq_depth,
                getattr(user_data, "windows_queue_depth_max", 0),
            )
            if getattr(user_data, "windows_sender", None) is not None:
                user_data.windows_sender.shutdown()
    except Exception as e:
        hailo_logger.warning("Could not print final stats: %s", e)


# -----------------------------------------------------------------------------------------------
# Skeleton Export Callback (for dataset export mode)
# -----------------------------------------------------------------------------------------------
def skeleton_export_callback(element, buffer, user_data):
    """
    Callback for skeleton export mode: extract pose, write to exporter.
    
    - Selects single person with highest bbox_conf per frame
    - Writes normalized keypoints to JSONL via SkeletonExporter
    - No cloud, no windows, no tracking complexity
    - frame_index starts from 0 (standard convention)
    """
    if buffer is None:
        return

    user_data.fps_tracker.update()
    current_time = time.time()
    timestamp_ms = int(current_time * 1000.0)
    
    if not hasattr(user_data, "_export_frame_index"):
        user_data._export_frame_index = 0
    frame_index = user_data._export_frame_index
    user_data._export_frame_index += 1

    pad = element.get_static_pad("src")
    format_caps, width, height = get_caps_from_pad(pad)
    if width is None or height is None or width <= 0 or height <= 0:
        user_data.invalid_caps_count += 1
        user_data.invalid_frame_count += 1
        return

    width, height = int(width), int(height)
    
    if not hasattr(user_data, "_export_caps_set") or not user_data._export_caps_set:
        user_data._export_image_w = width
        user_data._export_image_h = height
        user_data._export_caps_set = True

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    persons = []
    for det in detections:
        if det.get_label() != "person":
            continue
        try:
            pose = PersonPose.from_hailo_detection(det, width, height, store_raw_sample=False, track_id=1)
            persons.append(pose)
        except Exception:
            continue

    user_data.frame_events_count += 1
    if persons:
        user_data.frames_with_persons += 1
    else:
        user_data.frames_no_persons += 1
    user_data.persons_total += len(persons)

    exporter = getattr(user_data, "skeleton_exporter", None)
    if exporter is not None:
        exporter.write_frame(frame_index, timestamp_ms, persons, width, height)


# -----------------------------------------------------------------------------------------------
# Batch Export Functions
# -----------------------------------------------------------------------------------------------
def run_batch_export(opts):
    """
    Run skeleton export on all videos in video_dir.
    
    For each video:
    1. Create SkeletonExporter output file
    2. Run pipeline with skeleton_export_callback
    3. Collect stats and write summary.csv
    """
    from pathlib import Path
    
    video_dir = Path(opts.video_dir)
    export_out = Path(opts.export_out)
    export_format = getattr(opts, "export_format", "jsonl") or "jsonl"
    max_videos = getattr(opts, "max_videos", 0) or 0
    skip_existing = getattr(opts, "skip_existing", 0) == 1
    
    if not video_dir.exists():
        print(f"ERROR: video_dir does not exist: {video_dir}")
        return
    
    export_out.mkdir(parents=True, exist_ok=True)
    
    videos = sorted(video_dir.rglob("*.avi"))
    if max_videos > 0:
        videos = videos[:max_videos]
    
    print(f"Found {len(videos)} videos to export")
    
    all_stats = []
    
    for i, video_path in enumerate(videos):
        action_id = extract_action_from_filename(video_path.name)
        
        out_dir = export_out / action_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        stem = video_path.stem
        ext = "jsonl" if export_format == "jsonl" else "json"
        out_file = out_dir / f"{stem}.skeleton.{ext}"
        
        if skip_existing and out_file.exists():
            print(f"[{i + 1}/{len(videos)}] SKIP {video_path.name} (exists)")
            continue
        
        print(f"[{i + 1}/{len(videos)}] Processing {video_path.name} (action={action_id})")
        
        stats = process_single_video_export(
            video_path=video_path,
            output_path=out_file,
            action_id=action_id,
            export_format=export_format,
            opts=opts,
        )
        
        if stats:
            all_stats.append(stats)
            print(
                f"exported video={stats.video_name} action={stats.action_id} "
                f"frames={stats.frames_total} people_frames={stats.frames_with_people} "
                f"mean_conf={stats.mean_conf:.2f} out={stats.output_path}"
            )
    
    if all_stats:
        summary_path = export_out / "summary.csv"
        write_summary_csv(summary_path, all_stats)
        print(f"Summary written to {summary_path} ({len(all_stats)} videos)")


def process_single_video_export(video_path, output_path, action_id, export_format, opts):
    """
    Process a single video for skeleton export.
    
    Creates a new app instance, runs the pipeline, and returns ExportStats.
    """
    from pathlib import Path
    
    user_data = HARUserData()
    user_data.enable_cloud = False
    user_data.skeleton_exporter = SkeletonExporter(
        output_dir=str(output_path.parent.parent),
        format=export_format,
    )
    user_data._export_caps_set = False
    user_data._export_image_w = 0
    user_data._export_image_h = 0
    
    parser = get_har_parser()
    
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [
        sys.argv[0],
        "--input", str(video_path),
        "--no-display",
    ]
    
    try:
        app = HARPoseEstimationApp(skeleton_export_callback, user_data, parser)
        
        fps = getattr(app, "frame_rate", 30.0) or 30.0
        image_w = getattr(app, "video_width", 1920) or 1920
        image_h = getattr(app, "video_height", 1080) or 1080
        
        user_data.skeleton_exporter.start_video(
            video_name=video_path.name,
            action_id=action_id,
            fps=fps,
            image_w=image_w,
            image_h=image_h,
        )
        
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        except SystemExit:
            pass
        
        if user_data._export_caps_set:
            image_w = user_data._export_image_w
            image_h = user_data._export_image_h
        
        stats = user_data.skeleton_exporter.finish_video()
        return stats
        
    except Exception as e:
        print(f"ERROR processing {video_path.name}: {e}")
        import traceback
        traceback.print_exc()
        if user_data.skeleton_exporter:
            user_data.skeleton_exporter.close()
        return None
    finally:
        sys.argv = original_argv


# -----------------------------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------------------------
def main():
    """Application main entry point."""
    hailo_logger.info("Starting HAR Pose Estimation App...")
    # Set working directory to project root so face_gallery and other project paths are used consistently
    try:
        os.chdir(_PROJECT_ROOT)
        hailo_logger.info("Working directory set to project root: %s", os.getcwd())
    except OSError as e:
        hailo_logger.warning("Could not set working directory to %s: %s", _PROJECT_ROOT, e)

    import argparse
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--export-skeleton", action="store_true")
    temp_parser.add_argument("--video-dir", type=str, default=None)
    temp_parser.add_argument("--export-out", type=str, default=None)
    temp_parser.add_argument("--export-format", type=str, default="jsonl")
    temp_parser.add_argument("--max-videos", type=int, default=0)
    temp_parser.add_argument("--skip-existing", type=int, default=0)
    temp_args, _ = temp_parser.parse_known_args()
    
    if temp_args.export_skeleton:
        if not temp_args.video_dir or not temp_args.export_out:
            print("ERROR: --export-skeleton requires --video-dir and --export-out")
            return
        print("=" * 60)
        print("SKELETON EXPORT MODE")
        print("=" * 60)
        print(f"  video_dir: {temp_args.video_dir}")
        print(f"  export_out: {temp_args.export_out}")
        print(f"  format: {temp_args.export_format}")
        print(f"  max_videos: {temp_args.max_videos if temp_args.max_videos > 0 else 'all'}")
        print(f"  skip_existing: {bool(temp_args.skip_existing)}")
        print("=" * 60)
        run_batch_export(temp_args)
        return

    parser = get_har_parser()
    user_data = HARUserData()
    app = HARPoseEstimationApp(pose_extraction_callback, user_data, parser)
    opts = app.options_menu
    user_data.log_pose_summary = getattr(opts, "log_pose_summary", False)
    user_data.dump_frames_path = getattr(opts, "dump_frames", None)
    user_data.no_display = getattr(opts, "no_display", False)
    # Phase 2 tracking config and fallback tracker
    user_data.tracking_config = TrackingConfig(
        tracking_enabled=True,
        tracking_source=getattr(opts, "tracking_source", "metadata"),
        max_missing_frames=getattr(opts, "max_missing_frames", 15),
        iou_match_threshold=getattr(opts, "iou_threshold", 0.3),
        max_track_age_seconds=None,
        min_bbox_area=getattr(opts, "min_bbox_area", 0.0),
        min_bbox_height=getattr(opts, "min_bbox_height", None),
        min_pose_confidence=getattr(opts, "min_pose_confidence", None),
        debug_first_n_switches=5,
        debug_first_n_created=5,
        debug_first_n_ended=5,
    )
    user_data.fallback_tracker = FallbackTracker(user_data.tracking_config)
    user_data.log_tracking_summary = getattr(opts, "log_tracking_summary", False)

    # Phase 3 cloud streaming
    user_data.enable_cloud = getattr(opts, "enable_cloud", False)
    user_data.dry_run = getattr(opts, "dry_run", False)
    user_data.send_every_n_frames = max(1, int(getattr(opts, "send_every_n_frames", 1)))
    user_data.device_id = os.environ.get("DEVICE_ID", "")
    if not user_data.device_id:
        try:
            import socket
            user_data.device_id = socket.gethostname() or "edge-device"
        except Exception:
            user_data.device_id = "edge-device"
    user_data.session_id = str(uuid.uuid4())
    user_data.model_name = getattr(opts, "model_name", None) or "yolov8m_pose"
    user_data.tracking_source_str = getattr(opts, "tracking_source", "metadata")
    user_data.cloud_mode = getattr(opts, "cloud_mode", "frames") or "frames"
    user_data.camera_id = os.environ.get("CAMERA_ID", "default")

    if user_data.enable_cloud and user_data.cloud_mode == "frames":
        if not user_data.dry_run and getattr(opts, "cloud_url", "").strip():
            cloud_base_url = opts.cloud_url.strip().rstrip("/")
            cloud_config = CloudConfig(
                cloud_base_url=cloud_base_url,
                cloud_ingest_path=getattr(opts, "cloud_ingest_path", "/v1/edge/events") or "/v1/edge/events",
                api_key=getattr(opts, "cloud_api_key", "") or "",
                timeout_ms=getattr(opts, "send_timeout_ms", 5000),
                max_retries=getattr(opts, "max_retries", 2),
                backoff_seconds=0.5,
                verify_tls=not getattr(opts, "no_verify_tls", False),
                compression=None,
                max_queue_size=getattr(opts, "max_queue_size", 1000),
                drop_policy=getattr(opts, "drop_policy", "oldest") or "oldest",
            )
            user_data.cloud_config = cloud_config
            phase3_counters = {"events_sent": 0, "events_failed": 0, "events_dropped": 0, "queue_depth_max": 0}
            user_data.cloud_sender = CloudSender(cloud_config)
            user_data.cloud_queue = CloudSendQueue(cloud_config, user_data.cloud_sender, phase3_counters)
            hailo_logger.info("Phase 3 cloud enabled: url=%s, queue_size=%s", cloud_base_url, cloud_config.max_queue_size)
        else:
            user_data.cloud_queue = None
            user_data.cloud_sender = None
            user_data.cloud_config = None
            hailo_logger.info("Phase 3 cloud dry-run or no URL: building payloads only, no POST")

    if user_data.enable_cloud and user_data.cloud_mode == "windows":
        window_size = max(1, int(getattr(opts, "window_size", 30)))
        window_stride = max(1, int(getattr(opts, "window_stride", 30)))
        window_max_buffers = max(1, int(getattr(opts, "window_max_buffers", 50)))
        user_data.window_assembler = WindowAssembler(
            window_size=window_size,
            window_stride=window_stride,
            window_max_buffers=window_max_buffers,
        )
        if not user_data.dry_run and getattr(opts, "cloud_url", "").strip():
            cloud_base_url = opts.cloud_url.strip().rstrip("/")
            windows_config = WindowsConfig(
                cloud_base_url=cloud_base_url,
                cloud_windows_path=getattr(opts, "cloud_windows_path", "/v1/windows/ingest") or "/v1/windows/ingest",
                api_key=getattr(opts, "cloud_api_key", "") or "",
                connect_timeout_sec=0.5,
                read_timeout_sec=2.0,
                verify_tls=not getattr(opts, "no_verify_tls", False),
                max_queue_size=getattr(opts, "max_windows_queue_size", 500),
                drop_policy=getattr(opts, "windows_drop_policy", "oldest") or "oldest",
            )
            phase4_counters = {
                "windows_sent": 0,
                "windows_failed": 0,
                "windows_dropped": 0,
                "windows_queue_depth_max": 0,
            }
            windows_sender = WindowsSender(windows_config)
            user_data.windows_sender = WindowsSendQueue(windows_config, windows_sender, phase4_counters)
            user_data.windows_sender.start()
            hailo_logger.info(
                "Phase 4 windows enabled: url=%s, path=%s, queue_size=%s",
                cloud_base_url, windows_config.cloud_windows_path, windows_config.max_queue_size,
            )
        else:
            user_data.windows_sender = None
            hailo_logger.info("Phase 4 windows dry-run or no URL: building windows only, no POST")

    # Face recognition config (actual init in face package when enable_face)
    user_data.enable_face = getattr(opts, "enable_face", False)
    user_data.log_face_summary = getattr(opts, "log_face_summary", False)
    if user_data.enable_face:
        user_data._face_opts = {
            "cloud_face_gallery_path": getattr(opts, "cloud_face_gallery_path", "/v1/face-gallery") or "/v1/face-gallery",
            "cloud_face_gallery_version_path": getattr(opts, "cloud_face_gallery_version_path", "/v1/face-gallery/version") or "/v1/face-gallery/version",
            "face_gallery_cache": getattr(opts, "face_gallery_cache", DEFAULT_FACE_GALLERY_DIR) or DEFAULT_FACE_GALLERY_DIR,
            "face_gallery_refresh_s": max(1.0, float(getattr(opts, "face_gallery_refresh_s", 60))),
            "face_gallery_timeout_s": max(1.0, float(getattr(opts, "face_gallery_timeout_s", 5))),
            "face_model": getattr(opts, "face_model", "insightface") or "insightface",
            # Default 256 to reduce detection time; 320 is more accurate but slower
            "face_det_size": max(160, int(getattr(opts, "face_det_size", 256))),
            # Default 1 for better FPS when a single person is present
            "face_max_faces": max(1, int(getattr(opts, "face_max_faces", 1))),
            "face_sim_threshold": float(getattr(opts, "face_sim_threshold", 0.45)),
            "face_min_det_conf": float(getattr(opts, "face_min_det_conf", 0.6)),
            # Run face recognition every 10 frames by default to improve FPS
            "face_skip_frames": max(1, int(getattr(opts, "face_skip_frames", 10))),
            "face_recheck_every_s": max(0.1, float(getattr(opts, "face_recheck_every_s", 2.0))),
            "face_track_ttl_s": max(1.0, float(getattr(opts, "face_track_ttl_s", 10.0))),
            "window_attach_person": getattr(opts, "window_attach_person", "auto") or "auto",
            # Face gallery: fetched from cloud on each run when URL is set (priority: --face-gallery-url, --cloud-url, FACE_GALLERY_URL, CLOUD_URL)
            "cloud_url": (
                getattr(opts, "face_gallery_url", "")
                or getattr(opts, "cloud_url", "")
                or os.environ.get("FACE_GALLERY_URL", "")
                or os.environ.get("CLOUD_URL", "")
            ).strip().rstrip("/"),
            "cloud_api_key": getattr(opts, "cloud_api_key", "") or os.environ.get("CLOUD_API_KEY", ""),
        }
        _init_face_recognition(user_data)
        # Dedicated face recognition thread so the pipeline stays at ~30 FPS
        if user_data.enable_face and getattr(user_data, "face_recognizer", None) is not None:
            user_data.face_queue = queue.Queue(maxsize=1)
            user_data.face_binding_lock = threading.Lock()
            user_data.face_worker_thread = threading.Thread(target=_face_worker_loop, args=(user_data,), daemon=True)
            user_data.face_worker_thread.start()
            hailo_logger.info("Face recognition running in background thread (pipeline target ~30 FPS)")
        else:
            user_data.face_queue = None
            user_data.face_binding_lock = None
            user_data.face_worker_thread = None

    hailo_logger.info("Running pipeline...")
    hailo_logger.info("Press Ctrl+C to stop")

    try:
        app.run()
    except KeyboardInterrupt:
        hailo_logger.info("Stopping application...")
    finally:
        _print_final_stats(user_data)


if __name__ == "__main__":
    main()
