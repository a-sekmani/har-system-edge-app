#!/usr/bin/env python3
"""
HAR System Edge App - Pose Estimation Application
Uses hailo-apps as a library for Pose analysis from Raspberry Pi camera.
"""

# region imports
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root (parent of src) is on path for "from src.frame_event import ..."
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad
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
    """Log FPS every interval (shared by callback)."""
    if current_time - user_data.last_fps_log_time >= user_data.fps_log_interval:
        hailo_logger.info(
            f"FPS Stats - Current: {user_data.fps_tracker.get_fps():.2f} FPS, "
            f"Average: {user_data.fps_tracker.get_average_fps():.2f} FPS, "
            f"Frames: {user_data.get_count()}, "
            f"frame_events: {getattr(user_data, 'frame_events_count', 0)}, "
            f"invalid_caps: {getattr(user_data, 'invalid_caps_count', 0)}, "
            f"invalid_validate: {getattr(user_data, 'invalid_validate_count', 0)}"
        )
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
    except Exception as e:
        hailo_logger.warning("Could not print final stats: %s", e)


# -----------------------------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------------------------
def main():
    """Application main entry point."""
    hailo_logger.info("Starting HAR Pose Estimation App...")

    parser = get_har_parser()
    user_data = HARUserData()
    app = HARPoseEstimationApp(pose_extraction_callback, user_data, parser)
    opts = app.options_menu
    user_data.log_pose_summary = getattr(opts, "log_pose_summary", False)
    user_data.dump_frames_path = getattr(opts, "dump_frames", None)
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
