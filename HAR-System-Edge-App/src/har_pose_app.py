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
from src.frame_event import FrameEvent, PersonPose, validate_frame_event
# endregion imports


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
    """User data class with FPS tracking and Phase 1 pose extraction state."""
    
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
# Pose extraction callback (Phase 1): FPS + FrameEvent build + validation
# -----------------------------------------------------------------------------------------------
def pose_extraction_callback(element, buffer, user_data):
    """Callback: FPS tracking + extract pose per frame, build FrameEvent, validate; skip invalid."""
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
    persons = []
    for det in detections:
        if det.get_label() != "person":
            continue
        try:
            pose = PersonPose.from_hailo_detection(
                det, width, height, store_raw_sample=(len(persons) == 0)
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
        # Task A: print first 5 validation errors only once
        if not getattr(user_data, "_validation_errors_printed", False):
            user_data._validation_errors_printed = True
            for err in errors[:5]:
                hailo_logger.info("[validation] %s", err)
        # Task B: print raw sample for first invalid frame only
        if not getattr(user_data, "_first_invalid_sample_logged", False) and event.persons:
            user_data._first_invalid_sample_logged = True
            _log_first_invalid_sample(event, width, height)
        _log_fps_if_due(user_data, current_time)
        return
    
    user_data.frame_events_count += 1
    # Phase 1 acceptance counters
    if len(persons) > 0:
        user_data.frames_with_persons += 1
    else:
        user_data.frames_no_persons += 1
    user_data.persons_total += len(persons)
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
    if getattr(user_data, "dump_frames_path", None):
        _dump_frame_event(user_data, event)
    _log_fps_if_due(user_data, current_time)
    return


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
    """Print final FPS and Phase1 counters (called on exit; base class uses SIGINT so run() returns without raising)."""
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
    # Sync Phase 1 options from app (parser is parsed in app init)
    user_data.log_pose_summary = getattr(app.options_menu, "log_pose_summary", False)
    user_data.dump_frames_path = getattr(app.options_menu, "dump_frames", None)
    
    hailo_logger.info("Running pipeline...")
    hailo_logger.info("Press Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        hailo_logger.info("Stopping application...")
    finally:
        # Base class handles SIGINT and quits the loop without raising; run() returns.
        # Always print final stats when the app exits (Ctrl+C or normal).
        _print_final_stats(user_data)


if __name__ == "__main__":
    main()
