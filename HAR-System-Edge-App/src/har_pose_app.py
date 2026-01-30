#!/usr/bin/env python3
"""
HAR System Edge App - Pose Estimation Application
Uses hailo-apps as a library for Pose analysis from Raspberry Pi camera.
"""

# region imports
import argparse
import time
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
    """User data class with FPS tracking."""
    
    def __init__(self):
        super().__init__()
        self.fps_tracker = FPSTracker()
        self.last_fps_log_time = time.time()
        self.fps_log_interval = 5.0  # Log FPS every 5 seconds


# -----------------------------------------------------------------------------------------------
# Simple callback function for Phase 0
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
# Main function
# -----------------------------------------------------------------------------------------------
def main():
    """Application main entry point."""
    hailo_logger.info("Starting HAR Pose Estimation App...")
    
    parser = get_har_parser()
    user_data = HARUserData()
    app = HARPoseEstimationApp(simple_callback, user_data, parser)
    
    hailo_logger.info("Running pipeline...")
    hailo_logger.info("Press Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        hailo_logger.info("Stopping application...")
        # Print final FPS stats
        final_fps = user_data.fps_tracker.get_average_fps()
        final_count = user_data.get_count()
        hailo_logger.info(
            f"Final Stats - Total Frames: {final_count}, "
            f"Average FPS: {final_fps:.2f}"
        )


if __name__ == "__main__":
    main()
