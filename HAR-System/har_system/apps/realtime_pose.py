#!/usr/bin/env python3
"""
HAR-System: Real-time Pose Activity Recognition
================================================
Main application for real-time human activity recognition using Hailo Pose Estimation

Usage:
    python3 -m har_system --input rpi --show-fps
    python3 har_system/apps/realtime_pose.py --input rpi --show-fps
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import GStreamer
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Import Hailo components
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)

# Import HAR-System components
from har_system.core import TemporalActivityTracker
from har_system.core.callbacks import HARCallbackHandler, process_frame_callback
from har_system.utils import (
    parse_arguments,
    setup_output_directory,
    print_configuration,
    save_final_data,
    print_final_summary
)

# Setup logger
hailo_logger = get_logger(__name__)

class HARPoseEstimationApp(GStreamerPoseEstimationApp):
    """Custom Pose Estimation App with no-display option support"""
    
    def __init__(self, app_callback, user_data, parser=None, no_display=False):
        self.no_display = no_display
        super().__init__(app_callback, user_data, parser)
    
    def get_pipeline_string(self):
        """Override to support no-display mode safely."""
        hailo_logger.debug("Building HAR pipeline string...")
        
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
        
        # Use a non-rendering sink if display is disabled, otherwise use regular display.
        #
        # NOTE: GStreamerApp connects to an element named "hailo_display" for FPS measurements
        # when --show-fps is enabled. A plain fakesink doesn't have the "fps-measurements"
        # signal, so we keep a fpsdisplaysink element named "hailo_display" but set its
        # internal video-sink to fakesink and disable text overlay.
        if self.no_display:
            display_pipeline = (
                "fpsdisplaysink name=hailo_display "
                "video-sink=fakesink "
                "sync=false "
                "text-overlay=false "
                "signal-fps-measurements=true"
            )
        else:
            display_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink, 
                sync=self.sync, 
                show_fps=self.show_fps
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

def main():
    """Main entry point for HAR-System"""
    print("="*60)
    print("HAR-System: Real-time Human Activity Recognition")
    print("="*60)
    
    # Parse CLI arguments for the HAR wrapper (not the underlying hailo-apps pipeline CLI).
    args = parse_arguments()
    
    # Optionally load defaults from config/default.yaml.
    # CLI flags still take precedence for runtime behavior.
    from pathlib import Path
    config_file = Path(project_root) / "config" / "default.yaml"
    full_config = {}
    face_recognition_config = {}
    temporal_tracker_config = {}
    activity_thresholds_config = {}
    fall_detector_config = {}
    data_export_config = {}
    video_config = {}
    
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                full_config = yaml.safe_load(f) or {}
                har_cfg = full_config.get('har_system', {}) or {}
                face_recognition_config = har_cfg.get('face_recognition', {}) or {}
                temporal_tracker_config = har_cfg.get('temporal_tracker', {}) or {}
                activity_thresholds_config = (har_cfg.get('activity_classifier', {}) or {}).get('thresholds', {}) or {}
                fall_detector_config = har_cfg.get('fall_detector', {}) or {}
                data_export_config = full_config.get('data_export', {}) or {}
                video_config = full_config.get('video', {}) or {}
        except Exception as e:
            print(f"[WARNING] Could not load config file: {e}")
    
    # Consolidated runtime config passed into the callback handler.
    # This keeps GStreamer callbacks stateless and avoids global variables.
    config = {
        'input': args.input,
        'show_fps': args.show_fps,
        'verbose': args.verbose,
        'print_every_n_frames': args.print_interval,
        'save_data': args.save_data,
        'output_dir': args.output_dir,
        'save_interval': int(data_export_config.get('save_interval', 300)) if data_export_config else 300,
        'no_display': args.no_display,
        'face_recognition': face_recognition_config,
    }
    
    # Print configuration
    print_configuration(config)
    
    # Setup output directory
    setup_output_directory(args.output_dir, args.save_data)
    
    # Create the temporal tracker (pure Python logic on top of pose keypoints).
    hailo_logger.info("Creating HAR temporal tracker...")
    history_seconds = float(temporal_tracker_config.get('history_seconds', 3.0)) if temporal_tracker_config else 3.0
    fps_estimate = int(temporal_tracker_config.get('fps_estimate', 15)) if temporal_tracker_config else 15
    tracker = TemporalActivityTracker(history_seconds=history_seconds, fps_estimate=fps_estimate)

    # Apply threshold overrides from YAML (optional).
    # If keys are missing, tracker defaults remain unchanged.
    try:
        if activity_thresholds_config:
            tracker.thresholds.update(activity_thresholds_config)
        if fall_detector_config:
            tracker.thresholds.update({
                'fall_drop_ratio': fall_detector_config.get('fall_drop_ratio', tracker.thresholds.get('fall_drop_ratio')),
                'fall_time_threshold': fall_detector_config.get('fall_time_threshold', tracker.thresholds.get('fall_time_threshold')),
            })
    except Exception as e:
        print(f"[WARNING] Could not apply threshold overrides from config: {e}")
    
    # Optional: enable face recognition (requires a trained database).
    face_identity_manager = None
    face_processor = None
    
    if args.enable_face_recognition:
        print("\n[FACE-RECOG] Initializing face recognition system...")
        try:
            from har_system.core import FaceIdentityManager
            from har_system.core.face_processor import FaceRecognitionProcessor
            
            # Get database directory
            database_dir = args.database_dir or face_recognition_config.get('database_dir', './database')
            samples_dir = f"{database_dir}/samples"
            confidence_threshold = face_recognition_config.get('confidence_threshold', 0.70)
            
            # Create face identity manager
            face_identity_manager = FaceIdentityManager(
                min_confirmations=face_recognition_config.get('min_confirmations', 2),
                identity_timeout=face_recognition_config.get('identity_timeout', 5.0)
            )
            
            # Create face processor
            face_processor = FaceRecognitionProcessor(
                database_dir=database_dir,
                samples_dir=samples_dir,
                confidence_threshold=confidence_threshold
            )
            
            if face_processor.is_enabled():
                stats = face_processor.get_database_stats()
                print(f"[FACE-RECOG] Database loaded: {stats.get('total_persons', 0)} persons")
                if stats.get('persons'):
                    print(f"[FACE-RECOG] Known persons: {', '.join(stats.get('persons', []))}")
            else:
                print("[FACE-RECOG] Face recognition disabled (database not available)")
                face_identity_manager = None
                face_processor = None
                
        except Exception as e:
            print(f"[FACE-RECOG] Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            face_identity_manager = None
            face_processor = None
    
    # Callback handler bridges GStreamer detections → tracker updates → logging/export.
    user_data = HARCallbackHandler(tracker, config, face_identity_manager, face_processor)
    
    # Initialize GStreamer Pipeline
    hailo_logger.info("Initializing GStreamer Pipeline...")
    
    # The hailo-apps GStreamer app parses some settings from sys.argv.
    # We populate sys.argv here to reuse the upstream argument parsing as-is.
    sys.argv = ['har_system']
    if args.input:
        sys.argv.extend(['--input', args.input])
    if args.show_fps:
        sys.argv.append('--show-fps')
    # Apply video settings from YAML (optional).
    try:
        if video_config.get('width') is not None:
            sys.argv.extend(['--width', str(int(video_config['width']))])
        if video_config.get('height') is not None:
            sys.argv.extend(['--height', str(int(video_config['height']))])
        if video_config.get('frame_rate') is not None:
            sys.argv.extend(['--frame-rate', str(int(video_config['frame_rate']))])
    except Exception as e:
        print(f"[WARNING] Could not apply video settings from config: {e}")
    
    try:
        # Create and run application with custom class
        app = HARPoseEstimationApp(
            process_frame_callback, 
            user_data,
            no_display=args.no_display
        )
        
        print("\n[READY] HAR-System ready to run!")
        if face_processor and face_processor.is_enabled():
            print("[READY] Face recognition: ENABLED")
        if args.no_display:
            print("[READY] Display: DISABLED (performance mode)")
        else:
            print("[READY] Display: ENABLED")
        print("   Press Ctrl+C to stop\n")
        
        # Run application
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n[STOP] HAR-System stopped by user")
    except Exception as e:
        hailo_logger.error(f"Application error: {e}")
        raise
    finally:
        # Print final summary
        print_final_summary(user_data.get_tracker(), user_data.get_face_identity_manager())
        
        # Save final data
        if args.save_data:
            save_final_data(user_data.get_tracker(), args.output_dir)


if __name__ == "__main__":
    main()
