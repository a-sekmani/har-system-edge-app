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


def main():
    """Main entry point for HAR-System"""
    print("="*60)
    print("HAR-System: Real-time Human Activity Recognition")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration from file if exists
    from pathlib import Path
    config_file = Path(project_root) / "config" / "default.yaml"
    face_recognition_config = {}
    
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                full_config = yaml.safe_load(f)
                face_recognition_config = full_config.get('har_system', {}).get('face_recognition', {})
        except Exception as e:
            print(f"[WARNING] Could not load config file: {e}")
    
    # Setup configuration
    config = {
        'input': args.input,
        'show_fps': args.show_fps,
        'verbose': args.verbose,
        'print_every_n_frames': args.print_interval,
        'save_data': args.save_data,
        'output_dir': args.output_dir,
        'face_recognition': face_recognition_config,
    }
    
    # Print configuration
    print_configuration(config)
    
    # Setup output directory
    setup_output_directory(args.output_dir, args.save_data)
    
    # Create temporal activity tracker
    hailo_logger.info("Creating HAR temporal tracker...")
    tracker = TemporalActivityTracker(history_seconds=3.0, fps_estimate=15)
    
    # Initialize face recognition if enabled
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
    
    # Create callback handler
    user_data = HARCallbackHandler(tracker, config, face_identity_manager, face_processor)
    
    # Initialize GStreamer Pipeline
    hailo_logger.info("Initializing GStreamer Pipeline...")
    
    # Modify sys.argv for GStreamer app
    sys.argv = ['har_system']
    if args.input:
        sys.argv.extend(['--input', args.input])
    if args.show_fps:
        sys.argv.append('--show-fps')
    
    try:
        # Create and run application
        app = GStreamerPoseEstimationApp(process_frame_callback, user_data)
        
        print("\n[READY] HAR-System ready to run!")
        if face_processor and face_processor.is_enabled():
            print("[READY] Face recognition: ENABLED")
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
