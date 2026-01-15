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
    
    # Setup configuration
    config = {
        'input': args.input,
        'show_fps': args.show_fps,
        'verbose': args.verbose,
        'print_every_n_frames': args.print_interval,
        'save_data': args.save_data,
        'output_dir': args.output_dir,
    }
    
    # Print configuration
    print_configuration(config)
    
    # Setup output directory
    setup_output_directory(args.output_dir, args.save_data)
    
    # Create temporal activity tracker
    hailo_logger.info("Creating HAR temporal tracker...")
    tracker = TemporalActivityTracker(history_seconds=3.0, fps_estimate=15)
    
    # Create callback handler
    user_data = HARCallbackHandler(tracker, config)
    
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
        print_final_summary(user_data.get_tracker())
        
        # Save final data
        if args.save_data:
            save_final_data(user_data.get_tracker(), args.output_dir)


if __name__ == "__main__":
    main()
