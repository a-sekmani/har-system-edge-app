"""
HAR-System: Utility Functions
==============================
Helper utilities for HAR-System
"""

import os
import argparse


def parse_arguments():
    """Parse command line arguments for HAR application"""
    parser = argparse.ArgumentParser(
        description="HAR-System: Real-time Human Activity Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # With Raspberry Pi Camera
  python3 realtime_pose_har.py --input rpi --show-fps
  
  # With USB Camera
  python3 realtime_pose_har.py --input usb --show-fps
  
  # With video file
  python3 realtime_pose_har.py --input video.mp4
  
  # Save data
  python3 realtime_pose_har.py --input rpi --save-data
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='rpi',
        help='Video source: rpi, usb, /dev/videoX, or video file'
    )
    
    parser.add_argument(
        '--show-fps', '-f',
        action='store_true',
        help='Show FPS counter'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save data to JSON files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/camera',
        help='Data save directory (default: ./results/camera)'
    )
    
    parser.add_argument(
        '--print-interval',
        type=int,
        default=30,
        help='Print summary every N frames (default: 30)'
    )
    
    return parser.parse_args()


def setup_output_directory(output_dir: str, save_data: bool):
    """Setup output directory for saving data"""
    if save_data:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[SETUP] Output directory: {output_dir}")


def print_configuration(config: dict):
    """Print current configuration"""
    print(f"\n[CONFIG] Settings:")
    print(f"   Video Source: {config.get('input', 'N/A')}")
    print(f"   Show FPS: {config.get('show_fps', False)}")
    print(f"   Save Data: {config.get('save_data', False)}")
    if config.get('save_data'):
        print(f"   Save Directory: {config.get('output_dir', 'N/A')}")
    print(f"   Print Interval: {config.get('print_every_n_frames', 30)} frames")
    print()


def save_final_data(temporal_tracker, output_dir: str):
    """Save final tracking data"""
    print(f"\n  [SAVE] Saving final data...")
    for track_id in temporal_tracker.tracks.keys():
        filepath = os.path.join(output_dir, f"track_{track_id}_final.json")
        temporal_tracker.save_to_json(track_id, filepath)
    print(f"  [DONE] Data saved to: {output_dir}")


def print_final_summary(temporal_tracker):
    """Print final statistics summary"""
    print("\n" + "="*60)
    print("[SUMMARY] Final Summary")
    print("="*60)
    
    global_stats = temporal_tracker.get_global_stats()
    print(f"\n  Total People Detected: {global_stats['total_tracks_seen']}")
    print(f"  Total Falls: {global_stats['total_falls_detected']}")
    print(f"  Total Activity Changes: {global_stats['total_activity_changes']}")
    print("\n[EXIT] Thank you for using HAR-System!\n")
