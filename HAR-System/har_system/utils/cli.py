"""
HAR-System: Utility Functions
==============================
Helper utilities for HAR-System
"""

import os
import argparse

def add_realtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add realtime CLI arguments to an existing argparse parser."""
    
    # This argument is used to specify the input source for the video.
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='rpi',
        help='Video source: rpi, usb, /dev/videoX, or video file'
    )

    # This argument is used to show the FPS counter on the screen.
    parser.add_argument(
        '--show-fps', '-f',
        action='store_true',
        help='Show FPS counter'
    )

    # This argument is used to show the detailed information.
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )

    # This argument is used to save the data to JSON files.
    parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save data to JSON files'
    )

    # This argument is used to specify the output directory for the data.
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/camera',
        help='Data save directory (default: ./results/camera)'
    )

    # This argument is used to specify the print interval for the data.
    parser.add_argument(
        '--print-interval',
        type=int,
        default=30,
        help='Print summary every N frames (default: 30)'
    )

    # This argument is used to enable face recognition.
    parser.add_argument(
        '--enable-face-recognition',
        action='store_true',
        help='Enable face recognition (requires trained database)'
    )

    # This argument is used to specify the database directory for face recognition
    parser.add_argument(
        '--database-dir',
        type=str,
        default=None,
        help='Face recognition database directory (default: from config/default.yaml or ./database)'
    )

    # This argument is used to disable the video display.
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (improves performance, uses fakesink)'
    )

def add_train_faces_arguments(parser: argparse.ArgumentParser) -> None:
    """Add train-faces CLI arguments to an existing argparse parser."""
    
    # Used by both the top-level dispatcher and the standalone train-faces entry point.
    parser.add_argument(
        '--train-dir',
        type=str,
        default='./train_faces',
        help='Directory with training images (default: ./train_faces)'
    )
    # This argument is used to specify the database directory for face recognition.
    parser.add_argument(
        '--database-dir',
        type=str,
        default='./database',
        help='Database directory (default: ./database)'
    )

    # This argument is used to specify the confidence threshold for face recognition.
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.70,
        help='Recognition confidence threshold (default: 0.70)'
    )

def add_faces_arguments(parser: argparse.ArgumentParser) -> None:
    """Add faces management CLI arguments to an existing argparse parser."""

    # This mirrors the options supported by `har_system/apps/manage_faces.py`.
    parser.add_argument(
        '--database-dir',
        type=str,
        default='./database',
        help='Database directory (default: ./database)'
    )

    # This argument is used to list all known persons.
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all known persons'
    )

    # This argument is used to remove a person from the database.
    parser.add_argument(
        '--remove',
        type=str,
        metavar='NAME',
        help='Remove a person from database'
    )

    # This argument is used to clear the entire database.
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear entire database (WARNING: cannot be undone!)'
    )

    # This argument is used to show the database statistics.
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )

# This function is used to add the arguments to the chokepoint command.
def add_chokepoint_arguments(parser: argparse.ArgumentParser) -> None:
    """Add chokepoint CLI arguments to an existing argparse parser."""
    
    # Used by both the top-level dispatcher and `har_system/apps/chokepoint_analyzer.py`.
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./test_dataset',
        help='Path to test_dataset folder (default: ./test_dataset)'
    )

    # This argument is used to specify the results directory for the data.
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Results output directory (default: ./results)'
    )

    # This argument is used to enable face recognition.
    parser.add_argument(
        '--enable-face-recognition',
        action='store_true',
        help='Enable face recognition (person_id will be name or -1)'
    )

    # This argument is used to specify the database directory for face recognition.
    parser.add_argument(
        '--database-dir',
        type=str,
        default='./database',
        help='Face recognition database directory (default: ./database)'
    )
    
    # This argument is used to disable the video display.
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (improves performance)'
    )

def build_realtime_parser() -> argparse.ArgumentParser:
    """Build a standalone parser for the realtime app."""
    # Standalone parser (used when running realtime_pose directly).
    parser = argparse.ArgumentParser(
        description="HAR-System: Real-time Human Activity Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_realtime_arguments(parser)
    return parser


def parse_arguments():
    """Parse command line arguments for HAR application"""
    # This parser is used by `har_system/apps/realtime_pose.py`.
    # The top-level dispatcher (`python3 -m har_system realtime ...`) forwards compatible flags.
    parser = build_realtime_parser()
    return parser.parse_args()


def setup_output_directory(output_dir: str, save_data: bool):
    """Setup output directory for saving data"""
    # Only create directories when exporting is enabled.
    if save_data:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[SETUP] Output directory: {output_dir}")

def print_configuration(config: dict):
    """Print current configuration"""
    # Keep this output stable: scripts/users rely on it for quick validation.
    print(f"\n[CONFIG] Settings:")
    print(f"   Video Source: {config.get('input', 'N/A')}")
    print(f"   Show FPS: {config.get('show_fps', False)}")
    print(f"   Display Enabled: {not config.get('no_display', False)}")
    print(f"   Save Data: {config.get('save_data', False)}")
    if config.get('save_data'):
        print(f"   Save Directory: {config.get('output_dir', 'N/A')}")
    print(f"   Print Interval: {config.get('print_every_n_frames', 30)} frames")
    print()

def save_final_data(temporal_tracker, output_dir: str):
    """Save final tracking data"""
    # Persist a final snapshot per track_id for offline inspection/debugging.
    print(f"\n  [SAVE] Saving final data...")
    for track_id in temporal_tracker.tracks.keys():
        filepath = os.path.join(output_dir, f"track_{track_id}_final.json")
        temporal_tracker.save_to_json(track_id, filepath)
    print(f"  [DONE] Data saved to: {output_dir}")

def print_final_summary(temporal_tracker, face_identity_manager=None):
    """Print final statistics summary"""
    # Printed on shutdown (Ctrl+C) to summarize what happened during the run.
    print("\n" + "="*60)
    print("[SUMMARY] Final Summary")
    print("="*60)
    
    global_stats = temporal_tracker.get_global_stats()
    print(f"\n  Total People Detected: {global_stats['total_tracks_seen']}")
    print(f"  Total Falls: {global_stats['total_falls_detected']}")
    print(f"  Total Activity Changes: {global_stats['total_activity_changes']}")
    
    # Add face recognition statistics if available
    if face_identity_manager:
        face_stats = face_identity_manager.get_statistics()
        print(f"\n  [Face Recognition]")
        print(f"  People Recognized: {face_stats['identified_tracks']}")
        print(f"  Unique Persons: {face_stats['unique_persons']}")
        if face_stats['person_names']:
            print(f"  Names: {', '.join(face_stats['person_names'])}")
    
    print("\n[EXIT] Thank you for using HAR-System!\n")
