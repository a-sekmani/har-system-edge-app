#!/usr/bin/env python3
"""
HAR-System: Simple Temporal Tracking Demo
==========================================
Demonstrates direct use of TemporalActivityTracker

Usage:
    python3 demo_temporal_tracking.py
"""

import time
import numpy as np
from sys import path
from pathlib import Path

# Add parent directory to path
path.insert(0, str(Path(__file__).resolve().parent.parent))
from har_system import TemporalActivityTracker


def generate_fake_keypoints(position_x, position_y, standing=True):
    """Generate fake keypoints for testing"""
    person_height = 250 if standing else 180
    
    keypoints = {
        'nose': (position_x, position_y - person_height + 20, 0.9),
        'left_eye': (position_x - 10, position_y - person_height + 15, 0.85),
        'right_eye': (position_x + 10, position_y - person_height + 15, 0.85),
        'left_shoulder': (position_x - 30, position_y - person_height + 60, 0.9),
        'right_shoulder': (position_x + 30, position_y - person_height + 60, 0.9),
        'left_elbow': (position_x - 35, position_y - person_height + 110, 0.85),
        'right_elbow': (position_x + 35, position_y - person_height + 110, 0.85),
        'left_wrist': (position_x - 40, position_y - person_height + 160, 0.8),
        'right_wrist': (position_x + 40, position_y - person_height + 160, 0.8),
        'left_hip': (position_x - 20, position_y - person_height + 150, 0.9),
        'right_hip': (position_x + 20, position_y - person_height + 150, 0.9),
        'left_knee': (position_x - 25, position_y - person_height + 200, 0.85),
        'right_knee': (position_x + 25, position_y - person_height + 200, 0.85),
        'left_ankle': (position_x - 30, position_y - 10, 0.8),
        'right_ankle': (position_x + 30, position_y - 10, 0.8),
        'left_ear': (position_x - 20, position_y - person_height + 20, 0.8),
        'right_ear': (position_x + 20, position_y - person_height + 20, 0.8),
    }
    
    return keypoints


def main():
    """
    Simple demonstration: person walking from left to right
    """
    print("\n" + "="*60)
    print("HAR-System: Temporal Tracking Demo")
    print("="*60)
    
    print("\n[DEMO] Simulating person walking left to right\n")
    
    # Create temporal tracker
    tracker = TemporalActivityTracker(history_seconds=5.0)
    
    track_id = 1
    start_x = 100
    y = 400
    
    # Simulate 100 frames (~6.6 seconds at 15 FPS)
    for frame in range(100):
        x = start_x + frame * 5  # Move 5 pixels per frame
        
        frame_data = {
            'timestamp': time.time(),
            'bbox': {
                'xmin': x - 40,
                'ymin': y - 250,
                'xmax': x + 40,
                'ymax': y
            },
            'keypoints': generate_fake_keypoints(x, y, standing=True),
            'confidence': 0.95
        }
        
        activity = tracker.update(track_id, frame_data)
        
        # Print periodic updates
        if frame % 20 == 0:
            summary = tracker.get_summary(track_id)
            speed = tracker._calculate_normalized_speed(tracker.tracks[track_id])
            print(f"Frame {frame:3d}: "
                  f"Activity={activity:12s} | "
                  f"Distance={summary['stats']['total_distance_normalized']:6.2f} | "
                  f"Speed={speed:5.2f}")
        
        # Detect activity change
        change = tracker.detect_activity_change(track_id)
        if change:
            print(f"\n   [CHANGE] {change['from']} → {change['to']}\n")
        
        time.sleep(0.066)  # ~15 FPS
    
    # Final results
    print("\n" + "="*60)
    print("[RESULTS] Final Summary")
    print("="*60)
    
    summary = tracker.get_summary(track_id)
    print(f"\nFinal Activity: {summary['current_activity']}")
    print(f"Total Duration: {summary['duration_seconds']:.1f}s")
    print(f"Total Frames: {summary['total_frames']}")
    print(f"Normalized Distance: {summary['stats']['total_distance_normalized']:.2f}")
    print(f"Moving: {summary['stats']['percent_moving']:.1f}%")
    print(f"Stationary: {summary['stats']['percent_stationary']:.1f}%")
    
    print("\n[DONE] Example completed successfully!")
    print("\nTip: You can now modify this file to test different scenarios\n")


if __name__ == "__main__":
    main()
