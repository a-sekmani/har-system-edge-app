#!/usr/bin/env python3
"""
HAR-System: Temporal Tracker Test Suite
========================================
Comprehensive tests for temporal activity tracking

Usage:
    python3 test_har_tracker.py
"""

import time
import numpy as np
from sys import path
from pathlib import Path

# Allow running this file directly without installing the package.
# If you installed HAR-System via pip, this sys.path tweak is not required.
path.insert(0, str(Path(__file__).resolve().parent.parent))
from har_system import TemporalActivityTracker


def generate_fake_keypoints(position_x, position_y, standing=True):
    """Generate fake keypoints for testing"""
    person_height = 250 if standing else 180
    
    # For sitting: hip should be closer to ankle (higher hip_ratio ~0.65-0.80)
    # For standing: hip is further from ankle (lower hip_ratio ~0.45-0.55)
    if standing:
        hip_offset = 150  # Hip at position_y - 100, hip_ratio ~0.36
    else:
        # For sitting: hip_ratio should be > 0.62
        # hip_ratio = (ankle_y - hip_y) / bbox_height
        # We want: (position_y - 10 - (position_y - 180 + hip_offset)) / 180 > 0.62
        # (170 - hip_offset) / 180 > 0.62
        # 170 - hip_offset > 111.6
        # hip_offset < 58.4
        hip_offset = 50  # Hip at position_y - 130, hip_ratio ~0.67
    
    keypoints = {
        'nose': (position_x, position_y - person_height + 20, 0.9),
        'left_shoulder': (position_x - 30, position_y - person_height + 60, 0.9),
        'right_shoulder': (position_x + 30, position_y - person_height + 60, 0.9),
        'left_hip': (position_x - 20, position_y - person_height + hip_offset, 0.9),
        'right_hip': (position_x + 20, position_y - person_height + hip_offset, 0.9),
        'left_ankle': (position_x - 30, position_y - 10, 0.8),
        'right_ankle': (position_x + 30, position_y - 10, 0.8),
        'left_eye': (position_x - 10, position_y - person_height + 15, 0.85),
        'right_eye': (position_x + 10, position_y - person_height + 15, 0.85),
        'left_ear': (position_x - 20, position_y - person_height + 20, 0.8),
        'right_ear': (position_x + 20, position_y - person_height + 20, 0.8),
        'left_elbow': (position_x - 35, position_y - person_height + 110, 0.85),
        'right_elbow': (position_x + 35, position_y - person_height + 110, 0.85),
        'left_wrist': (position_x - 40, position_y - person_height + 160, 0.8),
        'right_wrist': (position_x + 40, position_y - person_height + 160, 0.8),
        'left_knee': (position_x - 25, position_y - person_height + 200, 0.85),
        'right_knee': (position_x + 25, position_y - person_height + 200, 0.85),
    }
    
    return keypoints


def test_stationary_person():
    """Test: Stationary person"""
    print("\n" + "="*60)
    print("[TEST 1] Stationary Person")
    print("="*60)
    
    tracker = TemporalActivityTracker()
    track_id = 1
    base_x, base_y = 300, 400
    
    # Simulate 50 frames of stationary person
    # Use smaller noise to avoid false movement detection
    for i in range(50):
        noise_x = np.random.uniform(-0.5, 0.5)
        noise_y = np.random.uniform(-0.5, 0.5)
        
        frame_data = {
            'timestamp': time.time(),
            'bbox': {
                'xmin': base_x - 40 + noise_x,
                'ymin': base_y - 250 + noise_y,
                'xmax': base_x + 40 + noise_x,
                'ymax': base_y + noise_y
            },
            'keypoints': generate_fake_keypoints(base_x + noise_x, base_y + noise_y, standing=True),
            'confidence': 0.95
        }
        
        tracker.update(track_id, frame_data)
        time.sleep(0.033)
    
    summary = tracker.get_summary(track_id)
    print(f"\n[RESULT]")
    print(f"   Activity: {summary['current_activity']}")
    print(f"   Stationary: {summary['stats']['percent_stationary']:.1f}%")
    print(f"   Normalized Distance: {summary['stats']['total_distance_normalized']:.3f}")
    
    assert summary['current_activity'] == 'stationary', "Activity should be 'stationary'"
    print("\n[PASS] Test passed!")


def test_moving_person():
    """Test: Moving person"""
    print("\n" + "="*60)
    print("[TEST 2] Moving Person")
    print("="*60)
    
    tracker = TemporalActivityTracker()
    track_id = 2
    base_y = 400
    
    # Simulate 50 frames of person moving left to right
    for i in range(50):
        base_x = 100 + i * 5
        
        frame_data = {
            'timestamp': time.time(),
            'bbox': {
                'xmin': base_x - 40,
                'ymin': base_y - 250,
                'xmax': base_x + 40,
                'ymax': base_y
            },
            'keypoints': generate_fake_keypoints(base_x, base_y, standing=True),
            'confidence': 0.95
        }
        
        tracker.update(track_id, frame_data)
        time.sleep(0.033)
    
    summary = tracker.get_summary(track_id)
    print(f"\n[RESULT]")
    print(f"   Activity: {summary['current_activity']}")
    print(f"   Moving: {summary['stats']['percent_moving']:.1f}%")
    print(f"   Normalized Distance: {summary['stats']['total_distance_normalized']:.2f}")
    
    assert summary['current_activity'] == 'moving', "Activity should be 'moving'"
    assert summary['stats']['percent_moving'] > 80, "Moving percentage should be > 80%"
    print("\n[PASS] Test passed!")


def test_sitting_person():
    """Test: Sitting person"""
    print("\n" + "="*60)
    print("[TEST 3] Sitting Person")
    print("="*60)
    
    tracker = TemporalActivityTracker()
    track_id = 3
    base_x, base_y = 300, 400
    
    # Simulate 50 frames of sitting person
    # Use smaller noise to avoid false movement detection
    for i in range(50):
        noise = np.random.uniform(-0.5, 0.5)
        
        frame_data = {
            'timestamp': time.time(),
            'bbox': {
                'xmin': base_x - 40,
                'ymin': base_y - 180 + noise,
                'xmax': base_x + 40,
                'ymax': base_y + noise
            },
            'keypoints': generate_fake_keypoints(base_x, base_y + noise, standing=False),
            'confidence': 0.95
        }
        
        tracker.update(track_id, frame_data)
        time.sleep(0.033)
    
    summary = tracker.get_summary(track_id)
    print(f"\n[RESULT]")
    print(f"   Activity: {summary['current_activity']}")
    print(f"   Sitting: {summary['stats']['percent_sitting']:.1f}%")
    
    assert summary['current_activity'] == 'sitting', "Activity should be 'sitting'"
    print("\n[PASS] Test passed!")


def run_all_tests():
    """Run all HAR-System tests"""
    print("\n" + "="*60)
    print("HAR-System: Temporal Tracker Tests")
    print("="*60)
    
    tests = [
        test_stationary_person,
        test_moving_person,
        test_sitting_person,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n[FAIL] Test failed: {e}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] Test error: {e}")
    
    # Final results
    print("\n" + "="*60)
    print("[SUMMARY] Test Results")
    print("="*60)
    print(f"[PASS] Passed: {passed}")
    print(f"[FAIL] Failed: {failed}")
    print(f"[TOTAL] Total: {passed + failed}")
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {failed} test(s) failed")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    run_all_tests()
