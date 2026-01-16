"""
HAR-System: Temporal Activity Tracker
======================================
Temporal tracking and analysis module for human activity recognition

Core Principles:
1. All measurements are normalized relative to person size
2. Use real timestamps instead of frame counts
3. Store only raw data + compute derivatives on demand
4. Simple and stable classification rules

Usage:
    tracker = TemporalActivityTracker()
    activity = tracker.update(track_id, frame_data)
"""

from collections import defaultdict, deque
import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple, Any


class TemporalActivityTracker:
    """HAR-System: Temporal Activity Tracker - Tracks human activities over time"""
    
    def __init__(self, history_seconds: float = 3.0, fps_estimate: int = 15):
        """
        Initialize the temporal layer
        
        Args:
            history_seconds: How many seconds of history to keep (3 seconds is sufficient for start)
            fps_estimate: Approximate FPS estimate for calculating deque size
        """
        self.history_frames = int(history_seconds * fps_estimate)
        self.fps_estimate = fps_estimate
        
        # Store data for each track_id
        self.tracks = defaultdict(lambda: self._create_new_track())
        
        # Threshold settings (Thresholds) - adjustable
        self.thresholds = {
            'speed_stationary': 0.1,      # Below this = stationary
            'speed_slow': 0.5,             # Between this and next = slow
            'speed_fast': 1.5,             # Above this = fast
            'hip_ratio_sitting': 0.62,     # Above this = sitting
            'fall_drop_ratio': 0.30,       # Drop > 30% = potential fall
            'fall_time_threshold': 0.5,    # Within less than 0.5 seconds
        }
        
        # Global statistics
        self.global_stats = {
            'total_tracks_seen': 0,
            'active_tracks': 0,
            'total_falls_detected': 0,
            'total_activity_changes': 0,
        }
    
    def _create_new_track(self) -> Dict:
        """Create a new record for a new person"""
        return {
            # ════════════════════════════════════
            # Raw Data
            # ════════════════════════════════════
            'timestamps': deque(maxlen=self.history_frames),
            'positions': deque(maxlen=self.history_frames),    # (x, y) center
            'bboxes': deque(maxlen=self.history_frames),       # {xmin, ymin, xmax, ymax}
            'keypoints': deque(maxlen=self.history_frames),    # dict of 17 points
            'confidences': deque(maxlen=self.history_frames),
            
            # ════════════════════════════════════
            # Metadata
            # ════════════════════════════════════
            'first_seen': None,
            'last_seen': None,
            'is_active': True,
            'total_frames': 0,
            
            # ════════════════════════════════════
            # Current State
            # ════════════════════════════════════
            'current_activity': 'unknown',
            'previous_activity': 'unknown',
            
            # ════════════════════════════════════
            # Cumulative Statistics
            # ════════════════════════════════════
            'stats': {
                'total_distance_norm': 0.0,
                'frames_stationary': 0,
                'frames_moving': 0,
                'frames_sitting': 0,
                'fall_detected': False,
                'fall_timestamp': None,
                'activity_changes': [],  # Change history only
            }
        }
    
    def update(self, track_id: int, frame_data: Dict[str, Any]) -> str:
        """
        Update data for a specific person in a new frame
        
        Args:
            track_id: Unique tracking number
            frame_data: {
                'timestamp': float,
                'bbox': {'xmin': float, 'ymin': float, 'xmax': float, 'ymax': float},
                'keypoints': dict of 17 points: {name: (x, y, confidence)},
                'confidence': float
            }
        
        Returns:
            Current detected activity (str)
        """
        track = self.tracks[track_id]
        timestamp = frame_data['timestamp']
        bbox = frame_data['bbox']
        
        # ════════════════════════════════════
        # 1. Record new appearance
        # ════════════════════════════════════
        if track['first_seen'] is None:
            track['first_seen'] = timestamp
            self.global_stats['total_tracks_seen'] += 1
            self.global_stats['active_tracks'] += 1
            print(f"[NEW] Person entered scene: Track ID {track_id}")
        
        # ════════════════════════════════════
        # 2. Store raw data
        # ════════════════════════════════════
        track['last_seen'] = timestamp
        track['total_frames'] += 1
        track['timestamps'].append(timestamp)
        track['bboxes'].append(bbox)
        track['keypoints'].append(frame_data['keypoints'])
        track['confidences'].append(frame_data['confidence'])
        
        # Calculate and store bbox center
        center = self._get_bbox_center(bbox)
        track['positions'].append(center)
        
        # ════════════════════════════════════
        # 3. Calculate derivatives (if sufficient history)
        # ════════════════════════════════════
        if len(track['positions']) >= 2:
            # Normalized speed
            speed_norm = self._calculate_normalized_speed(track)
            
            # Update statistics
            if speed_norm < self.thresholds['speed_stationary']:
                track['stats']['frames_stationary'] += 1
            else:
                track['stats']['frames_moving'] += 1
                track['stats']['total_distance_norm'] += speed_norm
        
        # ════════════════════════════════════
        # 4. Classify activity
        # ════════════════════════════════════
        if len(track['positions']) >= 10:
            track['previous_activity'] = track['current_activity']
            track['current_activity'] = self._classify_activity_simple(track)
            
            # Record activity change
            if (track['previous_activity'] != 'unknown' and 
                track['current_activity'] != track['previous_activity']):
                
                track['stats']['activity_changes'].append({
                    'timestamp': timestamp,
                    'from': track['previous_activity'],
                    'to': track['current_activity']
                })
                self.global_stats['total_activity_changes'] += 1
            
            # Update activity counters
            if track['current_activity'] == 'sitting':
                track['stats']['frames_sitting'] += 1
        
        # ════════════════════════════════════
        # 5. Fall detection
        # ════════════════════════════════════
        if len(track['keypoints']) >= 15:
            if self._detect_fall_simple(track):
                if not track['stats']['fall_detected']:
                    track['stats']['fall_detected'] = True
                    track['stats']['fall_timestamp'] = timestamp
                    self.global_stats['total_falls_detected'] += 1
                    print(f"[WARNING] Potential fall detected - Track ID {track_id} at time {timestamp:.2f}")
        
        return track['current_activity']
    
    # ════════════════════════════════════════════════════════
    # Helper Functions - Normalized Measurements
    # ════════════════════════════════════════════════════════
    
    def _get_bbox_center(self, bbox: Dict) -> Tuple[float, float]:
        """Calculate bounding box center"""
        cx = (bbox['xmin'] + bbox['xmax']) / 2
        cy = (bbox['ymin'] + bbox['ymax']) / 2
        return (cx, cy)
    
    def _get_bbox_height(self, bbox: Dict) -> float:
        """Calculate bounding box height"""
        return bbox['ymax'] - bbox['ymin']
    
    def _get_bbox_width(self, bbox: Dict) -> float:
        """Calculate bounding box width"""
        return bbox['xmax'] - bbox['xmin']
    
    def _calculate_normalized_speed(self, track: Dict, window: int = 10) -> float:
        """
        Calculate normalized speed
        
        Normalized relative to:
        - Real time (seconds)
        - Person size (bbox height)
        
        Returns:
            Relative speed (0.0 = stationary, 1.0 = normal speed, 2.0+ = fast)
        """
        positions = list(track['positions'])
        timestamps = list(track['timestamps'])
        bboxes = list(track['bboxes'])
        
        if len(positions) < 2:
            return 0.0
        
        # Use short window to reduce noise
        window = min(window, len(positions))
        recent_positions = positions[-window:]
        recent_timestamps = timestamps[-window:]
        recent_bboxes = bboxes[-window:]
        
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            total_distance += distance
        
        # Real time
        dt = recent_timestamps[-1] - recent_timestamps[0]
        if dt <= 0:
            return 0.0
        
        # Speed in pixels/second
        speed_px_per_sec = total_distance / dt
        
        # Normalization: divide by person height
        avg_height = np.mean([self._get_bbox_height(b) for b in recent_bboxes])
        if avg_height <= 0:
            return 0.0
        
        # Normalized speed (relative to person height/second)
        speed_normalized = speed_px_per_sec / avg_height
        
        return speed_normalized
    
    def _calculate_pose_height_normalized(self, keypoints: Dict, bbox: Dict) -> Optional[float]:
        """
        Calculate normalized pose height
        
        Returns:
            Ratio from 0.0 to ~1.0 (standing: 0.85-0.95, sitting: 0.6-0.75)
        """
        nose = keypoints.get('nose')
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')
        
        # Check for required points
        if not nose or not left_ankle or not right_ankle:
            return None
        
        # Average ankle position (y coordinate)
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Distance from nose to ankle
        pose_height_px = abs(ankle_y - nose[1])
        
        # Normalize relative to bbox height
        bbox_height = self._get_bbox_height(bbox)
        if bbox_height <= 0:
            return None
        
        height_ratio = pose_height_px / bbox_height
        
        return height_ratio
    
    def _calculate_hip_ratio(self, keypoints: Dict, bbox: Dict) -> Optional[float]:
        """
        Calculate hip position ratio (to distinguish between standing/sitting)
        
        Returns:
            - standing: ~0.45-0.55
            - sitting: ~0.65-0.80
        """
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')
        
        if not all([left_hip, right_hip, left_ankle, right_ankle]):
            return None
        
        # Average hip position
        hip_y = (left_hip[1] + right_hip[1]) / 2
        
        # Average ankle position
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Bbox height
        bbox_height = self._get_bbox_height(bbox)
        if bbox_height <= 0:
            return None
        
        # Ratio: distance from ankle to hip (ankle is lower, so ankle_y > hip_y)
        # For standing: ~0.45-0.55, for sitting: ~0.65-0.80
        hip_ratio = (ankle_y - hip_y) / bbox_height
        
        return hip_ratio
    
    # ════════════════════════════════════════════════════════
    # Classification and Detection
    # ════════════════════════════════════════════════════════
    
    def _classify_activity_simple(self, track: Dict, window: int = 30) -> str:
        """
        Classify activity - 3 basic categories
        
        - stationary: stationary/standing
        - moving: moving (walking/running)
        - sitting: sitting
        
        Returns:
            Activity name (str)
        """
        # Calculate speed in short window
        speed = self._calculate_normalized_speed(track, window)
        
        # Get latest keypoints
        if len(track['keypoints']) == 0:
            return 'unknown'
        
        # Use last 5 frames to calculate average hip_ratio
        recent_keypoints = list(track['keypoints'])[-5:]
        recent_bboxes = list(track['bboxes'])[-5:]
        
        hip_ratios = []
        for kp, bbox in zip(recent_keypoints, recent_bboxes):
            ratio = self._calculate_hip_ratio(kp, bbox)
            if ratio is not None:
                hip_ratios.append(ratio)
        
        avg_hip_ratio = np.mean(hip_ratios) if hip_ratios else 0.5
        
        # ════════════════════════════════════
        # Classification rules (simple and stable)
        # ════════════════════════════════════
        
        # Sitting: hip relatively high + low speed
        if (avg_hip_ratio > self.thresholds['hip_ratio_sitting'] and 
            speed < self.thresholds['speed_stationary'] * 1.5):
            return 'sitting'
        
        # Stationary/standing: very low speed
        if speed < self.thresholds['speed_stationary']:
            return 'stationary'
        
        # Moving: any noticeable speed
        return 'moving'
    
    def _detect_fall_simple(self, track: Dict, window: int = 15) -> bool:
        """
        Simple fall detection
        
        Fall = rapid drop in pose height within short time
        
        Returns:
            True if potential fall detected
        """
        if len(track['keypoints']) < window:
            return False
        
        recent_keypoints = list(track['keypoints'])[-window:]
        recent_bboxes = list(track['bboxes'])[-window:]
        recent_timestamps = list(track['timestamps'])[-window:]
        
        # Calculate pose height for each frame
        heights = []
        for kp, bbox in zip(recent_keypoints, recent_bboxes):
            h = self._calculate_pose_height_normalized(kp, bbox)
            if h is not None:
                heights.append(h)
        
        if len(heights) < 5:
            return False
        
        # Compare start and end of window
        start_height = np.mean(heights[:3])
        end_height = np.mean(heights[-3:])
        
        # Drop ratio
        drop_ratio = (start_height - end_height) / start_height if start_height > 0 else 0
        
        # Time elapsed
        dt = recent_timestamps[-1] - recent_timestamps[0]
        
        # Fall rule: large drop within short time
        if (drop_ratio > self.thresholds['fall_drop_ratio'] and 
            dt < self.thresholds['fall_time_threshold']):
            return True
        
        return False
    
    # ════════════════════════════════════════════════════════
    # Query Functions - Get Information
    # ════════════════════════════════════════════════════════
    
    def get_activity(self, track_id: int) -> str:
        """Get current activity for a specific person"""
        return self.tracks[track_id]['current_activity']
    
    def get_summary(self, track_id: int) -> Optional[Dict]:
        """
        Get comprehensive summary for a specific person
        
        Returns:
            dict with all statistics or None if not seen yet
        """
        track = self.tracks[track_id]
        
        if track['first_seen'] is None:
            return None
        
        duration = track['last_seen'] - track['first_seen']
        total_frames = track['total_frames']
        
        # Calculate percentages
        percent_moving = (track['stats']['frames_moving'] / total_frames * 100) if total_frames > 0 else 0
        percent_stationary = (track['stats']['frames_stationary'] / total_frames * 100) if total_frames > 0 else 0
        percent_sitting = (track['stats']['frames_sitting'] / total_frames * 100) if total_frames > 0 else 0
        
        return {
            'track_id': track_id,
            'duration_seconds': duration,
            'total_frames': total_frames,
            'current_activity': track['current_activity'],
            'stats': {
                'total_distance_normalized': track['stats']['total_distance_norm'],
                'percent_moving': percent_moving,
                'percent_stationary': percent_stationary,
                'percent_sitting': percent_sitting,
                'fall_detected': track['stats']['fall_detected'],
                'fall_timestamp': track['stats']['fall_timestamp'],
                'total_activity_changes': len(track['stats']['activity_changes']),
            },
            'activity_history': track['stats']['activity_changes'][-5:]  # Last 5 changes
        }
    
    def detect_activity_change(self, track_id: int) -> Optional[Dict]:
        """
        Detect change in activity
        
        Returns:
            dict with change details or None if no change
        """
        track = self.tracks[track_id]
        
        if (track['current_activity'] != track['previous_activity'] and 
            track['previous_activity'] != 'unknown'):
            return {
                'track_id': track_id,
                'from': track['previous_activity'],
                'to': track['current_activity'],
                'timestamp': track['last_seen']
            }
        
        return None
    
    def get_all_active_tracks(self) -> List[int]:
        """Get list of all currently active people"""
        current_time = time.time()
        active = []
        
        for track_id, track in self.tracks.items():
            # Consider person active if seen within last 2 seconds
            if track['last_seen'] and (current_time - track['last_seen']) < 2.0:
                active.append(track_id)
        
        return active
    
    def get_global_stats(self) -> Dict:
        """Get global system statistics"""
        return {
            'total_tracks_seen': self.global_stats['total_tracks_seen'],
            'active_tracks': len(self.get_all_active_tracks()),
            'total_falls_detected': self.global_stats['total_falls_detected'],
            'total_activity_changes': self.global_stats['total_activity_changes'],
        }
    
    def export_track_data(self, track_id: int) -> Dict:
        """
        Export all data for a specific person (for saving or sending)
        
        Returns:
            dict with all raw and derived data
        """
        track = self.tracks[track_id]
        
        if track['first_seen'] is None:
            return {}
        
        return {
            'track_id': track_id,
            'metadata': {
                'first_seen': track['first_seen'],
                'last_seen': track['last_seen'],
                'total_frames': track['total_frames'],
                'duration_seconds': track['last_seen'] - track['first_seen'],
            },
            'current_state': {
                'activity': track['current_activity'],
                'last_position': list(track['positions'])[-1] if track['positions'] else None,
                'last_bbox': list(track['bboxes'])[-1] if track['bboxes'] else None,
            },
            'statistics': track['stats'],
            'raw_data': {
                'timestamps': list(track['timestamps']),
                'positions': list(track['positions']),
                # Can add more as needed
            }
        }
    
    def save_to_json(self, track_id: int, filepath: str):
        """Save data for a specific person to JSON file"""
        data = self.export_track_data(track_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Track {track_id} data saved to {filepath}")
