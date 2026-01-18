"""
HAR-System: Callback Handlers
==============================
GStreamer callback handlers for HAR processing
"""

import time
from typing import Dict, Any
import numpy as np
import hailo
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.core.common.buffer_utils import get_numpy_from_buffer_efficient, get_caps_from_pad

class HARCallbackHandler(app_callback_class):
    """Callback handler for HAR-System processing"""
    
    def __init__(self, temporal_tracker, config: Dict[str, Any], 
                 face_identity_manager=None, face_processor=None):
        """
        Initialize callback handler
        
        Args:
            temporal_tracker: TemporalActivityTracker instance
            config: Configuration dictionary
            face_identity_manager: FaceIdentityManager instance (optional)
            face_processor: FaceRecognitionProcessor instance (optional)
        """
        super().__init__()
        
        self.temporal_tracker = temporal_tracker
        self.config = config
        self.verbose = config.get('verbose', False)
        self.print_every_n_frames = config.get('print_every_n_frames', 30)
        self.save_data = config.get('save_data', False)
        self.output_dir = config.get('output_dir', './results/camera')
        self.save_interval = int(config.get('save_interval', 300))
        
        # Face recognition components
        self.face_identity_manager = face_identity_manager
        self.face_processor = face_processor
        self.face_recognition_enabled = (face_identity_manager is not None and 
                                        face_processor is not None and
                                        face_processor.is_enabled())
        
        # Face recognition throttling:
        # we do not attempt recognition on every frame to reduce CPU load.
        self.recognition_interval = config.get('face_recognition', {}).get('recognition_interval_frames', 15)
        self.skip_first_frames = config.get('face_recognition', {}).get('skip_first_frames', 5)
        self.track_frame_counts = {}  # Track frame count per track_id
        
        # Statistics
        # Used for lightweight timing/FPS estimates in the callback path.
        self.frame_times = []
        self.last_summary_time = time.time()
        
        if self.face_recognition_enabled:
            print("[CALLBACK] Face recognition enabled with processor")
            stats = self.face_processor.get_database_stats()
            print(f"[CALLBACK] Known persons: {stats.get('total_persons', 0)}")
            if stats.get('persons'):
                print(f"[CALLBACK] Trained: {', '.join(stats.get('persons', []))}")
    
    def get_tracker(self):
        """Get temporal tracker instance"""
        return self.temporal_tracker

    def get_face_identity_manager(self):
        """Get face identity manager instance"""
        return self.face_identity_manager

def get_keypoint_mapping():
    """Get mapping of keypoint names to indices"""
    return {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

def extract_frame_data(detection, keypoint_map: Dict) -> Dict[str, Any]:
    """
    Extract frame data from Hailo detection
    
    Args:
        detection: Hailo detection object
        keypoint_map: Keypoint name to index mapping
    
    Returns:
        Dictionary with frame data
    """
    # Extract Track ID
    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
    if len(track) != 1:
        return None
    track_id = track[0].get_id()
    
    # Extract bounding box (normalized coordinates in [0..1]).
    bbox_obj = detection.get_bbox()
    bbox = {
        'xmin': bbox_obj.xmin(),
        'ymin': bbox_obj.ymin(),
        'xmax': bbox_obj.xmax(),
        'ymax': bbox_obj.ymax()
    }
    
    # Extract keypoints (COCO-17 style via Hailo landmarks).
    keypoints = {}
    landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
    
    if landmarks:
        points = landmarks[0].get_points()
        for name, idx in keypoint_map.items():
            if idx < len(points):
                p = points[idx]
                keypoints[name] = (p.x(), p.y(), p.confidence())
    
    return {
        'track_id': track_id,
        'timestamp': time.time(),
        'bbox': bbox,
        'keypoints': keypoints,
        'confidence': detection.get_confidence()
    }

def extract_eye_positions(detection, keypoint_map: Dict, frame_width: int, frame_height: int) -> tuple:
    """
    Extract eye positions from detection
    
    Args:
        detection: Hailo detection object
        keypoint_map: Keypoint name to index mapping
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        (person_id, left_eye_x, left_eye_y, right_eye_x, right_eye_y) or None
    """
    # Extract Track ID
    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
    if len(track) != 1:
        return None
    person_id = track[0].get_id()
    
    # Extract Bounding Box
    bbox_obj = detection.get_bbox()
    
    # Extract Keypoints
    landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
    if not landmarks:
        return None
    
    points = landmarks[0].get_points()
    
    # Extract left eye position
    left_eye_idx = keypoint_map.get("left_eye", 1)
    if left_eye_idx >= len(points):
        return None
    left_eye_point = points[left_eye_idx]
    left_eye_x = int((left_eye_point.x() * bbox_obj.width() + bbox_obj.xmin()) * frame_width)
    left_eye_y = int((left_eye_point.y() * bbox_obj.height() + bbox_obj.ymin()) * frame_height)
    
    # Extract right eye position
    right_eye_idx = keypoint_map.get("right_eye", 2)
    if right_eye_idx >= len(points):
        return None
    right_eye_point = points[right_eye_idx]
    right_eye_x = int((right_eye_point.x() * bbox_obj.width() + bbox_obj.xmin()) * frame_width)
    right_eye_y = int((right_eye_point.y() * bbox_obj.height() + bbox_obj.ymin()) * frame_height)
    
    return (person_id, left_eye_x, left_eye_y, right_eye_x, right_eye_y)

def print_frame_summary(frame_count: int, active_tracks: list, temporal_tracker):
    """Print periodic frame summary"""
    print(f"\n{'='*60}")
    print(f"[FRAME] {frame_count} | Active People: {len(active_tracks)}")
    print(f"{'='*60}")
    
    for track_id in active_tracks:
        summary = temporal_tracker.get_summary(track_id)
        if summary:
            name = summary.get('name', 'Unknown')
            display_name = f"{track_id} - {name}" if name != 'Unknown' else str(track_id)
            
            print(f"\n  [TRACK] {display_name}:")
            print(f"     Activity: {summary['current_activity']}")
            print(f"     Duration: {summary['duration_seconds']:.1f}s")
            print(f"     Normalized Distance: {summary['stats']['total_distance_normalized']:.2f}")
            print(f"     Moving: {summary['stats']['percent_moving']:.1f}%")
            print(f"     Stationary: {summary['stats']['percent_stationary']:.1f}%")
            print(f"     Sitting: {summary['stats']['percent_sitting']:.1f}%")
            
            if summary['stats']['fall_detected']:
                print(f"     [WARNING] Fall detected!")
    
    # Global statistics
    global_stats = temporal_tracker.get_global_stats()
    print(f"\n  [GLOBAL] Statistics:")
    print(f"     Total People: {global_stats['total_tracks_seen']}")
    print(f"     Falls Detected: {global_stats['total_falls_detected']}")
    print(f"     Activity Changes: {global_stats['total_activity_changes']}")

def process_frame_callback(element, buffer, user_data):
    """
    Main callback for processing each frame
    
    Args:
        element: GStreamer element
        buffer: GStreamer buffer
        user_data: HARCallbackHandler instance
    """
    if buffer is None:
        return
    
    frame_start_time = time.time()
    frame_count = user_data.get_count()
    
    # Extract detections from buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    keypoint_map = get_keypoint_mapping()
    
    # Process each detected person
    for detection in detections:
        label = detection.get_label()
        
        if label != "person":
            continue
        
        # Extract frame data
        frame_data = extract_frame_data(detection, keypoint_map)
        if frame_data is None:
            continue
        
        track_id = frame_data.pop('track_id')
        
        # Update temporal tracker
        try:
            activity = user_data.temporal_tracker.update(track_id, frame_data)
            
            # Face recognition processing
            if user_data.face_recognition_enabled:
                # Initialize frame count for new tracks
                if track_id not in user_data.track_frame_counts:
                    user_data.track_frame_counts[track_id] = 0
                
                user_data.track_frame_counts[track_id] += 1
                
                # Check if we should attempt face recognition
                track_frames = user_data.track_frame_counts[track_id]
                
                # Skip first few frames (usually blurry)
                if track_frames > user_data.skip_first_frames:
                    # Check if recognition is needed (at intervals)
                    if (track_frames - user_data.skip_first_frames) % user_data.recognition_interval == 0:
                        # Check if track needs recognition
                        if user_data.face_identity_manager.needs_recognition(track_id):
                            # Extract frame image from buffer
                            try:
                                pad = element.get_static_pad("src")
                                if pad:
                                    format_str, frame_width, frame_height = get_caps_from_pad(pad)
                                    frame_image = get_numpy_from_buffer_efficient(buffer, format_str, frame_width, frame_height)
                                    
                                    # Extract bbox for face region
                                    bbox_obj = detection.get_bbox()
                                    bbox = {
                                        'xmin': bbox_obj.xmin(),
                                        'ymin': bbox_obj.ymin(),
                                        'xmax': bbox_obj.xmax(),
                                        'ymax': bbox_obj.ymax()
                                    }
                                    
                                    # Perform face recognition
                                    name, confidence, global_id = user_data.face_processor.recognize_from_keypoints(
                                        frame_image, frame_data['keypoints'], bbox, frame_width, frame_height
                                    )
                                    
                                    # Update identity
                                    if name != "Unknown":
                                        updated = user_data.face_identity_manager.update_identity(
                                            track_id, name, confidence, global_id
                                        )
                                        if updated:
                                            print(f"[RECOGNITION] Track #{track_id} recognized as: {name} ({confidence:.2f})")
                            except Exception as e:
                                print(f"[ERROR] Face recognition failed: {e}")
            
            # Update identity in tracker
            if user_data.face_recognition_enabled:
                name = user_data.face_identity_manager.get_identity(track_id)
                user_data.temporal_tracker.update_identity(track_id, name)
                
                # Note: We cannot modify detection.label in Hailo (read-only)
                # The name will be shown in terminal output and saved in JSON
                # For video overlay, you would need to use a custom overlay element
            
            # Detect activity change
            change = user_data.temporal_tracker.detect_activity_change(track_id)
            if change:
                name = user_data.temporal_tracker.get_identity(track_id)
                display_id = f"{track_id} ({name})" if name != 'Unknown' else str(track_id)
                print(f"\n[CHANGE] Track {display_id}: {change['from']} → {change['to']}")
            
        except Exception as e:
            print(f"[ERROR] Error updating tracker: {e}")
    
    # Print periodic summary
    if frame_count % user_data.print_every_n_frames == 0:
        active_tracks = user_data.temporal_tracker.get_all_active_tracks()
        print_frame_summary(frame_count, active_tracks, user_data.temporal_tracker)
    
    # Save data (optional)
    if user_data.save_data and frame_count % user_data.save_interval == 0:
        import os
        for track_id in user_data.temporal_tracker.get_all_active_tracks():
            filepath = os.path.join(
                user_data.output_dir, 
                f"track_{track_id}_frame_{frame_count}.json"
            )
            user_data.temporal_tracker.save_to_json(track_id, filepath)
    
    # Calculate processing time
    frame_time = time.time() - frame_start_time
    user_data.frame_times.append(frame_time)
    
    # Show frame rate every 5 seconds
    if time.time() - user_data.last_summary_time > 5.0:
        if user_data.frame_times:
            avg_time = sum(user_data.frame_times) / len(user_data.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\n[PERF] Average processing time: {avg_time*1000:.1f}ms | FPS: {fps:.1f}")
            user_data.frame_times = []
            user_data.last_summary_time = time.time()
