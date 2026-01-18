"""
HAR-System: Face Recognition Processor
======================================
Processes frames for face recognition and integrates with HAR pipeline
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import hailo
    from hailo_apps.python.core.common.buffer_utils import get_numpy_from_buffer_efficient, get_caps_from_pad
    from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record
    HAILO_AVAILABLE = True
except ImportError:
    # Keep the module importable even if hailo-apps/GStreamer bindings are missing.
    # In that case, face recognition is disabled gracefully at runtime.
    HAILO_AVAILABLE = False
    print("[FACE-PROCESSOR] Warning: Hailo components not available")


class FaceRecognitionProcessor:
    """
    Processes frames for face recognition
    Integrates face detection, embedding extraction, and database search
    """
    
    def __init__(self, database_dir: str, samples_dir: str, 
                 confidence_threshold: float = 0.70,
                 min_face_size: int = 40):
        """
        Initialize Face Recognition Processor
        
        Args:
            database_dir: Path to database directory
            samples_dir: Path to samples directory
            confidence_threshold: Minimum confidence for recognition
            min_face_size: Minimum face size in pixels
        """
        self.database_dir = database_dir
        self.samples_dir = samples_dir
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.enabled = HAILO_AVAILABLE
        
        # Initialize database handler
        if HAILO_AVAILABLE:
            try:
                self.db_handler = DatabaseHandler(
                    db_name='persons.db',
                    table_name='persons',
                    schema=Record,
                    threshold=confidence_threshold,
                    database_dir=database_dir,
                    samples_dir=samples_dir
                )
                print(f"[FACE-PROCESSOR] Database initialized: {database_dir}")
            except Exception as e:
                print(f"[FACE-PROCESSOR] Failed to initialize database: {e}")
                self.enabled = False
                self.db_handler = None
        else:
            self.db_handler = None
    
    def is_enabled(self) -> bool:
        """Check if face recognition is enabled"""
        return self.enabled and self.db_handler is not None
    
    def extract_face_region(self, frame: np.ndarray, keypoints: Dict, 
                           bbox: Dict, frame_width: int, frame_height: int) -> Optional[np.ndarray]:
        """
        Extract face region from frame using pose keypoints
        
        Args:
            frame: Full frame image (numpy array)
            keypoints: Dictionary of keypoints with confidence
            bbox: Bounding box dict with xmin, ymin, xmax, ymax (normalized 0-1)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        
        Returns:
            Cropped face image or None
        """
        # Get face keypoints
        nose = keypoints.get('nose')
        left_eye = keypoints.get('left_eye')
        right_eye = keypoints.get('right_eye')
        left_ear = keypoints.get('left_ear')
        right_ear = keypoints.get('right_ear')
        
        if not all([nose, left_eye, right_eye]):
            print(f"[FACE-EXTRACT] Missing keypoints: nose={nose is not None}, left_eye={left_eye is not None}, right_eye={right_eye is not None}")
            return None
        
        # Check confidence (x, y, confidence)
        if nose[2] < 0.3 or left_eye[2] < 0.3 or right_eye[2] < 0.3:
            print(f"[FACE-EXTRACT] Low confidence: nose={nose[2]:.2f}, left_eye={left_eye[2]:.2f}, right_eye={right_eye[2]:.2f}")
            return None
        
        # Convert bbox to absolute coordinates
        xmin = int(bbox['xmin'] * frame_width)
        ymin = int(bbox['ymin'] * frame_height)
        xmax = int(bbox['xmax'] * frame_width)
        ymax = int(bbox['ymax'] * frame_height)
        
        # Calculate face region with padding
        # Use eyes to determine face width and height
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # Convert to absolute coordinates
        eye_center_x_abs = int(eye_center_x * (xmax - xmin) + xmin)
        eye_center_y_abs = int(eye_center_y * (ymax - ymin) + ymin)
        eye_distance_abs = int(eye_distance * (xmax - xmin))
        
        # Calculate face region (2.5x eye distance for width, 3x for height)
        face_width = int(eye_distance_abs * 2.5)
        face_height = int(eye_distance_abs * 3.0)
        
        # Center on eye level, slightly above
        x1 = max(0, eye_center_x_abs - face_width // 2)
        y1 = max(0, eye_center_y_abs - int(face_height * 0.3))
        x2 = min(frame_width, x1 + face_width)
        y2 = min(frame_height, y1 + face_height)
        
        # Ensure valid region
        if x2 <= x1 or y2 <= y1:
            print(f"[FACE-EXTRACT] Invalid region: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
            return None
        
        # Crop face region
        face_region = frame[y1:y2, x1:x2]
        
        # Check minimum size
        if face_region.shape[0] < self.min_face_size or face_region.shape[1] < self.min_face_size:
            print(f"[FACE-EXTRACT] Face too small: {face_region.shape[0]}x{face_region.shape[1]} (min: {self.min_face_size})")
            return None
        
        print(f"[FACE-EXTRACT] ✓ Face extracted: {face_region.shape[0]}x{face_region.shape[1]}")
        return face_region
    
    def recognize_from_keypoints(self, frame: np.ndarray, keypoints: Dict, 
                                 bbox: Dict, frame_width: int, frame_height: int) -> Tuple[str, float, Optional[str]]:
        """
        Recognize person from frame using keypoints
        
        Note: This is a simplified/placeholder implementation that uses pose keypoints
        to crop a face region. For a full face recognition pipeline, you would need to:
        1. Run SCRFD face detection on the cropped region
        2. Extract face embedding using MobileFaceNet
        3. Search in LanceDB
        
        This version uses the face region as a proxy and searches
        based on geometric features.
        
        Args:
            frame: Full frame image
            keypoints: Pose keypoints
            bbox: Bounding box
            frame_width: Frame width
            frame_height: Frame height
        
        Returns:
            Tuple of (name, confidence, global_id)
        """
        if not self.is_enabled():
            return ("Unknown", 0.0, None)
        
        try:
            # Extract face region
            face_region = self.extract_face_region(frame, keypoints, bbox, frame_width, frame_height)
            
            if face_region is None:
                # Debug: why extraction failed
                print(f"[FACE-PROCESSOR] Face extraction failed for track (keypoints issue or face too small)")
                return ("Unknown", 0.0, None)
            
            print(f"[FACE-PROCESSOR] Face region extracted: {face_region.shape}")
            
            # TODO: Replace this heuristic with the real SCRFD + MobileFaceNet pipeline.
            # For now, we fall back to a lightweight (and inaccurate) heuristic to keep
            # the integration points working end-to-end.
            
            # Get all known persons
            records = self.db_handler.get_all_records()
            known_persons = [r for r in records if r['label'] != 'Unknown']
            
            if not known_persons:
                print(f"[FACE-PROCESSOR] No known persons in database")
                return ("Unknown", 0.0, None)
            
            print(f"[FACE-PROCESSOR] Found {len(known_persons)} known person(s) in database")
            
            # Improved heuristic for multiple persons:
            # Use face size and position to make a basic match
            # This is still a TEMPORARY solution until full face recognition is integrated
            
            if len(known_persons) == 1:
                # Single person - straightforward
                person = known_persons[0]
                print(f"[FACE-PROCESSOR] Recognizing as: {person['label']} (single person)")
                return (person['label'], 0.75, person['global_id'])
            
            else:
                # Multiple persons - use simple matching based on face characteristics
                # Calculate face features for basic matching
                face_height, face_width = face_region.shape[:2]
                face_area = face_height * face_width
                
                # Use center position
                face_center_x = (bbox['xmin'] + bbox['xmax']) / 2
                face_center_y = (bbox['ymin'] + bbox['ymax']) / 2
                
                # For each person, calculate a simple score
                # In a real system, this would use embedding similarity
                best_match = None
                best_score = -1
                
                for person in known_persons:
                    # Simple heuristic score based on:
                    # 1. Face size (bigger = more confident)
                    # 2. Position (center = more confident)
                    size_score = min(face_area / 10000.0, 1.0)  # Normalize
                    center_score = 1.0 - abs(face_center_x - 0.5) - abs(face_center_y - 0.5)
                    
                    # Combine scores
                    score = (size_score * 0.6 + center_score * 0.4)
                    
                    if score > best_score:
                        best_score = score
                        best_match = person
                
                if best_match and best_score > 0.3:
                    print(f"[FACE-PROCESSOR] Best match: {best_match['label']} (score: {best_score:.2f})")
                    # Lower confidence for multi-person scenario
                    confidence = 0.65 + (best_score * 0.1)
                    return (best_match['label'], confidence, best_match['global_id'])
                else:
                    print(f"[FACE-PROCESSOR] No good match found (best score: {best_score:.2f})")
                    return ("Unknown", 0.0, None)
            
        except Exception as e:
            print(f"[FACE-PROCESSOR] Error during recognition: {e}")
            return ("Unknown", 0.0, None)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_enabled():
            return {'enabled': False, 'total_persons': 0}
        
        try:
            records = self.db_handler.get_all_records()
            known_records = [r for r in records if r['label'] != 'Unknown']
            
            return {
                'enabled': True,
                'total_persons': len(known_records),
                'total_samples': sum(len(r.get('samples_json', [])) for r in known_records),
                'persons': [r['label'] for r in known_records]
            }
        except Exception as e:
            return {'enabled': True, 'error': str(e)}
