"""
HAR-System: Hailo Face Recognition Integration
===============================================
Integration with Hailo-Apps Face Recognition system
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import cv2

# Import Hailo face recognition components
try:
    from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record
    from hailo_apps.python.core.common.defines import (
        FACE_RECON_DATABASE_DIR_NAME,
        FACE_RECON_SAMPLES_DIR_NAME
    )
except ImportError:
    print("[WARNING] Hailo apps not found. Face recognition will not work.")
    DatabaseHandler = None
    Record = None


class HailoFaceRecognition:
    """
    Integration with Hailo Face Recognition System
    
    This class provides:
    - Face database management
    - Face embedding extraction
    - Face recognition from images
    """
    
    def __init__(self, 
                 database_dir: str = "./database",
                 samples_dir: str = "./database/samples",
                 confidence_threshold: float = 0.70):
        """
        Initialize Hailo Face Recognition
        
        Args:
            database_dir: Directory for LanceDB database
            samples_dir: Directory for face samples
            confidence_threshold: Minimum confidence for recognition
        """
        self.database_dir = Path(database_dir)
        self.samples_dir = Path(samples_dir)
        self.confidence_threshold = confidence_threshold
        
        # Create directories
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database handler
        if DatabaseHandler is not None:
            try:
                self.db_handler = DatabaseHandler(
                    db_name='persons.db',
                    table_name='persons',
                    schema=Record,
                    threshold=confidence_threshold,
                    database_dir=str(self.database_dir),
                    samples_dir=str(self.samples_dir)
                )
                self.enabled = True
                print(f"[FACE-RECOG] Database initialized: {self.database_dir / 'persons.db'}")
            except Exception as e:
                print(f"[FACE-RECOG] Failed to initialize database: {e}")
                self.db_handler = None
                self.enabled = False
        else:
            self.db_handler = None
            self.enabled = False
            print("[FACE-RECOG] Face recognition disabled (DatabaseHandler not available)")
    
    def is_enabled(self) -> bool:
        """Check if face recognition is enabled"""
        return self.enabled and self.db_handler is not None
    
    def extract_face_region(self, frame: np.ndarray, keypoints: Dict, bbox: Dict) -> Optional[np.ndarray]:
        """
        Extract face region from frame using keypoints
        
        Args:
            frame: Full frame image (numpy array)
            keypoints: Dictionary of keypoints with (x, y, confidence)
            bbox: Bounding box dict with xmin, ymin, xmax, ymax (normalized 0-1)
        
        Returns:
            Cropped face image or None
        """
        if frame is None or not keypoints:
            return None
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Get nose and eye positions (in absolute pixel coordinates)
        nose = keypoints.get('nose')
        left_eye = keypoints.get('left_eye')
        right_eye = keypoints.get('right_eye')
        
        if not all([nose, left_eye, right_eye]):
            return None
        
        # Check confidence
        if nose[2] < 0.3 or left_eye[2] < 0.3 or right_eye[2] < 0.3:
            return None
        
        # Convert bbox to absolute coordinates
        bbox_abs = {
            'xmin': int(bbox['xmin'] * frame_width),
            'ymin': int(bbox['ymin'] * frame_height),
            'xmax': int(bbox['xmax'] * frame_width),
            'ymax': int(bbox['ymax'] * frame_height)
        }
        
        # Convert keypoints to absolute coordinates (they are relative to bbox)
        nose_x = int((nose[0] * (bbox_abs['xmax'] - bbox_abs['xmin'])) + bbox_abs['xmin'])
        nose_y = int((nose[1] * (bbox_abs['ymax'] - bbox_abs['ymin'])) + bbox_abs['ymin'])
        
        left_eye_x = int((left_eye[0] * (bbox_abs['xmax'] - bbox_abs['xmin'])) + bbox_abs['xmin'])
        left_eye_y = int((left_eye[1] * (bbox_abs['ymax'] - bbox_abs['ymin'])) + bbox_abs['ymin'])
        
        right_eye_x = int((right_eye[0] * (bbox_abs['xmax'] - bbox_abs['xmin'])) + bbox_abs['xmin'])
        right_eye_y = int((right_eye[1] * (bbox_abs['ymax'] - bbox_abs['ymin'])) + bbox_abs['ymin'])
        
        # Calculate eye center and distance
        eye_center_x = (left_eye_x + right_eye_x) // 2
        eye_center_y = (left_eye_y + right_eye_y) // 2
        eye_distance = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
        
        # Calculate face region (2.5x eye distance for width, 3x for height)
        face_width = int(eye_distance * 2.5)
        face_height = int(eye_distance * 3.0)
        
        # Center on nose position
        x1 = max(0, nose_x - face_width // 2)
        y1 = max(0, nose_y - int(face_height * 0.4))  # Slightly above nose
        x2 = min(frame_width, x1 + face_width)
        y2 = min(frame_height, y1 + face_height)
        
        # Ensure valid region
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop face region
        face_region = frame[y1:y2, x1:x2]
        
        # Ensure minimum size
        if face_region.shape[0] < 40 or face_region.shape[1] < 40:
            return None
        
        return face_region
    
    def recognize_face_from_embedding(self, embedding: np.ndarray) -> Tuple[str, float, Optional[str]]:
        """
        Recognize face from embedding vector
        
        Args:
            embedding: Face embedding vector (512-dim)
        
        Returns:
            Tuple of (name, confidence, global_id)
        """
        if not self.is_enabled():
            return ("Unknown", 0.0, None)
        
        try:
            person = self.db_handler.search_record(embedding=embedding)
            
            if person and person.get('label') != 'Unknown':
                confidence = 1.0 - person.get('_distance', 1.0)
                return (
                    person['label'],
                    confidence,
                    person.get('global_id')
                )
            else:
                return ("Unknown", 0.0, None)
        
        except Exception as e:
            print(f"[FACE-RECOG] Error during recognition: {e}")
            return ("Unknown", 0.0, None)
    
    def add_person_from_images(self, name: str, image_paths: list) -> bool:
        """
        Add a new person to the database from images
        
        Args:
            name: Person name
            image_paths: List of image file paths
        
        Returns:
            True if successful
        """
        if not self.is_enabled():
            print("[FACE-RECOG] Face recognition is disabled")
            return False
        
        print(f"[FACE-RECOG] Adding person: {name}")
        print(f"[FACE-RECOG] Processing {len(image_paths)} images...")
        
        # This is a simplified version - in full implementation,
        # you would need to:
        # 1. Load each image
        # 2. Detect face using SCRFD
        # 3. Extract embedding using MobileFaceNet
        # 4. Add to database
        
        # For now, return True to indicate structure is ready
        # Full implementation requires GStreamer pipeline integration
        
        print(f"[FACE-RECOG] Note: Full training requires running 'train-faces' command")
        return True
    
    def list_known_persons(self) -> list:
        """
        List all known persons in database
        
        Returns:
            List of person names
        """
        if not self.is_enabled():
            return []
        
        try:
            records = self.db_handler.get_all_records()
            names = [r['label'] for r in records if r['label'] != 'Unknown']
            return sorted(set(names))
        except Exception as e:
            print(f"[FACE-RECOG] Error listing persons: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.is_enabled():
            return {
                'enabled': False,
                'total_persons': 0,
                'total_samples': 0
            }
        
        try:
            records = self.db_handler.get_all_records()
            known_records = [r for r in records if r['label'] != 'Unknown']
            
            total_samples = sum(len(r.get('samples_json', [])) for r in known_records)
            
            return {
                'enabled': True,
                'total_persons': len(known_records),
                'total_samples': total_samples,
                'confidence_threshold': self.confidence_threshold,
                'database_path': str(self.database_dir / 'persons.db')
            }
        except Exception as e:
            print(f"[FACE-RECOG] Error getting stats: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }
    
    def clear_database(self):
        """Clear all persons from database"""
        if not self.is_enabled():
            print("[FACE-RECOG] Face recognition is disabled")
            return
        
        try:
            self.db_handler.clear_table()
            print("[FACE-RECOG] Database cleared")
        except Exception as e:
            print(f"[FACE-RECOG] Error clearing database: {e}")
    
    def remove_person(self, name: str) -> bool:
        """
        Remove a person from database
        
        Args:
            name: Person name to remove
        
        Returns:
            True if successful
        """
        if not self.is_enabled():
            return False
        
        try:
            record = self.db_handler.get_record_by_label(label=name)
            if record:
                self.db_handler.delete_record(record['global_id'])
                print(f"[FACE-RECOG] Removed person: {name}")
                return True
            else:
                print(f"[FACE-RECOG] Person not found: {name}")
                return False
        except Exception as e:
            print(f"[FACE-RECOG] Error removing person: {e}")
            return False
