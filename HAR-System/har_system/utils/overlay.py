"""
HAR-System: Custom Video Overlay
=================================
Custom overlay to display person names and activity on video
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


class PersonOverlay:
    """
    Custom overlay for displaying person information on video
    """
    
    def __init__(self, font_scale=0.6, thickness=2):
        """
        Initialize overlay
        
        Args:
            font_scale: Font size scale
            thickness: Text thickness
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'known': (0, 255, 0),      # Green for known persons
            'unknown': (0, 165, 255),  # Orange for unknown
            'text_bg': (0, 0, 0),      # Black background
            'bbox': (0, 255, 0),       # Green bbox
        }
    
    def draw_person_info(self, frame: np.ndarray, bbox: Dict, 
                        track_id: int, name: str, confidence: float,
                        activity: str) -> np.ndarray:
        """
        Draw person information on frame
        
        Args:
            frame: Video frame (numpy array)
            bbox: Bounding box dict with xmin, ymin, xmax, ymax (normalized)
            track_id: Track ID
            name: Person name
            confidence: Recognition confidence (0-1)
            activity: Current activity
        
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        
        # Convert normalized bbox to pixels
        x1 = int(bbox['xmin'] * w)
        y1 = int(bbox['ymin'] * h)
        x2 = int(bbox['xmax'] * w)
        y2 = int(bbox['ymax'] * h)
        
        # Choose color based on recognition
        color = self.colors['known'] if name != 'Unknown' else self.colors['unknown']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        if name != 'Unknown' and confidence > 0:
            text = f"ID:{track_id} {name} ({int(confidence*100)}%)"
        elif name != 'Unknown':
            text = f"ID:{track_id} {name}"
        else:
            text = f"ID:{track_id}"
        
        # Add activity
        activity_text = f"{activity}"
        
        # Calculate text size
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        (act_w, act_h), _ = cv2.getTextSize(activity_text, self.font, self.font_scale-0.1, self.thickness-1)
        
        # Draw text background (above bbox)
        bg_y1 = max(0, y1 - text_h - act_h - 15)
        bg_y2 = y1 - 5
        bg_x2 = x1 + max(text_w, act_w) + 10
        
        cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), self.colors['text_bg'], -1)
        
        # Draw text
        text_y = y1 - act_h - 10
        cv2.putText(frame, text, (x1 + 5, text_y), 
                   self.font, self.font_scale, color, self.thickness)
        
        # Draw activity
        act_y = y1 - 5
        cv2.putText(frame, activity_text, (x1 + 5, act_y),
                   self.font, self.font_scale-0.1, (255, 255, 255), self.thickness-1)
        
        return frame
    
    def draw_stats(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """
        Draw global statistics on frame
        
        Args:
            frame: Video frame
            stats: Statistics dictionary
        
        Returns:
            Frame with stats overlay
        """
        h, w = frame.shape[:2]
        
        # Prepare stats text
        lines = [
            f"People: {stats.get('total_tracks_seen', 0)}",
            f"Falls: {stats.get('total_falls_detected', 0)}",
            f"Changes: {stats.get('total_activity_changes', 0)}",
        ]
        
        # Calculate dimensions
        line_height = 25
        padding = 10
        bg_height = len(lines) * line_height + 2 * padding
        bg_width = 200
        
        # Draw background (top-right corner)
        x1 = w - bg_width - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + bg_height
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors['text_bg'], -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw text
        for i, line in enumerate(lines):
            y = y1 + padding + (i + 1) * line_height
            cv2.putText(frame, line, (x1 + padding, y),
                       self.font, 0.5, (255, 255, 255), 1)
        
        return frame


# Global instance
_overlay = None

def get_overlay() -> PersonOverlay:
    """Get or create overlay instance"""
    global _overlay
    if _overlay is None:
        _overlay = PersonOverlay()
    return _overlay
