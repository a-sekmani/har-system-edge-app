"""
HAR-System: Human Activity Recognition System
==============================================
Real-time human activity recognition using Hailo-8 and Raspberry Pi

Core components:
- core: Temporal tracking and activity recognition
- utils: Utility functions and CLI tools
- apps: Applications (realtime_pose, etc.)
"""

__version__ = "1.0.0"
__author__ = "HAR-System Team"

from .core.tracker import TemporalActivityTracker

__all__ = ['TemporalActivityTracker']
