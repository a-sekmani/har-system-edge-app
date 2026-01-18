"""
HAR-System: Human Activity Recognition System
==============================================
Real-time human activity recognition using Hailo-8 and Raspberry Pi
"""

__version__ = "1.0.0"
__author__ = "HAR-System Team"

# Public API re-export (so callers can `from har_system import TemporalActivityTracker`).
from .core.tracker import TemporalActivityTracker

# Keep the public surface small and stable.
__all__ = ['TemporalActivityTracker']
