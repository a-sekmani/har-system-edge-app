"""
HAR-System Core Module
======================
Core tracking and activity recognition components
"""

from .tracker import TemporalActivityTracker

__all__ = [
    'TemporalActivityTracker',
]

# Lazy import for callbacks (requires hailo_apps)
def _get_callbacks():
    from .callbacks import HARCallbackHandler, process_frame_callback
    return HARCallbackHandler, process_frame_callback
