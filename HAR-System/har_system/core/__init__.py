"""
HAR-System Core Module
======================
Core tracking and activity recognition components
"""

from .tracker import TemporalActivityTracker
from .face_identity_manager import FaceIdentityManager

try:
    # FaceRecognitionProcessor depends on optional hailo-apps / DB dependencies.
    from .face_processor import FaceRecognitionProcessor
    FACE_PROCESSOR_AVAILABLE = True
except ImportError:
    FaceRecognitionProcessor = None
    FACE_PROCESSOR_AVAILABLE = False

__all__ = [
    'TemporalActivityTracker',
    'FaceIdentityManager',
    'FaceRecognitionProcessor'
]

# Lazy import for callbacks (requires hailo-apps / GStreamer bindings).
# This avoids import-time failures when running pure-Python tests/examples.
def _get_callbacks():
    from .callbacks import HARCallbackHandler, process_frame_callback
    return HARCallbackHandler, process_frame_callback
