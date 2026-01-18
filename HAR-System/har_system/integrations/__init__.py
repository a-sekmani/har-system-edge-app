"""
HAR-System: Integrations
=========================
Integration modules for external systems
"""

from .hailo_face_recognition import HailoFaceRecognition

# Public integration surface.
__all__ = ['HailoFaceRecognition']
