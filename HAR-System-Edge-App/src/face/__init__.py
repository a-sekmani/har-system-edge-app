"""
Face recognition on edge: gallery sync, CPU inference, track binding.

- gallery_client: fetch face gallery and version from cloud
- gallery_store: load/save gallery to local cache
- recognizer: InsightFace-based detect + embedding + match
- tracker_binding: bind face identity to pose track_id with IoU and TTL
- schemas: FaceDetection, FaceIdentity, FaceGallery, PersonAttachment
"""

from src.face.schemas import (
    FaceDetection,
    FaceGallery,
    FaceIdentity,
    PersonAttachment,
)

__all__ = [
    "FaceDetection",
    "FaceGallery",
    "FaceIdentity",
    "PersonAttachment",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_FACE_POSE_IOU_THRESHOLD",
]

# Default embedding dimension (InsightFace buffalo_l typically 512)
DEFAULT_EMBED_DIM = 512
# IoU threshold to associate face bbox with pose bbox
DEFAULT_FACE_POSE_IOU_THRESHOLD = 0.2
