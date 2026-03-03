"""
Face recognition schemas: detection, identity, gallery, and window person attachment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Embedding as list of floats; numpy array in runtime
Embedding = List[float]


@dataclass
class FaceDetection:
    """Single face detection: bbox and confidence."""

    bbox_xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2
    det_conf: float
    landmarks: Optional[List[Tuple[float, float]]] = None


@dataclass
class FaceIdentity:
    """
    Identity bound to a pose track_id.
    TTL and voting managed in tracker_binding.
    """

    person_id: Optional[str]
    person_name: Optional[str]
    score: float
    last_seen_ts: float
    last_verified_ts: float
    votes: Dict[str, int] = field(default_factory=dict)


@dataclass
class GalleryPerson:
    """One person in the gallery with one or more embeddings."""

    person_id: str
    name: str
    embeddings: List[Embedding]  # list of 512-dim (or embed_dim) vectors


@dataclass
class FaceGallery:
    """In-memory face gallery from cloud or cache."""

    version: str  # e.g. "v12" from cloud gallery_version or version endpoint
    updated_at: str  # ISO timestamp or created_at from cloud
    persons: List[GalleryPerson]
    threshold: Optional[float] = None  # suggested match threshold from cloud (e.g. 0.45)
    embedding_dim: Optional[int] = None  # from cloud (e.g. 512)

    def total_embeddings(self) -> int:
        return sum(len(p.embeddings) for p in self.persons)


@dataclass
class PersonAttachment:
    """
    Person metadata attached to a window payload.
    Sent to cloud as window.person.
    """

    person_id: Optional[str]
    name: Optional[str]
    face_conf: float
    source: str  # e.g. "edge_face"
    verified_at_ms: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        conf = round(self.face_conf, 4)
        return {
            "person_id": self.person_id,
            "name": self.name,
            "face_conf": conf,
            "person_conf": conf,  # cloud API expects person_conf
            "source": self.source,
            "verified_at_ms": self.verified_at_ms,
        }
