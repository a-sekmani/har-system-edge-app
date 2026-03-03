"""
Face recognizer: InsightFace on CPU — detect faces, extract embedding, match to gallery.
Lazy init; uses det_size and min_det_conf from config.

Note: Embedding dimension (512) is fixed by the model. Most of the cost is the NN
forward pass (detect + embed), not the 512-dim cosine similarity; reducing embed
dim would have minimal FPS impact unless the model itself is changed (e.g. buffalo_s).
"""

import logging
import os
from typing import List, Optional, Tuple

from src.face.schemas import FaceDetection, FaceGallery

_LOG = logging.getLogger(__name__)

# Cosine similarity: (a . b) / (||a|| ||b||). For unit vectors same as dot product.
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


class FaceRecognizer:
    """
    InsightFace-based face detection and recognition on CPU.
    detect_faces(frame_bgr) -> list[FaceDetection]
    get_embedding(frame_bgr, face) -> list[float] (or ndarray in impl)
    match(embedding, gallery) -> Optional[(person_id, name, score)]
    """

    def __init__(
        self,
        det_size: int = 320,
        max_faces: int = 5,
        min_det_conf: float = 0.6,
        sim_threshold: float = 0.35,
    ):
        self.det_size = max(160, min(640, det_size))
        self.max_faces = max(1, max_faces)
        self.min_det_conf = min_det_conf
        self.sim_threshold = sim_threshold
        self._app = None  # lazy

    def _ensure_app(self) -> bool:
        if self._app is not None:
            return True
        try:
            from insightface.app import FaceAnalysis
            # root must be a path (InsightFace uses it for model cache); None causes TypeError
            root = os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface"))
            self._app = FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            _LOG.info("Face recognizer initialized (InsightFace buffalo_l, det_size=%s)", self.det_size)
            return True
        except Exception as e:
            _LOG.warning("Face recognizer init failed: %s", e)
            return False

    def detect_faces(self, frame_bgr) -> List[FaceDetection]:
        """
        Run face detection on BGR frame (numpy HxWx3).
        Returns list of FaceDetection (bbox_xyxy, det_conf) above min_det_conf, capped at max_faces.
        """
        if not self._ensure_app():
            return []
        try:
            faces = self._app.get(frame_bgr, max_num=self.max_faces)
            out = []
            for f in faces:
                conf = float(getattr(f, "det_score", 0.5))
                if conf < self.min_det_conf:
                    continue
                bbox = getattr(f, "bbox", None)
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                landmarks = None
                if hasattr(f, "kps") and f.kps is not None:
                    landmarks = [(float(k[0]), float(k[1])) for k in f.kps]
                out.append(FaceDetection(bbox_xyxy=(x1, y1, x2, y2), det_conf=conf, landmarks=landmarks))
            return out[: self.max_faces]
        except Exception as e:
            _LOG.debug("face detect error: %s", e)
            return []

    def get_embedding(self, frame_bgr, face: FaceDetection):
        """
        Extract 512-dim embedding for one face crop.
        frame_bgr: full frame; face.bbox_xyxy used to crop. Returns list of floats.
        """
        if not self._ensure_app():
            return []
        try:
            import numpy as np
            x1, y1, x2, y2 = [int(round(x)) for x in face.bbox_xyxy]
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                return []
            crop = frame_bgr[y1:y2, x1:x2]
            faces = self._app.get(crop, max_num=1)
            if not faces:
                return []
            f = faces[0]
            emb = getattr(f, "embedding", None)
            if emb is None:
                return []
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return list(emb)
        except Exception as e:
            _LOG.debug("face embedding error: %s", e)
            return []

    def match(
        self,
        embedding: List[float],
        gallery: Optional[FaceGallery],
    ) -> Optional[Tuple[str, str, float]]:
        """
        Match embedding to gallery. Returns (person_id, name, score) for best match
        if score >= threshold (gallery.threshold from cloud if set, else self.sim_threshold).
        """
        if not gallery or not embedding:
            return None
        threshold = getattr(gallery, "threshold", None)
        use_threshold = threshold if threshold is not None else self.sim_threshold
        best: Optional[Tuple[str, str, float]] = None
        for person in gallery.persons:
            for emb in person.embeddings:
                sim = _cosine_similarity(embedding, emb)
                if sim >= use_threshold and (best is None or sim > best[2]):
                    best = (person.person_id, person.name, sim)
        return best
