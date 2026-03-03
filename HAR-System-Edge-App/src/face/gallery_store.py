"""
Face gallery local cache: save/load gallery and version to disk.
Uses gallery.json for persons + embeddings and version.json for version/updated_at.
Validates embedding dimension (default 512) and dtype.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.face.schemas import FaceGallery, GalleryPerson

_LOG = logging.getLogger(__name__)

GALLERY_FILENAME = "gallery.json"
VERSION_FILENAME = "version.json"
DEFAULT_EMBED_DIM = 512


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_gallery(cache_dir: str, embed_dim: int = DEFAULT_EMBED_DIM) -> Optional[FaceGallery]:
    """
    Load gallery from cache_dir/gallery.json and version from version.json.
    Validates embedding length; skips invalid embeddings. Returns None if files missing or invalid.
    """
    base = Path(cache_dir)
    gallery_path = base / GALLERY_FILENAME
    version_path = base / VERSION_FILENAME
    if not gallery_path.exists():
        _LOG.debug("face gallery cache missing: %s", gallery_path)
        return None
    try:
        with open(gallery_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _LOG.warning("face gallery load error: %s", e)
        return None

    try:
        v = data.get("version", "")
        version = str(v) if v is not None else ""
        updated_at = str(data.get("updated_at", ""))
        persons_raw = data.get("persons", [])
        threshold = data.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        embedding_dim_cache = data.get("embedding_dim")
        if embedding_dim_cache is not None:
            embedding_dim_cache = int(embedding_dim_cache)
        dim = embedding_dim_cache if embedding_dim_cache is not None else embed_dim
        persons = []
        for p in persons_raw:
            person_id = str(p.get("person_id", ""))
            name = str(p.get("name", ""))
            embs = p.get("embeddings", [])
            if not person_id or not isinstance(embs, list):
                continue
            embeddings = []
            for e in embs:
                if isinstance(e, list) and (dim is None or len(e) == dim):
                    try:
                        embeddings.append([float(x) for x in e])
                    except (TypeError, ValueError):
                        continue
            if embeddings:
                persons.append(GalleryPerson(person_id=person_id, name=name, embeddings=embeddings))
        return FaceGallery(
            version=version,
            updated_at=updated_at,
            persons=persons,
            threshold=threshold,
            embedding_dim=embedding_dim_cache,
        )
    except (TypeError, ValueError) as e:
        _LOG.warning("face gallery cache schema error: %s", e)
        return None


def save_gallery(cache_dir: str, gallery: FaceGallery) -> bool:
    """
    Save gallery to cache_dir/gallery.json and version to version.json.
    Returns True on success.
    """
    _ensure_dir(cache_dir)
    base = Path(cache_dir)
    gallery_path = base / GALLERY_FILENAME
    version_path = base / VERSION_FILENAME
    try:
        payload = {
            "version": gallery.version,
            "updated_at": gallery.updated_at,
            "persons": [
                {
                    "person_id": p.person_id,
                    "name": p.name,
                    "embeddings": p.embeddings,
                }
                for p in gallery.persons
            ],
        }
        if gallery.threshold is not None:
            payload["threshold"] = gallery.threshold
        if gallery.embedding_dim is not None:
            payload["embedding_dim"] = gallery.embedding_dim
        with open(gallery_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        with open(version_path, "w", encoding="utf-8") as f:
            json.dump({"version": gallery.version, "updated_at": gallery.updated_at}, f)
        return True
    except OSError as e:
        _LOG.warning("face gallery save error: %s", e)
        return False
