"""
Face gallery HTTP client: GET version and GET face-gallery from cloud.
Uses X-API-Key header; same base URL as cloud ingest.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

from src.face.schemas import FaceGallery, GalleryPerson

_LOG = logging.getLogger(__name__)


def _build_request(
    url: str,
    api_key: str,
    timeout_sec: float,
    verify_tls: bool = True,
) -> urllib.request.Request:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    return req


def fetch_gallery_updated_at(
    base_url: str,
    version_path: str,
    api_key: str = "",
    timeout_sec: float = 5.0,
    verify_tls: bool = True,
) -> Optional[str]:
    """
    GET /v1/face-gallery/version (or custom path).
    Cloud returns updated_at (ISO 8601). Used to decide if local gallery should be updated.
    Returns updated_at string if response is valid, else None. Empty string is returned as None.
    """
    url = base_url.rstrip("/") + (version_path if version_path.startswith("/") else "/" + version_path)
    try:
        req = _build_request(url, api_key, timeout_sec, verify_tls)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                _LOG.warning("face gallery updated_at GET status=%s", resp.status)
                return None
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("updated_at", data.get("created_at"))
            if raw is None:
                return None
            s = str(raw).strip()
            return s if s else None
    except urllib.error.URLError as e:
        _LOG.warning("face gallery updated_at fetch failed: %s", e)
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        _LOG.warning("face gallery updated_at parse error: %s", e)
        return None


def fetch_face_gallery(
    base_url: str,
    gallery_path: str,
    api_key: str = "",
    timeout_sec: float = 5.0,
    verify_tls: bool = True,
) -> Optional[FaceGallery]:
    """
    GET /v1/face-gallery and parse into FaceGallery.
    Cloud returns: gallery_version (e.g. "v12"), embedding_dim (512), threshold,
    people: [ { person_id, name, embeddings: [[...], ...] } ] (only is_active with ≥1 face).
    Also accepts legacy: version (int), updated_at, persons.
    """
    url = base_url.rstrip("/") + (gallery_path if gallery_path.startswith("/") else "/" + gallery_path)
    try:
        req = _build_request(url, api_key, timeout_sec, verify_tls)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                _LOG.warning("face gallery GET status=%s", resp.status)
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        _LOG.warning("face gallery fetch failed: %s", e)
        return None
    except json.JSONDecodeError as e:
        _LOG.warning("face gallery JSON error: %s", e)
        return None

    try:
        version = str(data.get("gallery_version", data.get("version", "")))
        updated_at = str(data.get("updated_at", data.get("created_at", "")))
        persons_raw = data.get("people", data.get("persons", []))
        threshold = data.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        embedding_dim = data.get("embedding_dim")
        if embedding_dim is not None:
            embedding_dim = int(embedding_dim)
        persons: list = []
        for p in persons_raw:
            person_id = str(p.get("person_id", ""))
            name = str(p.get("name", ""))
            embs = p.get("embeddings", [])
            if not person_id or not isinstance(embs, list):
                continue
            embeddings = []
            for e in embs:
                if isinstance(e, list):
                    embeddings.append([float(x) for x in e])
                else:
                    continue
            if embeddings:
                persons.append(GalleryPerson(person_id=person_id, name=name, embeddings=embeddings))
        return FaceGallery(
            version=version,
            updated_at=updated_at,
            persons=persons,
            threshold=threshold,
            embedding_dim=embedding_dim,
        )
    except (TypeError, ValueError) as e:
        _LOG.warning("face gallery schema error: %s", e)
        return None
