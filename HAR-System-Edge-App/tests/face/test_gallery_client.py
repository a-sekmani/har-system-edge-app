"""Unit tests for face gallery_client: fetch_gallery_version, fetch_face_gallery (with mocked HTTP)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.face.gallery_client import fetch_gallery_version, fetch_face_gallery
from src.face.schemas import FaceGallery


def test_fetch_gallery_version_success():
    with patch("urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"version": "v12"}).encode("utf-8")
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None
        mock_open.return_value = mock_resp
        out = fetch_gallery_version("http://localhost:8000", "/v1/face-gallery/version", timeout_sec=2.0)
    assert out == "v12"


def test_fetch_gallery_version_non_200():
    with patch("urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 404
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None
        mock_open.return_value = mock_resp
        out = fetch_gallery_version("http://localhost:8000", "/v1/face-gallery/version", timeout_sec=2.0)
    assert out is None


def test_fetch_face_gallery_success():
    # Cloud format: gallery_version, people, optional threshold/embedding_dim
    payload = {
        "gallery_version": "v12",
        "embedding_dim": 512,
        "threshold": 0.45,
        "updated_at": "2026-03-02T10:00:00Z",
        "people": [
            {"person_id": "p1", "name": "Alice", "embeddings": [[0.1] * 512]},
            {"person_id": "p2", "name": "Bob", "embeddings": [[0.2] * 512, [0.21] * 512]},
        ],
    }
    with patch("urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None
        mock_open.return_value = mock_resp
        out = fetch_face_gallery("http://localhost:8000", "/v1/face-gallery", timeout_sec=2.0)
    assert out is not None
    assert isinstance(out, FaceGallery)
    assert out.version == "v12"
    assert out.threshold == 0.45
    assert out.embedding_dim == 512
    assert len(out.persons) == 2
    assert out.persons[0].person_id == "p1"
    assert out.persons[0].name == "Alice"
    assert len(out.persons[0].embeddings) == 1
    assert out.persons[1].person_id == "p2"
    assert len(out.persons[1].embeddings) == 2
    assert out.total_embeddings() == 3
