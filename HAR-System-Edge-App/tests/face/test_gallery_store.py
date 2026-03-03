"""Unit tests for face gallery_store: load_gallery, save_gallery, round-trip and invalid dimension."""

import json
import tempfile
import pytest
from pathlib import Path

from src.face.gallery_store import load_gallery, save_gallery, GALLERY_FILENAME, VERSION_FILENAME
from src.face.schemas import FaceGallery, GalleryPerson


def test_save_and_load_round_trip():
    gallery = FaceGallery(
        version="v3",
        updated_at="2026-03-02T12:00:00Z",
        persons=[
            GalleryPerson(person_id="id1", name="A", embeddings=[[0.1] * 512]),
            GalleryPerson(person_id="id2", name="B", embeddings=[[0.2] * 512, [0.21] * 512]),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        ok = save_gallery(d, gallery)
        assert ok
        assert (Path(d) / GALLERY_FILENAME).exists()
        assert (Path(d) / VERSION_FILENAME).exists()
        loaded = load_gallery(d)
        assert loaded is not None
        assert loaded.version == gallery.version
        assert len(loaded.persons) == 2
        assert loaded.persons[0].person_id == "id1"
        assert len(loaded.persons[1].embeddings) == 2


def test_load_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        out = load_gallery(d)
    assert out is None


def test_load_invalid_dimension_skips_embedding():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / GALLERY_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        # Embedding length 10 instead of 512
        payload = {
            "version": "v1",
            "updated_at": "2026-03-02T00:00:00Z",
            "persons": [{"person_id": "p1", "name": "X", "embeddings": [[0.0] * 10]}],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_gallery(d, embed_dim=512)
        assert loaded is not None
        assert loaded.version == "v1"
        assert len(loaded.persons) == 0
