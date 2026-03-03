"""Unit tests for face recognizer match logic (cosine similarity and threshold)."""

import pytest
from src.face.recognizer import FaceRecognizer, _cosine_similarity
from src.face.schemas import FaceGallery, GalleryPerson


def test_cosine_similarity_same_vector():
    a = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    a = [1.0, 0.0, 0.0]
    b = [-1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_match_above_threshold():
    rec = FaceRecognizer(sim_threshold=0.35)
    emb = [0.1] * 512
    gallery = FaceGallery(
        version="v1",
        updated_at="",
        persons=[GalleryPerson(person_id="pid1", name="Alice", embeddings=[[0.1] * 512])],
    )
    out = rec.match(emb, gallery)
    assert out is not None
    person_id, name, score = out
    assert person_id == "pid1"
    assert name == "Alice"
    assert score >= 0.35


def test_match_below_threshold_returns_none():
    rec = FaceRecognizer(sim_threshold=0.99)
    emb = [1.0] + [0.0] * 511
    gallery_emb = [0.0, 1.0] + [0.0] * 510
    gallery = FaceGallery(
        version="v1",
        updated_at="",
        persons=[GalleryPerson(person_id="pid1", name="Alice", embeddings=[gallery_emb])],
    )
    out = rec.match(emb, gallery)
    assert out is None


def test_match_empty_gallery_returns_none():
    rec = FaceRecognizer()
    out = rec.match([0.1] * 512, FaceGallery(version="", updated_at="", persons=[]))
    assert out is None


def test_match_none_gallery_returns_none():
    rec = FaceRecognizer()
    out = rec.match([0.1] * 512, None)
    assert out is None


def test_match_uses_gallery_threshold_when_present():
    """When gallery has threshold from cloud, recognizer uses it instead of sim_threshold."""
    rec = FaceRecognizer(sim_threshold=0.35)
    emb = [0.1] * 512
    # Same embedding → similarity 1.0; would pass 0.35 but we set gallery threshold 0.99
    gallery = FaceGallery(
        version="v1",
        updated_at="",
        persons=[GalleryPerson(person_id="p1", name="X", embeddings=[[0.1] * 512])],
        threshold=0.99,
    )
    out = rec.match(emb, gallery)
    assert out is not None
    assert out[2] >= 0.99
    # With threshold 0.9999 same vector still matches (sim=1.0)
    gallery_strict = FaceGallery(
        version="v1", updated_at="", persons=gallery.persons, threshold=0.9999
    )
    out2 = rec.match(emb, gallery_strict)
    assert out2 is not None
