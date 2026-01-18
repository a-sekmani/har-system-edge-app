"""
Unit tests for `har_system.core.face_identity_manager.FaceIdentityManager`.

Focus:
- Confirmation logic across repeated observations
- Time-based expiry (via monkeypatched `time.time`)
- Identity switching when a new name becomes dominant
"""

from __future__ import annotations

import time

import pytest

from har_system.core.face_identity_manager import FaceIdentityManager


@pytest.mark.unit
def test_unknown_identity_does_not_confirm():
    mgr = FaceIdentityManager(min_confirmations=2, identity_timeout=5.0)
    updated = mgr.update_identity(track_id=1, name="Unknown", confidence=0.0)
    assert updated is False
    assert mgr.get_identity(1) == "Unknown"
    assert mgr.is_identified(1) is False


@pytest.mark.unit
def test_identity_is_confirmed_after_min_confirmations(monkeypatch):
    mgr = FaceIdentityManager(min_confirmations=2, identity_timeout=5.0)

    base_time = 1000.0
    monkeypatch.setattr(time, "time", lambda: base_time)
    assert mgr.update_identity(1, "Ahmed", 0.8, global_id="g1") is False

    monkeypatch.setattr(time, "time", lambda: base_time + 0.1)
    assert mgr.update_identity(1, "Ahmed", 0.9, global_id="g1") is True

    assert mgr.get_identity(1) == "Ahmed"
    assert mgr.is_identified(1) is True
    assert mgr.get_confidence(1) > 0.0


@pytest.mark.unit
def test_identity_timeout_requires_reconfirmation(monkeypatch):
    mgr = FaceIdentityManager(min_confirmations=2, identity_timeout=1.0)

    base_time = 2000.0
    monkeypatch.setattr(time, "time", lambda: base_time)
    mgr.update_identity(1, "Sara", 0.8, global_id="g2")
    monkeypatch.setattr(time, "time", lambda: base_time + 0.1)
    mgr.update_identity(1, "Sara", 0.9, global_id="g2")

    assert mgr.is_identified(1) is True

    # After timeout, identity is no longer "confirmed" (but name stays available).
    monkeypatch.setattr(time, "time", lambda: base_time + 2.0)
    assert mgr.get_identity(1) == "Sara"
    assert mgr.is_identified(1) is False


@pytest.mark.unit
def test_identity_change_is_applied(monkeypatch):
    mgr = FaceIdentityManager(min_confirmations=2, identity_timeout=5.0)

    base_time = 3000.0
    monkeypatch.setattr(time, "time", lambda: base_time)
    mgr.update_identity(1, "Ahmed", 0.8)
    monkeypatch.setattr(time, "time", lambda: base_time + 0.1)
    mgr.update_identity(1, "Ahmed", 0.85)
    assert mgr.get_identity(1) == "Ahmed"

    # Provide enough confirmations for a different name.
    monkeypatch.setattr(time, "time", lambda: base_time + 1.0)
    mgr.update_identity(1, "Sara", 0.9)
    monkeypatch.setattr(time, "time", lambda: base_time + 1.1)
    mgr.update_identity(1, "Sara", 0.92)
    # Break ties: ensure the new name becomes the most common candidate.
    monkeypatch.setattr(time, "time", lambda: base_time + 1.2)
    mgr.update_identity(1, "Sara", 0.93)

    assert mgr.get_identity(1) == "Sara"

