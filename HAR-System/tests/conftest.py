"""
Pytest fixtures for HAR-System tests.

Design goals:
- Tests in HAR-System should run independently of the repo-level hailo-apps tests.
- Unit tests must not require a Hailo device nor an active GStreamer pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Ensure the HAR-System project root is importable during test collection.
# (Fixtures run after imports, so we must do this at module import time.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _restore_sys_argv():
    """Prevent sys.argv mutations from leaking across tests."""
    original = sys.argv.copy()
    try:
        yield
    finally:
        sys.argv = original


@pytest.fixture(scope="session")
def har_project_root() -> Path:
    """Expose HAR-System project root to tests."""
    return PROJECT_ROOT

