"""
Pytest configuration and shared fixtures for HAR-System-Edge-App tests.
"""
import sys
from pathlib import Path

import pytest

# Ensure parent repo (hailo-apps root) is on path for hailo_apps imports
REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REPO = Path(__file__).resolve().parents[2]
if str(PARENT_REPO) not in sys.path:
    sys.path.insert(0, str(PARENT_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def har_app_module():
    """Import har_pose_app module (may skip if hailo_apps not available)."""
    try:
        from src import har_pose_app
        return har_pose_app
    except ImportError as e:
        pytest.skip(f"hailo_apps or dependencies not available: {e}")


@pytest.fixture
def fps_tracker_class(har_app_module):
    """FPSTracker class from har_pose_app."""
    return har_app_module.FPSTracker


@pytest.fixture
def har_user_data_class(har_app_module):
    """HARUserData class from har_pose_app."""
    return har_app_module.HARUserData


@pytest.fixture
def get_har_parser_func(har_app_module):
    """get_har_parser function from har_pose_app."""
    return har_app_module.get_har_parser


@pytest.fixture
def simple_callback_func(har_app_module):
    """simple_callback function from har_pose_app."""
    return har_app_module.simple_callback
