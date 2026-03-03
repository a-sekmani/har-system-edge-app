#!/usr/bin/env python3
"""
Request the face gallery only from the cloud (no camera or other app logic).
Fetches version then full gallery and saves to local cache.

Usage:
  python scripts/request_face_gallery.py [base_url]
  Or: FACE_GALLERY_URL=http://... python scripts/request_face_gallery.py
  Or: CLOUD_URL=http://... python scripts/request_face_gallery.py

Run from HAR-System-Edge-App directory.
"""
import os
import sys
from pathlib import Path

# Add project root to path for src imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reduce log noise when only requesting the gallery
import logging
logging.getLogger("src.face.gallery_client").setLevel(logging.ERROR)

from src.face.gallery_client import fetch_face_gallery, fetch_gallery_version
from src.face.gallery_store import save_gallery

DEFAULT_BASE_URL = "http://192.168.1.106:8000"
DEFAULT_CACHE_DIR = "/var/lib/har/face_gallery/"
VERSION_PATH = "/v1/face-gallery/version"
GALLERY_PATH = "/v1/face-gallery"
TIMEOUT_S = 10.0


def main():
    base_url = (
        (sys.argv[1] if len(sys.argv) > 1 else "").strip().rstrip("/")
        or os.environ.get("FACE_GALLERY_URL", "").strip().rstrip("/")
        or os.environ.get("CLOUD_URL", "").strip().rstrip("/")
        or DEFAULT_BASE_URL
    )
    api_key = os.environ.get("CLOUD_API_KEY", "")
    cache_dir = os.environ.get("FACE_GALLERY_CACHE", DEFAULT_CACHE_DIR)

    print("=== Request face gallery only (no other execution) ===\n")
    print(f"Base URL:   {base_url}")
    print(f"Cache dir:  {cache_dir}\n")

    # 1) Fetch gallery version
    print("[1] GET gallery version...")
    version = fetch_gallery_version(base_url, VERSION_PATH, api_key, TIMEOUT_S)
    if version is None:
        print("    Failed to fetch version.\n")
        sys.exit(1)
    print(f"    -> version = {version}\n")

    # 2) Fetch full gallery
    print("[2] GET full gallery...")
    gallery = fetch_face_gallery(base_url, GALLERY_PATH, api_key, TIMEOUT_S)
    if gallery is None:
        print("    Failed to fetch gallery.\n")
        sys.exit(1)
    print(f"    -> persons = {len(gallery.persons)}, embeddings = {gallery.total_embeddings()}\n")

    # 3) Save to local cache
    print("[3] Saving gallery to local cache...")
    save_gallery(cache_dir, gallery)
    print(f"    -> saved to {cache_dir}\n")

    print("=== Done (no other execution) ===")


if __name__ == "__main__":
    main()
