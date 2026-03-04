#!/usr/bin/env python3
"""
Run 10 random video files from the NTU library, send results to the cloud, then print a report.
"""
import os
import random
import subprocess
import sys
from pathlib import Path

# Project root and app directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
NTU_LIB = Path("/home/admin/Desktop/ntu_filtered_by_action")
CLOUD_URL = "http://192.168.1.105:8000"
CLOUD_API_KEY = "dev-key"
NUM_VIDEOS = 10


def find_avi_files(lib_path: Path):
    """Collect all .avi files in the library."""
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found: {lib_path}")
    return sorted(lib_path.rglob("*.avi"))


def run_video(video_path: Path, report_lines: list) -> bool:
    """Run the app on a single video with cloud upload. Returns True on success."""
    env = {
        "PYTHONPATH": str(PROJECT_ROOT),
        "CLOUD_API_KEY": CLOUD_API_KEY,
    }
    cmd = [
        sys.executable,
        str(APP_DIR / "src" / "har_pose_app.py"),
        "--input", str(video_path),
        "--no-display",
        "--enable-cloud",
        "--cloud-mode", "windows",
        "--cloud-url", CLOUD_URL,
        "--cloud-api-key", CLOUD_API_KEY,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(APP_DIR),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max per video
        )
        success = result.returncode == 0
        if success:
            report_lines.append(f"  [OK] {video_path}")
        else:
            tail = (result.stderr or result.stdout or "")[-500:]
            report_lines.append(f"  [FAIL] {video_path}")
            report_lines.append(f"    returncode={result.returncode}")
            report_lines.append(f"    last 500 chars of output:\n{tail}")
        return success
    except subprocess.TimeoutExpired:
        report_lines.append(f"  [TIMEOUT] {video_path}")
        return False
    except Exception as e:
        report_lines.append(f"  [ERROR] {video_path}: {e}")
        return False


def main():
    print("Collecting video file list...")
    all_avis = find_avi_files(NTU_LIB)
    if len(all_avis) < NUM_VIDEOS:
        print(f"Warning: found only {len(all_avis)} file(s), will run all.")
        selected = all_avis
    else:
        random.seed(42)
        selected = random.sample(all_avis, NUM_VIDEOS)

    report_lines = [
        "=" * 60,
        "Report: 10 random NTU videos run with cloud upload",
        "=" * 60,
        f"Library: {NTU_LIB}",
        f"Cloud: {CLOUD_URL}",
        f"Videos selected: {len(selected)}",
        "",
    ]

    success_count = 0
    for i, video_path in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Running: {video_path.name}")
        if run_video(video_path, report_lines):
            success_count += 1

    report_lines.extend([
        "",
        "=" * 60,
        f"Summary: {success_count} of {len(selected)} succeeded",
        "=" * 60,
        "",
        "Files that were run:",
    ])
    for v in selected:
        report_lines.append(f"  - {v}")

    report_text = "\n".join(report_lines)
    report_file = PROJECT_ROOT / "run_10_videos_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nReport saved to: {report_file}")
    return 0 if success_count == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
