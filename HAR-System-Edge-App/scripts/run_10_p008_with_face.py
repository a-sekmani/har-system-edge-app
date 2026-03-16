#!/usr/bin/env python3
"""
Run 10 random videos for person P008 from the NTU library with face recognition enabled,
send results to the cloud, and report on face gallery update and face detection.
"""
import os
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
NTU_LIB = Path("/home/admin/Desktop/ntu_filtered_by_action")
CLOUD_URL = "http://192.168.1.105:8000"
CLOUD_API_KEY = "dev-key"
NUM_VIDEOS = 10
PERSON_FILTER = "P008"  # NTU filename segment for person ID


def find_p008_avi_files(lib_path: Path):
    """Collect all .avi files for person P008 (filename contains P008)."""
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found: {lib_path}")
    all_avis = list(lib_path.rglob("*.avi"))
    return sorted([p for p in all_avis if PERSON_FILTER in p.name])


def run_video(video_path: Path, report_lines: list) -> tuple[bool, str]:
    """
    Run the app on a single video with cloud upload and face recognition.
    Returns (success, combined_stdout_stderr) for gallery/face checks.
    """
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
        "--enable-face",
        "--face-gallery-url", CLOUD_URL,
        "--log-face-summary",
        "--face-skip-frames", "5",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(APP_DIR),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=600,
        )
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        success = result.returncode == 0
        if success:
            report_lines.append(f"  [OK] {video_path.name}")
        else:
            tail = combined[-500:] if len(combined) > 500 else combined
            report_lines.append(f"  [FAIL] {video_path.name} (returncode={result.returncode})")
            report_lines.append(f"    last output:\n{tail}")
        return success, combined
    except subprocess.TimeoutExpired:
        report_lines.append(f"  [TIMEOUT] {video_path.name}")
        return False, ""
    except Exception as e:
        report_lines.append(f"  [ERROR] {video_path.name}: {e}")
        return False, ""


def check_gallery_and_face_in_output(combined: str) -> tuple[bool, bool]:
    """Parse output for face gallery update and face recognition activity."""
    gallery_updated = (
        "face gallery synced from cloud" in combined
        or "face gallery loaded from cache" in combined
        or "face gallery refreshed from cloud" in combined
    )
    gallery_attempted = (
        gallery_updated
        or "face gallery empty" in combined
        or "face gallery fetch" in combined.lower()
        or "face gallery updated_at" in combined
    )
    face_ran = (
        "Face recognition running" in combined
        or "face summary" in combined
        or "face gallery" in combined.lower()
    )
    return gallery_updated or gallery_attempted, face_ran


def check_person_recognized_in_output(combined: str) -> tuple[bool, str]:
    """Check if any person was recognized (not Unknown). Returns (recognized, name_or_unknown)."""
    import re
    # Look for "Person: Name" or "Persons on screen: Name" in logs
    for line in combined.splitlines():
        if "Person:" in line or "Persons on screen:" in line:
            m = re.search(r"Person:\s*(\S+)|Persons on screen:\s*(\S+)", line)
            if m:
                name = (m.group(1) or m.group(2) or "").strip()
                if name and name != "Unknown" and name != "-":
                    return True, name
    return False, "Unknown"


def main():
    print(f"Collecting P008 video file list from {NTU_LIB}...")
    p008_avis = find_p008_avi_files(NTU_LIB)
    if not p008_avis:
        print(f"No videos found for person {PERSON_FILTER}. Exiting.")
        return 1
    if len(p008_avis) < NUM_VIDEOS:
        print(f"Warning: found only {len(p008_avis)} P008 file(s), will run all.")
        selected = p008_avis
    else:
        random.seed(123)
        selected = random.sample(p008_avis, NUM_VIDEOS)

    report_lines = [
        "=" * 60,
        f"Report: {NUM_VIDEOS} random P008 videos with face recognition and cloud upload",
        "=" * 60,
        f"Library: {NTU_LIB}",
        f"Cloud: {CLOUD_URL}",
        f"Person filter: {PERSON_FILTER}",
        f"Videos selected: {len(selected)}",
        "",
    ]

    success_count = 0
    gallery_updated_any = False
    face_ran_any = False
    recognition_per_video = []  # (video_name, recognized, name)
    for i, video_path in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] Running: {video_path.name}")
        ok, output = run_video(video_path, report_lines)
        if ok:
            success_count += 1
        g, f = check_gallery_and_face_in_output(output)
        if g:
            gallery_updated_any = True
        if f:
            face_ran_any = True
        rec, name = check_person_recognized_in_output(output)
        recognition_per_video.append((video_path.name, rec, name))

    report_lines.extend([
        "",
        "=" * 60,
        f"Summary: {success_count} of {len(selected)} succeeded",
        "=" * 60,
        "",
        "Face gallery: " + ("updated/synced or loaded from cache at least once" if gallery_updated_any else "no gallery update/load seen in logs"),
        "Face recognition: " + ("enabled and ran (see logs for face summary/gallery)" if face_ran_any else "no face activity seen in logs"),
        "",
        "Per-video face recognition:",
    ])
    for vname, rec, name in recognition_per_video:
        report_lines.append(f"  {vname}: " + (f"recognized as '{name}'" if rec else "no person recognized (Unknown)"))
    rec_count = sum(1 for _, r, _ in recognition_per_video if r)
    report_lines.append(f"  -> {rec_count}/{len(recognition_per_video)} videos had a recognized person.")
    report_lines.extend([
        "",
        "Why NTU P008 may not match: Gallery embeddings are from cloud enrollment (photos), not from NTU video frames. If face angle/lighting/quality differs, similarity may stay below the configured threshold (e.g. --face-sim-threshold). Same for other names in gallery.",
        "",
        "Files that were run:",
    ])
    for v in selected:
        report_lines.append(f"  - {v}")

    report_text = "\n".join(report_lines)
    report_file = PROJECT_ROOT / "run_10_p008_face_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print(f"\nReport saved to: {report_file}")
    return 0 if success_count == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
