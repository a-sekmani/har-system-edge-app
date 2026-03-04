#!/usr/bin/env python3
"""
Phase 2 acceptance test script.
Runs the app for ~30 seconds with --no-display and --tracking-source fallback,
parses Phase 1 + Phase 2 counters from log, prints conditions with PASS/FAIL, then overall result.
Exit: 0 = PASS, 1 = FAIL.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_PERSON_FRAMES = 30
MIN_MULTI_PERSON_FRAMES = 30  # optional two-person mode


def run_app_and_capture(duration_sec=30, input_source="rpi", tracking_source="fallback"):
    """Run app with --no-display and --tracking-source for duration_sec; return full stdout."""
    cmd = [
        sys.executable,
        "src/har_pose_app.py",
        "--input", input_source,
        "--no-display",
        "--show-fps",
        "--tracking-source", tracking_source,
    ]
    process = subprocess.Popen(
        cmd,
        cwd=_PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    start = time.time()
    output_lines = []
    while time.time() - start < duration_sec:
        if process.poll() is not None:
            break
        line = process.stdout.readline()
        if line:
            output_lines.append(line)
        time.sleep(0.1)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    return "".join(output_lines)


def parse_counters(full_output):
    """
    Parse FPS Stats, Phase1 and Phase2 summary/final lines.
    FPS/Phase1: use snapshot where total_frames is highest.
    Phase2: use last Phase2 summary (cumulative).
    Returns dict with Phase 1 + Phase 2 counters.
    """
    out = {
        "total_frames": 0,
        "frame_events": 0,
        "invalid_caps": 0,
        "invalid_validate": 0,
        "frames_with_persons": 0,
        "frames_with_landmarks": 0,
        "frames_keypoints_len_not_17": 0,
        "unique_track_ids": 0,
        "new_tracks_created": 0,
        "tracks_ended": 0,
        "id_switch_suspected": 0,
        "multi_person_frames": 0,
        "detections_total": 0,
        "filtered_detections_total": 0,
    }
    fps_matches = list(re.finditer(
        r"Frames:\s*(\d+).*?frame_events:\s*(\d+).*?invalid_caps:\s*(\d+).*?invalid_validate:\s*(\d+)",
        full_output,
        re.DOTALL,
    ))
    if fps_matches:
        best = max(fps_matches, key=lambda m: int(m.group(1)))
        out["total_frames"] = int(best.group(1))
        out["frame_events"] = int(best.group(2))
        out["invalid_caps"] = int(best.group(3))
        out["invalid_validate"] = int(best.group(4))

    phase1_matches = list(re.finditer(
        r"Phase1\s+(?:summary|final):\s*.*?frames_with_persons=(\d+).*?frames_with_landmarks=(\d+).*?frames_keypoints_len_not_17=(\d+)",
        full_output,
        re.DOTALL,
    ))
    if phase1_matches:
        m = phase1_matches[-1]
        out["frames_with_persons"] = int(m.group(1))
        out["frames_with_landmarks"] = int(m.group(2))
        out["frames_keypoints_len_not_17"] = int(m.group(3))

    # Phase2: require first 5 fields; optional detections_total, filtered_detections_total
    phase2_matches = list(re.finditer(
        r"Phase2\s+(?:summary|final):\s*unique_track_ids=(\d+).*?new_tracks_created=(\d+).*?tracks_ended=(\d+).*?id_switch_suspected=(\d+).*?multi_person_frames=(\d+)(?:.*?detections_total=(\d+).*?filtered_detections_total=(\d+))?",
        full_output,
        re.DOTALL,
    ))
    if phase2_matches:
        m = phase2_matches[-1]
        out["unique_track_ids"] = int(m.group(1))
        out["new_tracks_created"] = int(m.group(2))
        out["tracks_ended"] = int(m.group(3))
        out["id_switch_suspected"] = int(m.group(4))
        out["multi_person_frames"] = int(m.group(5))
        if m.lastindex >= 7 and m.group(6) is not None and m.group(7) is not None:
            out["detections_total"] = int(m.group(6))
            out["filtered_detections_total"] = int(m.group(7))

    return out


def check_phase2_single_person_conditions(counters, min_person_frames=MIN_PERSON_FRAMES):
    """Return (all_ok, failed_reasons) for Phase 2 single-person acceptance."""
    reasons = []
    total = counters.get("total_frames", 0)
    fe = counters.get("frame_events", 0)
    ic = counters.get("invalid_caps", 0)
    iv = counters.get("invalid_validate", 0)
    fwp = counters.get("frames_with_persons", 0)
    utid = counters.get("unique_track_ids", 0)
    isw = counters.get("id_switch_suspected", 0)
    if total == 0:
        reasons.append("total_frames is 0")
        return False, reasons
    if fe < 0.95 * total:
        reasons.append(f"frame_events ({fe}) < 0.95 * total_frames ({total})")
    if ic != 0:
        reasons.append(f"invalid_caps must be 0, got {ic}")
    if iv != 0:
        reasons.append(f"invalid_validate must be 0, got {iv}")
    if fwp < min_person_frames:
        reasons.append(f"frames_with_persons ({fwp}) < {min_person_frames}")
    if utid > 2:
        reasons.append(f"unique_track_ids ({utid}) > 2 (single-person: expect <= 1 normally, <= 2 with brief gaps)")
    if isw != 0:
        reasons.append(f"id_switch_suspected ({isw}) != 0 (single-person: expect 0)")
    return (len(reasons) == 0, reasons)


def check_phase2_two_person_conditions(counters, min_multi=MIN_MULTI_PERSON_FRAMES):
    """Return (all_ok, failed_reasons) for Phase 2 two-person optional acceptance."""
    reasons = []
    mpf = counters.get("multi_person_frames", 0)
    utid = counters.get("unique_track_ids", 0)
    isw = counters.get("id_switch_suspected", 0)
    if mpf < min_multi:
        reasons.append(f"multi_person_frames ({mpf}) < {min_multi}")
    if utid < 2 or utid > 6:
        reasons.append(f"unique_track_ids ({utid}) not in [2, 6]")
    if isw > 10:
        reasons.append(f"id_switch_suspected ({isw}) > 10")
    return (len(reasons) == 0, reasons)


def main():
    """Run Phase 2 acceptance: run app with fallback tracking, parse counters, print conditions and PASS/FAIL."""
    print("=" * 60)
    print("Phase 2 Acceptance Test (single-person)")
    print("=" * 60)

    if not (_PROJECT_ROOT / "src" / "har_pose_app.py").exists():
        print("[FAIL] src/har_pose_app.py not found. Run from HAR-System-Edge-App directory.")
        sys.exit(1)

    duration = 30
    print(f"\nRunning app with --no-display --tracking-source fallback for {duration} seconds...")
    print("(One person in view recommended)\n")

    try:
        full_output = run_app_and_capture(duration_sec=duration, input_source="rpi", tracking_source="fallback")
    except Exception as e:
        print(f"[FAIL] Error running app: {e}")
        sys.exit(1)

    counters = parse_counters(full_output)
    total = counters["total_frames"]
    fe = counters["frame_events"]
    ic = counters["invalid_caps"]
    iv = counters["invalid_validate"]
    fwp = counters["frames_with_persons"]
    utid = counters["unique_track_ids"]
    ntc = counters["new_tracks_created"]
    te = counters["tracks_ended"]
    isw = counters["id_switch_suspected"]
    mpf = counters["multi_person_frames"]

    print("Counters (Phase 1 + Phase 2):")
    print(f"  total_frames           = {total}")
    print(f"  frame_events            = {fe}")
    print(f"  invalid_caps            = {ic}")
    print(f"  invalid_validate       = {iv}")
    print(f"  frames_with_persons     = {fwp}")
    print(f"  unique_track_ids        = {utid}")
    print(f"  new_tracks_created      = {ntc}")
    print(f"  tracks_ended            = {te}")
    print(f"  id_switch_suspected     = {isw}")
    print(f"  multi_person_frames     = {mpf}")
    print()

    print("Conditions (single-person):")
    all_ok = True
    threshold_95 = 0.95 * total if total else 0
    c1 = fe >= threshold_95
    all_ok = all_ok and c1
    print(f"  frame_events >= 0.95 * total_frames : {fe} >= {threshold_95:.0f}  [{'PASS' if c1 else 'FAIL'}]")
    c2 = ic == 0
    all_ok = all_ok and c2
    print(f"  invalid_caps == 0                    : {ic} == 0  [{'PASS' if c2 else 'FAIL'}]")
    c3 = iv == 0
    all_ok = all_ok and c3
    print(f"  invalid_validate == 0               : {iv} == 0  [{'PASS' if c3 else 'FAIL'}]")
    c4 = fwp >= MIN_PERSON_FRAMES
    all_ok = all_ok and c4
    print(f"  frames_with_persons >= {MIN_PERSON_FRAMES}             : {fwp} >= {MIN_PERSON_FRAMES}  [{'PASS' if c4 else 'FAIL'}]")
    c5 = utid <= 2
    all_ok = all_ok and c5
    print(f"  unique_track_ids <= 2               : {utid} <= 2  [{'PASS' if c5 else 'FAIL'}]")
    c6 = isw == 0
    all_ok = all_ok and c6
    print(f"  id_switch_suspected == 0             : {isw} == 0  [{'PASS' if c6 else 'FAIL'}]")
    print()

    if all_ok:
        print("[PASS] Phase 2 acceptance (single-person): all criteria met.")
        sys.exit(0)
    else:
        print("[FAIL] Phase 2 acceptance: one or more conditions failed (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
