#!/usr/bin/env python3
"""
Phase 1 acceptance test script.
Runs the app for ~30 seconds with --no-display, parses counters from log,
prints each acceptance condition with actual values and PASS/FAIL, then overall result.
Exit: 0 = PASS, 1 = FAIL.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_PERSON_FRAMES = 30


def run_app_and_capture(duration_sec=30, input_source="rpi"):
    """Run app with --no-display for duration_sec; return full stdout/stderr."""
    cmd = [
        sys.executable,
        "src/har_pose_app.py",
        "--input", input_source,
        "--no-display",
        "--show-fps",
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
    Parse all FPS Stats and Phase1 summary/final lines; use cumulative totals
    at the moment when total_frames is highest (full run), not just last interval.
    Returns dict with: total_frames, frame_events, invalid_caps, invalid_validate,
    frames_with_persons, frames_with_landmarks, frames_keypoints_len_not_17.
    """
    out = {
        "total_frames": 0,
        "frame_events": 0,
        "invalid_caps": 0,
        "invalid_validate": 0,
        "frames_with_persons": 0,
        "frames_with_landmarks": 0,
        "frames_keypoints_len_not_17": 0,
    }
    fps_matches = list(re.finditer(
        r"Frames:\s*(\d+).*?frame_events:\s*(\d+).*?invalid_caps:\s*(\d+).*?invalid_validate:\s*(\d+)",
        full_output,
        re.DOTALL,
    ))
    # Use snapshot where total_frames is highest (cumulative totals over full run)
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
    # Phase1 is logged in same cycle as FPS Stats; last one corresponds to end of run (cumulative)
    if phase1_matches:
        m = phase1_matches[-1]
        out["frames_with_persons"] = int(m.group(1))
        out["frames_with_landmarks"] = int(m.group(2))
        out["frames_keypoints_len_not_17"] = int(m.group(3))

    return out


def check_phase1_conditions(counters, min_person_frames=MIN_PERSON_FRAMES):
    """
    Return (all_ok: bool, failed_reasons: list) for Phase 1 acceptance.
    Same logic as used in main() for printing conditions and PASS/FAIL.
    """
    reasons = []
    total = counters.get("total_frames", 0)
    fe = counters.get("frame_events", 0)
    ic = counters.get("invalid_caps", 0)
    iv = counters.get("invalid_validate", 0)
    fkp17 = counters.get("frames_keypoints_len_not_17", 0)
    fwp = counters.get("frames_with_persons", 0)
    fwl = counters.get("frames_with_landmarks", 0)
    if total == 0:
        reasons.append("total_frames is 0")
        return False, reasons
    if fe < 0.95 * total:
        reasons.append(f"frame_events ({fe}) < 0.95 * total_frames ({total})")
    if ic != 0:
        reasons.append(f"invalid_caps must be 0, got {ic}")
    if iv != 0:
        reasons.append(f"invalid_validate must be 0, got {iv}")
    if fkp17 != 0:
        reasons.append(f"frames_keypoints_len_not_17 must be 0, got {fkp17}")
    if fwp < min_person_frames:
        reasons.append(f"frames_with_persons ({fwp}) < MIN_PERSON_FRAMES ({min_person_frames})")
    if fwp > 0 and fwl < 0.8 * fwp:
        reasons.append(f"frames_with_landmarks ({fwl}) < 0.8 * frames_with_persons ({fwp})")
    return (len(reasons) == 0, reasons)


def main():
    """Run Phase 1 acceptance: run app, parse counters, print conditions and PASS/FAIL."""
    print("=" * 60)
    print("Phase 1 Acceptance Test")
    print("=" * 60)

    if not (_PROJECT_ROOT / "src" / "har_pose_app.py").exists():
        print("[FAIL] src/har_pose_app.py not found. Run from HAR-System-Edge-App directory.")
        sys.exit(1)

    duration = 30
    print(f"\nRunning app with --no-display for {duration} seconds...")
    print("(One person in view recommended)\n")

    try:
        full_output = run_app_and_capture(duration_sec=duration, input_source="rpi")
    except Exception as e:
        print(f"[FAIL] Error running app: {e}")
        sys.exit(1)

    counters = parse_counters(full_output)
    total = counters["total_frames"]
    fe = counters["frame_events"]
    ic = counters["invalid_caps"]
    iv = counters["invalid_validate"]
    fkp17 = counters["frames_keypoints_len_not_17"]
    fwp = counters["frames_with_persons"]
    fwl = counters["frames_with_landmarks"]

    # Print summary of counters (cumulative over full run; snapshot where total_frames is highest)
    print("Counters (cumulative over full run, snapshot where total_frames is highest):")
    print(f"  total_frames               = {total}")
    print(f"  frame_events               = {fe}")
    print(f"  invalid_caps               = {ic}")
    print(f"  invalid_validate           = {iv}")
    print(f"  frames_with_persons        = {fwp}")
    print(f"  frames_with_landmarks      = {fwl}")
    print(f"  frames_keypoints_len_not_17 = {fkp17}")
    print()

    # Print each condition with actual values and PASS/FAIL
    print("Conditions:")
    all_ok = True
    threshold_95 = 0.95 * total if total else 0
    c1 = fe >= threshold_95
    all_ok = all_ok and c1
    print(f"  frame_events >= 0.95 * total_frames     : {fe} >= {threshold_95:.0f}  [{'PASS' if c1 else 'FAIL'}]")

    c2 = ic == 0
    all_ok = all_ok and c2
    print(f"  invalid_caps == 0                       : {ic} == 0  [{'PASS' if c2 else 'FAIL'}]")

    c3 = iv == 0
    all_ok = all_ok and c3
    print(f"  invalid_validate == 0                  : {iv} == 0  [{'PASS' if c3 else 'FAIL'}]")

    c4 = fkp17 == 0
    all_ok = all_ok and c4
    print(f"  frames_keypoints_len_not_17 == 0        : {fkp17} == 0  [{'PASS' if c4 else 'FAIL'}]")

    c5 = fwp >= MIN_PERSON_FRAMES
    all_ok = all_ok and c5
    print(f"  frames_with_persons >= MIN_PERSON_FRAMES ({MIN_PERSON_FRAMES}) : {fwp} >= {MIN_PERSON_FRAMES}  [{'PASS' if c5 else 'FAIL'}]")

    threshold_80 = 0.8 * fwp if fwp else 0
    c6 = fwl >= threshold_80
    all_ok = all_ok and c6
    print(f"  frames_with_landmarks >= 0.8 * frames_with_persons : {fwl} >= {threshold_80:.0f}  [{'PASS' if c6 else 'FAIL'}]")
    print()

    if all_ok:
        print("[PASS] Phase 1 acceptance: all criteria met.")
        sys.exit(0)
    else:
        print("[FAIL] Phase 1 acceptance: one or more conditions failed (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
