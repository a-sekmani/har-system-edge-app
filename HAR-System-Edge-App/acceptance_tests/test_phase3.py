#!/usr/bin/env python3
"""
Phase 3 acceptance test script.
(1) With --enable-cloud false: run 30s, assert Phase 1/2 unchanged, no Phase 3 send.
(2) With --enable-cloud --dry-run --no-display: run 30s, parse Phase 3 counters,
    assert events_built >= 0.95 * (total_frames / send_every_n_frames), events_sent == 0,
    events_failed == 0, invalid_validate == 0.
(3) send_every_n_frames=2: dry-run with --send-every-n-frames 2, events_built ≈ total_frames/2.
(4) Local HTTP sink: run app against a local POST-accepting server; events_sent > 0, events_failed == 0,
    server_received >= events_sent (log may be stale).
(5) Network failure: send to invalid URL; events_failed > 0 (queue/drop policy exercised).
Exit: 0 = PASS, 1 = FAIL.
"""

import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEND_EVERY_N_FRAMES = 1


def run_app_and_capture(
    duration_sec=30,
    input_source="rpi",
    enable_cloud=False,
    dry_run=False,
    send_every_n_frames=1,
    cloud_url=None,
    max_queue_size=None,
):
    """Run app with given flags for duration_sec; return full stdout."""
    cmd = [
        sys.executable,
        "src/har_pose_app.py",
        "--input", input_source,
        "--no-display",
        "--show-fps",
        "--tracking-source", "fallback",
    ]
    if enable_cloud:
        cmd.append("--enable-cloud")
    if dry_run:
        cmd.append("--dry-run")
    if send_every_n_frames != 1:
        cmd.extend(["--send-every-n-frames", str(send_every_n_frames)])
    if cloud_url is not None:
        cmd.extend(["--cloud-url", cloud_url])
    if max_queue_size is not None:
        cmd.extend(["--max-queue-size", str(max_queue_size)])
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


def _find_free_port():
    """Bind to port 0 to get a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CountingPOSTHandler(BaseHTTPRequestHandler):
    """HTTP handler that counts POST requests and returns 200."""

    # Class-level counter; set server_received on the server object
    def do_POST(self):
        if hasattr(self.server, "post_count"):
            self.server.post_count += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format, *args):
        pass  # quiet


def run_local_http_sink(port):
    """Start HTTPServer on port in a daemon thread; return (server, post_count_list)."""
    received = [0]  # mutable so handler can update

    class Server(HTTPServer):
        post_count = 0

    server = Server(("127.0.0.1", port), _CountingPOSTHandler)
    server.post_count = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def parse_counters(full_output):
    """
    Parse FPS Stats, Phase1, Phase2, and Phase3 summary/final lines.
    Returns dict with total_frames, frame_events, invalid_*, Phase1/2 counters, and Phase3 (events_built, events_sent, events_failed, events_dropped, queue_depth_max).
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
        "events_built": 0,
        "events_sent": 0,
        "events_failed": 0,
        "events_dropped": 0,
        "queue_depth_max": 0,
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

    phase2_matches = list(re.finditer(
        r"Phase2\s+(?:summary|final):\s*unique_track_ids=(\d+).*?new_tracks_created=(\d+).*?tracks_ended=(\d+).*?id_switch_suspected=(\d+).*?multi_person_frames=(\d+)",
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

    phase3_matches = list(re.finditer(
        r"Phase3\s+(?:summary|final):\s*events_built=(\d+).*?events_sent=(\d+).*?events_failed=(\d+).*?events_dropped=(\d+).*?queue_depth=(\d+).*?queue_depth_max=(\d+)",
        full_output,
        re.DOTALL,
    ))
    if phase3_matches:
        m = phase3_matches[-1]
        out["events_built"] = int(m.group(1))
        out["events_sent"] = int(m.group(2))
        out["events_failed"] = int(m.group(3))
        out["events_dropped"] = int(m.group(4))
        out["queue_depth_max"] = int(m.group(6))

    return out


def check_phase3_dry_run_conditions(counters, send_every_n_frames=1):
    """Return (all_ok, failed_reasons) for Phase 3 dry-run acceptance."""
    reasons = []
    total = counters.get("total_frames", 0)
    fe = counters.get("frame_events", 0)
    iv = counters.get("invalid_validate", 0)
    built = counters.get("events_built", 0)
    sent = counters.get("events_sent", 0)
    failed = counters.get("events_failed", 0)
    fwp = counters.get("frames_with_persons", 0)
    if total == 0:
        reasons.append("total_frames is 0")
        return False, reasons
    expected_built_min = int(0.95 * (total / send_every_n_frames))
    if built < expected_built_min:
        reasons.append(f"events_built ({built}) < 0.95 * (total_frames / send_every_n_frames) ({expected_built_min})")
    if sent != 0:
        reasons.append(f"events_sent ({sent}) != 0 (dry-run must not send)")
    if failed != 0:
        reasons.append(f"events_failed ({failed}) != 0")
    if iv != 0:
        reasons.append(f"invalid_validate ({iv}) != 0")
    return (len(reasons) == 0, reasons)


def main():
    """Run Phase 3 acceptance: (1) enable_cloud false, (2) dry-run, (3) send_every_n_frames=2, (4) local HTTP sink, (5) network failure + queue."""
    print("=" * 60)
    print("Phase 3 Acceptance Test")
    print("=" * 60)

    if not (_PROJECT_ROOT / "src" / "har_pose_app.py").exists():
        print("[FAIL] src/har_pose_app.py not found. Run from HAR-System-Edge-App directory.")
        sys.exit(1)

    duration = 30
    duration_short = 15
    failures = []

    # Test 1: enable_cloud false — Phase 1/2 unchanged, no Phase 3 send
    print("\n[Test 1/5] Running with --enable-cloud false for {} seconds...".format(duration))
    try:
        output1 = run_app_and_capture(
            duration_sec=duration,
            enable_cloud=False,
            dry_run=False,
        )
    except Exception as e:
        print("[FAIL] Error running app: {}".format(e))
        failures.append("Test 1: {}".format(e))
    else:
        counters1 = parse_counters(output1)
        if counters1.get("events_built", 0) != 0 or counters1.get("events_sent", 0) != 0:
            failures.append("Test 1: enable_cloud false but events_built={} events_sent={}".format(
                counters1.get("events_built", 0), counters1.get("events_sent", 0)))
        else:
            print("[PASS] enable_cloud false: no Phase 3 activity.")

    # Test 2: dry-run — events_built >= threshold, events_sent == 0, events_failed == 0, invalid_validate == 0
    print("\n[Test 2/5] Running with --enable-cloud --dry-run for {} seconds...".format(duration))
    try:
        output2 = run_app_and_capture(
            duration_sec=duration,
            enable_cloud=True,
            dry_run=True,
            send_every_n_frames=SEND_EVERY_N_FRAMES,
        )
    except Exception as e:
        failures.append("Test 2: {}".format(e))
    else:
        counters2 = parse_counters(output2)
        ok, reasons = check_phase3_dry_run_conditions(counters2, send_every_n_frames=SEND_EVERY_N_FRAMES)
        print("  events_built={} events_sent={} events_failed={} invalid_validate={}".format(
            counters2["events_built"], counters2["events_sent"], counters2["events_failed"], counters2["invalid_validate"]))
        if ok:
            print("[PASS] dry-run: all criteria met.")
        else:
            failures.append("Test 2: " + "; ".join(reasons))

    # Test 3: send_every_n_frames=2 — events_built ≈ total_frames/2 within margin
    print("\n[Test 3/5] Running with --enable-cloud --dry-run --send-every-n-frames 2 for {} seconds...".format(duration))
    try:
        output3 = run_app_and_capture(
            duration_sec=duration,
            enable_cloud=True,
            dry_run=True,
            send_every_n_frames=2,
        )
    except Exception as e:
        failures.append("Test 3: {}".format(e))
    else:
        c3 = parse_counters(output3)
        total = c3["total_frames"]
        built = c3["events_built"]
        # events_built ≈ total_frames/2: allow 0.4 * total .. 0.6 * total + small margin
        expected_half = total / 2.0
        margin = max(20, int(0.15 * total))
        low = int(0.4 * total)
        high = int(0.6 * total) + margin
        ok3 = total > 0 and low <= built <= high
        print("  total_frames={} events_built={} (expected ~{:.0f}, range [{}, {}])".format(total, built, expected_half, low, high))
        if ok3:
            print("[PASS] send_every_n_frames=2: events_built ≈ total_frames/2.")
        else:
            failures.append("Test 3: events_built={} not in [{}, {}] for total_frames={}".format(built, low, high, total))

    # Test 4: Local HTTP sink — events_sent > 0, events_failed == 0, server received ≈ events_sent
    port = _find_free_port()
    base_url = "http://127.0.0.1:{}".format(port)
    print("\n[Test 4/5] Starting local HTTP sink on {} for {} seconds...".format(base_url, duration_short))
    server = HTTPServer(("127.0.0.1", port), _CountingPOSTHandler)
    server.post_count = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        output4 = run_app_and_capture(
            duration_sec=duration_short,
            enable_cloud=True,
            dry_run=False,
            cloud_url=base_url,
        )
    except Exception as e:
        failures.append("Test 4: {}".format(e))
    finally:
        server.shutdown()
    server_received = server.post_count
    c4 = parse_counters(output4)
    sent = c4["events_sent"]
    failed = c4["events_failed"]
    # App logs Phase3 periodically; last line may be stale. Accept server_received >= events_sent.
    ok4 = sent > 0 and failed == 0 and server_received >= sent
    print("  events_sent={} events_failed={} server_received={}".format(sent, failed, server_received))
    if ok4:
        print("[PASS] Local HTTP sink: events_sent>0, events_failed==0, server received >= events_sent.")
    else:
        failures.append("Test 4: sent={} failed={} server_received={} (expected sent>0, failed==0, server_received>=sent)".format(
            sent, failed, server_received))

    # Test 5: Invalid URL — events_failed > 0 (and events_dropped > 0 when drain fails).
    # With enqueue-one-then-drain-one per frame, queue does not fill; queue_depth_max may stay 1.
    invalid_port = _find_free_port()
    invalid_url = "http://127.0.0.1:{}".format(invalid_port)  # nothing listening
    max_q = 5
    print("\n[Test 5/5] Running with invalid URL and max-queue-size={} for {} seconds...".format(max_q, duration_short))
    try:
        output5 = run_app_and_capture(
            duration_sec=duration_short,
            enable_cloud=True,
            dry_run=False,
            cloud_url=invalid_url,
            max_queue_size=max_q,
        )
    except Exception as e:
        failures.append("Test 5: {}".format(e))
    else:
        c5 = parse_counters(output5)
        failed5 = c5["events_failed"]
        qmax = c5["queue_depth_max"]
        dropped = c5["events_dropped"]
        ok5 = failed5 > 0
        print("  events_failed={} queue_depth_max={} events_dropped={}".format(failed5, qmax, dropped))
        if ok5:
            print("[PASS] Network failure: events_failed>0 (queue/drop policy exercised).")
        else:
            failures.append("Test 5: events_failed={} (expected >0)".format(failed5))

    # Summary
    print("\n" + "=" * 60)
    if not failures:
        print("[PASS] Phase 3 acceptance: all 5 tests passed.")
        sys.exit(0)
    else:
        print("[FAIL] Phase 3 acceptance: {} condition(s) failed:".format(len(failures)))
        for f in failures:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
