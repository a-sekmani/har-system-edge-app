# HAR-System-Edge-App

Edge application for the HAR (Human Activity Recognition) system. It uses the `hailo-apps` library for Pose analysis from a Raspberry Pi camera.

## Structure

```
HAR-System-Edge-App/
├── src/
│   ├── frame_event.py            # FrameEvent, PersonPose (with track_id), COCO-17, validation
│   ├── tracker.py                # Phase 2: TrackingConfig, FallbackTracker, get_metadata_track_id
│   ├── cloud_schema.py           # Phase 3: Cloud Event schema, build_cloud_payload()
│   ├── cloud_client.py           # Phase 3: CloudSender (HTTP POST), CloudSendQueue, retry, drop policy
│   └── har_pose_app.py           # Main application (Phase 1 + Phase 2 + Phase 3)
├── tests/
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_fps_tracker.py
│   ├── test_har_user_data.py
│   ├── test_callback.py
│   ├── test_har_pose_app.py
│   ├── test_frame_event.py
│   ├── test_pose_extraction.py
│   ├── test_validation.py
│   ├── test_phase1_logic.py
│   ├── test_tracker.py            # Phase 2: tracker core tests
│   ├── test_phase2_logic.py       # Phase 2: parse_counters, check_phase2_conditions
│   ├── test_cloud_schema.py       # Phase 3: payload schema, keypoints length 17
│   └── test_cloud_client.py       # Phase 3: queue drop policy, retry, CloudSender
├── test_phase0.py                # Phase 0 acceptance test script
├── test_phase1.py                # Phase 1 acceptance test script
├── test_phase2.py                # Phase 2 acceptance test script (tracking)
├── test_phase3.py                # Phase 3 acceptance test script (cloud streaming dry-run / enable_cloud false)
├── tools/
│   └── mock_cloud_server.py      # Mock HTTP server for Phase 3 E2E (POST -> 200, no deps)
├── pytest.ini
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- `hailo-apps` library installed and configured
- Raspberry Pi with camera
- Hailo device connected

## Setup

1. Ensure `hailo-apps` is installed and configured:
```bash
cd /path/to/har-system-edge-app-v0.2
source setup_env.sh
```

2. Activate the virtual environment:
```bash
source venv_hailo_apps/bin/activate  # or your venv name
```

## Usage

### Phase 0 - Baseline

#### Run with display:
```bash
cd HAR-System-Edge-App
python src/har_pose_app.py --input rpi --show-fps
```

#### Run without display (for better performance):
```bash
python src/har_pose_app.py --input rpi --no-display --show-fps
```

#### Run acceptance test script:
```bash
python test_phase0.py
```

### Phase 1 - Raw Pose Data (FrameEvent)

The app produces a **FrameEvent** per frame: frame number, timestamp, image size, and a list of **PersonPose** (bbox in pixels, detection confidence, 17 keypoints in COCO order). No tracking (`track_id`) or cloud upload in this phase.

- **COCO-17 keypoint order** (indices 0–16): nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
- **Missing keypoint sentinel**: `[-1, -1, 0.0]` (same type for all keypoints; no null).
- Each person has `bbox` `[x1, y1, x2, y2]` in pixels, `bbox_conf`, and exactly 17 keypoints `[x, y, c]`.
- Invalid frames are skipped with an optional counter; no crash.

#### Phase 1 acceptance test

```bash
python test_phase1.py
```

Runs the app with `--no-display` for 30 seconds, parses counters from the log, and exits 0 = PASS or 1 = FAIL. **One person in view** is recommended so that person/landmark counters are non-zero.

**Output:** The script prints a summary of counters (total_frames, frame_events, invalid_caps, invalid_validate, frames_with_persons, frames_with_landmarks, frames_keypoints_len_not_17), then **PASS** or **FAIL** with clear reasons (e.g. which condition was not met).

**Phase 1 passes only if all of the following hold:**

| Condition | Requirement |
|-----------|-------------|
| frame_events | `frame_events >= 0.95 * total_frames` |
| invalid_caps | `invalid_caps == 0` |
| invalid_validate | `invalid_validate == 0` |
| frames_keypoints_len_not_17 | `== 0` |
| frames_with_persons | `>= MIN_PERSON_FRAMES` (30) |
| frames_with_landmarks | `>= 0.8 * frames_with_persons` |

**Summary counters** (logged every FPS interval and at exit by the app): `frames_with_persons`, `frames_no_persons`, `persons_total`, `frames_with_landmarks`, `frames_keypoints_len_not_17`.

### Phase 2 — Tracking (Track IDs)

Every person in a **FrameEvent** has a **track_id** (int). Keypoints and validation are unchanged; missing track_id does not make the frame invalid.

- **track_id in PersonPose**: Required; use `-1` (TRACK_ID_UNKNOWN) only when tracking cannot assign an id (e.g. fallback failed or policy allows).
- **Source**: `metadata` — use Hailo `HAILO_UNIQUE_ID` from detection when available; otherwise fallback tracker for that detection. `fallback` — use IoU-based tracker for all detections.
- **Lifecycle**: Tracks expire after `max_missing_frames` (or optional `max_track_age_seconds`); cleanup keeps the number of stored tracks bounded. Re-appearing detections get a new id after expiry (no reuse of expired id).

- **Detection filter (reduce ghost tracks):** Before tracking, detections can be filtered so that small or low-confidence ones are excluded and do not create tracks. Use `--min-bbox-area A` (pixels²), `--min-bbox-height H` (pixels), and/or `--min-pose-confidence C` (0–1, average keypoint confidence). This helps single-person acceptance when the model outputs extra “ghost” detections.

**Phase 2 counters** (cumulative, logged in FPS interval and at exit):

| Counter | Meaning |
|--------|--------|
| unique_track_ids | Count of distinct track ids ever seen |
| new_tracks_created | Number of track IDs seen for the first time (metadata or fallback) |
| tracks_ended | Tracks removed after max_missing_frames / max_track_age |
| id_switch_suspected | Heuristic: same frame ids but assignment flip (e.g. two persons, ids swap) |
| multi_person_frames | Frames where number of persons ≥ 2 |
| detections_total | Raw person detections from ROI (before filter) |
| filtered_detections_total | Detections excluded by min_bbox_area / min_bbox_height / min_pose_confidence |

**CLI flags (Phase 2):**

- `--tracking-source metadata|fallback` (default: `metadata`) — Track ID source.
- `--max-missing-frames N` (default: 15) — Expire track after N frames without detection (fallback).
- `--iou-threshold X` (default: 0.3) — IoU threshold for fallback tracker matching.
- `--min-bbox-area A` (default: 0) — Filter detections with bbox area below A (pixels²).
- `--min-bbox-height H` — Filter detections with bbox height below H pixels (reduces ghost tracks).
- `--min-pose-confidence C` — Filter detections with avg keypoint confidence below C (0–1).
- `--log-tracking-summary` — Log Phase 2 tracking summary periodically.

#### Phase 2 acceptance test

```bash
python test_phase2.py
```

Runs the app with `--no-display` and `--tracking-source fallback` for 30 seconds, parses Phase 1 + Phase 2 counters from the log, and exits 0 = PASS or 1 = FAIL. **One person in view** is recommended.

**Single-person criteria (required):**

| Condition | Requirement |
|-----------|-------------|
| frame_events | `>= 0.95 * total_frames` |
| invalid_caps | `== 0` |
| invalid_validate | `== 0` |
| frames_with_persons | `>= 30` |
| unique_track_ids | `<= 2` (single-person: ≤1 normally, ≤2 with brief disappearance) |
| id_switch_suspected | `== 0` (single-person: no id swap expected) |

**Optional two-person mode:** Run with two persons in view; criteria include `multi_person_frames >= 30`, `unique_track_ids` in 2–6, `id_switch_suspected <= 10` (script can be extended with a flag for "two" mode).

### Phase 3 — Cloud streaming

Phase 3 streams **tracks and 17 keypoints** (no images, no analytics) from the edge to a configurable cloud endpoint. The app acts as a **producer only**; analytics run in the cloud. When `--enable-cloud` is false, Phase 1 and Phase 2 behaviour and counters are unchanged.

**Cloud Event schema (JSON payload):**

- **event_type**: `"frame_event"` (constant).
- **source**: `device_id` (string, stable per device), `session_id` (UUID per run), `model` (e.g. from config), `tracking_source` (`"metadata"` | `"fallback"`).
- **frame**: `frame_index`, `ts_monotonic_ms` or `ts_unix_ms`, `image_w`, `image_h`, optional `fps_current`, `fps_avg`.
- **persons**: list of objects with `track_id` (int), `bbox_xyxy` (or documented bbox format), optional `score` (bbox confidence), `keypoints` (array of 17; each element `{name, x, y, c}` with fixed COCO-17 order), `keypoints_format`: `"coco17"`, `coords`: `"pixel"`.

No image data is sent in Phase 3.

**Transport:** HTTP POST to a configurable base URL + ingest path. Auth via `Authorization: Bearer <api_key>` or `X-API-Key`; configurable timeout, retries, optional TLS verification. WS/SSE can be added later.

**Queue and drop policy:** In-memory queue with configurable `max_queue_size`. When the queue is full, one event is dropped: **drop oldest** (default) so the newest is kept, or **drop newest** (configurable via `--drop-policy`). On send failure, the client retries up to `max_retries` with backoff; after exhaustion the event is dropped and `events_failed` / `events_dropped` are incremented. Sending does not block the pipeline: the callback enqueues and drains at most one item per valid frame so FPS stays stable.

**Rate control:** `send_every_n_frames` (default 1) controls how often a cloud event is built and enqueued (e.g. 3 = every 3rd valid frame).

**Phase 3 counters** (logged in FPS interval and at exit when `enable_cloud`):

| Counter | Meaning |
|--------|--------|
| events_built | Events built and (if not dry-run) enqueued |
| events_sent | Successfully sent to cloud |
| events_failed | Send failed after max_retries (dropped) |
| events_dropped | Dropped due to full queue or retry exhaustion |
| queue_depth | Current queue size |
| queue_depth_max | Maximum queue size seen |

**CLI flags (Phase 3):**

- `--enable-cloud`: Enable cloud streaming (default: False).
- `--cloud-url`: Base URL for the cloud endpoint.
- `--cloud-api-key`: API key; if empty, read from env `CLOUD_API_KEY`.
- `--cloud-ingest-path`: Path appended to base URL (e.g. `/v1/edge/events`).
- `--send-every-n-frames`: Send every N valid frames (default: 1).
- `--max-queue-size`: Maximum in-memory queue size.
- `--send-timeout-ms`: HTTP send timeout in milliseconds.
- `--max-retries`: Number of retries on send failure.
- `--drop-policy`: `oldest` or `newest` when queue is full (default: oldest).
- `--dry-run`: Build and count payloads, do not POST (events_sent remains 0).
- `--verify-tls` / `--no-verify-tls`: TLS certificate verification (default: verify).

**Acceptance criteria (Phase 3):**

- With `--enable-cloud false`: Phase 1/2 behaviour and counters unchanged; no Phase 3 activity.
- **Dry-run:** With `--enable-cloud --dry-run`, payloads are built and counted (`events_built` meets threshold), not sent (`events_sent == 0`, `events_failed == 0`), and validation errors are zero.
- **Live send:** With a working endpoint, `events_sent > 0`, `events_failed` minimal, `queue_depth_max` bounded, and FPS not significantly worse than Phase 2 (e.g. within ~10%).
- Optional: On network disconnect, the app does not crash; `events_failed` and `events_dropped` increase as expected.

#### Phase 3 acceptance test

```bash
python test_phase3.py
```

Runs five checks: (1) with `--enable-cloud false`, Phase 3 counters zero; (2) with `--enable-cloud --dry-run`, events_built meets threshold, events_sent == 0, events_failed == 0, invalid_validate == 0; (3) with `--send-every-n-frames 2`, events_built ≈ total_frames/2 within margin; (4) local HTTP sink: app sends to a local POST-accepting server, events_sent > 0, events_failed == 0, server_received >= events_sent; (5) invalid URL: events_failed > 0 (queue/drop policy exercised).

#### Mock cloud server (E2E)

A mock HTTP server accepts POST and returns 200 immediately so you can verify send/receive end-to-end and confirm FPS stays high when the backend is fast (no blocking on timeouts). Stdlib only; no extra dependencies.

**Terminal 1 — start mock server:**

```bash
python tools/mock_cloud_server.py --port 9999
```

**Terminal 2 — run app against mock:**

```bash
python src/har_pose_app.py --input rpi --no-display --show-fps \
  --enable-cloud --cloud-url http://127.0.0.1:9999 --send-every-n-frames 10
```

**Expected:** `events_sent > 0`, `events_failed == 0`, `events_dropped == 0`, FPS close to ~30 (may drop slightly with payload). The mock prints `POST /v1/edge/events -> 200 (received=N)` for each request.

### Options

- `--input rpi`: Use Raspberry Pi camera
- `--input usb`: Use USB camera
- `--no-display`: Disable video display (use fakesink)
- `--show-fps`: Show or log FPS (includes frame_events and invalid_frames counts)
- `--log-pose-summary`: Log pose summary every N seconds (persons count, sample bbox/keypoints)
- `--dump-frames path`: Write each FrameEvent (or every K-th) to a JSON file for debugging
- `--tracking-source metadata|fallback`: Track ID source (default: metadata)
- `--max-missing-frames N`: Expire track after N frames without detection (default: 15)
- `--iou-threshold X`: IoU threshold for fallback tracker (default: 0.3)
- `--min-bbox-area A`: Filter detections below this bbox area in pixels² (default: 0)
- `--min-bbox-height H`: Filter detections with bbox height below H pixels
- `--min-pose-confidence C`: Filter detections with avg keypoint confidence below C (0–1)
- `--log-tracking-summary`: Log Phase 2 tracking summary periodically
- `--enable-cloud`: Enable Phase 3 cloud streaming (default: False)
- `--cloud-url`: Base URL for cloud endpoint
- `--cloud-api-key`: API key (or use env CLOUD_API_KEY)
- `--cloud-ingest-path`: Ingest path (e.g. /v1/edge/events)
- `--send-every-n-frames`: Send every N valid frames (default: 1)
- `--max-queue-size`: Max in-memory queue size for cloud events
- `--send-timeout-ms`: HTTP send timeout in ms
- `--max-retries`: Retries on send failure
- `--drop-policy`: oldest or newest when queue full (default: oldest)
- `--dry-run`: Build and count payloads, do not POST
- `--verify-tls` / `--no-verify-tls`: TLS verification (default: verify)
- `--help`: Show all options

## Testing

### Unit tests

Run all unit tests from the project root (parent directory must contain `hailo_apps`):

```bash
cd HAR-System-Edge-App
pytest tests/ -v
```

Or from the repository root `har-system-edge-app-v0.2`:

```bash
cd har-system-edge-app-v0.2
pytest HAR-System-Edge-App/tests/ -v
```

**Coverage:**
- **test_parser.py**: `get_har_parser()`, `--no-display`, `--log-pose-summary`, `--dump-frames`, Phase 2 flags (`--tracking-source`, `--max-missing-frames`, `--min-bbox-height`, `--min-pose-confidence`, `--log-tracking-summary`), defaults
- **test_fps_tracker.py**: `FPSTracker` (update, get_fps, get_average_fps, frame window)
- **test_har_user_data.py**: `HARUserData` (fps_tracker, inheritance; Phase 1 and Phase 2 counters including `detections_total`, `filtered_detections_total`; optional log_pose_summary/dump_frames_path)
- **test_callback.py**: `simple_callback` (buffer=None, FPS update, no raise)
- **test_har_pose_app.py**: `HARPoseEstimationApp` (inheritance, pipeline with fakesink); `_print_final_stats()` (no raise, logger called); `_pose_confidence_from_detection()` (filter helper)
- **test_frame_event.py**: `FrameEvent`, `PersonPose` (with `track_id`), COCO-17 order, missing keypoint sentinel, bbox format
- **test_pose_extraction.py**: Mock hailo detection → `PersonPose`/`FrameEvent`; Policy 1 clamp (keypoints out of bounds); `track_id` in `from_hailo_detection`
- **test_validation.py**: `validate_frame_event` (valid/invalid keypoints, bbox, image; does not depend on `track_id`; skip policy)
- **test_phase1_logic.py**: `parse_counters()` and `check_phase1_conditions()` from test_phase1 (log parsing, condition logic)
- **test_tracker.py**: `TrackingConfig` (including min_bbox_height, min_pose_confidence); `get_metadata_track_id`; FallbackTracker (stable id, missing/recover, expire, two persons, IoU threshold, min_bbox_area)
- **test_phase2_logic.py**: Phase 2 `parse_counters` (including detections_total, filtered_detections_total) and `check_phase2_single_person_conditions` / `check_phase2_two_person_conditions` (log parsing, single/two-person criteria)
- **test_cloud_schema.py**: Phase 3 payload schema; `build_cloud_payload()` (event_type, source, frame, persons with len(keypoints)==17, keypoints_format coco17, coords pixel); schema sanity (device_id, session_id, frame_index, timestamp, persons list, per-person track_id, 17 keypoints, x,y within image bounds or sentinel)
- **test_cloud_client.py**: Phase 3 CloudConfig (API key from env), CloudSender (POST URL/headers, failure path), CloudSendQueue (drop policy oldest/newest when full, drain_one retry and counters)

### Phase 0 acceptance tests

1. **With display**: Run 30 seconds with display enabled
2. **Without display**: Run 30 seconds with `--no-display`
3. **Long run**: Run 5–10 minutes without errors

```bash
python test_phase0.py
```

### Phase 1 acceptance test

- Run `test_phase1.py`: app runs with `--no-display` for 30 seconds; script parses counters and reports PASS/FAIL (see Phase 1 section above for full criteria).

```bash
python test_phase1.py
```

### Phase 2 acceptance test

- Run `test_phase2.py`: app runs with `--no-display` and `--tracking-source fallback` for 30 seconds; script parses Phase 1 + Phase 2 counters and reports PASS/FAIL (single-person: frame_events ≥ 95%, invalid_caps/validate 0, frames_with_persons ≥ 30, unique_track_ids ≤ 2, id_switch_suspected == 0).

```bash
python test_phase2.py
```

### Phase 3 acceptance test

- Run `test_phase3.py`: (1) enable_cloud false; (2) dry-run criteria; (3) send_every_n_frames=2; (4) local HTTP sink (events_sent > 0, server_received >= events_sent); (5) invalid URL (events_failed > 0).

```bash
python test_phase3.py
```

### Acceptance criteria

- Application runs without errors for 5–10 minutes
- FPS is shown or logged consistently (Phase 0 and Phase 1: frame_events, invalid_caps, invalid_validate)
- No errors in logs
- Phase 1: All Phase 1 conditions met (frame_events ≥ 95% of total_frames, invalid_caps/validate 0, keypoints_len_not_17 = 0, frames_with_persons ≥ 30, frames_with_landmarks ≥ 80% of frames_with_persons)
- Phase 2: All Phase 2 single-person conditions met (same as Phase 1 plus unique_track_ids ≤ 2, id_switch_suspected == 0)
- Phase 3: With `--enable-cloud false`, Phase 1/2 unchanged; dry-run: events_built meets threshold, events_sent == 0, events_failed == 0; optional live send: events_sent > 0, FPS within ~10% of Phase 2

## Notes

- The app uses `GStreamerPoseEstimationApp` from `hailo-apps` without modification
- `--no-display` is implemented by overriding `get_pipeline_string()` to use `fakesink`
- The callback builds `FrameEvent` per frame (with `track_id`), validates (see `src/frame_event.py`), applies the detection filter (min_bbox_area / min_bbox_height / min_pose_confidence), updates Phase 1 and Phase 2 counters, optionally builds and enqueues Phase 3 cloud payloads (rate-limited by send_every_n_frames) and drains the queue (non-blocking), then logs FPS
