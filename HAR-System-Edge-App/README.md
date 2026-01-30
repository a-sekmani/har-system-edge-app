# HAR-System-Edge-App

Edge application for the HAR (Human Activity Recognition) system. It uses the `hailo-apps` library for Pose analysis from a Raspberry Pi camera.

## Structure

```
HAR-System-Edge-App/
├── src/
│   ├── frame_event.py            # FrameEvent, PersonPose, COCO-17, validation (Phase 1)
│   └── har_pose_app.py           # Main application
├── tests/
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_fps_tracker.py
│   ├── test_har_user_data.py
│   ├── test_callback.py
│   ├── test_har_pose_app.py
│   ├── test_frame_event.py       # FrameEvent, PersonPose, COCO-17, sentinel (Phase 1)
│   ├── test_pose_extraction.py   # Mock detection → PersonPose (Phase 1)
│   ├── test_validation.py        # validate_frame_event (Phase 1)
│   └── test_phase1_logic.py      # parse_counters, check_phase1_conditions (Phase 1)
├── test_phase0.py                # Phase 0 acceptance test script
├── test_phase1.py                # Phase 1 acceptance test script (FrameEvent production)
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

### Options

- `--input rpi`: Use Raspberry Pi camera
- `--input usb`: Use USB camera
- `--no-display`: Disable video display (use fakesink)
- `--show-fps`: Show or log FPS (includes frame_events and invalid_frames counts)
- `--log-pose-summary`: Log pose summary every N seconds (persons count, sample bbox/keypoints)
- `--dump-frames path`: Write each FrameEvent (or every K-th) to a JSON file for debugging
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
- **test_parser.py**: `get_har_parser()`, `--no-display` option, default values
- **test_fps_tracker.py**: `FPSTracker` (update, get_fps, get_average_fps, frame window)
- **test_har_user_data.py**: `HARUserData` (fps_tracker, inheritance; Phase 1 counters, optional log_pose_summary/dump_frames_path)
- **test_callback.py**: `simple_callback` (buffer=None, FPS update, no raise)
- **test_har_pose_app.py**: `HARPoseEstimationApp` (inheritance, pipeline with fakesink); `_print_final_stats()` (no raise, logger called)
- **test_parser.py**: `get_har_parser()`, `--no-display`, `--log-pose-summary`, `--dump-frames` (presence, defaults)
- **test_frame_event.py**: `FrameEvent`, `PersonPose`, COCO-17 order, missing keypoint sentinel, bbox format
- **test_pose_extraction.py**: Mock hailo detection → `PersonPose`/`FrameEvent`; Policy 1 clamp (keypoints out of bounds)
- **test_validation.py**: `validate_frame_event` (valid/invalid keypoints, bbox, image; skip policy)
- **test_phase1_logic.py**: `parse_counters()` and `check_phase1_conditions()` from test_phase1 (log parsing, condition logic)

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

### Acceptance criteria

- Application runs without errors for 5–10 minutes
- FPS is shown or logged consistently (Phase 0 and Phase 1: frame_events, invalid_caps, invalid_validate)
- No errors in logs
- Phase 1: All Phase 1 conditions met (frame_events ≥ 95% of total_frames, invalid_caps/validate 0, keypoints_len_not_17 = 0, frames_with_persons ≥ 30, frames_with_landmarks ≥ 80% of frames_with_persons)

## Notes

- The app uses `GStreamerPoseEstimationApp` from `hailo-apps` without modification
- `--no-display` is implemented by overriding `get_pipeline_string()` to use `fakesink`
- FPS is measured in the callback; Phase 1 callback builds `FrameEvent` per frame and validates (see `src/frame_event.py`)
