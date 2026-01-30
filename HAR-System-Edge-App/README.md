# HAR-System-Edge-App

Edge application for the HAR (Human Activity Recognition) system. It uses the `hailo-apps` library for Pose analysis from a Raspberry Pi camera.

## Structure

```
HAR-System-Edge-App/
├── src/
│   └── har_pose_app.py          # Main application
├── tests/                        # Unit tests
│   ├── conftest.py               # Pytest config and fixtures
│   ├── test_parser.py            # Tests for get_har_parser and --no-display
│   ├── test_fps_tracker.py       # Tests for FPSTracker
│   ├── test_har_user_data.py     # Tests for HARUserData
│   ├── test_callback.py          # Tests for simple_callback
│   └── test_har_pose_app.py      # Tests for HARPoseEstimationApp
├── test_phase0.py                # Phase 0 acceptance test script
├── pytest.ini                    # Pytest config
├── README.md                     # This file
└── requirements.txt              # Dependencies
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

### Options

- `--input rpi`: Use Raspberry Pi camera
- `--input usb`: Use USB camera
- `--no-display`: Disable video display (use fakesink)
- `--show-fps`: Show or log FPS
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
- **test_har_user_data.py**: `HARUserData` (fps_tracker, inheritance, attributes)
- **test_callback.py**: `simple_callback` (buffer=None, FPS update, no raise)
- **test_har_pose_app.py**: `HARPoseEstimationApp` (inheritance, pipeline with fakesink, main)

### Phase 0 acceptance tests

1. **With display**: Run 30 seconds with display enabled
2. **Without display**: Run 30 seconds with `--no-display`
3. **Long run**: Run 5–10 minutes without errors

```bash
python test_phase0.py
```

### Acceptance criteria

- Application runs without errors for 5–10 minutes
- FPS is shown or logged consistently
- No errors in logs

## Notes

- The app uses `GStreamerPoseEstimationApp` from `hailo-apps` without modification
- `--no-display` is implemented by overriding `get_pipeline_string()` to use `fakesink`
- FPS is measured in the callback function
