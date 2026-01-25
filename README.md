# 🧠 HAR System - Human Activity Recognition (Edge application)

**Real-time human activity recognition using Raspberry Pi and Hailo-8 AI Accelerator**

---

## 📋 Overview

HAR-System is an intelligent edge AI system for **real-time human activity recognition** on Raspberry Pi.

### ✨ Key Features

**Core Recognition & Tracking:**
- ✅ **Stable Multi-Person Tracking** - Persistent Track IDs across frames with temporal consistency
- ✅ **17 Keypoints Extraction** - Full body pose estimation per person using Hailo-8 AI accelerator
- ✅ **Activity Classification** - Automatic detection: standing, moving, sitting with temporal analysis
- ✅ **Fall Detection** - Configurable sensitivity for safety applications with time-based validation

**Face Recognition System:**
- ✅ **Face Recognition** - Real-time person identification by name using LanceDB
- ✅ **Face Training** - Train the system with custom images for specific persons
- ✅ **Face Database Management** - Add, remove, list, and manage recognized persons

**Advanced Capabilities:**
- ✅ **ChokePoint Dataset Analysis** - Analyze pedestrian datasets for person tracking evaluation
- ✅ **Normalized Metrics** - Camera-independent measurements (resolution and distance independent)
- ✅ **Temporal Tracking** - History-based activity tracking with configurable time windows
- ✅ **Multiple Input Sources** - Support for Raspberry Pi camera and USB cameras

**Performance & Data:**
- ✅ **Performance Mode** - No-display mode for maximum performance in production
- ✅ **Real-time FPS Monitoring** - Built-in performance metrics and frame rate display
- ✅ **Data Export** - JSON export for post-processing, analysis, and integration
- ✅ **Configurable Thresholds** - Customizable sensitivity for all detection algorithms

**Developer Experience:**
- ✅ **Unified CLI** - Single entry point with multiple commands (realtime, train-faces, faces, chokepoint)
- ✅ **Comprehensive Testing** - Full test suite with unit and component tests using pytest
- ✅ **Modular Architecture** - Easy to extend and customize with clean separation of concerns
- ✅ **YAML Configuration** - Flexible configuration system for all system parameters

---

## 🏗️ Architecture

```
HAR-System/
│
├── 📦 har_system/              # Main package
│   ├── __init__.py             # Package initialization
│   ├── __main__.py             # CLI entry point (python -m har_system)
│   ├── core/                   # Core tracking engine
│   │   ├── __init__.py
│   │   ├── tracker.py          # TemporalActivityTracker
│   │   ├── callbacks.py        # Frame processing
│   │   ├── face_identity_manager.py  # Face identity management
│   │   └── face_processor.py    # Face recognition processor
│   ├── integrations/           # External system integrations (NEW)
│   │   ├── __init__.py
│   │   └── hailo_face_recognition.py  # Face recognition integration
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── cli.py              # CLI tools
│   │   └── overlay.py          # Overlay helpers
│   └── apps/                   # Applications
│       ├── __init__.py
│       ├── realtime_pose.py    # Main real-time app
│       ├── train_faces.py      # Face training app 
│       ├── manage_faces.py     # Face management app 
│       └── chokepoint_analyzer.py  # ChokePoint dataset analyzer
│
├── 🧪 examples/                # Examples & demos
│   ├── demo_temporal_tracking.py
│   └── test_har_tracker.py
│
├── 🧪 tests/                   # Test suite
│   ├── conftest.py             # Pytest configuration and fixtures
│   ├── unit/                   # Unit tests (fast, no dependencies)
│   │   ├── test_cli.py
│   │   ├── test_face_identity_manager.py
│   │   ├── test_face_processor.py
│   │   ├── test_hailo_integration.py
│   │   ├── test_main_dispatcher.py
│   │   ├── test_overlay.py
│   │   └── test_tracker.py
│   └── component/              # Component tests (module contracts)
│       ├── test_callbacks_extractors.py
│       ├── test_manage_faces_operations.py
│       └── test_train_faces_validation.py
│
├── 📜 scripts/                 # Shell scripts
│   ├── run_with_camera.sh          # Quick start with camera
│   ├── run_performance_mode.sh     # Run in performance mode 
│   ├── test_performance.sh         # Performance benchmarking
│   ├── run_chokepoint_analysis.sh  # ChokePoint analyzer
│   ├── run_examples.sh             # Run examples
│   ├── run_tests.sh                # Run tests
│   └── train_faces_auto.sh         # Automated face training
│
├── ⚙️ config/                   # Configuration files
│   └── default.yaml            # Main configuration
│
├── 📄 pytest.ini               # Pytest configuration

```

---

## 🚀 Quick Start

### 1. Installation

#### Prerequisites
```bash
# Install hailo-apps
cd ~/har-system-edge-app
sudo ./install.sh

# Activating the environment
source setup_env.sh

# Installing dependencies
pip install -e .

# Move to the project library
cd HAR-System
```

### 2. Run with Camera

#### Basic Usage
```bash
# Standard mode (with display)
python3 -m har_system realtime --input rpi --show-fps

# Performance mode (no display, faster) ⚡ RECOMMENDED
python3 -m har_system realtime --input rpi --no-display

# Quick start script
cd scripts
./run_with_camera.sh              # Standard mode
./run_performance_mode.sh         # Performance mode
```

#### Advanced Usage
```bash
# With face recognition (requires trained database)
python3 -m har_system realtime --input rpi --enable-face-recognition

# Maximum performance (for production)
python3 -m har_system realtime --input rpi --no-display --print-interval 60

# With data collection
python3 -m har_system realtime --input rpi --no-display --save-data

# Using USB camera
python3 -m har_system realtime --input usb --show-fps
```

### 3. Face Recognition

Face recognition allows you to identify specific persons by name in real-time.

#### Training the System
```bash
# Step 1: Organize training images
mkdir -p train_faces/Ahmed train_faces/Sara
# Add 5-10 photos per person (different angles, lighting)

# Step 2: Train the system
python3 -m har_system train-faces --train-dir ./train_faces

# Step 3: Verify training
python3 -m har_system faces --list
```

#### Train only a limited number of persons (testing)

Use `--max-persons` to limit how many person folders are trained (useful for quick testing on a subset, e.g. 5 persons):

```bash
python3 -m har_system train-faces --train-dir ./train_faces --max-persons 5
```

#### Managing Faces Database
```bash
# List all known persons
python3 -m har_system faces --list

# Remove a specific person
python3 -m har_system faces --remove Ahmed

# Clear entire database
python3 -m har_system faces --clear
```

#### Running with Face Recognition
```bash
# Enable face recognition
python3 -m har_system realtime --input rpi --enable-face-recognition

# With performance mode (recommended)
python3 -m har_system realtime --input rpi --enable-face-recognition --no-display

# Custom database directory
python3 -m har_system realtime --input rpi --enable-face-recognition --database-dir ./custom_db
```

### 4. Performance Testing

Test and compare system performance:

```bash
# Automated performance comparison
cd scripts
./test_performance.sh

# Monitor CPU usage (in separate terminal)
top -H -p $(pgrep -f har_system)

# Check FPS in output
# Look for: [PERF] Average processing time: X ms | FPS: Y
```

### 5. Run ChokePoint Dataset Analysis

Analyze the ChokePoint pedestrian dataset:

```bash
# Basic analysis
python3 -m har_system chokepoint --dataset-path ./test_dataset --results-dir ./results

# Performance mode (no display)
python3 -m har_system chokepoint --dataset-path ./test_dataset --results-dir ./results --no-display

# Using script
cd scripts
./run_chokepoint_analysis.sh
```

### 6. Run Examples

```bash
# Using script (recommended)
cd scripts
./run_examples.sh              # Run all examples
./run_examples.sh test          # Run test_har_tracker.py only
./run_examples.sh demo          # Run demo_temporal_tracking.py only

# Or directly with Python
python3 examples/test_har_tracker.py
python3 examples/demo_temporal_tracking.py
```

---

## 📖 Usage Guide

### Command Line Interface

HAR-System provides a unified CLI with multiple commands:

```bash
# Main entry point
python3 -m har_system <command> [options]

# Available commands:
commands:
  realtime       Real-time pose tracking with activity recognition
  train-faces    Train face recognition system with images
  faces          Manage face recognition database
  chokepoint     Analyze ChokePoint pedestrian dataset

# Installed commands (after pip install)
har-system [options]          # Shortcut for: python3 -m har_system realtime
har-chokepoint [options]      # Shortcut for: python3 -m har_system chokepoint
```

#### Quick Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `realtime` | Live tracking | `python3 -m har_system realtime --input rpi` |
| `train-faces` | Train faces | `python3 -m har_system train-faces --train-dir ./faces` |
| `faces --list` | List persons | `python3 -m har_system faces --list` |
| `chokepoint` | Dataset analysis | `python3 -m har_system chokepoint --dataset-path ./data` |

### Real-time Pose Tracking

#### Command Options

```bash
python3 -m har_system realtime [OPTIONS]

Video Input:
  -i, --input TEXT              Video source (rpi/usb/file) [default: rpi]
  
Display & Performance:
  -f, --show-fps                Show FPS counter on display
  --no-display                  Disable video display (improves performance)
  
Output & Logging:
  -v, --verbose                 Show detailed debug information
  --print-interval INT          Print summary every N frames [default: 30]
  --save-data                   Save tracking data to JSON files
  --output-dir TEXT             Output directory [default: ./results/camera]
  
Face Recognition:
  --enable-face-recognition     Enable face recognition
  --database-dir TEXT           Face database directory [default: from config/default.yaml or ./database]
```

#### Usage Examples

**Basic Usage:**
```bash
# Standard mode (with display)
python3 -m har_system realtime --input rpi --show-fps

# Performance mode (recommended for production) ⚡
python3 -m har_system realtime --input rpi --no-display
```

**Different Video Sources:**
```bash
# Raspberry Pi Camera (default)
python3 -m har_system realtime --input rpi

# USB Camera
python3 -m har_system realtime --input usb

# Specific device
python3 -m har_system realtime --input /dev/video0

# Video file
python3 -m har_system realtime --input video.mp4
```

**Advanced Features:**
```bash
# With face recognition
python3 -m har_system realtime --input rpi --enable-face-recognition --no-display

# Data collection mode
python3 -m har_system realtime --input rpi --no-display --save-data --output-dir ./data

# Verbose debugging
python3 -m har_system realtime --input rpi --verbose --print-interval 30

# Maximum performance (production)
python3 -m har_system realtime --input rpi --no-display --print-interval 60
```

### ChokePoint Dataset Analysis

Analyze pedestrian activity patterns from the ChokePoint dataset.

#### Command Options

```bash
python3 -m har_system chokepoint [OPTIONS]

Options:
  --dataset-path TEXT          Path to test_dataset folder [default: ./test_dataset]
  --results-dir TEXT           Results output directory [default: ./results]
  --enable-face-recognition    Enable face recognition (optional)
  --database-dir TEXT          Face database directory (if face recognition enabled)
  --no-display                 Disable video display (improves performance)
```

#### Examples

```bash
# Basic analysis
python3 -m har_system chokepoint --dataset-path ./test_dataset

# Performance mode (no display)
python3 -m har_system chokepoint --dataset-path ./test_dataset --no-display

# Custom output directory
python3 -m har_system chokepoint --dataset-path ./test_dataset --results-dir ./my_results

# With face recognition
python3 -m har_system chokepoint --dataset-path ./test_dataset --enable-face-recognition

# Full features with performance mode
python3 -m har_system chokepoint \
    --dataset-path ./test_dataset \
    --enable-face-recognition \
    --database-dir ./database \
    --no-display
```

---

## 🔧 Python API

### Basic Usage

```python
from har_system import TemporalActivityTracker

# Create tracker
tracker = TemporalActivityTracker(
    history_seconds=3.0,  # Keep 3 seconds of history
    fps_estimate=15       # Expect ~15 FPS
)

# Update with frame data
frame_data = {
    'timestamp': time.time(),
    'bbox': {'xmin': 100, 'ymin': 150, 'xmax': 200, 'ymax': 400},
    'keypoints': {...},  # 17 keypoints
    'confidence': 0.95
}

activity = tracker.update(track_id=1, frame_data=frame_data)
print(f"Current activity: {activity}")
```

### Get Statistics

```python
# Get summary for person
summary = tracker.get_summary(track_id=1)
print(f"Activity: {summary['current_activity']}")
print(f"Moving: {summary['stats']['percent_moving']:.1f}%")

# Detect activity changes
change = tracker.detect_activity_change(track_id=1)
if change:
    print(f"Changed from {change['from']} to {change['to']}")

# Get all active people
active = tracker.get_all_active_tracks()
print(f"Active people: {len(active)}")

# Export data
tracker.save_to_json(track_id=1, filepath='person_1.json')
```

---

## ⚙️ Configuration

### Configuration File

`config/default.yaml` is used to define **defaults** for:
- Temporal tracker (`history_seconds`, `fps_estimate`)
- Activity / fall thresholds
- Face recognition parameters (thresholds, intervals, db paths)
- Export save interval (`data_export.save_interval`)
- Video size and frame rate passed to the underlying Hailo GStreamer app (`video.width/height/frame_rate`)

Some behaviors are still controlled via CLI flags (for example: enabling face recognition, enabling saving, and `--no-display`).

```yaml
har_system:
  # Temporal Tracking Settings
  temporal_tracker:
    history_seconds: 3.0        # How many seconds of history to keep
    fps_estimate: 15            # Expected FPS (for temporal calculations)
  
  # Activity Classification Thresholds
  activity_classifier:
    thresholds:
      speed_stationary: 0.1     # Below this = stationary
      speed_slow: 0.5           # Between stationary and fast
      speed_fast: 1.5           # Above this = fast movement
      hip_ratio_sitting: 0.62   # Hip position threshold for sitting
  
  # Fall Detection Settings
  fall_detector:
    fall_drop_ratio: 0.30       # Drop > 30% of height = potential fall
    fall_time_threshold: 0.5    # Must occur within 0.5 seconds
  
  # Face Recognition Settings
  face_recognition:
    enabled: false                      # Parameters only; enabling is done via --enable-face-recognition
    database_dir: "./database"          # Database directory
    samples_dir: "./database/samples"   # Samples directory
    confidence_threshold: 0.60          # Minimum confidence (0.0-1.0)
    recognition_interval_frames: 10     # Check face every N frames
    skip_first_frames: 3                # Skip first N frames (usually blurry)
    min_confirmations: 1                # Confirmations before trusting identity
    identity_timeout: 5.0               # Seconds before re-confirmation
    max_faces_per_frame: 5              # Maximum faces to process per frame

# Display Settings (overridden by CLI flags)
display:
  verbose: false                  # Show detailed information
  print_every_n_frames: 30        # Print summary every N frames
  show_fps: true                  # Prefer enabling via --show-fps

# Data Export Settings
data_export:
  save_data: false                # Prefer enabling via --save-data
  output_dir: ./results/camera    # Save directory
  save_interval: 300              # Save every N frames

# Video Settings (passed to GStreamer)
video:
  input: rpi                      # Video source: rpi, usb, /dev/videoX, or file
  width: 1280                     # Video width (16:9 aspect ratio, default)
  height: 720                     # Video height (16:9 aspect ratio, default)
  frame_rate: 30                  # Target frame rate
```

### Performance Tuning

For better performance, adjust these settings:

```yaml
# Maximum Performance Configuration
video:
  width: 640                      # Lower resolution (keep 16:9 ratio)
  height: 360

har_system:
  temporal_tracker:
    history_seconds: 2.0          # Less history

display:
  print_every_n_frames: 60        # Less frequent output

face_recognition:
  enabled: false                  # Disable if not needed
  recognition_interval_frames: 20 # Less frequent recognition
```

Then run with:
```bash
python3 -m har_system realtime --input rpi --no-display --print-interval 60
```

---

## 📊 Output Format

### Terminal Output

When running with display or in verbose mode, you'll see:

```
============================================================
[FRAME] 30 | Active People: 2
============================================================

  [TRACK] 1 - Ahmed:               ← Identified by face recognition
     Activity: moving
     Duration: 12.3s
     Normalized Distance: 45.67
     Moving: 85.2% | Stationary: 14.8%

  [TRACK] 2 - Unknown:             ← Face not recognized
     Activity: standing
     Duration: 5.8s
     Stationary: 92.5%

  [GLOBAL] Statistics:
     Total People: 3
     Falls Detected: 0
     Activity Changes: 5

[PERF] Average processing time: 66.7ms | FPS: 15.0
```

### JSON Export

When using `--save-data`, each tracked person is saved to JSON:

```json
{
  "track_id": 1,
  "name": "Ahmed",                      // Person name (if recognized)
  "metadata": {
    "first_seen": 1736463421.234,
    "last_seen": 1736463433.567,
    "duration_seconds": 12.333,
    "frame_count": 185
  },
  "current_state": {
    "activity": "moving",
    "bbox": {
      "xmin": 0.25,
      "ymin": 0.15,
      "xmax": 0.55,
      "ymax": 0.95
    },
    "confidence": 0.94
  },
  "statistics": {
    "total_distance_normalized": 45.67,
    "percent_moving": 85.2,
    "percent_stationary": 14.8,
    "percent_sitting": 0.0,
    "fall_detected": false,
    "activity_changes": 3
  },
  "history": {
    "total_frames": 185,
    "keypoints_available": true
  }
}
```

**Output Files Location:**
- Default: `./results/camera/`
- Filenames: `track_{id}_frame_{frame_number}.json` (periodic), `track_{id}_final.json` (on exit)

---

## 📐 Normalized Measurements

All measurements are **resolution and distance independent**:

| Metric | Formula | Benefits |
|--------|---------|----------|
| Speed | `(distance/dt) / bbox_height` | Works on any resolution |
| Pose Height | `(nose_y - ankle_y) / bbox_height` | Distance independent |
| Hip Ratio | `(hip_y - ankle_y) / bbox_height` | Angle independent |

**Why normalized?**
- ✅ Same thresholds work on 640×480 or 1920×1080
- ✅ Works whether person is 2m or 10m from camera
- ✅ Robust to different camera angles

---

## 🧪 Testing

### Running Tests

```bash
# Activate the virtual environment first
source ../venv/bin/activate

# Run all tests (unit + component)
pytest

# Run only unit tests (fast, no device/GStreamer required)
pytest -m unit

# Run only component tests
pytest -m component

# Run specific test file
pytest tests/unit/test_tracker.py

# Run specific test
pytest tests/unit/test_tracker.py::test_temporal_tracking

# Run with verbose output
pytest -v

# Run with output capture disabled (see print statements)
pytest -s

# Run legacy tracker tests (script-style)
python3 examples/test_har_tracker.py
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Fast tests that don't require GStreamer or Hailo device
  - CLI parsing and validation
  - Core tracker logic
  - Face identity management
  - Overlay utilities
  
- **Component Tests** (`tests/component/`): Tests for module contracts and integrations
  - Callback extractors
  - Face management operations
  - Face training validation

### Test Markers

Tests are categorized using pytest markers:
- `@pytest.mark.unit` - Fast tests without device/GStreamer
- `@pytest.mark.component` - Tests for module contracts (often mocked)
- `@pytest.mark.integration` - Tests that may require gstreamer/device/resources
- `@pytest.mark.requires_gstreamer` - Requires GStreamer runtime
- `@pytest.mark.requires_device` - Requires a Hailo device

### Configuration

Test configuration is defined in `pytest.ini`:
- Default quiet mode (`-q`)
- Test paths: `tests/`
- Custom markers for test categorization

---

## 🐛 Troubleshooting

### Common Issues

#### Import Error / Module Not Found
```bash
# Solution: Activate hailo-apps environment
cd /home/admin/hailo-apps
source setup_env.sh

# Reinstall HAR-System
cd HAR-System
pip install -e .
```

#### Low FPS / High CPU Usage

**Quick Fixes:**
1. ⚡ **Use `--no-display` flag** (often improves performance noticeably, Because it reduces the load on the CPU, which, according to experience, is a bottleneck.)
   ```bash
   python3 -m har_system realtime --input rpi --no-display
   ```

2. 📉 **Reduce terminal output**
   ```bash
   python3 -m har_system realtime --input rpi --print-interval 60
   ```

3. 📺 **Lower video resolution** (edit `config/default.yaml`)
   ```yaml
   video:
     width: 640     # Instead of 1280
     height: 360    # Instead of 720 (keep 16:9 ratio)
   ```

4. 🔄 **Disable face recognition** (if not needed)
   ```bash
   # Just don't use --enable-face-recognition flag
   python3 -m har_system realtime --input rpi --no-display
   ```

**Advanced Optimization:**
- Use smaller model: `yolov8s_pose.hef` instead of `yolov8m_pose.hef`
- Reduce tracking history: Set `history_seconds: 2.0` in config
- Check for multiple processes: `ps aux | grep har_system`
- Verify Hailo device: `hailortcli fw-control identify`

#### False Fall Detection

Adjust sensitivity in `config/default.yaml`:
```yaml
fall_detector:
  fall_drop_ratio: 0.35      # Higher = less sensitive
  fall_time_threshold: 0.6   # Longer time window
```

#### Camera Not Detected

```bash
# Check available cameras
ls /dev/video*

# Try specific device
python3 -m har_system realtime --input /dev/video0

# For USB camera
python3 -m har_system realtime --input usb
```

#### Face Recognition Issues

```bash
# Verify database exists and has persons
python3 -m har_system faces --list

# Retrain if needed
python3 -m har_system train-faces --train-dir ./train_faces

# Check database directory
ls -la ./database/persons.db/
```

### Performance Benchmarking

```bash
# Test your system performance
cd scripts
./test_performance.sh

# Monitor CPU in real-time
top -H -p $(pgrep -f har_system)

# Check GPU/NPU usage (Hailo)
hailortcli fw-control identify
```