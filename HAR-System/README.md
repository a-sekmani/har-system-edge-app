# 🧠 HAR System Edge App

**Real-time human activity recognition using Raspberry Pi and Hailo-8 accelerator**

---

## 📋 Overview

HAR-System is an intelligent edge AI system for real-time human activity recognition. It combines:
- **Hailo-8 AI Accelerator** for fast pose estimation
- **Temporal tracking** to understand behavior over time
- **Normalized measurements** that work on any resolution/distance/angle
- **Real-time processing** at 10-15 FPS 

### ✨ Key Features

- ✅ **Stable tracking** with persistent Track IDs
- ✅ **17 keypoints** per person extraction
- ✅ **Activity classification** (standing, moving, sitting)
- ✅ **Fall detection** with configurable sensitivity
- ✅ **Normalized metrics** independent of camera setup
- ✅ **Data export** to JSON for analysis
- ✅ **Modular architecture** easy to extend

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
│   │   └── callbacks.py        # Frame processing
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   └── cli.py              # CLI tools
│   └── apps/                   # Applications
│       ├── __init__.py
│       ├── realtime_pose.py    # Main real-time app
│       └── chokepoint_analyzer.py  # ChokePoint dataset analyzer
│
├── 🧪 examples/                # Examples & tests
│   ├── demo_temporal_tracking.py
│   └── test_har_tracker.py
│
├── 📜 scripts/                 # Shell scripts
│   ├── run_with_camera.sh
│   ├── run_chokepoint_analysis.sh
│   ├── run_examples.sh
│   └── run_tests.sh
│
├── ⚙️ config/                   # Configuration files
│   └── default.yaml
│
├── 📄 setup.py                 # Installation script
├── 📄 pyproject.toml           # Python project configuration
├── 📄 requirements.txt         # Python dependencies
├── 📄 CHOKEPOINT_README.md     # ChokePoint analyzer documentation
└── 📄 README.md                # This file
```

---

## 🚀 Quick Start

### 1. Installation

#### Prerequisites
```bash
# Install hailo-apps first
cd ~ /hailo-apps
sudo ./install.sh
source setup_env.sh
```

#### Install HAR-System
```bash
cd HAR-System

# Option A: Development mode (recommended)
pip install -e .

# Option B: Production install
pip install .
```

### 2. Run with Camera

```bash
# Using module
python3 -m har_system realtime --input rpi --show-fps

# Using installed command
har-system --input rpi --show-fps

# Using script
cd scripts
./run_with_camera.sh
```

### 3. Run ChokePoint Dataset Analysis

```bash
# Analyze ChokePoint dataset
python3 -m har_system chokepoint --dataset-path ./test_dataset --results-dir ./results

# Or using installed command (after reinstall)
har-chokepoint --dataset-path ./test_dataset --results-dir ./results

# Or using script
cd scripts
./run_chokepoint_analysis.sh
```

For more details, see [CHOKEPOINT_README.md](CHOKEPOINT_README.md).

### 4. Run Examples

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

## 📖 Usage

### Command Line Interface

HAR-System supports multiple commands via the unified CLI:

```bash
# Main entry point
python3 -m har_system <command> [options]

# Or using installed commands (after pip install)
har-system [options]          # Real-time app
har-chokepoint [options]      # ChokePoint analyzer
```

### Real-time Pose Tracking

```bash
# Command options
python3 -m har_system realtime [OPTIONS]

Options:
  -i, --input TEXT        Video source (rpi/usb/file) [default: rpi]
  -f, --show-fps          Show FPS counter
  -v, --verbose           Show detailed information
  --save-data             Save tracking data to JSON
  --output-dir TEXT       Output directory [default: ./results/camera]
  --print-interval INT    Print summary every N frames [default: 30]
```

**Examples:**
```bash
# With Raspberry Pi camera
python3 -m har_system realtime --input rpi --show-fps

# With USB camera
python3 -m har_system realtime --input usb --show-fps

# With video file
python3 -m har_system realtime --input video.mp4

# Save data
python3 -m har_system realtime --input rpi --save-data --output-dir ./my_data

# Verbose mode
python3 -m har_system realtime --input rpi --verbose --print-interval 60
```

### ChokePoint Dataset Analysis

```bash
# Command options
python3 -m har_system chokepoint [OPTIONS]

Options:
  --dataset-path TEXT     Path to test_dataset folder [default: ./test_dataset]
  --results-dir TEXT      Results output directory [default: ./results]
```

**Examples:**
```bash
# Analyze ChokePoint dataset
python3 -m har_system chokepoint --dataset-path ./test_dataset

# Custom results directory
python3 -m har_system chokepoint --dataset-path ./test_dataset --results-dir ./my_results
```

For detailed ChokePoint documentation, see [CHOKEPOINT_README.md](CHOKEPOINT_README.md).

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

Edit `config/default.yaml`:

```yaml
har_system:
  temporal_tracker:
    history_seconds: 3.0
    fps_estimate: 15
  
  activity_classifier:
    thresholds:
      speed_stationary: 0.1
      hip_ratio_sitting: 0.62
  
  fall_detector:
    fall_drop_ratio: 0.30
    fall_time_threshold: 0.5
```

---

## 📊 Output Format

### Terminal Output

```
============================================================
[FRAME] 30 | Active People: 2
============================================================

  [TRACK] 1:
     Activity: moving
     Duration: 12.3s
     Normalized Distance: 45.67
     Moving: 85.2% | Stationary: 14.8%

  [GLOBAL] Statistics:
     Total People: 3
     Falls Detected: 0
     Activity Changes: 5
```

### JSON Export

```json
{
  "track_id": 1,
  "metadata": {
    "first_seen": 1736463421.234,
    "duration_seconds": 12.333
  },
  "current_state": {
    "activity": "moving"
  },
  "statistics": {
    "total_distance_normalized": 45.67,
    "percent_moving": 85.2,
    "fall_detected": false
  }
}
```

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

```bash
# Run all tests
python3 examples/test_har_tracker.py

# Run specific test
python3 -c "from examples.test_har_tracker import test_moving_person; test_moving_person()"
```

---

## 🔨 Development

### Project Structure

```python
har_system/
├── __init__.py           # Package exports
├── __main__.py           # Unified CLI entry point (python -m har_system)
│                         # Supports: realtime, chokepoint commands
├── core/                 # Core components
│   ├── __init__.py
│   ├── tracker.py        # Main tracking algorithm (~540 lines)
│   └── callbacks.py      # Frame processing (~210 lines)
├── utils/                # Utilities
│   ├── __init__.py
│   └── cli.py            # CLI helper functions (~100 lines)
└── apps/                 # Applications
    ├── __init__.py
    ├── realtime_pose.py  # Main real-time app (~115 lines)
    └── chokepoint_analyzer.py  # ChokePoint dataset analyzer (~580 lines)
```

### Entry Points

The system provides multiple entry points:

1. **Module entry point**: `python3 -m har_system <command>` (via `__main__.py`)
2. **Console scripts** (after `pip install`):
   - `har-system` → `har_system.apps.realtime_pose:main`
   - `har-chokepoint` → `har_system.apps.chokepoint_analyzer:main`

### Adding New Features

1. **New activity classifier**: Edit `har_system/core/tracker.py`
2. **New callback**: Edit `har_system/core/callbacks.py`
3. **New application**: 
   - Add to `har_system/apps/`
   - Register in `har_system/__main__.py` as a new command
   - Add entry point in `setup.py` (optional)
4. **New utility**: Add to `har_system/utils/`

---

## 🐛 Troubleshooting

### Import Error
```bash
# Activate environment first
cd /home/admin/hailo-apps
source setup_env.sh
```

### Low FPS
- Use smaller model (yolov8s instead of yolov8m)
- Reduce video resolution
- Decrease `history_seconds` in config

### False Fall Detection
```yaml
# Increase thresholds in config/default.yaml
fall_detector:
  fall_drop_ratio: 0.35      # Less sensitive
  fall_time_threshold: 0.6
```
