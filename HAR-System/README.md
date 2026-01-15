# 🧠 HAR-System: Human Activity Recognition

**Real-time human activity recognition using Hailo-8 and Raspberry Pi**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

HAR-System is an intelligent edge AI system for real-time human activity recognition. It combines:
- **Hailo-8 AI Accelerator** for fast pose estimation
- **Temporal tracking** to understand behavior over time
- **Normalized measurements** that work on any resolution/distance/angle
- **Real-time processing** at 10-15 FPS on Raspberry Pi 5

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
│   ├── core/                   # Core tracking engine
│   │   ├── tracker.py          # TemporalActivityTracker
│   │   └── callbacks.py        # Frame processing
│   ├── utils/                  # Utility functions
│   │   └── cli.py              # CLI tools
│   └── apps/                   # Applications
│       └── realtime_pose.py    # Main app
│
├── 🧪 examples/                # Examples & tests
├── 📜 scripts/                 # Shell scripts
├── ⚙️ config/                   # Configuration files
└── 📄 setup.py                 # Installation script
```

---

## 🚀 Quick Start

### 1. Installation

#### Prerequisites
```bash
# Install hailo-apps first
cd /home/admin/hailo-apps
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
python3 -m har_system --input rpi --show-fps

# Using installed command
har-system --input rpi --show-fps

# Using script
cd scripts
./run_with_camera.sh
```

### 3. Run Examples

```bash
# Test tracker
python3 examples/test_har_tracker.py

# Simple demo
python3 examples/demo_temporal_tracking.py
```

---

## 📖 Usage

### Command Line Options

```bash
har-system [OPTIONS]

Options:
  -i, --input TEXT        Video source (rpi/usb/file) [default: rpi]
  -f, --show-fps          Show FPS counter
  -v, --verbose           Show detailed information
  --save-data             Save tracking data to JSON
  --output-dir TEXT       Output directory [default: ./temporal_data]
  --print-interval INT    Print summary every N frames [default: 30]
```

### Examples

```bash
# With Raspberry Pi camera
har-system --input rpi --show-fps

# With USB camera
har-system --input usb --show-fps

# With video file
har-system --input video.mp4

# Save data
har-system --input rpi --save-data --output-dir ./my_data

# Verbose mode
har-system --input rpi --verbose --print-interval 60
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
├── __main__.py           # Entry point for python -m
├── core/                 # Core components
│   ├── tracker.py        # Main tracking algorithm (~540 lines)
│   └── callbacks.py      # Frame processing (~210 lines)
├── utils/                # Utilities
│   └── cli.py            # CLI functions (~80 lines)
└── apps/                 # Applications
    └── realtime_pose.py  # Main app (~115 lines)
```

### Adding New Features

1. **New activity classifier**: Edit `har_system/core/tracker.py`
2. **New callback**: Edit `har_system/core/callbacks.py`
3. **New application**: Add to `har_system/apps/`
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

---

## 📦 Dependencies

- **Python** >= 3.8
- **numpy** >= 1.21.0
- **opencv-python** >= 4.5.0
- **hailo-apps** (for Hailo integration)
- **GStreamer** with Hailo plugins

All dependencies are installed via `setup.py`.

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- Built on top of [hailo-apps](https://github.com/hailo-ai/hailo-rpi5-examples)
- Uses Hailo-8 AI Accelerator
- Designed for Raspberry Pi 5

---

## 📞 Support

- **Issues**: Check examples and tests first
- **Questions**: Review this README
- **Community**: [Hailo Community Forum](https://community.hailo.ai/)

---

**Built with ❤️ for Edge AI**  
**Version**: 1.0.0  
**Status**: 🟢 Production Ready
