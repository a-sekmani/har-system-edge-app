# Setup and Requirements

## Requirements

- Python 3.10 or later
- hailo-apps library installed and configured
- Raspberry Pi with camera (for live input)
- Hailo device connected

## Setup

1. From the **repository root** (directory containing `HAR-System-Edge-App` and `venv_hailo_apps`):

   ```bash
   cd /path/to/har-system-edge-app-v0.2
   source setup_env.sh
   ```

   This sets `PYTHONPATH` to the repo root and activates the virtual environment (`venv_hailo_apps`).

2. Run the application from the **HAR-System-Edge-App** directory:

   ```bash
   cd HAR-System-Edge-App
   python src/har_pose_app.py --input rpi --no-display --show-fps
   ```

   Alternatively, from the repo root without changing directory:

   ```bash
   export PYTHONPATH=/path/to/har-system-edge-app-v0.2
   source venv_hailo_apps/bin/activate
   python HAR-System-Edge-App/src/har_pose_app.py --input rpi --no-display --show-fps
   ```

   (The app sets its working directory to `HAR-System-Edge-App` at startup so `face_gallery` and other project paths resolve correctly.)

## Optional: Face recognition

If you use `--enable-face`, install: insightface, onnxruntime, opencv-python.
