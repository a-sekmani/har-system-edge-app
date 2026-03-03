# Setup and Requirements

## Requirements

- Python 3.10 or later
- hailo-apps library installed and configured
- Raspberry Pi with camera (for live input)
- Hailo device connected

## Setup

1. Ensure hailo-apps is installed. From the repository root (parent of HAR-System-Edge-App):

   ```bash
   cd /path/to/har-system-edge-app-v0.2
   source setup_env.sh
   ```

2. Activate the virtual environment:

   ```bash
   source venv_hailo_apps/bin/activate
   ```

3. Run the application from the HAR-System-Edge-App directory.

## Optional: Face recognition

If you use `--enable-face`, install: insightface, onnxruntime, opencv-python.
