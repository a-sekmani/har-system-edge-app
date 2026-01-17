#!/bin/bash
# HAR-System: Run with Raspberry Pi Camera

echo "=================================================="
echo "HAR-System: Real-time Activity Recognition"
echo "=================================================="
echo ""

# Navigate to correct directory
cd "$(dirname "$0")/.."

# Check environment
if [ ! -d "../venv_hailo_apps" ]; then
    echo "[ERROR] Virtual environment not found"
    echo "   Please run: cd ~/hailo-apps && sudo ./install.sh"
    exit 1
fi

# Activate environment
echo "[SETUP] Activating virtual environment..."
source ../setup_env.sh

# Run application
echo ""
echo "[RUN] Starting HAR-System..."
echo "   Press Ctrl+C to stop"
echo ""
python3 -m har_system realtime --input rpi --show-fps

echo ""
echo "=================================================="
echo "[EXIT] HAR-System stopped"
echo "=================================================="
