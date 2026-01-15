#!/bin/bash
# HAR-System: Quick test script

echo "=================================================="
echo "HAR-System: Testing Temporal Tracker"
echo "=================================================="
echo ""

# Navigate to correct directory
cd "$(dirname "$0")/.."

# Check environment
if [ ! -d "../venv_hailo_apps" ]; then
    echo "[ERROR] Virtual environment not found"
    echo "   Please run: cd /home/admin/hailo-apps && source setup_env.sh"
    exit 1
fi

# Activate environment
echo "[SETUP] Activating virtual environment..."
source ../setup_env.sh

# Run tests
echo ""
echo "[RUN] Running HAR-System tests..."
echo ""
python3 examples/test_har_tracker.py

echo ""
echo "=================================================="
echo "[DONE] Tests completed"
echo "=================================================="
