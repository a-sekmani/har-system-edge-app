#!/bin/bash
# HAR-System: Run Examples

echo "=================================================="
echo "HAR-System: Examples"
echo "=================================================="
echo ""

# Navigate to correct directory
cd "$(dirname "$0")/.."

# Check environment
if [ ! -d "../venv_hailo_apps" ]; then
    echo "[ERROR] Virtual environment not found"
    echo "   Please run: cd ~/hailo-apps && source setup_env.sh"
    exit 1
fi

# Activate environment
echo "[SETUP] Activating virtual environment..."
source ../setup_env.sh

# Check if specific example is requested
if [ "$1" = "test" ] || [ "$1" = "tracker" ]; then
    echo ""
    echo "[RUN] Running test_har_tracker.py..."
    echo ""
    python3 examples/test_har_tracker.py
elif [ "$1" = "demo" ] || [ "$1" = "temporal" ]; then
    echo ""
    echo "[RUN] Running demo_temporal_tracking.py..."
    echo ""
    python3 examples/demo_temporal_tracking.py
else
    # Run both examples
    echo ""
    echo "[RUN] Running all examples..."
    echo ""
    
    echo "=================================================="
    echo "1. Running test_har_tracker.py"
    echo "=================================================="
    python3 examples/test_har_tracker.py
    
    echo ""
    echo "=================================================="
    echo "2. Running demo_temporal_tracking.py"
    echo "=================================================="
    python3 examples/demo_temporal_tracking.py
fi

echo ""
echo "=================================================="
echo "[DONE] Examples completed"
echo "=================================================="
