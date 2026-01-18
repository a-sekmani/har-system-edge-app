#!/bin/bash
# HAR-System: Run Examples

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "HAR-System: Examples"
echo "=================================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

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
