#!/bin/bash
# HAR-System: Quick test script

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "HAR-System: Testing Temporal Tracker"
echo "=================================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run tests
echo ""
echo "[RUN] Running HAR-System tests..."
echo ""
python3 examples/test_har_tracker.py

echo ""
echo "=================================================="
echo "[DONE] Tests completed"
echo "=================================================="
