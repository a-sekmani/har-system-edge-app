#!/bin/bash
# HAR-System: Run with Raspberry Pi Camera

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "HAR-System: Real-time Activity Recognition"
echo "=================================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

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
