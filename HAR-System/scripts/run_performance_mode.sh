#!/bin/bash
# HAR-System Performance Mode - Quick Start Script
# This script runs HAR-System in optimal performance mode

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting HAR-System in Performance Mode"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  ✓ Display: DISABLED (--no-display)"
echo "  ✓ Video Source: Raspberry Pi Camera"
echo "  ✓ Print Interval: Every 60 frames"
echo "  ✓ Face Recognition: DISABLED"
echo ""
echo "Expected Performance:"
echo "  • CPU Usage: ~50-70% (vs 80-95% with display)"
echo "  • FPS: 12-15 (vs 8-12 with display)"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Change to project root to ensure imports work
cd "$PROJECT_ROOT"

# Run HAR-System in performance mode
python3 -m har_system realtime \
    --input rpi \
    --no-display \
    --print-interval 60 \
    --show-fps

echo ""
echo "✅ HAR-System stopped"
