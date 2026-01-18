#!/bin/bash
# Quick test script to compare performance with and without display

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "======================================"
echo "HAR-System Performance Test"
echo "======================================"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Test duration in seconds
DURATION=30

echo "Test 1: Running WITH display for ${DURATION} seconds..."
echo "Press Ctrl+C after observing for a while"
echo ""
timeout ${DURATION} python3 -m har_system realtime --input rpi --show-fps --print-interval 30 || true

echo ""
echo "======================================"
echo ""
echo "Test 2: Running WITHOUT display for ${DURATION} seconds..."
echo "Press Ctrl+C after observing for a while"
echo ""
timeout ${DURATION} python3 -m har_system realtime --input rpi --no-display --print-interval 30 || true

echo ""
echo "======================================"
echo "Performance Test Complete!"
echo "======================================"
echo ""
echo "Compare the FPS values from both tests:"
echo "- Test 1 (with display): Typically 8-12 FPS"
echo "- Test 2 (no display): Typically 12-15 FPS"
echo ""
echo "Monitor CPU usage during tests:"
echo "  top -H -p \$(pgrep -f har_system)"
echo ""
