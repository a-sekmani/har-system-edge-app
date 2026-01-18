#!/bin/bash
# Run ChokePoint dataset analysis

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "HAR-System: ChokePoint Dataset Analyzer"
echo "=================================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check that the dataset folder exists
if [ ! -d "test_dataset/choke_point" ]; then
    echo "[ERROR] مجلد test_dataset/choke_point غير موجود!"
    echo "يرجى إنشاء المجلد ووضع البيانات فيه:"
    echo "  test_dataset/choke_point/video_001/00000000.jpg"
    echo "  test_dataset/choke_point/video_001/00000001.jpg"
    echo "  ..."
    exit 1
fi

# Run the analysis via the packaged CLI entry point
echo ""
echo "[RUN] Starting ChokePoint analysis..."
echo ""
python3 -m har_system chokepoint \
    --dataset-path ./test_dataset \
    --results-dir ./results

echo ""
echo "=================================================="
echo "[DONE] اكتمل التحليل! النتائج في مجلد ./results/choke_point"
echo "=================================================="
