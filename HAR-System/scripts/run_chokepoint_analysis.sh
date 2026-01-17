#!/bin/bash
# تشغيل تحليل ChokePoint Dataset

echo "=================================================="
echo "HAR-System: ChokePoint Dataset Analyzer"
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

# التحقق من وجود مجلد test_dataset
if [ ! -d "test_dataset/choke_point" ]; then
    echo "[ERROR] مجلد test_dataset/choke_point غير موجود!"
    echo "يرجى إنشاء المجلد ووضع البيانات فيه:"
    echo "  test_dataset/choke_point/video_001/00000000.jpg"
    echo "  test_dataset/choke_point/video_001/00000001.jpg"
    echo "  ..."
    exit 1
fi

# تشغيل التحليل (باستخدام الوحدة الرئيسية المدمجة)
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
