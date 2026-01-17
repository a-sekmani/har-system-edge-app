#!/bin/bash
# HAR-System: Quick Start Script with Face Recognition
# =====================================================

echo "=================================================="
echo "HAR-System: دليل البدء السريع"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in HAR-System directory
if [ ! -f "har_system/__init__.py" ]; then
    echo "⚠️  يرجى تشغيل هذا السكريبت من مجلد HAR-System"
    exit 1
fi

echo "${GREEN}✓${NC} الموقع الصحيح"
echo ""

# Function to display menu
show_menu() {
    echo "اختر العملية:"
    echo "  1) إعداد مجلد التدريب"
    echo "  2) تدريب نظام التعرف على الوجوه"
    echo "  3) عرض الأشخاص المعروفين"
    echo "  4) تشغيل النظام (بدون تعرف على الوجوه)"
    echo "  5) تشغيل النظام (مع تعرف على الوجوه)"
    echo "  6) حذف شخص من القاعدة"
    echo "  7) اختبار النظام"
    echo "  0) خروج"
    echo ""
    read -p "اختيارك: " choice
    echo ""
}

# Setup training directory
setup_training() {
    echo "${YELLOW}[1] إعداد مجلد التدريب${NC}"
    echo ""
    
    read -p "أدخل اسم الشخص الأول: " person1
    read -p "أدخل اسم الشخص الثاني (اختياري): " person2
    
    mkdir -p train_faces/$person1
    
    if [ ! -z "$person2" ]; then
        mkdir -p train_faces/$person2
    fi
    
    echo ""
    echo "${GREEN}✓${NC} تم إنشاء المجلدات:"
    echo "  train_faces/$person1/"
    [ ! -z "$person2" ] && echo "  train_faces/$person2/"
    echo ""
    echo "الآن أضف 3-5 صور لكل شخص في مجلده"
    echo "مثال: cp ~/photos/*.jpg train_faces/$person1/"
    echo ""
}

# Train faces
train_faces() {
    echo "${YELLOW}[2] تدريب نظام التعرف على الوجوه${NC}"
    echo ""
    
    if [ ! -d "train_faces" ]; then
        echo "⚠️  مجلد train_faces غير موجود"
        echo "   قم بتشغيل الخيار 1 أولاً"
        echo ""
        return
    fi
    
    python3 -m har_system train-faces --train-dir ./train_faces
    echo ""
}

# List known persons
list_persons() {
    echo "${YELLOW}[3] الأشخاص المعروفين${NC}"
    echo ""
    
    python3 -m har_system faces --list
    echo ""
}

# Run without face recognition
run_basic() {
    echo "${YELLOW}[4] تشغيل النظام (بدون تعرف على الوجوه)${NC}"
    echo ""
    echo "سيتم تشغيل النظام مع تتبع النشاط فقط"
    echo "اضغط Ctrl+C للإيقاف"
    echo ""
    sleep 2
    
    python3 -m har_system realtime --input rpi --show-fps
}

# Run with face recognition
run_with_faces() {
    echo "${YELLOW}[5] تشغيل النظام (مع تعرف على الوجوه)${NC}"
    echo ""
    
    if [ ! -d "database" ]; then
        echo "⚠️  قاعدة البيانات غير موجودة"
        echo "   قم بتدريب النظام أولاً (الخيار 2)"
        echo ""
        return
    fi
    
    echo "سيتم تشغيل النظام مع التعرف على الوجوه"
    echo "اضغط Ctrl+C للإيقاف"
    echo ""
    sleep 2
    
    python3 -m har_system realtime --input rpi --show-fps --enable-face-recognition
}

# Remove person
remove_person() {
    echo "${YELLOW}[6] حذف شخص من القاعدة${NC}"
    echo ""
    
    read -p "أدخل اسم الشخص المراد حذفه: " person_name
    
    if [ -z "$person_name" ]; then
        echo "⚠️  الاسم فارغ"
        echo ""
        return
    fi
    
    python3 -m har_system faces --remove "$person_name"
    echo ""
}

# Test system
test_system() {
    echo "${YELLOW}[7] اختبار النظام${NC}"
    echo ""
    
    python3 -c "
import sys
sys.path.insert(0, '.')

print('اختبار النظام...')
print()

try:
    from har_system.core import TemporalActivityTracker, FaceIdentityManager
    from har_system.integrations import HailoFaceRecognition
    
    print('✓ جميع المكونات متوفرة')
    
    # Test tracker
    tracker = TemporalActivityTracker()
    tracker.update_identity(1, 'Test')
    assert tracker.get_identity(1) == 'Test'
    print('✓ Tracker يعمل بشكل صحيح')
    
    # Test face manager
    face_mgr = FaceIdentityManager()
    face_mgr.update_identity(1, 'Test', 0.9)
    print('✓ FaceIdentityManager يعمل بشكل صحيح')
    
    print()
    print('✅ النظام جاهز للعمل!')
    
except Exception as e:
    print(f'✗ خطأ: {e}')
    sys.exit(1)
"
    echo ""
}

# Main loop
while true; do
    show_menu
    
    case $choice in
        1) setup_training ;;
        2) train_faces ;;
        3) list_persons ;;
        4) run_basic ;;
        5) run_with_faces ;;
        6) remove_person ;;
        7) test_system ;;
        0) echo "وداعاً!"; exit 0 ;;
        *) echo "اختيار غير صحيح" ;;
    esac
    
    read -p "اضغط Enter للمتابعة..."
    clear
done
