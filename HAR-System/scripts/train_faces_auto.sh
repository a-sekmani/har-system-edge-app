#!/bin/bash
# HAR-System: Automatic Face Training Script
# This script handles training automatically using hailo-apps

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAR_SYSTEM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HAILO_APPS_ROOT="$(cd "$HAR_SYSTEM_ROOT/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "============================================================"
echo "HAR-System: Automatic Face Training"
echo "============================================================"
echo ""

# Parse arguments
TRAIN_DIR="${HAR_SYSTEM_ROOT}/train_faces"
DATABASE_DIR="${HAR_SYSTEM_ROOT}/database"

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-dir)
            TRAIN_DIR="$2"
            shift 2
            ;;
        --database-dir)
            DATABASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert to absolute paths
TRAIN_DIR="$(cd "$TRAIN_DIR" 2>/dev/null && pwd)" || {
    echo -e "${RED}[ERROR]${NC} Training directory not found: $TRAIN_DIR"
    exit 1
}

# Ensure DATABASE_DIR is absolute
if [[ "$DATABASE_DIR" != /* ]]; then
    DATABASE_DIR="${HAR_SYSTEM_ROOT}/${DATABASE_DIR}"
fi
DATABASE_DIR="$(cd "$(dirname "$DATABASE_DIR")" 2>/dev/null && pwd)/$(basename "$DATABASE_DIR")" || DATABASE_DIR="${HAR_SYSTEM_ROOT}/database"

echo -e "${GREEN}[CONFIG]${NC} Training Directory: $TRAIN_DIR"
echo -e "${GREEN}[CONFIG]${NC} Database Directory: $DATABASE_DIR"
echo ""

# Check for images
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A "$TRAIN_DIR")" ]; then
    echo -e "${RED}[ERROR]${NC} No training images found in $TRAIN_DIR"
    exit 1
fi

# Count persons and images
PERSON_COUNT=$(find "$TRAIN_DIR" -maxdepth 1 -type d ! -path "$TRAIN_DIR" | wc -l)
IMAGE_COUNT=$(find "$TRAIN_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)

echo -e "${GREEN}[SCAN]${NC} Found $PERSON_COUNT person(s) with $IMAGE_COUNT images"
echo ""

# Setup hailo-apps environment
HAILO_TRAIN_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/train"

echo -e "${YELLOW}[SETUP][NC} Preparing hailo-apps training environment..."

# Remove any existing train content
if [ -d "$HAILO_TRAIN_DIR" ]; then
    rm -rf "$HAILO_TRAIN_DIR"
fi

# Create fresh train directory  
mkdir -p "$HAILO_TRAIN_DIR"

# Copy training images
echo "  Copying training images..."
cp -r "$TRAIN_DIR"/* "$HAILO_TRAIN_DIR/"

# Clean filenames (remove spaces and special characters)
echo "  Cleaning filenames..."
find "$HAILO_TRAIN_DIR" -type f -name "* *" | while read file; do
    newfile=$(echo "$file" | tr ' ' '_' | sed 's/[^a-zA-Z0-9_\/.-]//g')
    mv "$file" "$newfile" 2>/dev/null || true
done

echo -e "  ${GREEN}✓${NC} Images prepared"

# List what was copied
echo "  Persons to train:"
for person_dir in "$HAILO_TRAIN_DIR"/*; do
    if [ -d "$person_dir" ]; then
        person_name=$(basename "$person_dir")
        img_count=$(find "$person_dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
        echo "    - $person_name: $img_count images"
    fi
done
echo ""

# Run hailo-apps training
echo "============================================================"
echo -e "${YELLOW}[TRAINING]${NC} Running hailo-apps face recognition training"
echo "============================================================"
echo ""
echo "This will:"
echo "  • Detect faces using SCRFD model"
echo "  • Extract embeddings using MobileFaceNet"
echo "  • Store in LanceDB database"
echo ""

cd "$HAILO_APPS_ROOT"

# Check if venv exists
if [ ! -d "venv_hailo_apps" ]; then
    echo -e "${RED}[ERROR]${NC} Virtual environment not found"
    echo "Please run: cd $HAILO_APPS_ROOT && ./install.sh"
    exit 1
fi

# Remove default training images from local_resources to prevent auto-copy
echo -e "${YELLOW}[SETUP]${NC} Preventing default samples from being copied..."
HAILO_LOCAL_FACES="${HAILO_APPS_ROOT}/local_resources/faces"
if [ -d "$HAILO_LOCAL_FACES" ]; then
    # Backup if exists
    if [ -d "${HAILO_LOCAL_FACES}_backup" ]; then
        rm -rf "${HAILO_LOCAL_FACES}_backup"
    fi
    mv "$HAILO_LOCAL_FACES" "${HAILO_LOCAL_FACES}_backup"
    echo -e "  ${GREEN}✓${NC} Default samples disabled"
fi
echo ""

# Activate venv and run training
source venv_hailo_apps/bin/activate
export PYTHONPATH="${HAILO_APPS_ROOT}:${PYTHONPATH}"

python3 -m hailo_apps.python.pipeline_apps.face_recognition.face_recognition --mode train

# Restore backup
if [ -d "${HAILO_LOCAL_FACES}_backup" ]; then
    mv "${HAILO_LOCAL_FACES}_backup" "$HAILO_LOCAL_FACES"
fi

TRAINING_STATUS=$?

if [ $TRAINING_STATUS -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} Training failed"
    exit 1
fi

echo ""
echo "============================================================"
echo -e "${GREEN}[SUCCESS]${NC} Training completed!"
echo "============================================================"
echo ""

# Copy database to HAR-System
HAILO_DB_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/database"
HAILO_SAMPLES_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/samples"

if [ -d "$HAILO_DB_DIR" ]; then
    echo -e "${YELLOW}[COPY]${NC} Copying database to HAR-System..."
    
    # Remove old database if exists
    if [ -d "$DATABASE_DIR" ]; then
        echo "  Removing old database..."
        rm -rf "$DATABASE_DIR"
    fi
    
    # Create directory structure
    mkdir -p "$DATABASE_DIR"
    
    # Copy entire database directory (persons.db is a directory in LanceDB)
    if [ -d "$HAILO_DB_DIR/persons.db" ]; then
        cp -r "$HAILO_DB_DIR/persons.db" "$DATABASE_DIR/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Database copied"
    else
        echo -e "  ${RED}✗${NC} Database not found!"
    fi
    
    # Copy samples directory
    if [ -d "$HAILO_SAMPLES_DIR" ] && [ "$(ls -A $HAILO_SAMPLES_DIR 2>/dev/null)" ]; then
        cp -r "$HAILO_SAMPLES_DIR" "$DATABASE_DIR/samples" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Samples copied"
    else
        echo -e "  ${YELLOW}⚠${NC} No samples found"
    fi
    
    echo ""
fi

# Show results
echo "============================================================"
echo -e "${GREEN}[RESULTS]${NC} Training Summary"
echo "============================================================"
echo ""
echo "  Database: $DATABASE_DIR/persons.db"
echo "  Samples: $DATABASE_DIR/samples/"
echo ""

# Try to show person count
if command -v python3 &> /dev/null; then
    python3 << EOF
import sys
sys.path.insert(0, '$HAR_SYSTEM_ROOT')
try:
    from har_system.integrations import HailoFaceRecognition
    face_recog = HailoFaceRecognition(
        database_dir='$DATABASE_DIR',
        samples_dir='$DATABASE_DIR/samples'
    )
    if face_recog.is_enabled():
        stats = face_recog.get_database_stats()
        persons = face_recog.list_known_persons()
        print(f"  Trained Persons: {stats.get('total_persons', 0)}")
        print(f"  Total Samples: {stats.get('total_samples', 0)}")
        if persons:
            print(f"  Names: {', '.join(persons)}")
except Exception as e:
    pass
EOF
fi

echo ""
echo "============================================================"
echo -e "${GREEN}[READY]${NC} Face recognition system is ready!"
echo "============================================================"
echo ""
echo "To use:"
echo "  python3 -m har_system realtime --input rpi --enable-face-recognition"
echo ""
