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
MAX_PERSONS=""

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
        --max-persons)
            MAX_PERSONS="$2"
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
if [ -n "$MAX_PERSONS" ]; then
    echo -e "${GREEN}[CONFIG]${NC} Max Persons: $MAX_PERSONS (for testing)"
fi
echo ""

# Check existing database (do NOT clear it - we'll update it intelligently)
echo -e "${YELLOW}[DATABASE]${NC} Checking existing database..."
if [ -d "$DATABASE_DIR" ] && [ -d "$DATABASE_DIR/persons.db" ]; then
    echo -e "  ${GREEN}✓${NC} Existing database found - will update with new images"
    echo -e "  ${GREEN}✓${NC} Old data will be preserved, new images will be added"
else
    mkdir -p "$DATABASE_DIR/samples"
    echo -e "  ${GREEN}✓${NC} No existing database - will create fresh database"
fi
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
FACE_RECOG_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition"
HAILO_TRAIN_DIR="${FACE_RECOG_DIR}/train"
HAILO_TRAIN_IMAGES_DIR="${FACE_RECOG_DIR}/train_images"
HAILO_TRAIN_TEMP_DIR="${FACE_RECOG_DIR}/train_images_temp"

echo -e "${YELLOW}[SETUP][NC} Preparing hailo-apps training environment..."
echo "  Ensuring ONLY train_faces directory is used as source..."

# Remove ALL existing train directories to prevent old images from being used
echo "  Cleaning ALL old training directories..."
if [ -d "$HAILO_TRAIN_DIR" ]; then
    rm -rf "$HAILO_TRAIN_DIR"
    echo "    ✓ Removed: train"
fi

if [ -d "$HAILO_TRAIN_IMAGES_DIR" ]; then
    rm -rf "$HAILO_TRAIN_IMAGES_DIR"
    echo "    ✓ Removed: train_images"
fi

if [ -d "$HAILO_TRAIN_TEMP_DIR" ]; then
    rm -rf "$HAILO_TRAIN_TEMP_DIR"
    echo "    ✓ Removed: train_images_temp"
fi

# Create fresh train directory (this will be the ONLY source)
mkdir -p "$HAILO_TRAIN_DIR"

# Copy training images from train_faces ONLY
echo "  Copying images from train_faces directory: $TRAIN_DIR"
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A "$TRAIN_DIR")" ]; then
    echo -e "${RED}[ERROR]${NC} train_faces directory is empty or does not exist: $TRAIN_DIR"
    exit 1
fi

# Apply max_persons limit if specified
if [ -n "$MAX_PERSONS" ] && [ "$MAX_PERSONS" -gt 0 ]; then
    echo "  Limiting to first $MAX_PERSONS persons for testing..."
    person_count=0
    for person_dir in "$TRAIN_DIR"/*; do
        if [ -d "$person_dir" ] && [ "$person_count" -lt "$MAX_PERSONS" ]; then
            cp -r "$person_dir" "$HAILO_TRAIN_DIR/"
            person_count=$((person_count + 1))
        fi
    done
    echo "    ✓ Copied $person_count persons (limited by --max-persons)"
else
    cp -r "$TRAIN_DIR"/* "$HAILO_TRAIN_DIR/"
    echo "    ✓ Copied all persons from train_faces to training directory"
fi
echo "    ✓ Training will use ONLY these images (no old images will be used)"

# Clean filenames (remove spaces and special characters, filter hidden files)
echo "  Cleaning filenames..."
# First, remove macOS hidden files (._*)
find "$HAILO_TRAIN_DIR" -type f -name "._*" -delete
# Remove other hidden/system files
find "$HAILO_TRAIN_DIR" -type f \( -name ".DS_Store" -o -name "Thumbs.db" \) -delete

# Clean all filenames: replace spaces and special characters
find "$HAILO_TRAIN_DIR" -type f | while read file; do
    dir=$(dirname "$file")
    filename=$(basename "$file")
    
    # Skip if already clean and no spaces/special chars
    if [[ "$filename" =~ ^[a-zA-Z0-9_./-]+$ ]] && [[ ! "$filename" =~ [[:space:]] ]]; then
        continue
    fi
    
    # Clean filename: replace spaces with _, remove special chars except . - _
    newfilename=$(echo "$filename" | tr ' ' '_' | sed 's/[^a-zA-Z0-9_.-]//g' | sed 's/__*/_/g')
    
    # Skip if empty or same
    if [ -z "$newfilename" ] || [ "$filename" = "$newfilename" ]; then
        continue
    fi
    
    newfile="$dir/$newfilename"
    if [ "$file" != "$newfile" ]; then
        mv "$file" "$newfile" 2>/dev/null || true
    fi
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
if [ ! -d "venv" ]; then
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

# Check hailo-apps database (preserve if exists)
echo -e "${YELLOW}[SETUP]${NC} Checking hailo-apps database..."
HAILO_DB_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/database"
HAILO_SAMPLES_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/samples"

# Copy existing database from HAR-System to hailo-apps if it exists
if [ -d "$DATABASE_DIR/persons.db" ]; then
    echo -e "  ${YELLOW}⚙${NC} Existing database found in HAR-System"
    echo -e "  ${YELLOW}⚙${NC} Copying to hailo-apps for training..."
    
    # Ensure directories exist
    mkdir -p "$HAILO_DB_DIR"
    mkdir -p "$HAILO_SAMPLES_DIR"
    
    # Copy database
    cp -r "$DATABASE_DIR/persons.db" "$HAILO_DB_DIR/" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Database copied"
    
    # Copy samples
    if [ -d "$DATABASE_DIR/samples" ] && [ "$(ls -A $DATABASE_DIR/samples 2>/dev/null)" ]; then
        cp -r "$DATABASE_DIR/samples"/* "$HAILO_SAMPLES_DIR/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Existing samples copied"
    fi
    
    echo -e "  ${GREEN}✓${NC} Will update existing database with new images"
else
    # Create fresh directories
    mkdir -p "$HAILO_DB_DIR"
    mkdir -p "$HAILO_SAMPLES_DIR"
    echo -e "  ${GREEN}✓${NC} Created fresh database and samples directories"
fi
echo ""

# Activate venv and run training
source venv/bin/activate
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

# Copy updated database back to HAR-System
HAILO_DB_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/database"
HAILO_SAMPLES_DIR="${HAILO_APPS_ROOT}/hailo_apps/python/pipeline_apps/face_recognition/samples"

if [ -d "$HAILO_DB_DIR" ]; then
    echo -e "${YELLOW}[COPY]${NC} Copying updated database back to HAR-System..."
    
    # Create directory structure if doesn't exist
    mkdir -p "$DATABASE_DIR"
    
    # Copy entire database directory (persons.db is a directory in LanceDB)
    if [ -d "$HAILO_DB_DIR/persons.db" ]; then
        # Remove old database first to ensure clean copy
        rm -rf "$DATABASE_DIR/persons.db" 2>/dev/null || true
        cp -r "$HAILO_DB_DIR/persons.db" "$DATABASE_DIR/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Updated database copied"
    else
        echo -e "  ${RED}✗${NC} Database not found!"
    fi
    
    # Copy samples directory (merge with existing)
    if [ -d "$HAILO_SAMPLES_DIR" ] && [ "$(ls -A $HAILO_SAMPLES_DIR 2>/dev/null)" ]; then
        mkdir -p "$DATABASE_DIR/samples"
        cp -r "$HAILO_SAMPLES_DIR"/* "$DATABASE_DIR/samples/" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Updated samples copied (merged with existing)"
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
