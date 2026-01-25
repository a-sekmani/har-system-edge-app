#!/usr/bin/env python3
"""
HAR-System: Face Training Application
======================================
Train the face recognition system with new persons using hailo-apps pipeline
"""

import sys
import os
import shutil
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from har_system.integrations import HailoFaceRecognition

def main(train_dir='./train_faces', database_dir='./database', confidence_threshold=0.70, max_persons=None):
    """
    Main training entry point - uses hailo-apps face_recognition for actual training
    
    Args:
        train_dir: Directory containing training images
        database_dir: Database directory
        confidence_threshold: Recognition confidence threshold
        max_persons: Maximum number of persons to train (None = all)
    """
    print("="*60)
    print("HAR-System: Face Recognition Training")
    print("="*60)
    print()
    
    # Validate input directories and basic structure (one folder per person).
    train_dir = Path(train_dir).resolve()
    database_dir = Path(database_dir).resolve()
    
    if not train_dir.exists():
        print(f"[ERROR] Training directory not found: {train_dir}")
        print()
        print("Please create the directory and add training images:")
        print(f"  mkdir -p {train_dir}/PersonName")
        print(f"  # Add 3-5 images to {train_dir}/PersonName/")
        sys.exit(1)
    
    # Check if directory is empty
    subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[ERROR] No person folders found in: {train_dir}")
        print()
        print("Please organize training images as:")
        print(f"  {train_dir}/PersonName/photo1.jpg")
        sys.exit(1)
    
    # Apply max_persons limit if specified
    if max_persons is not None and max_persons > 0:
        if len(subdirs) > max_persons:
            print(f"[LIMIT] Found {len(subdirs)} persons, limiting to first {max_persons} for testing")
            subdirs = sorted(subdirs)[:max_persons]
        else:
            print(f"[INFO] Found {len(subdirs)} persons (max_persons={max_persons} not reached)")
    
    print(f"[CONFIG] Training Directory: {train_dir}")
    print(f"[CONFIG] Database Directory: {database_dir}")
    print(f"[CONFIG] Confidence Threshold: {confidence_threshold}")
    if max_persons is not None:
        print(f"[CONFIG] Max Persons: {max_persons}")
    print()
    
    # Check existing database and show what will be updated
    print("[DATABASE] Checking existing database...")
    face_recog_temp = HailoFaceRecognition(
        database_dir=str(database_dir),
        samples_dir=str(database_dir / "samples")
    )
    if face_recog_temp.is_enabled():
        existing_persons = face_recog_temp.list_known_persons()
        if existing_persons:
            print(f"[DATABASE] Found {len(existing_persons)} existing person(s) in database:")
            for person in existing_persons:
                print(f"  - {person}")
            print("[DATABASE] Will add new images for existing persons and create new persons as needed")
        else:
            print("[DATABASE] Database is empty - will create fresh database")
    else:
        print("[DATABASE] No database found - will create fresh database")
    print()
    
    # Scan and display found persons (helps confirm the folder layout before training).
    print(f"[SCAN] Found {len(subdirs)} person(s) to train:")
    total_images = 0
    for person_dir in subdirs:
        image_files = list(person_dir.glob('*.jpg')) + \
                     list(person_dir.glob('*.jpeg')) + \
                     list(person_dir.glob('*.png'))
        total_images += len(image_files)
        print(f"  - {person_dir.name}: {len(image_files)} images")
    print()
    
    if total_images == 0:
        print("[ERROR] No images found!")
        sys.exit(1)
    
    # Training is delegated to hailo-apps' face_recognition pipeline.
    print("[TRAINING] Starting face recognition training...")
    print()
    
    # Prefer the provided shell script (it prepares hailo-apps directories and runs training).
    # This keeps Python-side glue minimal and matches the expected hailo-apps workflow.
    script_path = project_root / "scripts" / "train_faces_auto.sh"
    
    if script_path.exists():
        print("✓ Using automatic training script")
        print()
        
        import subprocess
        try:
            cmd = [str(script_path), "--train-dir", str(train_dir), "--database-dir", str(database_dir)]
            if max_persons is not None:
                cmd.extend(["--max-persons", str(max_persons)])
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            return  # Success!
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Automatic training failed")
            print(f"        Exit code: {e.returncode}")
            print()
            print("Falling back to manual instructions...")
            print()
    
    try:
        # Import hailo-apps face recognition
        from hailo_apps.python.pipeline_apps.face_recognition.face_recognition_pipeline import GStreamerFaceRecognitionApp
        from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
        
        print("✓ hailo-apps face_recognition found")
        print()
        
        # Setup temporary training directory in hailo-apps structure
        hailo_apps_root = Path(__file__).resolve().parents[4]  # Go up to hailo-apps root
        face_recog_dir = hailo_apps_root / "hailo_apps" / "python" / "pipeline_apps" / "face_recognition"
        
        # Define all possible training directories that might contain old images
        temp_train_dir = face_recog_dir / "train_images_temp"
        train_dir_default = face_recog_dir / "train"
        train_images_dir = face_recog_dir / "train_images"
        
        # Create temp directory and copy images
        print("[SETUP] Preparing training environment...")
        print("  Cleaning ALL old training directories to ensure only train_faces is used...")
        
        # Remove ALL old training directories (comprehensive cleanup)
        old_dirs = [temp_train_dir, train_dir_default, train_images_dir]
        for old_dir in old_dirs:
            if old_dir.exists():
                try:
                    shutil.rmtree(old_dir)
                    print(f"    ✓ Removed: {old_dir.name}")
                except Exception as e:
                    print(f"    ⚠ Warning: Could not remove {old_dir.name}: {e}")
        
        # Create fresh temp directory (this will be the ONLY source of training images)
        temp_train_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created fresh training directory: {temp_train_dir.name}")
        
        # Copy all person folders from train_faces ONLY (no other sources)
        print(f"  Copying images from train_faces directory: {train_dir}")
        for person_dir in subdirs:
            dest_person_dir = temp_train_dir / person_dir.name
            shutil.copytree(person_dir, dest_person_dir)
            
            # Clean filenames: remove hidden files and sanitize names
            for img_file in dest_person_dir.rglob("*"):
                if img_file.is_file():
                    # Remove macOS hidden files (._*)
                    if img_file.name.startswith("._"):
                        img_file.unlink()
                        continue
                    
                    # Remove system files
                    if img_file.name in [".DS_Store", "Thumbs.db"]:
                        img_file.unlink()
                        continue
                    
                    # Clean filename if it contains spaces or special characters
                    if " " in img_file.name or re.search(r'[^a-zA-Z0-9_.-]', img_file.name):
                        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', img_file.name)
                        clean_name = re.sub(r'_+', '_', clean_name)  # Replace multiple _ with single _
                        if clean_name != img_file.name:
                            new_path = img_file.parent / clean_name
                            img_file.rename(new_path)
            
            print(f"  ✓ Copied {person_dir.name}")
        print()
        
        # Reuse hailo-apps CLI-style entry point by temporarily populating sys.argv.
        original_argv = sys.argv.copy()
        try:
            # Set mode to train
            sys.argv = ['train_faces', '--mode', 'train']
            
            # Create a simple callback class
            class TrainingUserData(app_callback_class):
                def __init__(self):
                    super().__init__()
            
            user_data = TrainingUserData()
            
            # Create pipeline app
            print("[PIPELINE] Initializing GStreamer pipeline...")
            print("           This will process each image through:")
            print("           • SCRFD face detection")
            print("           • Face alignment")
            print("           • MobileFaceNet embedding extraction")
            print()
            
            # Ensure samples directory exists
            samples_dir_path = database_dir / "samples"
            if not samples_dir_path.exists():
                samples_dir_path.mkdir(parents=True, exist_ok=True)
                print(f"[SETUP] Created samples directory")
            print()
            
            # Create app but override train directory to use ONLY our temp_train_dir
            # This ensures no old images from other directories are used
            app = GStreamerFaceRecognitionApp(lambda e, b, u: None, user_data)
            app.train_images_dir = str(temp_train_dir)  # Use ONLY train_faces (copied to temp)
            app.database_dir = database_dir
            app.samples_dir = database_dir / "samples"
            app.db_handler.samples_dir = str(app.samples_dir)
            
            # Verify that train_images_dir contains ONLY our images from train_faces
            train_images_path = Path(app.train_images_dir)
            if not train_images_path.exists() or not any(train_images_path.iterdir()):
                print(f"[ERROR] Training directory is empty: {app.train_images_dir}")
                print("[ERROR] This should not happen - images should be copied from train_faces")
                raise RuntimeError("Training directory is empty - no images from train_faces")
            
            # Count images to verify
            person_dirs = [d for d in train_images_path.iterdir() if d.is_dir()]
            total_images = sum(len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.jpeg"))) + len(list(d.rglob("*.png"))) for d in person_dirs)
            
            print(f"[VERIFY] Training will use ONLY images from train_faces:")
            print(f"         Source: {train_dir}")
            print(f"         Destination: {app.train_images_dir}")
            print(f"         Persons: {len(person_dirs)}, Total images: {total_images}")
            
            # Ensure directories exist
            database_dir.mkdir(parents=True, exist_ok=True)
            app.samples_dir.mkdir(parents=True, exist_ok=True)
            
            print()
            print("[TRAINING] Processing images from train_faces ONLY...")
            print("="*60)
            
            # Run training (this will use ONLY the images we copied from train_faces)
            app.run_training()
            
            print()
            print("="*60)
            print("[SUCCESS] Training completed!")
            print()
            
        finally:
            sys.argv = original_argv
            # Cleanup temp directory
            if temp_train_dir.exists():
                shutil.rmtree(temp_train_dir)
                print("[CLEANUP] Temporary files removed")
        
        # Show results
        face_recog = HailoFaceRecognition(
            database_dir=str(database_dir),
            samples_dir=str(database_dir / "samples"),
            confidence_threshold=confidence_threshold
        )
        
        if face_recog.is_enabled():
            stats = face_recog.get_database_stats()
            print()
            print("="*60)
            print("[DATABASE] Training Results")
            print("="*60)
            print(f"  Database Path: {stats.get('database_path', 'N/A')}")
            print(f"  Known Persons: {stats.get('total_persons', 0)}")
            print(f"  Total Samples: {stats.get('total_samples', 0)}")
            print()
            
            if stats.get('total_persons', 0) > 0:
                known_persons = face_recog.list_known_persons()
                print(f"  Trained persons: {', '.join(known_persons)}")
                print()
        
        print("="*60)
        print("[READY] Face recognition system is ready!")
        print()
        print("To use face recognition:")
        print(f"  python3 -m har_system realtime --input rpi --enable-face-recognition")
        print()
        
    except ImportError as e:
        print(f"[ERROR] Could not import hailo-apps face_recognition")
        print(f"        {e}")
        print()
        print("This feature requires hailo-apps to be installed.")
        print()
        print("Alternative: Manual training")
        print("="*60)
        print()
        print("1. Copy training images to hailo-apps:")
        print(f"   cp -r {train_dir}/* \\")
        print("     /home/admin/hailo-apps/hailo_apps/python/pipeline_apps/face_recognition/train_images/")
        print()
        print("2. Run hailo-apps training:")
        print("   cd /home/admin/hailo-apps")
        print("   source setup_env.sh")
        print("   python3 -m hailo_apps.python.pipeline_apps.face_recognition.face_recognition --mode train")
        print()
        print("3. Copy database back:")
        print(f"   cp -r hailo_apps/python/pipeline_apps/face_recognition/database/* {database_dir}/")
        print()
        sys.exit(1)
    
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # For standalone execution
    import argparse
    parser = argparse.ArgumentParser(description="HAR-System: Train Face Recognition")
    parser.add_argument('--train-dir', type=str, default='./train_faces')
    parser.add_argument('--database-dir', type=str, default='./database')
    parser.add_argument('--confidence-threshold', type=float, default=0.70)
    parser.add_argument('--max-persons', type=int, default=None, 
                       help='Maximum number of persons to train (useful for testing)')
    args = parser.parse_args()
    
    main(
        train_dir=args.train_dir,
        database_dir=args.database_dir,
        confidence_threshold=args.confidence_threshold,
        max_persons=args.max_persons
    )
