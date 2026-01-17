#!/usr/bin/env python3
"""
HAR-System: Face Training Application
======================================
Train the face recognition system with new persons using hailo-apps pipeline
"""

import sys
import os
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from har_system.integrations import HailoFaceRecognition


def main(train_dir='./train_faces', database_dir='./database', confidence_threshold=0.70):
    """
    Main training entry point - uses hailo-apps face_recognition for actual training
    
    Args:
        train_dir: Directory containing training images
        database_dir: Database directory
        confidence_threshold: Recognition confidence threshold
    """
    print("="*60)
    print("HAR-System: Face Recognition Training")
    print("="*60)
    print()
    
    # Check if train directory exists
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
    
    print(f"[CONFIG] Training Directory: {train_dir}")
    print(f"[CONFIG] Database Directory: {database_dir}")
    print(f"[CONFIG] Confidence Threshold: {confidence_threshold}")
    print()
    
    # Scan and display found persons
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
    
    # Try to import hailo-apps face_recognition
    print("[TRAINING] Starting face recognition training...")
    print()
    
    # Try using the automatic training script first
    script_path = project_root / "scripts" / "train_faces_auto.sh"
    
    if script_path.exists():
        print("✓ Using automatic training script")
        print()
        
        import subprocess
        try:
            result = subprocess.run(
                [str(script_path), "--train-dir", str(train_dir), "--database-dir", str(database_dir)],
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
        temp_train_dir = hailo_apps_root / "hailo_apps" / "python" / "pipeline_apps" / "face_recognition" / "train_images_temp"
        
        # Create temp directory and copy images
        print("[SETUP] Preparing training environment...")
        if temp_train_dir.exists():
            shutil.rmtree(temp_train_dir)
        temp_train_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all person folders
        for person_dir in subdirs:
            dest_person_dir = temp_train_dir / person_dir.name
            shutil.copytree(person_dir, dest_person_dir)
            print(f"  ✓ Copied {person_dir.name}")
        print()
        
        # Prepare arguments for face_recognition
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
            
            # Create app but override train directory
            app = GStreamerFaceRecognitionApp(lambda e, b, u: None, user_data)
            app.train_images_dir = temp_train_dir
            app.database_dir = database_dir
            app.samples_dir = database_dir / "samples"
            app.db_handler.samples_dir = str(app.samples_dir)
            
            # Ensure directories exist
            database_dir.mkdir(parents=True, exist_ok=True)
            app.samples_dir.mkdir(parents=True, exist_ok=True)
            
            print("[TRAINING] Processing images...")
            print("="*60)
            
            # Run training
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
    args = parser.parse_args()
    
    main(
        train_dir=args.train_dir,
        database_dir=args.database_dir,
        confidence_threshold=args.confidence_threshold
    )
