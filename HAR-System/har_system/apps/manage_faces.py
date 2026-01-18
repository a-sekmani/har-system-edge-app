#!/usr/bin/env python3
"""
HAR-System: Face Management Application
========================================
Manage known faces in the database
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from har_system.integrations import HailoFaceRecognition
from har_system.utils.cli import add_faces_arguments


def main():
    """Main face management entry point"""
    parser = argparse.ArgumentParser(
        description="HAR-System: Manage Face Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all known persons
  python3 -m har_system faces --list
  
  # Show database statistics
  python3 -m har_system faces --stats
  
  # Remove a person
  python3 -m har_system faces --remove Ahmed
  
  # Clear entire database
  python3 -m har_system faces --clear
        """
    )

    # Use the shared argument definitions (single source of truth).
    add_faces_arguments(parser)
    
    args = parser.parse_args()
    
    # Initialize the face recognition wrapper (LanceDB via hailo-apps DatabaseHandler).
    face_recog = HailoFaceRecognition(
        database_dir=args.database_dir,
        samples_dir=f"{args.database_dir}/samples"
    )
    
    if not face_recog.is_enabled():
        # This typically means hailo-apps (or its DB deps) are not installed/available.
        print("[ERROR] Face recognition system could not be initialized")
        print()
        print("Please ensure:")
        print("  • hailo-apps is installed")
        print("  • Dependencies are installed (lancedb)")
        print()
        sys.exit(1)
    
    # Execute commands
    if args.list or args.stats:
        print("="*60)
        print("📋 Face Recognition Database")
        print("="*60)
        print()
        
        stats = face_recog.get_database_stats()
        print(f"Database Path: {stats.get('database_path', 'N/A')}")
        print(f"Total Persons: {stats.get('total_persons', 0)}")
        print(f"Total Samples: {stats.get('total_samples', 0)}")
        print(f"Confidence Threshold: {stats.get('confidence_threshold', 0.70):.2f}")
        print()
        
        if args.list:
            known_persons = face_recog.list_known_persons()
            if known_persons:
                print("Known Persons:")
                for i, name in enumerate(known_persons, 1):
                    print(f"  {i}. {name}")
            else:
                print("No known persons in database")
                print()
                print("To add persons:")
                print("  1. Organize images: train_faces/PersonName/*.jpg")
                print("  2. Run: python3 -m har_system train-faces --train-dir ./train_faces")
                print("  3. Use hailo-apps face_recognition for actual training")
        print()
    
    elif args.remove:
        print(f"[REMOVE] Removing person: {args.remove}")
        if face_recog.remove_person(args.remove):
            print(f"[SUCCESS] {args.remove} removed from database")
        else:
            print(f"[FAILED] Could not remove {args.remove}")
            print("         Person may not exist in database")
        print()
    
    elif args.clear:
        print("="*60)
        print("[WARNING] Clear Database")
        print("="*60)
        print()
        print("This will DELETE ALL persons from the database!")
        print("This action CANNOT be undone!")
        print()
        response = input("Type 'yes' to confirm: ")
        if response.lower() == 'yes':
            face_recog.clear_database()
            print()
            print("[SUCCESS] Database cleared")
        else:
            print()
            print("[CANCELLED] Database not cleared")
        print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
