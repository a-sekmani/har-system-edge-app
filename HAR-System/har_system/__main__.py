#!/usr/bin/env python3
"""
HAR-System: Main Entry Point
=============================
Run HAR-System as a module with different commands:

    python3 -m har_system realtime [options]    # Real-time pose tracking
    python3 -m har_system chokepoint [options]  # ChokePoint dataset analysis

- This file acts as a CLI dispatcher: it parses the top-level command and forwards it
  to the appropriate app inside `har_system/apps/`.
- We keep a single entry point (`python3 -m har_system ...`) while reusing each app's
  existing CLI logic (e.g., `realtime_pose.py`).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared CLI argument definitions 
from har_system.utils.cli import (
    add_realtime_arguments,
    add_train_faces_arguments,
    add_faces_arguments,
    add_chokepoint_arguments,
)


def main():
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        description="HAR-System: Human Activity Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Real-time pose command
    realtime_parser = subparsers.add_parser(
        'realtime',
        help='Real-time human pose tracking and activity recognition'
    )
    add_realtime_arguments(realtime_parser)
    
    # Face training command
    train_parser = subparsers.add_parser(
        'train-faces',
        help='Train face recognition with images'
    )
    add_train_faces_arguments(train_parser)
    
    # Face management command
    faces_parser = subparsers.add_parser(
        'faces',
        help='Manage face recognition database'
    )
    add_faces_arguments(faces_parser)
    
    # ChokePoint analysis command
    chokepoint_parser = subparsers.add_parser(
        'chokepoint',
        help='Analyze ChokePoint dataset for person tracking'
    )
    add_chokepoint_arguments(chokepoint_parser)
    
    # Parse CLI arguments.
    args = parser.parse_args()
    
    # If no sub-command was provided, show help and exit with code 1
    # (useful for scripts/automation).
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate application
    if args.command == 'train-faces':
        from har_system.apps.train_faces import main as train_main
        # Call training directly as a Python function (no need to manipulate sys.argv here).
        train_main(
            train_dir=args.train_dir,
            database_dir=args.database_dir,
            confidence_threshold=getattr(args, 'confidence_threshold', 0.70),
            max_persons=getattr(args, 'max_persons', None)
        )
    
    elif args.command == 'faces':
        from har_system.apps.manage_faces import main as faces_main
        # `manage_faces.main()` is implemented as a standalone CLI that reads sys.argv.
        # To reuse it, we populate sys.argv based on the parsed arguments above.
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['manage_faces']
            if getattr(args, 'list', False):
                sys.argv.append('--list')
            if getattr(args, 'stats', False):
                sys.argv.append('--stats')
            if getattr(args, 'remove', None):
                sys.argv.extend(['--remove', args.remove])
            if getattr(args, 'clear', False):
                sys.argv.append('--clear')
            sys.argv.extend(['--database-dir', args.database_dir])
            faces_main()
        finally:
            # Important: restore sys.argv to avoid side effects.
            sys.argv = original_argv
    
    elif args.command == 'realtime':
        from har_system.apps.realtime_pose import main as realtime_main
        # `realtime_pose.main()` is implemented as a CLI that reads sys.argv.
        # Here we translate this dispatcher's args into the CLI format that realtime_pose expects.
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['realtime_pose']
            if args.input:
                sys.argv.extend(['--input', args.input])
            if args.show_fps:
                sys.argv.append('--show-fps')
            if args.verbose:
                sys.argv.append('--verbose')
            if args.save_data:
                sys.argv.append('--save-data')
            sys.argv.extend(['--output-dir', args.output_dir])
            sys.argv.extend(['--print-interval', str(args.print_interval)])
            if args.enable_face_recognition:
                sys.argv.append('--enable-face-recognition')
            # Only pass database-dir if explicitly provided so realtime_pose can fall back to YAML.
            if args.database_dir is not None:
                sys.argv.extend(['--database-dir', args.database_dir])
            if args.no_display:
                sys.argv.append('--no-display')
            realtime_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'chokepoint':
        from har_system.apps.chokepoint_analyzer import main as chokepoint_main
        # Same pattern as realtime: reuse the CLI in `chokepoint_analyzer.py` by preparing sys.argv.
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['chokepoint_analyzer'] + [
                '--dataset-path', args.dataset_path,
                '--results-dir', args.results_dir
            ]
            if args.enable_face_recognition:
                sys.argv.append('--enable-face-recognition')
            if args.database_dir:
                sys.argv.extend(['--database-dir', args.database_dir])
            if args.no_display:
                sys.argv.append('--no-display')
            chokepoint_main()
        finally:
            sys.argv = original_argv
    
    else:
        # Defensive fallback in case args.command is unexpected.
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
