#!/usr/bin/env python3
"""
HAR-System: Main Entry Point
=============================
Run HAR-System as a module with different commands:

    python3 -m har_system realtime [options]    # Real-time pose tracking
    python3 -m har_system chokepoint [options]  # ChokePoint dataset analysis
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Main CLI dispatcher"""
    parser = argparse.ArgumentParser(
        description="HAR-System: Human Activity Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time pose tracking
  python3 -m har_system realtime --input rpi --show-fps
  
  # ChokePoint dataset analysis
  python3 -m har_system chokepoint --dataset-path ./test_dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Real-time pose command
    realtime_parser = subparsers.add_parser(
        'realtime',
        help='Real-time human pose tracking and activity recognition'
    )
    realtime_parser.add_argument(
        '--input', '-i',
        type=str,
        default='rpi',
        help='Video source: rpi, usb, /dev/videoX, or video file'
    )
    realtime_parser.add_argument(
        '--show-fps', '-f',
        action='store_true',
        help='Show FPS counter'
    )
    realtime_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    realtime_parser.add_argument(
        '--save-data',
        action='store_true',
        help='Save data to JSON files'
    )
    realtime_parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/camera',
        help='Data save directory (default: ./results/camera)'
    )
    realtime_parser.add_argument(
        '--print-interval',
        type=int,
        default=30,
        help='Print summary every N frames (default: 30)'
    )
    
    # ChokePoint analysis command
    chokepoint_parser = subparsers.add_parser(
        'chokepoint',
        help='Analyze ChokePoint dataset for person tracking'
    )
    chokepoint_parser.add_argument(
        '--dataset-path',
        type=str,
        default='./test_dataset',
        help='Path to test_dataset folder (default: ./test_dataset)'
    )
    chokepoint_parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Results output directory (default: ./results)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate application
    if args.command == 'realtime':
        from har_system.apps.realtime_pose import main as realtime_main
        # Convert args to format expected by realtime_pose
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
            realtime_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'chokepoint':
        from har_system.apps.chokepoint_analyzer import main as chokepoint_main
        # Convert args to format expected by chokepoint_analyzer
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['chokepoint_analyzer'] + [
                '--dataset-path', args.dataset_path,
                '--results-dir', args.results_dir
            ]
            chokepoint_main()
        finally:
            sys.argv = original_argv
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
