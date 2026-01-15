#!/usr/bin/env python3
"""
HAR-System: Main Entry Point
=============================
Run HAR-System as a module: python3 -m har_system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import and run main application
from har_system.apps.realtime_pose import main

if __name__ == "__main__":
    main()
