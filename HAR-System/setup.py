"""
HAR-System Setup
================
Installation script for HAR-System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="har-system",
    version="1.0.0",
    description="Human Activity Recognition System using Hailo-8 and Raspberry Pi",
    author="HAR-System Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'har-system=har_system.apps.realtime_pose:main',
            'har-chokepoint=har_system.apps.chokepoint_analyzer:main',
        ],
    },
    include_package_data=True,
    package_data={
        'har_system': ['config/*.yaml'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
