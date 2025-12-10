"""
CAD Component Models

This package contains CAD component definitions.
Components can be organized by category as the codebase grows.
"""

# Import CAD components from this directory
# All CAD component files are now in this directory
import sys
import os
from pathlib import Path

# Add this directory to path for imports
CAD_DIR = Path(__file__).parent
if str(CAD_DIR) not in sys.path:
    sys.path.insert(0, str(CAD_DIR))

# Import components from this directory
from robot_base import Rover
from electronics import type1 as Electronics
from pan_tilt import PanTilt
from wheel import BuiltWheel
from cqparts_motors.stepper import Stepper
from sensor_fork import SensorFork

__all__ = [
    "Rover",
    "Electronics", 
    "PanTilt",
    "BuiltWheel",
    "Stepper",
    "SensorFork",
]

