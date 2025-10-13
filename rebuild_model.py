#!/usr/bin/env python3
"""Quick script to rebuild the GLB from generated code."""

import sys
import os

# Add cqparts_bucket to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cqparts_bucket'))

# Import the rebuild function
from optim import _rebuild_and_save_glb, ROVER_GLB_PATH

print("=" * 80)
print("REBUILDING GLB FROM GENERATED CODE")
print("=" * 80)
print()

try:
    print("Using generated/robot_base_vlm.py (with 4 wheels per side)...")
    _rebuild_and_save_glb(use_generated=True)
    print()
    print("=" * 80)
    print(f"✅ SUCCESS! GLB rebuilt and saved to: {ROVER_GLB_PATH}")
    print("=" * 80)
    print()
    print("Refresh your browser to see the updated model with wheels!")
    print("Expected: 8 wheels (4 per side)")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

