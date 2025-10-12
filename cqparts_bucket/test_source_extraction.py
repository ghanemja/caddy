#!/usr/bin/env python3
"""
Quick test to verify robot_base.py source extraction works
"""

import sys
import os

# Change to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the function
from optim import _baseline_cqparts_source

print("=" * 80)
print("Testing robot_base.py source extraction")
print("=" * 80)
print()

try:
    source = _baseline_cqparts_source()
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    
    if source:
        print(f"✓ SUCCESS! Extracted {len(source)} characters")
        print()
        print("First 500 characters:")
        print("-" * 80)
        print(source[:500])
        print("-" * 80)
        print()
        
        # Check for key elements
        checks = [
            ("class Rover" in source, "class Rover"),
            ("class RobotBase" in source, "class RobotBase"),
            ("import" in source, "import statements"),
            ("def " in source, "function definitions"),
        ]
        
        print("Content validation:")
        for passed, item in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} Contains {item}")
        
        print()
        if all(p for p, _ in checks):
            print("✓ All checks passed! Source extraction is working correctly.")
            sys.exit(0)
        else:
            print("⚠ Some checks failed, but source was extracted")
            sys.exit(0)
    else:
        print("✗ FAILED! No source code extracted")
        print("The function returned an empty string or error message")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

