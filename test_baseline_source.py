#!/usr/bin/env python3
"""
Test script to verify baseline source extraction is working correctly.
This helps debug why the VLM might not be getting complete code.
"""

import sys
import os

# Add cqparts_bucket to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cqparts_bucket'))

# Now import the baseline source function
from optim import _baseline_cqparts_source

print("=" * 80)
print("BASELINE SOURCE EXTRACTION TEST")
print("=" * 80)
print()

# Extract baseline
baseline = _baseline_cqparts_source()

print(f"✓ Baseline extracted: {len(baseline)} characters")
print()

# Check for critical components
checks = {
    "Has imports": ("import" in baseline and "cadquery" in baseline),
    "Has Rover class": "class Rover" in baseline,
    "Has RobotBase class": "class RobotBase" in baseline,
    "Has make_components": "def make_components" in baseline,
    "Has make_constraints": "def make_constraints" in baseline,
    "Has MountedStepper": "MountedStepper" in baseline,
    "Has wheels_per_side": "wheels_per_side" in baseline,
    "Has _axle_offsets": "def _axle_offsets" in baseline,
    "Has PartRef": "PartRef" in baseline,
}

print("Content Checks:")
for check, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {check}")
print()

# Count lines
lines = baseline.split('\n')
print(f"Total lines: {len(lines)}")
print()

# Show first 30 lines
print("First 30 lines:")
print("-" * 80)
for i, line in enumerate(lines[:30], 1):
    print(f"{i:3d}| {line}")
print("-" * 80)
print()

# Show structure
print("Classes found:")
import re
class_matches = re.findall(r'^class (\w+)', baseline, re.MULTILINE)
for cls in class_matches:
    print(f"  - {cls}")
print()

# Check if it would fit in VLM context
chars_per_token = 4  # Rough estimate
tokens_needed = len(baseline) // chars_per_token
print(f"Estimated tokens needed: ~{tokens_needed}")
print(f"Will fit in 4096 token limit: {'✓' if tokens_needed < 4096 else '✗'}")
print()

# Save to file for inspection
output_file = "/tmp/baseline_source_test.py"
with open(output_file, "w") as f:
    f.write(baseline)
print(f"✓ Full baseline saved to: {output_file}")
print("  You can inspect it with: cat /tmp/baseline_source_test.py")
print()

# Final verdict
all_passed = all(checks.values())
if all_passed and len(lines) > 50:
    print("=" * 80)
    print("✅ BASELINE SOURCE IS COMPLETE AND READY")
    print("=" * 80)
    print()
    print("The VLM should receive complete code with:")
    print(f"  - {len(lines)} lines of code")
    print(f"  - All necessary imports, classes, and methods")
    print(f"  - Complete make_components() implementation")
    print()
    print("If generated code is still incomplete, the issue is with:")
    print("  1. VLM model quality (not following instructions)")
    print("  2. max_new_tokens too low (needs 4096+)")
    print("  3. Temperature too high (should be 0.01-0.05 for code)")
else:
    print("=" * 80)
    print("⚠️  BASELINE SOURCE MAY BE INCOMPLETE")
    print("=" * 80)
    print()
    print("Missing components:")
    for check, passed in checks.items():
        if not passed:
            print(f"  ✗ {check}")
    print()
    print("This will cause VLM to generate broken code!")

