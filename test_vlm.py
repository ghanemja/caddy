#!/usr/bin/env python3
"""
Standalone test script for VLM model loading and inference.
Run this to test the model without starting the full UI server.

Usage:
    # Activate conda environment first (if using conda):
    conda activate vlm_optimizer
    
    # Then run:
    python test_vlm.py              # Test VLM + mesh analysis
    python test_vlm.py --no-mesh    # Test only VLM (skip mesh analysis)
    
    # Or use the standalone version (doesn't require CAD dependencies):
    python test_vlm_standalone.py
    
    # Or run directly from cqparts_bucket:
    cd cqparts_bucket && python optim.py --test-vlm
"""

import os
import sys

# Determine the correct path to cqparts_bucket
script_dir = os.path.dirname(os.path.abspath(__file__))
cqparts_bucket_path = os.path.join(script_dir, "cqparts_bucket")

# Add cqparts_bucket to path
if os.path.exists(cqparts_bucket_path):
    if cqparts_bucket_path not in sys.path:
        sys.path.insert(0, cqparts_bucket_path)
    # Change to cqparts_bucket directory so relative imports work
    os.chdir(cqparts_bucket_path)
else:
    # Maybe we're already in cqparts_bucket or it's in a different location
    if os.path.basename(os.getcwd()) == "cqparts_bucket":
        # Already in cqparts_bucket
        pass
    else:
        print(f"ERROR: Could not find cqparts_bucket directory")
        print(f"Expected: {cqparts_bucket_path}")
        print(f"Current directory: {os.getcwd()}")
        print("\nTip: Make sure you're running from the project root directory")
        sys.exit(1)

# Import the test function
try:
    from optim import test_vlm_model
except ImportError as e:
    print(f"ERROR: Could not import optim module: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    print("\nTip: Make sure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("\nOr if using conda:")
    print("  conda activate vlm_optimizer")
    sys.exit(1)

if __name__ == "__main__":
    include_mesh = "--no-mesh" not in sys.argv
    try:
        success = test_vlm_model(include_mesh_analysis=include_mesh)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[test] Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[test] ✗ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

