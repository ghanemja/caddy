#!/usr/bin/env python3
"""
Quick test script for the segmentation backend abstraction.

Tests both PointNet and Hunyuan3D-Part backends (if available).
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import trimesh

from vlm_cad.segmentation import create_segmentation_backend, PartSegmentationResult


def create_test_mesh() -> Path:
    """Create a simple test mesh."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    temp_dir = Path("/tmp")
    mesh_path = temp_dir / "test_box.obj"
    box.export(str(mesh_path))
    return mesh_path


def test_backend(kind: str, mesh_path: Path):
    """Test a segmentation backend."""
    print(f"\n{'=' * 60}")
    print(f"Testing {kind} backend")
    print(f"{'=' * 60}\n")
    
    try:
        backend = create_segmentation_backend(kind=kind)
        print(f"✓ {kind} backend created")
        
        result = backend.segment(mesh_path, num_points=512)
        print(f"✓ Segmentation complete")
        print(f"  - Parts detected: {result.num_parts}")
        print(f"  - Points: {result.num_points}")
        print(f"  - Labels shape: {result.labels.shape}")
        if result.points is not None:
            print(f"  - Points shape: {result.points.shape}")
        
        # Verify result structure
        assert isinstance(result, PartSegmentationResult)
        assert result.labels is not None
        assert len(result.labels) > 0
        assert result.num_parts > 0
        
        print(f"✓ All checks passed for {kind}")
        return True
        
    except NotImplementedError as e:
        print(f"⚠ {kind} backend not yet implemented: {e}")
        return False
    except FileNotFoundError as e:
        print(f"⚠ {kind} model not found: {e}")
        return False
    except Exception as e:
        print(f"✗ {kind} backend failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Segmentation Backend Test")
    print("=" * 60)
    
    # Create test mesh
    print("\nCreating test mesh...")
    mesh_path = create_test_mesh()
    print(f"✓ Test mesh created: {mesh_path}")
    
    # Test PointNet backend
    pointnet_ok = test_backend("pointnet", mesh_path)
    
    # Test Hunyuan3D backend
    hunyuan_ok = test_backend("hunyuan3d", mesh_path)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print(f"PointNet++: {'✓ PASS' if pointnet_ok else '✗ FAIL/SKIP'}")
    print(f"Hunyuan3D-Part: {'✓ PASS' if hunyuan_ok else '✗ FAIL/SKIP'}")
    
    if pointnet_ok:
        print("\n✓ PointNet++ backend is working correctly!")
    else:
        print("\n⚠ PointNet++ backend needs setup (model checkpoint required)")
    
    if hunyuan_ok:
        print("✓ Hunyuan3D-Part backend is working correctly!")
    else:
        print("⚠ Hunyuan3D-Part backend needs implementation (see SEGMENTATION_BACKENDS.md)")
    
    # Cleanup
    try:
        mesh_path.unlink()
    except:
        pass


if __name__ == "__main__":
    main()

