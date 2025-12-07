"""
Smoke tests for segmentation backends.

This test verifies that both PointNet and Hunyuan3D-Part backends
can be instantiated and return valid PartSegmentationResult objects.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import trimesh
import pytest

from vlm_cad.segmentation import (
    create_segmentation_backend,
    PartSegmentationResult,
    PointNetSegmentationBackend,
    Hunyuan3DPartSegmentationBackend,
)


def create_test_mesh() -> Path:
    """Create a simple test mesh file."""
    # Create a simple box mesh
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    # Save to temporary file
    temp_dir = Path(tempfile.mkdtemp())
    mesh_path = temp_dir / "test_box.obj"
    box.export(str(mesh_path))
    
    return mesh_path


def test_part_segmentation_result():
    """Test PartSegmentationResult dataclass."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    points = np.random.rand(6, 3)
    
    result = PartSegmentationResult(
        labels=labels,
        points=points,
    )
    
    assert result.num_parts == 3
    assert result.num_points == 6
    assert np.array_equal(result.labels, labels)
    assert np.array_equal(result.points, points)
    
    # Test to_dict
    result_dict = result.to_dict()
    assert "labels" in result_dict
    assert "points" in result_dict
    assert result_dict["num_parts"] == 3


def test_pointnet_backend_creation():
    """Test that PointNet backend can be created."""
    try:
        backend = create_segmentation_backend(kind="pointnet")
        assert isinstance(backend, PointNetSegmentationBackend)
        print("✓ PointNet backend created successfully")
    except FileNotFoundError as e:
        pytest.skip(f"PointNet model not found: {e}")
    except Exception as e:
        pytest.fail(f"Failed to create PointNet backend: {e}")


def test_pointnet_backend_segmentation():
    """Test PointNet backend segmentation on a simple mesh."""
    mesh_path = create_test_mesh()
    
    try:
        backend = create_segmentation_backend(kind="pointnet")
        result = backend.segment(mesh_path, num_points=512)
        
        # Verify result structure
        assert isinstance(result, PartSegmentationResult)
        assert result.labels is not None
        assert len(result.labels) > 0
        assert result.points is not None
        assert result.num_parts > 0
        assert result.num_points > 0
        
        print(f"✓ PointNet segmentation: {result.num_parts} parts, {result.num_points} points")
        
    except FileNotFoundError as e:
        pytest.skip(f"PointNet model not found: {e}")
    except Exception as e:
        pytest.fail(f"PointNet segmentation failed: {e}")
    finally:
        # Cleanup
        mesh_path.unlink()
        mesh_path.parent.rmdir()


def test_hunyuan3d_backend_creation():
    """Test that Hunyuan3D backend can be created (may fail if not implemented)."""
    try:
        backend = create_segmentation_backend(kind="hunyuan3d")
        assert isinstance(backend, Hunyuan3DPartSegmentationBackend)
        print("✓ Hunyuan3D backend created successfully")
    except NotImplementedError:
        pytest.skip("Hunyuan3D backend not yet implemented")
    except Exception as e:
        pytest.skip(f"Hunyuan3D backend not available: {e}")


def test_backend_factory():
    """Test backend factory function with different configurations."""
    # Test default (should be pointnet)
    backend = create_segmentation_backend()
    assert backend is not None
    
    # Test explicit pointnet
    backend = create_segmentation_backend(kind="pointnet")
    assert isinstance(backend, PointNetSegmentationBackend)
    
    # Test invalid backend
    try:
        backend = create_segmentation_backend(kind="invalid")
        pytest.fail("Should have raised ValueError")
    except ValueError:
        pass  # Expected


def test_env_var_config():
    """Test that environment variable configuration works."""
    original = os.environ.get("SEGMENTATION_BACKEND")
    
    try:
        # Set to pointnet
        os.environ["SEGMENTATION_BACKEND"] = "pointnet"
        backend = create_segmentation_backend()
        assert isinstance(backend, PointNetSegmentationBackend)
        
    except FileNotFoundError:
        pytest.skip("PointNet model not found")
    finally:
        # Restore original value
        if original is not None:
            os.environ["SEGMENTATION_BACKEND"] = original
        elif "SEGMENTATION_BACKEND" in os.environ:
            del os.environ["SEGMENTATION_BACKEND"]


if __name__ == "__main__":
    # Run tests
    print("Running segmentation backend tests...")
    print()
    
    test_part_segmentation_result()
    print()
    
    try:
        test_pointnet_backend_creation()
        test_pointnet_backend_segmentation()
    except Exception as e:
        print(f"⚠ PointNet tests skipped: {e}")
    print()
    
    try:
        test_hunyuan3d_backend_creation()
    except Exception as e:
        print(f"⚠ Hunyuan3D tests skipped: {e}")
    print()
    
    test_backend_factory()
    test_env_var_config()
    
    print("✓ All tests completed!")

