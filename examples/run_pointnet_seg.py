"""
Example script for part segmentation using the segmentation backend abstraction.

This script demonstrates:
1. Creating a segmentation backend (PointNet++ or Hunyuan3D-Part)
2. Segmenting a mesh into parts
3. Computing geometric properties from segmented parts
"""

from pathlib import Path
import torch
import numpy as np
import os

from vlm_cad.segmentation import create_segmentation_backend
from vlm_cad.pointnet_seg.geometry import (
    compute_part_bounding_boxes,
    compute_part_statistics,
    find_largest_parts,
)


def main():
    # Configuration
    mesh_path = os.environ.get(
        "MESH_PATH",
        "examples/sample_plane.obj"  # Update this path to your mesh file
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_points = 2048
    
    # Get backend type from env var (default: pointnet)
    backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
    
    print("=" * 60)
    print("Part Segmentation Example (Using Backend Abstraction)")
    print("=" * 60)
    print(f"Mesh: {mesh_path}")
    print(f"Backend: {backend_kind}")
    print(f"Device: {device}")
    print()
    
    # Create segmentation backend
    print(f"Creating {backend_kind} segmentation backend...")
    try:
        backend = create_segmentation_backend(kind=backend_kind, device=device)
        print("✓ Backend created successfully")
    except Exception as e:
        print(f"✗ Failed to create backend: {e}")
        if backend_kind == "pointnet":
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "pointnet2", "pointnet2_part_seg_msg.pth"
            )
            print("\n" + "="*60)
            print("SETUP REQUIRED: PointNet++ model not found")
            print("="*60)
            print(f"Expected path: {checkpoint_path}")
            print("\nTo download:")
            print("1. Clone: git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git")
            print("2. Find the model in log/ directory (pointnet2_part_seg_msg.pth)")
            print(f"3. Copy it to: {checkpoint_path}")
            print("\nOr set POINTNET2_CHECKPOINT environment variable to your model path")
            print("="*60)
        print("\nCannot continue without the backend. Exiting.")
        return
    
    # Segment mesh
    print(f"\nSegmenting mesh: {mesh_path}")
    try:
        result = backend.segment(mesh_path, num_points=num_points)
        
        points = result.points
        labels = result.labels
        print(f"✓ Segmentation complete")
        print(f"  Parts detected: {result.num_parts}")
        print(f"  Points: {result.num_points}")
        print(f"  Unique labels: {len(np.unique(labels))}")
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute geometric properties
    print("\nComputing geometric properties...")
    bboxes = compute_part_bounding_boxes(points, labels)
    stats = compute_part_statistics(points, labels)
    
    print(f"\nPer-part bounding boxes:")
    for label_id, bbox in bboxes.items():
        print(f"  Part {label_id}:")
        print(f"    Center: {bbox['center']}")
        print(f"    Extent: {bbox['extent']}")
        print(f"    Points: {bbox['num_points']}")
    
    # Find largest parts
    print(f"\nLargest parts by volume:")
    largest = find_largest_parts(points, labels, top_k=3, metric="volume")
    for label_id, volume in largest:
        print(f"  Part {label_id}: volume={volume:.4f}")
    
    print("\n✓ Analysis complete")


if __name__ == "__main__":
    main()
