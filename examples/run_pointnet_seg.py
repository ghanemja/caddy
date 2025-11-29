"""
Example script for PointNet++ part segmentation.

This script demonstrates:
1. Loading a pretrained PointNet++ model
2. Segmenting a mesh into parts
3. Computing geometric properties from segmented parts
"""

from pathlib import Path
import torch
import numpy as np

from vlm_cad.pointnet_seg.model import load_pretrained_model
from vlm_cad.pointnet_seg.inference import segment_mesh
from vlm_cad.pointnet_seg.geometry import (
    compute_part_bounding_boxes,
    compute_part_statistics,
    find_largest_parts,
)


def main():
    # Configuration
    # Default path: models/pointnet2/pointnet2_part_seg_msg.pth
    # You can override this by setting the POINTNET2_CHECKPOINT environment variable
    import os
    checkpoint_path = os.environ.get(
        "POINTNET2_CHECKPOINT",
        os.path.join(os.path.dirname(__file__), "..", "models", "pointnet2", "pointnet2_part_seg_msg.pth")
    )
    mesh_path = os.environ.get(
        "MESH_PATH",
        "examples/sample_plane.obj"  # Update this path to your mesh file
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_points = 2048
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading PointNet++ model from {checkpoint_path}...")
    try:
        model = load_pretrained_model(
            checkpoint_path=checkpoint_path,
            num_classes=50,
            use_normals=True,
            device=device,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\n" + "="*60)
        print("SETUP REQUIRED: Download the pretrained model")
        print("="*60)
        print(f"Expected path: {checkpoint_path}")
        print("\nTo download:")
        print("1. Clone: git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git")
        print("2. Find the model in log/ directory (pointnet2_part_seg_msg.pth)")
        print(f"3. Copy it to: {checkpoint_path}")
        print("\nOr set POINTNET2_CHECKPOINT environment variable to your model path")
        print("="*60)
        return
    
    # Segment mesh
    print(f"\nSegmenting mesh: {mesh_path}")
    try:
        result = segment_mesh(
            mesh_path=mesh_path,
            model=model,
            device=device,
            num_points=num_points,
            return_logits=False,
        )
        points = result["points"]
        labels = result["labels"]
        print(f"✓ Segmentation complete")
        print(f"  Points: {len(points)}")
        print(f"  Unique labels: {len(np.unique(labels))}")
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
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

