#!/usr/bin/env python3
"""
Example script to export part labels as JSON for human labeling UI.

This script:
1. Loads a mesh
2. Runs segmentation (Hunyuan3D-Part or PointNet++)
3. Builds a PartTable
4. Exports to JSON format for labeling UI
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import trimesh

from vlm_cad.segmentation import create_segmentation_backend
from vlm_cad.parts.parts import (
    build_part_table_from_segmentation,
    part_table_to_labeling_json,
)


def main():
    # Configuration
    mesh_path = os.environ.get(
        "MESH_PATH",
        os.path.join(parent_dir, "examples", "sample_plane.obj")
    )
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
    
    print("=" * 60)
    print("Part Labels Export for Labeling UI")
    print("=" * 60)
    print(f"Mesh: {mesh_path}")
    print(f"Backend: {backend_kind}")
    print()
    
    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    print(f"✓ Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    
    # Run segmentation
    print(f"\nRunning {backend_kind} segmentation...")
    try:
        backend = create_segmentation_backend(kind=backend_kind)
        seg_result = backend.segment(mesh_path, num_points=len(vertices))
        
        print(f"✓ Segmentation complete: {seg_result.num_parts} parts")
        
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get vertex labels
    if seg_result.vertex_labels is not None:
        vertex_labels = seg_result.vertex_labels
    elif seg_result.vertices is not None:
        vertex_labels = seg_result.labels
    else:
        # Map point labels to vertices (approximate)
        print("Warning: Using point labels as vertex labels (approximate)")
        vertex_labels = seg_result.labels
        
        # Ensure length matches
        if len(vertex_labels) != len(vertices):
            if len(vertex_labels) < len(vertices):
                vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
            else:
                vertex_labels = vertex_labels[:len(vertices)]
    
    # Build PartTable
    print("\nBuilding PartTable...")
    part_table = build_part_table_from_segmentation(
        vertices=vertices,
        part_labels=vertex_labels,
        ground_plane_z=None,  # Auto-detect
    )
    print(f"✓ PartTable created with {len(part_table.parts)} parts")
    
    # Print part summary
    print("\nPart Summary:")
    for part_id, part_info in sorted(part_table.parts.items()):
        shape_hint = "long_thin" if part_info.extents[0] / part_info.extents[2] > 3 else "block_like"
        print(f"  Part {part_id}: {len(vertex_labels[vertex_labels == part_id])} vertices, "
              f"extents={part_info.extents}, shape={shape_hint}")
    
    # Convert to labeling JSON
    print("\nConverting to labeling JSON...")
    labeling_json = part_table_to_labeling_json(part_table)
    print(f"✓ JSON created with {len(labeling_json['parts'])} parts")
    
    # Save to file
    output_path = output_dir / f"{Path(mesh_path).stem}_part_labels.json"
    with open(output_path, 'w') as f:
        json.dump(labeling_json, f, indent=2)
    
    print(f"\n✓ Saved labeling JSON to: {output_path}")
    print(f"\nYou can now:")
    print(f"  1. Open {output_path} in a labeling UI")
    print(f"  2. Assign names and descriptions to each part")
    print(f"  3. Use apply_labels_from_json() to update the PartTable")
    
    # Show example of what the JSON looks like
    print(f"\nExample part entry:")
    if labeling_json['parts']:
        example = labeling_json['parts'][0]
        print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()

