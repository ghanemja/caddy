"""
Example script for parametric mesh deformation.

This script demonstrates:
1. Loading a mesh and running the ingestion pipeline
2. Building a ParametricMeshDeformer
3. Applying parameter-based deformations
4. Saving deformed meshes
"""

from pathlib import Path
from typing import Tuple, Dict
import os
import numpy as np
import torch
import trimesh

from vlm_cad.pointnet_seg.model import load_pretrained_model
from vlm_cad.pointnet_seg.inference import segment_mesh
from vlm_cad.pointnet_seg.labels import get_category_from_flat_label
from vlm_cad.semantics.vlm_client import DummyVLMClient
from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient
from vlm_cad.semantics.ingest_mesh import (
    ingest_mesh_to_semantic_params,
    build_deformer_from_ingest_result,
)
from vlm_cad.mesh_deform import MeshData, ParametricMeshDeformer


def load_full_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh file and return vertices and faces.
    
    Args:
        mesh_path: path to mesh file
        
    Returns:
        Tuple of (vertices [N, 3], faces [M, 3])
    """
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    return vertices, faces


def save_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: str):
    """
    Save a mesh to a file.
    
    Args:
        vertices: mesh vertices [N, 3]
        faces: mesh faces [M, 3]
        output_path: path to save mesh
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)


def build_part_label_names(
    labels: np.ndarray,
    category: str,
) -> dict[int, str]:
    """
    Build a mapping from part ID to part name.
    
    Args:
        labels: per-vertex part labels [N]
        category: object category
        
    Returns:
        Dictionary mapping part_id -> part_name
    """
    unique_labels = np.unique(labels)
    part_label_names = {}
    
    for label_id in unique_labels:
        label_id_int = int(label_id)
        # Try to get category and part name from flat label
        result = get_category_from_flat_label(label_id_int)
        if result:
            cat, part_name = result
            # Use part name if category matches, otherwise use generic name
            if cat.lower() == category.lower():
                part_label_names[label_id_int] = part_name
            else:
                part_label_names[label_id_int] = f"part_{label_id_int}"
        else:
            part_label_names[label_id_int] = f"part_{label_id_int}"
    
    return part_label_names


def main():
    # Configuration
    import os
    checkpoint_path = os.environ.get(
        "POINTNET2_CHECKPOINT",
        os.path.join(os.path.dirname(__file__), "..", "models", "pointnet2", "pointnet2_part_seg_msg.pth")
    )
    mesh_path = os.environ.get(
        "MESH_PATH",
        "examples/sample_plane.obj"  # Update this path to your mesh file
    )
    render_dir = os.path.join(os.path.dirname(__file__), "renders")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Parametric Mesh Deformation Example")
    print("=" * 60)
    print(f"Mesh: {mesh_path}")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if mesh exists
    if not os.path.exists(mesh_path):
        print(f"✗ Mesh file not found: {mesh_path}")
        print("\nPlease set MESH_PATH environment variable or update the default path")
        return
    
    # Load full mesh (vertices + faces)
    print(f"Loading mesh from {mesh_path}...")
    try:
        vertices, faces = load_full_mesh(mesh_path)
        print(f"✓ Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"✗ Failed to load mesh: {e}")
        return
    
    # Initialize VLM client
    print("\nInitializing VLM client...")
    try:
        vlm = FinetunedVLMClient()
        print("✓ Using fine-tuned VLM client")
    except Exception as e:
        print(f"⚠ Could not use fine-tuned VLM: {e}")
        print("Falling back to dummy VLM client...")
        vlm = DummyVLMClient()
        print("✓ Using dummy VLM client (for testing)")
    
    # Load PointNet++ model
    print(f"\nLoading PointNet++ model from {checkpoint_path}...")
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
        print("\nCannot continue without the model. Exiting.")
        return
    
    # Run ingestion pipeline
    print(f"\n{'=' * 60}")
    print("Running ingestion pipeline...")
    print(f"{'=' * 60}\n")
    
    try:
        ingest_result = ingest_mesh_to_semantic_params(
            mesh_path=mesh_path,
            vlm=vlm,
            model=model,
            render_output_dir=render_dir,
            num_points=2048,
        )
        
        print(f"\nCategory: {ingest_result.category}")
        print(f"Proposed parameters: {len(ingest_result.proposed_parameters)}")
        for param in ingest_result.proposed_parameters:
            print(f"  • {param.semantic_name} = {param.value:.4f}")
        
    except Exception as e:
        print(f"\n✗ Ingestion pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get part labels from segmentation
    print(f"\n{'=' * 60}")
    print("Getting part labels...")
    print(f"{'=' * 60}\n")
    
    try:
        seg_result = segment_mesh(
            mesh_path,
            model,
            num_points=2048,
            return_logits=False,
        )
        part_labels = seg_result["labels"]
        
        # Build part label names
        part_label_names = build_part_label_names(
            part_labels,
            ingest_result.category,
        )
        print(f"✓ Found {len(part_label_names)} unique parts")
        for part_id, part_name in part_label_names.items():
            count = np.sum(part_labels == part_id)
            print(f"  • Part {part_id} ({part_name}): {count} points")
        
    except Exception as e:
        print(f"✗ Failed to get part labels: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Build deformer
    print(f"\n{'=' * 60}")
    print("Building ParametricMeshDeformer...")
    print(f"{'=' * 60}\n")
    
    try:
        # Map point cloud labels to full mesh vertices
        # The segmentation was done on a sampled point cloud, but we need labels
        # for all mesh vertices. We'll use nearest-neighbor mapping.
        print("Mapping point cloud labels to mesh vertices...")
        
        # Load the point cloud that was used for segmentation
        from vlm_cad.pointnet_seg.mesh_io import load_mesh_as_point_cloud
        pc_points, _ = load_mesh_as_point_cloud(
            mesh_path,
            num_points=2048,
            normalize=True,
            return_normals=True,
        )
        
        # Normalize vertices to match point cloud normalization
        # (The point cloud was normalized, so we need to normalize vertices too)
        from vlm_cad.pointnet_seg.mesh_io import normalize_point_cloud
        vertices_normalized = normalize_point_cloud(vertices)
        
        # Map each vertex to nearest point in point cloud
        # This is a simple nearest-neighbor approach
        try:
            from scipy.spatial.distance import cdist
            distances = cdist(vertices_normalized, pc_points)
            nearest_pc_indices = np.argmin(distances, axis=1)
            vertex_labels = part_labels[nearest_pc_indices]
        except ImportError:
            # Fallback if scipy not available: use simple indexing
            print("  ⚠ scipy not available, using simplified mapping")
            if len(part_labels) <= len(vertices):
                # Repeat labels to match vertex count
                vertex_labels = np.zeros(len(vertices), dtype=part_labels.dtype)
                for i in range(len(vertices)):
                    idx = i % len(part_labels)
                    vertex_labels[i] = part_labels[idx]
            else:
                # Take first N labels
                vertex_labels = part_labels[:len(vertices)]
        
        print(f"✓ Mapped labels to {len(vertex_labels)} vertices")
        
        deformer = build_deformer_from_ingest_result(
            ingest_result=ingest_result,
            vertices=vertices,
            faces=faces,
            part_labels=part_labels,
            part_label_names=part_label_names,
        )
        print("✓ Deformer built successfully")
        print(f"  Base parameters: {list(deformer.base_parameters.keys())}")
        print(f"  Deformation configs: {list(deformer.config.keys())}")
        
    except Exception as e:
        print(f"✗ Failed to build deformer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Apply deformations
    print(f"\n{'=' * 60}")
    print("Applying deformations...")
    print(f"{'=' * 60}\n")
    
    # Get baseline parameters
    baseline_params = deformer.base_parameters.copy()
    print("Baseline parameters:")
    for name, value in baseline_params.items():
        print(f"  • {name} = {value:.4f}")
    
    # Create test parameter sets
    test_cases = [
        ("baseline", baseline_params),
    ]
    
    # Add category-specific test cases
    category_lower = ingest_result.category.lower()
    if "airplane" in category_lower or "plane" in category_lower:
        # Test cases for airplane
        test_params_1 = baseline_params.copy()
        if "wing_span" in test_params_1:
            test_params_1["wing_span"] *= 1.2  # +20%
        if "chord_length" in test_params_1:
            test_params_1["chord_length"] *= 1.1  # +10%
        test_cases.append(("wing_span_120pct_chord_110pct", test_params_1))
        
        test_params_2 = baseline_params.copy()
        if "wing_span" in test_params_2:
            test_params_2["wing_span"] *= 0.8  # -20%
        test_cases.append(("wing_span_80pct", test_params_2))
    
    elif "chair" in category_lower:
        # Test cases for chair
        test_params_1 = baseline_params.copy()
        if "seat_height" in test_params_1:
            test_params_1["seat_height"] *= 1.1  # +10%
        test_cases.append(("seat_height_110pct", test_params_1))
        
        test_params_2 = baseline_params.copy()
        if "back_height" in test_params_2:
            test_params_2["back_height"] *= 1.15  # +15%
        test_cases.append(("back_height_115pct", test_params_2))
    
    # Apply each test case
    for case_name, test_params in test_cases:
        print(f"\nDeforming: {case_name}")
        try:
            deformed_mesh = deformer.deform(test_params)
            
            # Save deformed mesh
            output_path = os.path.join(output_dir, f"{case_name}.obj")
            save_mesh(
                deformed_mesh.vertices,
                deformed_mesh.faces,
                output_path,
            )
            print(f"  ✓ Saved to {output_path}")
            
            # Print parameter changes
            print("  Parameter changes:")
            for name, value in test_params.items():
                if name in baseline_params:
                    change = ((value / baseline_params[name]) - 1.0) * 100
                    print(f"    • {name}: {baseline_params[name]:.4f} → {value:.4f} ({change:+.1f}%)")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("✓ Deformation complete!")
    print(f"Output meshes saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

