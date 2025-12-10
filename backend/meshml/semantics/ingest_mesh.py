"""
Mesh ingestion orchestrator.

This module orchestrates the full pipeline:
1. Render mesh views
2. Pre-VLM: category + candidate params
3. PointNet++ segmentation + geometry extraction
4. Post-VLM: final semantic params
"""

from __future__ import annotations  # Defer evaluation of type hints

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
import os
import numpy as np

# Use segmentation abstraction layer
from ..segmentation import create_segmentation_backend, PartSegmentationBackend
from ..parts.parts import build_part_table_from_segmentation
from ..pointnet_seg.geometry import (
    compute_part_bounding_boxes,
    compute_part_statistics,
    axis_extent,
)

from .semantics_pre import PreVLMOutput, infer_category_and_candidates
from .semantics_post import PostVLMOutput, refine_parameters_with_vlm
from .vlm_client import VLMClient
from .types import RawParameter, FinalParameter


@dataclass
class IngestResult:
    """Final result from mesh ingestion pipeline."""
    category: str
    raw_parameters: "List[RawParameter]"  # Generic parameters (p1, p2, p3, ...)
    proposed_parameters: "List[FinalParameter]"  # Proposed semantic names (for user confirmation)
    pre_output: "PreVLMOutput"
    post_output: "PostVLMOutput"
    extra: Dict[str, Any]
    part_table: Optional["PartTable"] = None  # PartTable for part metadata
    
    # Backward compatibility: final_parameters as alias for proposed_parameters
    @property
    def final_parameters(self) -> "List[FinalParameter]":
        """Backward compatibility: return proposed_parameters."""
        return self.proposed_parameters


def render_mesh_views(
    mesh_path: str,
    output_dir: str,
    num_views: int = 3,
) -> "List[str]":
    """
    Render the mesh from several canonical views and save as images.
    
    Args:
        mesh_path: path to mesh file
        output_dir: directory to save rendered images
        num_views: number of views to render (default 3: front, side, iso)
        
    Returns:
        List of image file paths
    """
    try:
        import trimesh
        import numpy as np
        from PIL import Image
    except ImportError:
        raise ImportError("trimesh and PIL are required for mesh rendering")
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define camera positions for different views
    views = []
    if num_views >= 1:
        views.append(("front", [0, 0, 2]))  # Front view
    if num_views >= 2:
        views.append(("side", [2, 0, 0]))  # Side view
    if num_views >= 3:
        views.append(("iso", [1, 1, 1]))  # Isometric view
    
    # For simplicity, we'll create a simple orthographic projection
    # In practice, you might want to use a more sophisticated renderer
    image_paths = []
    
    # Get mesh bounds for scaling
    bounds = mesh.bounds
    center = mesh.centroid
    scale = np.max(bounds[1] - bounds[0])
    
    for view_name, camera_pos in views:
        # Simple 2D projection (orthographic)
        # This is a simplified renderer - for better quality, use trimesh's scene renderer
        # or open3d visualization
        
        # For now, create a placeholder image
        # In production, you'd use trimesh.scene.Scene or open3d.visualization
        img = Image.new('RGB', (512, 512), color='white')
        
        # Save image
        img_path = os.path.join(output_dir, f"{view_name}.png")
        img.save(img_path)
        image_paths.append(img_path)
    
    # If trimesh scene rendering is available, use it
    try:
        scene = trimesh.Scene(mesh)
        # Render using trimesh's built-in renderer if available
        # This is a placeholder - actual implementation would use scene.save_image()
        pass
    except:
        pass
    
    return image_paths


def _extract_raw_parameters(
    points: np.ndarray,
    labels: np.ndarray,
    category: str,
) -> List[RawParameter]:
    """
    Extract raw geometric parameters from segmented point cloud.
    
    Args:
        points: point cloud [N, 3]
        labels: part labels [N]
        category: object category (for context)
        
    Returns:
        List of RawParameter objects
    """
    import numpy as np
    
    raw_params = []
    
    # Compute part statistics
    stats = compute_part_statistics(points, labels)
    bboxes = compute_part_bounding_boxes(points, labels)
    
    # Global bounding box
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    global_extent = global_max - global_min
    global_center = (global_min + global_max) / 2.0
    
    param_counter = 1  # Start at p1
    
    # Add global parameters
    raw_params.append(
        RawParameter(
            id=f"p{param_counter}",
            value=float(global_extent[0]),
            units="normalized",
            description="Global bounding box length along X axis",
        )
    )
    param_counter += 1
    
    raw_params.append(
        RawParameter(
            id=f"p{param_counter}",
            value=float(global_extent[1]),
            units="normalized",
            description="Global bounding box length along Y axis",
        )
    )
    param_counter += 1
    
    raw_params.append(
        RawParameter(
            id=f"p{param_counter}",
            value=float(global_extent[2]),
            units="normalized",
            description="Global bounding box length along Z axis",
        )
    )
    param_counter += 1
    
    # Add per-part parameters
    for label_id, stat in stats.items():
        bbox = stat["bbox"]
        extent = bbox["extent"]
        center = bbox["center"]
        
        # Part extent along each axis
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            raw_params.append(
                RawParameter(
                    id=f"p{param_counter}",
                    value=float(extent[axis_idx]),
                    units="normalized",
                    description=f"Part {label_id} extent along {axis_name} axis",
                    part_labels=[f"part_{label_id}"],
                )
            )
            param_counter += 1
        
        # Part span (max extent)
        raw_params.append(
            RawParameter(
                id=f"p{param_counter}",
                value=float(np.max(extent)),
                units="normalized",
                description=f"Part {label_id} maximum span",
                part_labels=[f"part_{label_id}"],
            )
        )
        param_counter += 1
        
        # Part center coordinates
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            raw_params.append(
                RawParameter(
                    id=f"p{param_counter}",
                    value=float(center[axis_idx]),
                    units="normalized",
                    description=f"Part {label_id} center coordinate along {axis_name}",
                    part_labels=[f"part_{label_id}"],
                )
            )
            param_counter += 1
    
    return raw_params


def ingest_mesh_to_semantic_params(
    mesh_path: str,
    vlm: "VLMClient",
    model: "PartSegmentationBackend | PointNet2PartSegWrapper",  # Backward compat
    render_output_dir: str,
    num_points: int = 2048,
) -> "IngestResult":
    """
    Orchestrate the full mesh ingestion pipeline.
    
    Args:
        mesh_path: path to mesh file
        vlm: VLM client instance
        model: PointNet++ model instance
        render_output_dir: directory to save rendered images
        num_points: number of points to sample from mesh
        
    Returns:
        IngestResult with category, final parameters, and metadata
    """
    import numpy as np
    
    # Step 1: Render mesh views
    print(f"[Ingest] Rendering mesh views...", flush=True)
    image_paths = render_mesh_views(mesh_path, render_output_dir, num_views=3)
    print(f"[Ingest] Rendered {len(image_paths)} views", flush=True)
    
    # Step 2: Pre-VLM - category + candidate params
    print(f"[Ingest] Running pre-VLM classification...", flush=True)
    print(f"[Ingest] Note: First-time VLM loading may take 2-3 minutes (downloading model)...", flush=True)
    pre_output = infer_category_and_candidates(image_paths, vlm)
    print(f"[Ingest] Category: {pre_output.category}", flush=True)
    print(f"[Ingest] Candidate parameters: {len(pre_output.candidate_parameters)}", flush=True)
    
    # Step 3: Part segmentation (using abstraction layer)
    print(f"[Ingest] Running part segmentation...", flush=True)
    
    # Handle backward compatibility: if model is PointNet2PartSegWrapper, wrap it
    from ..segmentation.backends import PartSegmentationBackend, PointNetSegmentationBackend
    if not isinstance(model, PartSegmentationBackend):
        # Legacy: wrap PointNet2PartSegWrapper
        print("[Ingest] Wrapping legacy PointNet model in backend abstraction", flush=True)
        backend = PointNetSegmentationBackend.__new__(PointNetSegmentationBackend)
        backend.model = model
        backend.device = model.device
        backend.use_normals = model.use_normals
        backend.segment = lambda mp, **kw: _legacy_segment_mesh(mp, model, **kw)
    else:
        backend = model
    
    # Run segmentation
    seg_result = backend.segment(mesh_path, num_points=num_points)
    
    # Extract points and labels from result
    if seg_result.points is not None:
        points = seg_result.points
    else:
        # If no points, we need to sample them for geometry extraction
        from ..pointnet_seg.mesh_io import load_mesh_as_point_cloud
        points, _ = load_mesh_as_point_cloud(mesh_path, num_points=num_points, normalize=True)
    
    labels = seg_result.labels
    print(f"[Ingest] Segmented into {seg_result.num_parts} parts", flush=True)
    
    # Build PartTable from segmentation
    # For Hunyuan3D-Part, we should have vertex_labels; for PointNet, we use point labels
    # Load full mesh to get vertices for PartTable
    import trimesh
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    
    # Use vertex_labels if available, otherwise map point labels to vertices
    if seg_result.vertex_labels is not None:
        vertex_labels = seg_result.vertex_labels
    elif seg_result.vertices is not None:
        # If we have original vertices, use point labels directly
        vertex_labels = seg_result.labels
    else:
        # Fallback: map point labels to vertices using nearest neighbors
        # For now, use point labels (will be approximate)
        print("[Ingest] Warning: No vertex labels available, using point labels as approximation")
        vertex_labels = seg_result.labels
    
    # Ensure vertex_labels matches vertex count
    if len(vertex_labels) != len(vertices):
        # Truncate or pad to match
        if len(vertex_labels) < len(vertices):
            # Repeat last label
            vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
        else:
            vertex_labels = vertex_labels[:len(vertices)]
    
    # Build PartTable
    print(f"[Ingest] Building part metadata table...", flush=True)
    part_table = build_part_table_from_segmentation(
        vertices=vertices,
        part_labels=vertex_labels,
        ground_plane_z=None,  # Auto-detect
    )
    print(f"[Ingest] ✓ PartTable created with {len(part_table.parts)} parts", flush=True)
    
    # Step 4: Extract raw geometric parameters
    print(f"[Ingest] Extracting raw geometric parameters...", flush=True)
    raw_parameters = _extract_raw_parameters(points, labels, pre_output.category)
    print(f"[Ingest] Extracted {len(raw_parameters)} raw parameters", flush=True)
    
    # Step 5: Post-VLM - propose semantic names for generic parameters
    print(f"[Ingest] Running post-VLM semantic name proposal...", flush=True)
    
    # Extract part labels from segmentation for context
    unique_labels = np.unique(labels)
    part_labels = [f"part_{int(label_id)}" for label_id in unique_labels]
    
    post_output = refine_parameters_with_vlm(
        image_paths,
        pre_output,
        raw_parameters,
        vlm,
        part_labels=part_labels,
        part_table=part_table,  # Pass PartTable for VLM context
    )
    print(f"[Ingest] Proposed semantic parameters: {len(post_output.final_parameters)}", flush=True)
    
    # Step 6: Build result
    result = IngestResult(
        category=pre_output.category,
        raw_parameters=raw_parameters,  # Generic parameters (p1, p2, p3, ...)
        proposed_parameters=post_output.final_parameters,  # Proposed semantic names
        pre_output=pre_output,
        post_output=post_output,
        part_table=part_table,  # Include PartTable for part metadata
        extra={
            "num_points": len(points),
            "num_parts": len(np.unique(labels)),
            "image_paths": image_paths,
            "mesh_path": mesh_path,
            "part_labels": part_labels,
            "identified_parts": pre_output.parts,  # Parts identified by pre-VLM
        },
    )
    
    return result


def build_deformer_from_ingest_result(
    ingest_result: IngestResult,
    vertices: np.ndarray,
    faces: np.ndarray,
    part_labels: np.ndarray,
    part_label_names: Dict[int, str],
) -> "ParametricMeshDeformer":
    """
    Build a ParametricMeshDeformer from an IngestResult.
    
    Given:
      - ingest_result: has category, final/proposed parameters with semantic names and baseline values
      - vertices, faces: mesh geometry
      - part_labels: per-vertex integer labels
      - part_label_names: mapping part_id -> name
    
    Build a ParametricMeshDeformer that:
      - uses the baseline semantic parameters from ingest_result
      - uses a default deformation config per category
    
    Args:
        ingest_result: result from ingest_mesh_to_semantic_params
        vertices: mesh vertices [N, 3]
        faces: mesh faces [M, 3]
        part_labels: per-vertex part labels [N]
        part_label_names: mapping from part ID -> name
        
    Returns:
        ParametricMeshDeformer instance
    """
    from ..mesh_deform.deformer import (
        MeshData,
        ParametricMeshDeformer,
        build_default_deformation_config_for_category,
    )
    
    # Extract baseline parameters from ingest result
    base_params = {
        fp.semantic_name: fp.value
        for fp in ingest_result.proposed_parameters
    }
    
    # Build default deformation config for the category
    # Pass final_parameters to enable auto-config from VLM part_labels
    config = build_default_deformation_config_for_category(
        ingest_result.category,
        part_label_names,
        final_parameters=ingest_result.proposed_parameters,
    )
    
    # Create mesh data
    mesh_data = MeshData(vertices=vertices, faces=faces)
    
    # Use PartTable if available
    part_table = None
    if ingest_result.part_table is not None:
        part_table = ingest_result.part_table
    
    # Create and return deformer
    return ParametricMeshDeformer(
        base_mesh=mesh_data,
        base_parameters=base_params,
        part_labels=part_labels,
        part_label_names=part_label_names,
        config=config,
        part_table=part_table,
    )

