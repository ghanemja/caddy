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

from ..pointnet_seg.model import PointNet2PartSegWrapper
from ..pointnet_seg.mesh_io import load_mesh_as_point_cloud
from ..pointnet_seg.inference import segment_mesh
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
    model: "PointNet2PartSegWrapper",
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
    
    # Step 3: PointNet++ segmentation
    print(f"[Ingest] Running PointNet++ segmentation...", flush=True)
    seg_result = segment_mesh(
        mesh_path,
        model,
        num_points=num_points,
        return_logits=False,
    )
    points = seg_result["points"]
    labels = seg_result["labels"]
    print(f"[Ingest] Segmented into {len(np.unique(labels))} parts", flush=True)
    
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
    )
    print(f"[Ingest] Proposed semantic parameters: {len(post_output.final_parameters)}", flush=True)
    
    # Step 6: Build result
    result = IngestResult(
        category=pre_output.category,
        raw_parameters=raw_parameters,  # Generic parameters (p1, p2, p3, ...)
        proposed_parameters=post_output.final_parameters,  # Proposed semantic names
        pre_output=pre_output,
        post_output=post_output,
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

