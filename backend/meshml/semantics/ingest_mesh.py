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
from typing import List, Dict, Optional, Any, TYPE_CHECKING
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
    proposed_parameters: (
        "List[FinalParameter]"  # Proposed semantic names (for user confirmation)
    )
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
    existing_images: Optional["List[str]"] = None,
) -> "List[str]":
    """
    Render the mesh from several canonical views and save as images.
    Can also use existing images if provided (e.g., from frontend canvas or uploads).

    Args:
        mesh_path: path to mesh file
        output_dir: directory to save rendered images
        num_views: number of views to render (default 3: front, side, iso)
        existing_images: Optional list of existing image paths to use instead of rendering

    Returns:
        List of image file paths
    """
    try:
        import trimesh
        import numpy as np
        from PIL import Image
    except ImportError:
        raise ImportError("trimesh and PIL are required for mesh rendering")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # If existing images are provided, use them
    if existing_images:
        print(
            f"[render_mesh_views] Using {len(existing_images)} existing images",
            flush=True,
        )
        # Copy existing images to output directory if they're not already there
        image_paths = []
        for i, existing_path in enumerate(existing_images):
            if os.path.exists(existing_path):
                # Copy to output directory with standardized name
                view_name = f"view_{i+1}" if i < num_views else f"extra_{i+1}"
                img_path = os.path.join(output_dir, f"{view_name}.png")
                if existing_path != img_path:
                    import shutil

                    shutil.copy2(existing_path, img_path)
                image_paths.append(img_path)
            else:
                print(
                    f"[render_mesh_views] Warning: Existing image not found: {existing_path}",
                    flush=True,
                )

        # If we have fewer images than requested views, render additional ones
        if len(image_paths) < num_views:
            print(
                f"[render_mesh_views] Rendering {num_views - len(image_paths)} additional views",
                flush=True,
            )
            additional_paths = _render_mesh_views_impl(
                mesh_path,
                output_dir,
                num_views - len(image_paths),
                start_index=len(image_paths),
            )
            image_paths.extend(additional_paths)

        return image_paths

    # Otherwise, render from mesh
    return _render_mesh_views_impl(mesh_path, output_dir, num_views)


def _render_mesh_views_impl(
    mesh_path: str,
    output_dir: str,
    num_views: int,
    start_index: int = 0,
) -> "List[str]":
    """Internal implementation of mesh rendering."""
    import trimesh
    import numpy as np
    from PIL import Image, ImageDraw
    import os

    # Load mesh
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Get mesh bounds for scaling and centering
    bounds = mesh.bounds
    center = mesh.centroid
    scale = np.max(bounds[1] - bounds[0])
    if scale == 0:
        scale = 1.0

    # Normalize mesh to unit scale centered at origin
    mesh_centered = mesh.copy()
    mesh_centered.vertices -= center
    mesh_centered.vertices /= scale

    # Define camera positions and rotations for different views
    views = []
    if num_views >= 1:
        views.append(("front", np.array([0, 0, 2]), np.array([0, 0, 0])))  # Front view
    if num_views >= 2:
        views.append(
            ("side", np.array([2, 0, 0]), np.array([0, np.pi / 2, 0]))
        )  # Side view
    if num_views >= 3:
        views.append(
            ("iso", np.array([1.5, 1.5, 1.5]), np.array([np.pi / 4, np.pi / 4, 0]))
        )  # Isometric view

    image_paths = []
    image_size = (512, 512)

    # Try to use pyrender for high-quality rendering
    try:
        import pyrender

        print(f"[render_mesh_views] Using pyrender for mesh rendering", flush=True)

        # Create scene
        scene = pyrender.Scene(
            ambient_light=[0.5, 0.5, 0.5], bg_color=[30, 41, 59]
        )  # Dark blue-gray background

        # Add mesh to scene
        mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh_centered))

        # Add lights
        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=3.0,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=np.eye(4))
        directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        scene.add(directional_light, pose=np.eye(4))

        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

        # Render each view
        for i, (view_name, camera_pos, camera_rot) in enumerate(views[:num_views]):
            # Set up camera pose
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_pos

            # Apply rotation (simplified - just position for now)
            camera_node = scene.add(camera, pose=camera_pose)

            # Render
            r = pyrender.OffscreenRenderer(image_size[0], image_size[1])
            color, depth = r.render(scene)
            r.delete()

            # Remove camera for next iteration
            scene.remove_node(camera_node)

            # Save image
            img = Image.fromarray(color)
            view_idx = start_index + i
            img_path = os.path.join(output_dir, f"view_{view_idx+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
            print(
                f"[render_mesh_views] Rendered {view_name} view: {img_path}", flush=True
            )

        return image_paths

    except ImportError:
        print(
            f"[render_mesh_views] pyrender not available, using fallback rendering",
            flush=True,
        )
    except Exception as e:
        print(
            f"[render_mesh_views] pyrender failed: {e}, using fallback rendering",
            flush=True,
        )

    # Fallback: Simple orthographic projection using PIL
    print(f"[render_mesh_views] Using fallback orthographic projection", flush=True)

    for i, (view_name, camera_pos, camera_rot) in enumerate(views[:num_views]):
        # Create image with dark background
        img = Image.new("RGB", image_size, color=(30, 41, 59))  # Dark blue-gray
        draw = ImageDraw.Draw(img)

        # Simple orthographic projection: project 3D vertices to 2D
        # Rotate vertices based on camera position
        vertices = mesh_centered.vertices.copy()

        # Simple rotation based on camera position (normalize to unit vector)
        camera_dir = camera_pos / (np.linalg.norm(camera_pos) + 1e-8)

        # Project to 2D (simple orthographic)
        # Use camera direction to determine which axes to project
        if abs(camera_dir[2]) > 0.7:  # Front/back view
            proj_vertices = vertices[:, [0, 1]]  # Project to XY plane
        elif abs(camera_dir[0]) > 0.7:  # Side view
            proj_vertices = vertices[:, [1, 2]]  # Project to YZ plane
        else:  # Isometric
            # Simple isometric projection
            proj_vertices = np.array(
                [
                    vertices[:, 0] - vertices[:, 2] * 0.5,
                    vertices[:, 1] - vertices[:, 2] * 0.5,
                ]
            ).T

        # Scale and center projection
        if len(proj_vertices) > 0:
            proj_min = proj_vertices.min(axis=0)
            proj_max = proj_vertices.max(axis=0)
            proj_range = proj_max - proj_min
            if np.any(proj_range > 0):
                proj_scale = min(image_size) * 0.8 / np.max(proj_range)
                proj_center = (proj_min + proj_max) / 2
                proj_vertices = (proj_vertices - proj_center) * proj_scale
                proj_vertices[:, 0] += image_size[0] / 2
                proj_vertices[:, 1] += image_size[1] / 2

                # Draw mesh edges (simplified - just draw some edges)
                # For better visualization, we'd need to extract edges from faces
                if hasattr(mesh_centered, "faces") and len(mesh_centered.faces) > 0:
                    # Draw a subset of edges
                    faces = mesh_centered.faces[
                        : min(1000, len(mesh_centered.faces))
                    ]  # Limit for performance
                    for face in faces:
                        if len(face) >= 3:
                            points = [
                                (int(proj_vertices[v][0]), int(proj_vertices[v][1]))
                                for v in face
                                if 0 <= proj_vertices[v][0] < image_size[0]
                                and 0 <= proj_vertices[v][1] < image_size[1]
                            ]
                            if len(points) >= 3:
                                draw.polygon(points, outline=(200, 200, 200), fill=None)
                else:
                    # Draw vertices as points
                    for v in proj_vertices[
                        :: max(1, len(proj_vertices) // 1000)
                    ]:  # Sample vertices
                        x, y = int(v[0]), int(v[1])
                        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                            draw.ellipse(
                                [x - 2, y - 2, x + 2, y + 2], fill=(200, 200, 200)
                            )

        # Save image
        view_idx = start_index + i
        img_path = os.path.join(output_dir, f"view_{view_idx+1}.png")
        img.save(img_path)
        image_paths.append(img_path)
        print(
            f"[render_mesh_views] Rendered {view_name} view (fallback): {img_path}",
            flush=True,
        )

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
    model: Optional[
        "Union[PartSegmentationBackend, PointNet2PartSegWrapper]"
    ] = None,  # Optional - can be None if part_table provided
    render_output_dir: str = None,
    num_points: Optional[int] = None,  # None = use environment variable or default
    part_table: Optional[
        "PartTable"
    ] = None,  # Optional: provide PartTable to skip segmentation
    points: Optional[
        np.ndarray
    ] = None,  # Optional: provide points to skip segmentation
    labels: Optional[
        np.ndarray
    ] = None,  # Optional: provide labels to skip segmentation
    existing_images: Optional[
        "List[str]"
    ] = None,  # Optional: existing images (from frontend or uploads)
    reference_image_path: Optional[
        str
    ] = None,  # Optional: path to reference image from Step 1 (prioritized for classification)
) -> "IngestResult":
    """
    Orchestrate the full mesh ingestion pipeline.

    Args:
        mesh_path: path to mesh file
        vlm: VLM client instance
        model: Optional segmentation backend (only needed if part_table not provided)
        render_output_dir: directory to save rendered images
        num_points: number of points to sample from mesh (None = use env var or default 5000)
        part_table: Optional PartTable from step 1 (if provided, segmentation is skipped)
        points: Optional points array from step 1 (if provided, used for raw parameter extraction)
        labels: Optional labels array from step 1 (if provided, used for raw parameter extraction)

    Returns:
        IngestResult with category, final parameters, and metadata
    """
    import numpy as np
    import os

    # Use environment variable or reasonable default (5000 minimum for P3-SAM)
    if num_points is None:
        num_points = int(os.environ.get("P3SAM_INFERENCE_POINT_NUM", "5000"))
        if num_points < 5000:
            num_points = 5000  # Minimum for reliable segmentation

    # Step 1: Render mesh views (or use existing images)
    print(f"[Ingest] Rendering mesh views...", flush=True)
    if existing_images:
        print(
            f"[Ingest] Using {len(existing_images)} existing images provided",
            flush=True,
        )
    image_paths = render_mesh_views(
        mesh_path, render_output_dir, num_views=3, existing_images=existing_images
    )
    print(f"[Ingest] Rendered/collected {len(image_paths)} views", flush=True)

    # Step 2: Pre-VLM - category + candidate params
    print(f"[Ingest] Running pre-VLM classification...", flush=True)
    print(
        f"[Ingest] Note: First-time VLM loading may take 2-3 minutes (downloading model)...",
        flush=True,
    )
    pre_output = infer_category_and_candidates(
        image_paths, vlm, reference_image_path=reference_image_path
    )
    print(f"[Ingest] Category: {pre_output.category}", flush=True)
    print(
        f"[Ingest] Candidate parameters: {len(pre_output.candidate_parameters)}",
        flush=True,
    )

    # Step 3: Part segmentation (using abstraction layer) - SKIP if part_table provided
    if part_table is not None:
        print(
            f"[Ingest] Using provided PartTable (skipping segmentation)...", flush=True
        )
        print(
            f"[Ingest] ✓ PartTable provided with {len(part_table.parts)} parts",
            flush=True,
        )

        # If points/labels not provided, we need them for raw parameter extraction
        # But we can extract them from the PartTable geometry if needed
        if points is None or labels is None:
            # We need points and labels for raw parameter extraction
            # Sample points from mesh if not provided
            import trimesh

            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            if points is None:
                # Sample points from mesh vertices
                vertices = np.array(mesh.vertices, dtype=np.float32)
                if num_points and len(vertices) > num_points:
                    # Sample subset of vertices
                    indices = np.random.choice(len(vertices), num_points, replace=False)
                    points = vertices[indices]
                else:
                    points = vertices

            if labels is None:
                # Use vertex labels from PartTable - CRITICAL: these must match part_ids
                vertex_labels = part_table.vertex_part_labels
                print(
                    f"[Ingest] Using PartTable vertex_labels: shape {vertex_labels.shape}, unique labels: {np.unique(vertex_labels)}",
                    flush=True,
                )
                print(
                    f"[Ingest] PartTable part IDs: {list(part_table.parts.keys())}",
                    flush=True,
                )

                if len(vertex_labels) == len(points):
                    labels = vertex_labels
                elif len(vertex_labels) > len(points):
                    # Sample labels to match points - use same indices as point sampling
                    # We need to use the same random seed and indices that were used for points
                    # For now, just take first N labels (assuming points were sampled from first N vertices)
                    labels = vertex_labels[: len(points)]
                else:
                    # Pad labels
                    labels = np.pad(
                        vertex_labels,
                        (0, len(points) - len(vertex_labels)),
                        mode="edge",
                    )

                print(
                    f"[Ingest] Final labels shape: {labels.shape}, unique labels: {np.unique(labels)}",
                    flush=True,
                )
                print(
                    f"[Ingest] Labels match PartTable IDs: {set(np.unique(labels)).issubset(set(part_table.parts.keys()))}",
                    flush=True,
                )
    else:
        # Run segmentation only if PartTable not provided
        print(f"[Ingest] Running part segmentation...", flush=True)
        print(
            f"[Ingest] NOTE: Segmentation should have been done in step 1 - this is unexpected",
            flush=True,
        )

        if model is None:
            raise ValueError("model is required when part_table is not provided")

        # Handle backward compatibility: if model is PointNet2PartSegWrapper, wrap it
        from ..segmentation.backends import (
            PartSegmentationBackend,
            PointNetSegmentationBackend,
        )

        if not isinstance(model, PartSegmentationBackend):
            # Legacy: wrap PointNet2PartSegWrapper
            print(
                "[Ingest] Wrapping legacy PointNet model in backend abstraction",
                flush=True,
            )
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

            points, _ = load_mesh_as_point_cloud(
                mesh_path, num_points=num_points, normalize=True
            )

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
            print(
                "[Ingest] Warning: No vertex labels available, using point labels as approximation"
            )
            vertex_labels = seg_result.labels

        # Ensure vertex_labels matches vertex count
        if len(vertex_labels) != len(vertices):
            # Truncate or pad to match
            if len(vertex_labels) < len(vertices):
                # Repeat last label
                vertex_labels = np.pad(
                    vertex_labels, (0, len(vertices) - len(vertex_labels)), mode="edge"
                )
            else:
                vertex_labels = vertex_labels[: len(vertices)]

        # Build PartTable
        print(f"[Ingest] Building part metadata table...", flush=True)
        part_table = build_part_table_from_segmentation(
            vertices=vertices,
            part_labels=vertex_labels,
            ground_plane_z=None,  # Auto-detect
        )
        print(
            f"[Ingest] ✓ PartTable created with {len(part_table.parts)} parts",
            flush=True,
        )

    # Step 4: Extract raw geometric parameters
    print(f"[Ingest] Extracting raw geometric parameters...", flush=True)
    print(
        f"[Ingest] Points shape: {points.shape if points is not None else 'None'}, Labels shape: {labels.shape if labels is not None else 'None'}",
        flush=True,
    )
    if part_table:
        print(
            f"[Ingest] PartTable has {len(part_table.parts)} parts with IDs: {list(part_table.parts.keys())}",
            flush=True,
        )
    raw_parameters = _extract_raw_parameters(points, labels, pre_output.category)
    print(f"[Ingest] Extracted {len(raw_parameters)} raw parameters", flush=True)
    # Debug: Check part_labels in raw parameters
    for rp in raw_parameters[:5]:  # Check first 5
        print(
            f"[Ingest] Raw param {rp.id} has part_labels: {rp.part_labels}", flush=True
        )

    # Step 5: Post-VLM - propose semantic names for generic parameters
    # Use per-part processing if PartTable has semantic names, otherwise use global processing
    print(f"[Ingest] Running post-VLM semantic name proposal...", flush=True)
    print(
        f"[Ingest] NOTE: This step uses VLM only - no Hunyuan3D/P3-SAM involved here",
        flush=True,
    )
    print(
        f"[Ingest] VLM will receive: category, part names (user input), geometry data, raw parameters",
        flush=True,
    )

    from .semantics_post import refine_parameters_per_part

    # Extract part labels for metadata (used in both branches)
    unique_labels = np.unique(labels)
    part_labels = [f"part_{int(label_id)}" for label_id in unique_labels]

    # Check if PartTable has semantic names (user-provided labels)
    has_semantic_names = part_table and any(
        part.name or (part.extra and part.extra.get("provisional_name"))
        for part in part_table.parts.values()
    )

    if has_semantic_names:
        print(
            f"[Ingest] Using per-part parameter assignment based on semantic names...",
            flush=True,
        )
        # Debug: Print what parts and names we're processing
        named_parts = {
            pid: (
                info.name or info.extra.get("provisional_name") if info.extra else None
            )
            for pid, info in part_table.parts.items()
            if info.name or (info.extra and info.extra.get("provisional_name"))
        }
        print(
            f"[Ingest] Processing {len(named_parts)} parts with semantic names: {named_parts}",
            flush=True,
        )

        post_output = refine_parameters_per_part(
            image_paths,
            pre_output,
            raw_parameters,
            vlm,
            part_table=part_table,
        )
    else:
        # Use global processing with part labels for context
        from .semantics_post import refine_parameters_with_vlm

        post_output = refine_parameters_with_vlm(
            image_paths,
            pre_output,
            raw_parameters,
            vlm,
            part_labels=part_labels,
            part_table=part_table,  # Pass PartTable for VLM context
        )
    print(
        f"[Ingest] Proposed semantic parameters: {len(post_output.final_parameters)}",
        flush=True,
    )

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
        fp.semantic_name: fp.value for fp in ingest_result.proposed_parameters
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
