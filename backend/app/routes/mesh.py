"""
Mesh processing routes blueprint
"""

from flask import Blueprint, request, jsonify, current_app
import os
import sys
import tempfile
import numpy as np
import trimesh

bp = Blueprint("mesh", __name__)

# Import from run.py for now - will be moved to service modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


@bp.post("/api/mesh/clear_gpu_memory")
def clear_gpu_memory():
    """Clear GPU memory by unloading segmentation models."""
    try:
        from meshml.segmentation import create_segmentation_backend
        import torch

        if not torch.cuda.is_available():
            return jsonify({"status": "info", "message": "CUDA not available"}), 200

        before_allocated = torch.cuda.memory_allocated() / 1024**3
        before_reserved = torch.cuda.memory_reserved() / 1024**3

        # Try to get and clear the Hunyuan3D backend if it exists
        try:
            backend_kind = os.environ.get("SEGMENTATION_BACKEND", "hunyuan3d").lower()
            if backend_kind == "hunyuan3d":
                # Create a temporary backend instance to access clear_gpu_memory method
                device = "cuda" if torch.cuda.is_available() else "cpu"
                temp_backend = create_segmentation_backend(
                    kind=backend_kind, device=device
                )
                if hasattr(temp_backend, "clear_gpu_memory"):
                    result = temp_backend.clear_gpu_memory()
                    return (
                        jsonify(
                            {
                                "status": "success",
                                "message": f"Cleared {result['freed_gb']:.2f} GB GPU memory",
                                **result,
                            }
                        ),
                        200,
                    )
        except Exception as e:
            print(f"[clear_gpu_memory] Could not use backend clear method: {e}")

        # Fallback: simple cache clear
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        after_allocated = torch.cuda.memory_allocated() / 1024**3
        after_reserved = torch.cuda.memory_reserved() / 1024**3
        freed = before_reserved - after_reserved

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Cleared {freed:.2f} GB GPU memory (cache only)",
                    "before": {
                        "allocated_gb": round(before_allocated, 2),
                        "reserved_gb": round(before_reserved, 2),
                    },
                    "after": {
                        "allocated_gb": round(after_allocated, 2),
                        "reserved_gb": round(after_reserved, 2),
                    },
                    "freed_gb": round(freed, 2),
                    "note": "Note: Model may still be in memory. Restart server to fully clear.",
                }
            ),
            200,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {"status": "error", "message": f"Failed to clear GPU memory: {str(e)}"}
            ),
            500,
        )


@bp.post("/ingest_mesh_segment")
def ingest_mesh_segment():
    """
    Step 1: Run segmentation only (fast, ~1-5 minutes).
    Returns part information for user labeling.

    Accepts:
    - mesh: mesh file (OBJ/STL/PLY) - required

    Returns:
    - segmentation: part segmentation results
    - part_table: PartTable JSON for labeling UI
    - mesh_path: path to uploaded mesh (for step 2)
    """
    try:
        from pathlib import Path

        # Get uploaded mesh file
        mesh_file = request.files.get("mesh")
        if not mesh_file:
            return jsonify({"ok": False, "error": "mesh file required"}), 400

        # Save uploaded file to temp directory
        temp_dir = tempfile.mkdtemp(prefix="mesh_ingest_")
        mesh_filename = mesh_file.filename or "mesh.obj"
        mesh_path = os.path.join(temp_dir, mesh_filename)
        mesh_file.save(mesh_path)

        print(f"[ingest_mesh_segment] Processing mesh: {mesh_path}", flush=True)

        # Import segmentation only (no VLM)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from meshml.segmentation import create_segmentation_backend
        from meshml.pointnet_seg.labels import get_category_from_flat_label
        from meshml.pointnet_seg.geometry import (
            compute_part_statistics,
            compute_part_bounding_boxes,
        )
        from meshml.parts.parts import (
            build_part_table_from_segmentation,
            part_table_to_labeling_json,
        )

        # Create segmentation backend
        print(f"[ingest_mesh_segment] Initializing segmentation backend...")
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "hunyuan3d").lower()
        print(f"[ingest_mesh_segment] Using segmentation backend: {backend_kind}")

        try:
            model = create_segmentation_backend(kind=backend_kind, device=device)
        except Exception as e:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Failed to initialize segmentation backend '{backend_kind}': {str(e)}",
                    }
                ),
                500,
            )

        # Run segmentation only (fast, ~1-5 minutes)
        # Use environment variable or reasonable default (5000 minimum for P3-SAM)
        default_num_points = int(os.environ.get("P3SAM_INFERENCE_POINT_NUM", "5000"))
        if default_num_points < 5000:
            default_num_points = 5000  # Minimum for reliable segmentation
        print(f"[ingest_mesh_segment] Running part segmentation...")
        seg_result = model.segment(mesh_path, num_points=default_num_points)
        points = seg_result.points
        labels = seg_result.labels
        unique_labels = np.unique(labels)

        print(f"[ingest_mesh_segment] ✓ Part segmentation complete!")
        print(f"[ingest_mesh_segment]   Segmented into {seg_result.num_parts} parts")
        print(f"[ingest_mesh_segment]   Point cloud: {seg_result.num_points} points")

        # Build part statistics for visualization
        part_stats = compute_part_statistics(points, labels)
        part_bboxes = compute_part_bounding_boxes(points, labels)

        # Build part label names
        part_label_names = {}
        for label_id in unique_labels:
            label_id_int = int(label_id)
            result = get_category_from_flat_label(label_id_int)
            if result:
                cat, part_name = result
                part_label_names[label_id_int] = part_name
            else:
                part_label_names[label_id_int] = f"part_{label_id_int}"

        # Create segmentation summary
        segmentation_summary = {
            "num_parts": len(unique_labels),
            "num_points": len(points),
            "parts": [],
        }

        for label_id in unique_labels:
            label_id_int = int(label_id)
            part_name = part_label_names.get(label_id_int, f"part_{label_id_int}")
            count = np.sum(labels == label_id_int)
            bbox = part_bboxes.get(label_id_int, {})

            # Convert bbox to JSON-serializable format
            bbox_data = None
            if label_id_int in part_bboxes and bbox:

                def to_list(v):
                    """Convert NumPy arrays/scalars to native Python types."""
                    if v is None:
                        return [0.0, 0.0, 0.0]
                    if hasattr(v, "tolist"):
                        return [float(x) for x in v.tolist()]
                    if isinstance(v, (list, tuple)):
                        return [float(x) for x in v]
                    return [float(v)]

                bbox_data = {
                    "min": to_list(bbox.get("min")),
                    "max": to_list(bbox.get("max")),
                    "center": to_list(bbox.get("center")),
                    "extent": to_list(bbox.get("extent")),
                }

            segmentation_summary["parts"].append(
                {
                    "id": int(label_id_int),
                    "name": part_name,
                    "point_count": int(count),
                    "percentage": float(count / len(points) * 100),
                    "bbox": bbox_data,
                }
            )

        # Print part details with names prominently displayed
        print(f"[ingest_mesh_segment] Part breakdown:")
        for part in segmentation_summary["parts"]:
            name = part["name"]
            part_id = part["id"]
            point_count = part[
                "point_count"
            ]  # Renamed to avoid shadowing 'points' variable
            pct = part["percentage"]
            print(
                f"[ingest_mesh_segment]   • {name} (ID: {part_id}): {point_count} points ({pct:.1f}%)"
            )

        # Build PartTable for user labeling
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = np.array(mesh.vertices, dtype=np.float32)

        # Store original vertex count for comparison after unmerging
        original_mesh_vertex_count = len(mesh.vertices)

        # Re-calculate vertex_labels using point cloud segmentation (known-good data)
        # Use nearest-neighbor lookup to map point cloud labels to mesh vertices
        try:
            from scipy.spatial import cKDTree

            # Create cKDTree from segmented point cloud
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)

            print(
                f"[ingest_mesh_segment] Mapping {len(points_array)} point cloud labels to {len(vertices)} mesh vertices using nearest-neighbor lookup"
            )

            # Build KD-tree from point cloud
            tree = cKDTree(points_array)

            # Query mesh vertices to find nearest point cloud point for each vertex
            _, nearest_point_indices = tree.query(vertices, k=1)

            # Handle case where query returns 2D array (for k>1) or scalar (for single query)
            if nearest_point_indices.ndim > 1:
                nearest_point_indices = nearest_point_indices.flatten()
            elif np.isscalar(nearest_point_indices):
                nearest_point_indices = np.array([nearest_point_indices])

            # Assign labels from nearest point cloud points to mesh vertices
            vertex_labels = labels_array[nearest_point_indices]

            # Debug: Calculate and compare label distributions
            # 1. Distribution of unique labels in source (point cloud)
            unique_labels_pc, counts_pc = np.unique(labels_array, return_counts=True)
            label_distribution_pc = dict(zip(unique_labels_pc, counts_pc))

            # 2. Distribution of unique labels in target (mesh)
            unique_labels_mesh, counts_mesh = np.unique(
                vertex_labels, return_counts=True
            )
            label_distribution_mesh = dict(zip(unique_labels_mesh, counts_mesh))

            # 3. Print both distributions
            print(
                f"[ingest_mesh_segment] Point cloud label distribution: {len(unique_labels_pc)} unique labels"
            )
            for label_id, count in sorted(label_distribution_pc.items()):
                pct = (count / len(labels_array)) * 100
                print(
                    f"[ingest_mesh_segment]   • Label {label_id}: {count} points ({pct:.1f}%)"
                )

            print(
                f"[ingest_mesh_segment] Mesh vertex label distribution: {len(unique_labels_mesh)} unique labels"
            )
            for label_id, count in sorted(label_distribution_mesh.items()):
                pct = (count / len(vertex_labels)) * 100
                print(
                    f"[ingest_mesh_segment]   • Label {label_id}: {count} vertices ({pct:.1f}%)"
                )

            # 4. Check if mapping failed (mesh has only 1 label but point cloud has multiple)
            num_unique_pc = len(unique_labels_pc)
            num_unique_mesh = len(unique_labels_mesh)
            if num_unique_mesh == 1 and num_unique_pc > 1:
                print(
                    f"[ingest_mesh_segment] ⚠ WARNING: Label mapping may have failed! "
                    f"Point cloud has {num_unique_pc} unique labels but mesh only has {num_unique_mesh} unique label. "
                    f"This suggests the KDTree mapping did not preserve label diversity.",
                    flush=True,
                )

            print(
                f"[ingest_mesh_segment] ✓ Mapped vertex labels: {len(vertex_labels)} vertices, {len(np.unique(vertex_labels))} unique labels"
            )

        except ImportError:
            # Fallback if scipy not available: use simple assignment (less accurate)
            print(
                "[ingest_mesh_segment] ⚠ Warning: scipy.spatial.cKDTree not available, using fallback label mapping"
            )
            vertex_labels = seg_result.labels
            if len(vertex_labels) != len(vertices):
                if len(vertex_labels) < len(vertices):
                    vertex_labels = np.pad(
                        vertex_labels,
                        (0, len(vertices) - len(vertex_labels)),
                        mode="edge",
                    )
                else:
                    vertex_labels = vertex_labels[: len(vertices)]
        except Exception as e:
            # Fallback on any error
            print(
                f"[ingest_mesh_segment] ⚠ Warning: Error in nearest-neighbor label mapping: {e}, using fallback",
                flush=True,
            )
            vertex_labels = seg_result.labels
            if len(vertex_labels) != len(vertices):
                if len(vertex_labels) < len(vertices):
                    vertex_labels = np.pad(
                        vertex_labels,
                        (0, len(vertices) - len(vertex_labels)),
                        mode="edge",
                    )
                else:
                    vertex_labels = vertex_labels[: len(vertices)]

        # Build PartTable with preliminary names from segmentation
        part_table = build_part_table_from_segmentation(
            vertices=vertices,
            part_labels=vertex_labels,
            ground_plane_z=None,
            preliminary_names=part_label_names,
        )
        part_table_json = part_table_to_labeling_json(part_table)

        print(
            f"[ingest_mesh_segment] ✓ PartTable created with {len(part_table.parts)} parts for user labeling"
        )

        # Helper function to generate a fixed, distinct color palette
        def get_color_palette(n_classes):
            """
            Generate a fixed, distinct set of RGB colors for segmentation.
            Uses a deterministic approach to ensure consistency across point cloud and mesh.

            Args:
                n_classes: Number of distinct colors needed

            Returns:
                Dictionary mapping class_id (0 to n_classes-1) -> RGB color array [R, G, B] in [0, 255]
            """
            try:
                # Try using matplotlib's colormaps for better color distinction
                import matplotlib.cm as cm

                # Use tab20 for up to 20 classes, then cycle or use Set3
                if n_classes <= 20:
                    # Use get_cmap for older matplotlib, or direct access for newer
                    try:
                        colormap = cm.get_cmap("tab20")
                    except AttributeError:
                        colormap = cm.tab20
                    colors = [colormap(i / 20.0)[:3] for i in range(n_classes)]
                elif n_classes <= 40:
                    # Combine tab20 and Set3 for more classes
                    try:
                        colormap1 = cm.get_cmap("tab20")
                        colormap2 = cm.get_cmap("Set3")
                    except AttributeError:
                        colormap1 = cm.tab20
                        colormap2 = cm.Set3
                    colors1 = [colormap1(i / 20.0)[:3] for i in range(20)]
                    colors2 = [
                        colormap2(i / (n_classes - 20.0))[:3]
                        for i in range(n_classes - 20)
                    ]
                    colors = colors1 + colors2
                else:
                    # For many classes, use a cyclic colormap
                    try:
                        colormap = cm.get_cmap("tab20")
                    except AttributeError:
                        colormap = cm.tab20
                    colors = [colormap((i % 20) / 20.0)[:3] for i in range(n_classes)]

                # Create palette dictionary
                palette = {
                    i: (np.array(colors[i]) * 255).astype(np.uint8)
                    for i in range(n_classes)
                }
            except ImportError:
                # Fallback: Use golden angle HSL approach (deterministic, no matplotlib needed)
                import colorsys

                palette = {}
                for i in range(n_classes):
                    # Use golden angle for color distribution (deterministic)
                    hue = (i * 137.508) % 360
                    # Convert HSL to RGB: colorsys.hls_to_rgb(h, l, s) where h in [0,1], l in [0,1], s in [0,1]
                    rgb = colorsys.hls_to_rgb(hue / 360.0, 0.65, 0.8)
                    # Convert from [0,1] to [0,255] for trimesh vertex colors
                    palette[i] = (np.array(rgb) * 255).astype(np.uint8)

            return palette

        # Apply colors to mesh vertices and export colored mesh as GLB
        try:
            import colorsys

            # Ensure vertex_labels match mesh.vertices length
            if len(vertex_labels) != len(mesh.vertices):
                if len(vertex_labels) < len(mesh.vertices):
                    vertex_labels = np.pad(
                        vertex_labels,
                        (0, len(mesh.vertices) - len(vertex_labels)),
                        mode="edge",
                    )
                else:
                    vertex_labels = vertex_labels[: len(mesh.vertices)]

            print(
                f"[ingest_mesh_segment] Vertex labels length: {len(vertex_labels)}, Mesh vertices length: {len(mesh.vertices)}"
            )

            # Step 1: Unmerge all vertices so no vertices are shared between faces
            # This ensures sharp boundaries by giving each face its own independent vertices
            def unmerge_all_vertices(mesh_obj):
                """
                Duplicate all vertices so each face has its own independent vertices.
                This prevents any vertex sharing between faces.

                Args:
                    mesh_obj: trimesh.Trimesh object

                Returns:
                    new_mesh: trimesh.Trimesh with unmerged vertices (one vertex per face corner)
                """
                faces = mesh_obj.faces
                vertices_orig = mesh_obj.vertices

                # Create new vertices: one copy per face corner
                new_vertices = []
                new_faces = []

                for face in faces:
                    # For each face, create new vertex copies
                    face_vertex_indices = []
                    for orig_v_idx in face:
                        new_v_idx = len(new_vertices)
                        new_vertices.append(vertices_orig[orig_v_idx])
                        face_vertex_indices.append(new_v_idx)
                    new_faces.append(face_vertex_indices)

                # Create new mesh with unmerged vertices
                new_mesh = trimesh.Trimesh(
                    vertices=np.array(new_vertices),
                    faces=np.array(new_faces),
                    process=False,  # Don't process, we've already handled vertex unmerging
                )

                return new_mesh

            # Unmerge all vertices (no sharing between faces)
            mesh_unmerged = unmerge_all_vertices(mesh)

            print(
                f"[ingest_mesh_segment] Unmerged vertices: {len(mesh.vertices)} -> {len(mesh_unmerged.vertices)} vertices"
            )

            # Step 2: Re-run KDTree mapping on unmerged mesh vertices
            # Since unmerging changed the vertex count, we need to map labels from point cloud again
            try:
                from scipy.spatial import cKDTree

                # Create cKDTree from segmented point cloud (reuse from earlier)
                points_array = np.array(points, dtype=np.float32)
                labels_array = np.array(labels, dtype=np.int32)

                print(
                    f"[ingest_mesh_segment] Re-mapping {len(points_array)} point cloud labels to {len(mesh_unmerged.vertices)} unmerged mesh vertices"
                )

                # Build KD-tree from point cloud
                tree = cKDTree(points_array)

                # Query unmerged mesh vertices to find nearest point cloud point
                unmerged_vertices = np.array(mesh_unmerged.vertices, dtype=np.float32)
                _, nearest_point_indices = tree.query(unmerged_vertices, k=1)

                # Handle case where query returns 2D array or scalar
                if nearest_point_indices.ndim > 1:
                    nearest_point_indices = nearest_point_indices.flatten()
                elif np.isscalar(nearest_point_indices):
                    nearest_point_indices = np.array([nearest_point_indices])

                # Assign labels from nearest point cloud points to unmerged mesh vertices
                vertex_labels = labels_array[nearest_point_indices]

                print(
                    f"[ingest_mesh_segment] ✓ Re-mapped vertex labels: {len(vertex_labels)} vertices, {len(np.unique(vertex_labels))} unique labels"
                )

            except ImportError:
                # Fallback if scipy not available: use original labels (less accurate)
                print(
                    "[ingest_mesh_segment] ⚠ Warning: scipy.spatial.cKDTree not available, using original labels (may be inaccurate after unmerging)"
                )
                # This fallback is not ideal since vertex count changed, but better than crashing
                # Pad or truncate original labels to match new vertex count
                if len(vertex_labels) < len(mesh_unmerged.vertices):
                    vertex_labels = np.pad(
                        vertex_labels,
                        (0, len(mesh_unmerged.vertices) - len(vertex_labels)),
                        mode="edge",
                    )
                else:
                    vertex_labels = vertex_labels[: len(mesh_unmerged.vertices)]
            except Exception as e:
                # Fallback on any error
                print(
                    f"[ingest_mesh_segment] ⚠ Warning: Error in re-mapping labels after unmerging: {e}, using fallback",
                    flush=True,
                )
                # Pad or truncate original labels to match new vertex count
                if len(vertex_labels) < len(mesh_unmerged.vertices):
                    vertex_labels = np.pad(
                        vertex_labels,
                        (0, len(mesh_unmerged.vertices) - len(vertex_labels)),
                        mode="edge",
                    )
                else:
                    vertex_labels = vertex_labels[: len(mesh_unmerged.vertices)]

            # Use the unmerged mesh for coloring
            mesh = mesh_unmerged

            # Generate consistent color palette for unique label IDs using shared function
            unique_label_ids = np.unique(vertex_labels)
            num_unique_labels = len(unique_label_ids)

            # Get color palette for the number of unique labels
            color_palette = get_color_palette(num_unique_labels)

            # Map label IDs to colors (label IDs may not be contiguous 0..n-1)
            # Create a mapping from actual label IDs to palette indices
            sorted_label_ids = np.sort(unique_label_ids)
            label_id_to_color = {}
            for palette_idx, label_id in enumerate(sorted_label_ids):
                label_id_int = int(label_id)
                # Use palette index to get color (ensures consistent colors)
                label_id_to_color[label_id_int] = color_palette[palette_idx]

            # Step 3: Apply colors to unmerged mesh vertices based on re-mapped labels
            vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
            for i, label_id in enumerate(vertex_labels):
                label_id_int = int(label_id)
                vertex_colors[i] = label_id_to_color.get(
                    label_id_int, np.array([136, 136, 136], dtype=np.uint8)
                )  # Default gray for unmapped labels

            mesh.visual.vertex_colors = vertex_colors

            print(
                f"[ingest_mesh_segment] ✓ Applied vertex colors to unmerged mesh: {len(vertex_colors)} vertices"
            )

            # Create separate sub-meshes for each label and export as Scene
            # Step 1: Determine which part each face belongs to (majority vote of vertex labels)
            face_labels = np.zeros(len(mesh.faces), dtype=np.int32)
            for face_idx, face in enumerate(mesh.faces):
                face_vertex_labels = [vertex_labels[v_idx] for v_idx in face]
                # Use the most common label (mode)
                face_labels[face_idx] = int(np.bincount(face_vertex_labels).argmax())

            # Step 2: Group faces by label ID
            faces_by_label = {}
            for face_idx, label_id in enumerate(face_labels):
                label_id_int = int(label_id)
                if label_id_int not in faces_by_label:
                    faces_by_label[label_id_int] = []
                faces_by_label[label_id_int].append(face_idx)

            print(
                f"[ingest_mesh_segment] Creating {len(faces_by_label)} separate sub-meshes for parts"
            )

            # Step 3: Create a sub-mesh for each label and add to Scene
            scene = trimesh.Scene()
            for label_id_int, face_indices in faces_by_label.items():
                # Skip if no faces for this label
                if len(face_indices) == 0:
                    print(
                        f"[ingest_mesh_segment] Warning: Label {label_id_int} has no faces, skipping"
                    )
                    continue

                try:
                    # Create sub-mesh using trimesh.submesh
                    # submesh returns a list of meshes (one per connected component)
                    submesh_list = mesh.submesh([face_indices], append=False)

                    # Get the color for this label
                    part_color = label_id_to_color.get(
                        label_id_int, np.array([136, 136, 136], dtype=np.uint8)
                    )

                    # Add each connected component as a separate geometry in the scene
                    for submesh_idx, part_mesh in enumerate(submesh_list):
                        # Apply uniform color to this sub-mesh (as face colors for uniform appearance)
                        if hasattr(part_mesh.visual, "face_colors"):
                            # Apply color to all faces uniformly
                            part_mesh.visual.face_colors = np.tile(
                                part_color, (len(part_mesh.faces), 1)
                            )
                        elif hasattr(part_mesh.visual, "vertex_colors"):
                            # Fallback: apply to vertices if face colors not available
                            part_mesh.visual.vertex_colors = np.tile(
                                part_color, (len(part_mesh.vertices), 1)
                            )

                        # Add to scene with a unique node name
                        node_name = (
                            f"part_{label_id_int}"
                            if submesh_idx == 0
                            else f"part_{label_id_int}_{submesh_idx}"
                        )
                        scene.add_geometry(part_mesh, node_name=node_name)

                    print(
                        f"[ingest_mesh_segment] Created sub-mesh for label {label_id_int}: {len(face_indices)} faces, {len(submesh_list)} connected components"
                    )
                except Exception as e:
                    print(
                        f"[ingest_mesh_segment] Error creating sub-mesh for label {label_id_int}: {e}",
                        flush=True,
                    )
                    continue

            # Step 4: Export Scene as GLB
            if len(scene.geometry) == 0:
                raise ValueError(
                    "No valid sub-meshes were created. Cannot export empty scene."
                )

            # Log vertex counts before export to confirm unmerging was successful
            total_unmerged_vertices = sum(
                len(geom.vertices) for geom in scene.geometry.values()
            )
            print(
                f"[ingest_mesh_segment] Before GLB export: Original mesh had {original_mesh_vertex_count} vertices, "
                f"unmerged scene has {total_unmerged_vertices} total vertices across {len(scene.geometry)} sub-meshes "
                f"(unmerging successful: {total_unmerged_vertices} > {original_mesh_vertex_count})"
            )

            colored_mesh_path = os.path.join(temp_dir, "segmentation_colored.glb")
            # Export scene as GLB - trimesh.export() does not re-merge vertices by default
            # No process=True or merge flags needed - the unmerged vertices are preserved
            # file_type is inferred from .glb extension, no additional parameters needed
            scene.export(colored_mesh_path, file_type="glb")

            # Verify file was created and has size
            if not os.path.exists(colored_mesh_path):
                raise FileNotFoundError(
                    f"GLB file was not created at {colored_mesh_path}"
                )
            file_size = os.path.getsize(colored_mesh_path)
            if file_size == 0:
                raise ValueError(
                    f"GLB file was created but is empty: {colored_mesh_path}"
                )

            print(
                f"[ingest_mesh_segment] ✓ Scene with {len(scene.geometry)} sub-meshes exported to: {colored_mesh_path} ({file_size} bytes)"
            )
            segmentation_summary["colored_mesh_path"] = colored_mesh_path

        except Exception as e:
            print(
                f"[ingest_mesh_segment] Warning: Could not create colored mesh GLB: {e}",
                flush=True,
            )
            import traceback

            traceback.print_exc()
            segmentation_summary["colored_mesh_path"] = None

        # Color the mesh with segmentation colors and export as GLB (advanced splitting version)
        try:
            import colorsys

            # Utility function to split vertices at part boundaries for crisp coloring
            def split_vertices_by_parts(mesh_obj, vertex_part_labels):
                """
                Split vertices at part boundaries to prevent color interpolation.

                For vertices shared between faces of different parts, create separate
                vertex instances so each face can have its own color without blur.

                Args:
                    mesh_obj: trimesh.Trimesh object
                    vertex_part_labels: array of part IDs for each vertex

                Returns:
                    New trimesh.Trimesh with split vertices
                """
                faces = mesh_obj.faces
                vertices_orig = mesh_obj.vertices

                # Step 1: Determine which part each face belongs to
                # Use majority vote: a face belongs to the part that most of its vertices belong to
                face_part_labels = np.zeros(len(faces), dtype=np.int32)
                for face_idx, face in enumerate(faces):
                    face_vertex_labels = [vertex_part_labels[v_idx] for v_idx in face]
                    # Use the most common label (mode)
                    face_part_labels[face_idx] = int(
                        np.bincount(face_vertex_labels).argmax()
                    )

                # Step 2: Build mapping: vertex -> set of face parts that use it
                # A vertex needs to be split if it's used by faces from different parts
                vertex_to_face_parts = {}
                for face_idx, face in enumerate(faces):
                    face_part = face_part_labels[face_idx]
                    for v_idx in face:
                        if v_idx not in vertex_to_face_parts:
                            vertex_to_face_parts[v_idx] = set()
                        vertex_to_face_parts[v_idx].add(face_part)

                # Step 3: Create new vertex mapping
                # For each original vertex, create one copy per face part that uses it
                # Each copy will have the color of the face part that uses it
                vertex_mapping = {}  # (original_v_idx, face_part_id) -> new_v_idx
                new_vertices = []
                new_vertex_part_labels = (
                    []
                )  # Track part ID for each new vertex (for coloring)

                for orig_v_idx, face_parts_using_vertex in vertex_to_face_parts.items():
                    if len(face_parts_using_vertex) == 1:
                        # Vertex is only used by faces from one part - no need to split
                        face_part_id = list(face_parts_using_vertex)[0]
                        new_v_idx = len(new_vertices)
                        vertex_mapping[(orig_v_idx, face_part_id)] = new_v_idx
                        new_vertices.append(vertices_orig[orig_v_idx])
                        # Use the face part for color (faces determine the color)
                        new_vertex_part_labels.append(face_part_id)
                    else:
                        # Vertex is shared between faces from multiple parts - create copies
                        for face_part_id in face_parts_using_vertex:
                            new_v_idx = len(new_vertices)
                            vertex_mapping[(orig_v_idx, face_part_id)] = new_v_idx
                            new_vertices.append(vertices_orig[orig_v_idx])
                            # Use the face part for color
                            new_vertex_part_labels.append(face_part_id)

                # Step 4: Update faces to use the appropriate vertex copies
                new_faces = []
                for face_idx, face in enumerate(faces):
                    face_part = face_part_labels[face_idx]
                    new_face = [
                        vertex_mapping[(orig_v_idx, face_part)] for orig_v_idx in face
                    ]
                    new_faces.append(new_face)

                # Step 5: Create new mesh with split vertices
                new_mesh = trimesh.Trimesh(
                    vertices=np.array(new_vertices),
                    faces=np.array(new_faces),
                    process=False,  # Don't process, we've already handled vertex splitting
                )

                return new_mesh, np.array(new_vertex_part_labels, dtype=np.int32)

            # Use the shared color palette function for consistency
            # Get unique part IDs from PartTable (more reliable than raw vertex_labels)
            unique_part_ids = list(part_table.parts.keys())
            num_unique_parts = len(unique_part_ids)

            # Get color palette for the number of unique parts
            color_palette_parts = get_color_palette(num_unique_parts)

            # Map part IDs to colors (same mapping logic as other exports)
            sorted_part_ids = sorted(unique_part_ids)
            part_id_to_color = {}
            for palette_idx, part_id in enumerate(sorted_part_ids):
                part_id_int = int(part_id)
                # Use palette index to get color (ensures consistent colors)
                part_id_to_color[part_id_int] = color_palette_parts[palette_idx]

            # Split vertices at part boundaries to get crisp edges
            vertex_part_labels = part_table.vertex_part_labels
            mesh_split, vertex_part_labels_split = split_vertices_by_parts(
                mesh, vertex_part_labels
            )

            print(
                f"[ingest_mesh_segment] Split vertices: {len(mesh.vertices)} -> {len(mesh_split.vertices)} vertices"
            )

            # Create separate sub-meshes for each part by splitting the mesh by label
            # Step 1: Determine which part each face belongs to (majority vote of vertex labels)
            face_part_labels = np.zeros(len(mesh_split.faces), dtype=np.int32)
            for face_idx, face in enumerate(mesh_split.faces):
                face_vertex_labels = [vertex_part_labels_split[v_idx] for v_idx in face]
                # Use the most common label (mode) - all vertices should have the same label after splitting, but use mode for safety
                face_part_labels[face_idx] = int(
                    np.bincount(face_vertex_labels).argmax()
                )

            # Step 2: Group faces by part ID
            faces_by_part = {}
            for face_idx, part_id in enumerate(face_part_labels):
                part_id_int = int(part_id)
                if part_id_int not in faces_by_part:
                    faces_by_part[part_id_int] = []
                faces_by_part[part_id_int].append(face_idx)

            print(
                f"[ingest_mesh_segment] Creating {len(faces_by_part)} separate sub-meshes for parts"
            )

            # Step 3: Create a sub-mesh for each part
            scene_meshes = []
            for part_id_int, face_indices in faces_by_part.items():
                # Skip if no faces for this part
                if len(face_indices) == 0:
                    print(
                        f"[ingest_mesh_segment] Warning: Part {part_id_int} has no faces, skipping"
                    )
                    continue

                # Extract faces for this part
                part_faces = mesh_split.faces[face_indices]

                # Validate that we have valid faces
                if len(part_faces) == 0:
                    print(
                        f"[ingest_mesh_segment] Warning: Part {part_id_int} has empty face array, skipping"
                    )
                    continue

                # Find unique vertices used by these faces
                unique_vertex_indices = np.unique(part_faces.flatten())
                if len(unique_vertex_indices) == 0:
                    print(
                        f"[ingest_mesh_segment] Warning: Part {part_id_int} has no vertices, skipping"
                    )
                    continue

                vertex_map = {
                    old_idx: new_idx
                    for new_idx, old_idx in enumerate(unique_vertex_indices)
                }

                # Remap face indices to use the new vertex indices
                remapped_faces = np.array(
                    [[vertex_map[v_idx] for v_idx in face] for face in part_faces]
                )

                # Extract vertices for this sub-mesh
                part_vertices = mesh_split.vertices[unique_vertex_indices]

                # Validate vertices
                if len(part_vertices) == 0:
                    print(
                        f"[ingest_mesh_segment] Warning: Part {part_id_int} has no vertices after extraction, skipping"
                    )
                    continue

                # Create sub-mesh
                try:
                    part_mesh = trimesh.Trimesh(
                        vertices=part_vertices, faces=remapped_faces, process=False
                    )

                    # Validate the mesh is valid
                    if not part_mesh.is_valid or len(part_mesh.faces) == 0:
                        print(
                            f"[ingest_mesh_segment] Warning: Part {part_id_int} mesh is invalid or has no faces, skipping"
                        )
                        continue

                    # Apply color to this sub-mesh (as vertex colors)
                    part_color = part_id_to_color.get(
                        part_id_int, np.array([136, 136, 136], dtype=np.uint8)
                    )
                    part_mesh.visual.vertex_colors = np.tile(
                        part_color, (len(part_vertices), 1)
                    )

                    # Add to scene with a name for identification
                    scene_meshes.append((f"part_{part_id_int}", part_mesh))
                    print(
                        f"[ingest_mesh_segment] Created sub-mesh for part {part_id_int}: {len(part_faces)} faces, {len(part_vertices)} vertices"
                    )
                except Exception as e:
                    print(
                        f"[ingest_mesh_segment] Error creating sub-mesh for part {part_id_int}: {e}",
                        flush=True,
                    )
                    continue

            # Step 4: Create a Scene containing all sub-meshes
            colored_glb_path = os.path.join(temp_dir, "segmentation_colored.glb")

            # Prepare vertex colors for fallback (single mesh export)
            vertex_colors_fallback = np.zeros(
                (len(mesh_split.vertices), 3), dtype=np.uint8
            )
            for i, part_id in enumerate(vertex_part_labels_split):
                part_id_int = int(part_id)
                vertex_colors_fallback[i] = part_id_to_color.get(
                    part_id_int, np.array([136, 136, 136], dtype=np.uint8)
                )  # Default gray

            if len(scene_meshes) == 0:
                print(
                    "[ingest_mesh_segment] ⚠ Warning: No valid sub-meshes were created. "
                    "Falling back to single colored mesh export."
                )
                # Fallback: export the split mesh with vertex colors as a single mesh
                mesh_split.visual.vertex_colors = vertex_colors_fallback
                # Log vertex count before export to confirm unmerging was successful
                print(
                    f"[ingest_mesh_segment] Before fallback GLB export: Original mesh had {original_mesh_vertex_count} vertices, "
                    f"split mesh has {len(mesh_split.vertices)} vertices "
                    f"(unmerging successful: {len(mesh_split.vertices)} > {original_mesh_vertex_count})"
                )
                # Export mesh as GLB - trimesh.export() does not re-merge vertices by default
                # No process=True or merge flags needed - the unmerged vertices are preserved
                # file_type is inferred from .glb extension, no additional parameters needed
                mesh_split.export(colored_glb_path, file_type="glb")
            else:
                scene = trimesh.Scene()
                for mesh_name, part_mesh in scene_meshes:
                    scene.add_geometry(part_mesh, node_name=mesh_name)

                # Validate scene has geometry before export
                if len(scene.geometry) == 0:
                    print(
                        "[ingest_mesh_segment] ⚠ Warning: Scene has no geometry. "
                        "Falling back to single colored mesh export."
                    )
                    # Fallback: export the split mesh with vertex colors as a single mesh
                    mesh_split.visual.vertex_colors = vertex_colors_fallback
                    # Log vertex count before export to confirm unmerging was successful
                    print(
                        f"[ingest_mesh_segment] Before fallback GLB export: Original mesh had {original_mesh_vertex_count} vertices, "
                        f"split mesh has {len(mesh_split.vertices)} vertices "
                        f"(unmerging successful: {len(mesh_split.vertices)} > {original_mesh_vertex_count})"
                    )
                    # Export mesh as GLB - trimesh.export() does not re-merge vertices by default
                    mesh_split.export(colored_glb_path)
                else:
                    try:
                        # Log vertex counts before export to confirm unmerging was successful
                        total_unmerged_vertices_advanced = sum(
                            len(geom.vertices) for geom in scene.geometry.values()
                        )
                        print(
                            f"[ingest_mesh_segment] Before advanced GLB export: Original mesh had {original_mesh_vertex_count} vertices, "
                            f"unmerged scene has {total_unmerged_vertices_advanced} total vertices across {len(scene.geometry)} sub-meshes "
                            f"(unmerging successful: {total_unmerged_vertices_advanced} > {original_mesh_vertex_count})"
                        )
                        # Export scene as GLB - trimesh.export() does not re-merge vertices by default
                        # No process=True or merge flags needed - the unmerged vertices are preserved
                        # file_type is inferred from .glb extension, no additional parameters needed
                        scene.export(colored_glb_path, file_type="glb")

                        # Verify file was created and has size
                        if not os.path.exists(colored_glb_path):
                            raise FileNotFoundError(
                                f"GLB file was not created at {colored_glb_path}"
                            )
                        file_size = os.path.getsize(colored_glb_path)
                        if file_size == 0:
                            raise ValueError(
                                f"GLB file was created but is empty: {colored_glb_path}"
                            )

                        print(
                            f"[ingest_mesh_segment] ✓ Split mesh with {len(scene_meshes)} parts exported to: {colored_glb_path} ({file_size} bytes)"
                        )
                    except Exception as scene_export_error:
                        print(
                            f"[ingest_mesh_segment] ⚠ Warning: Scene export failed: {scene_export_error}. "
                            "Falling back to single colored mesh export.",
                            flush=True,
                        )
                        # Fallback: export the split mesh with vertex colors as a single mesh
                        mesh_split.visual.vertex_colors = vertex_colors_fallback
                        # Log vertex count before export to confirm unmerging was successful
                        print(
                            f"[ingest_mesh_segment] Before error fallback GLB export: Original mesh had {original_mesh_vertex_count} vertices, "
                            f"split mesh has {len(mesh_split.vertices)} vertices "
                            f"(unmerging successful: {len(mesh_split.vertices)} > {original_mesh_vertex_count})"
                        )
                        # Export mesh as GLB - trimesh.export() does not re-merge vertices by default
                        # file_type is inferred from .glb extension, no additional parameters needed
                        mesh_split.export(colored_glb_path, file_type="glb")

            # Final verification
            if not os.path.exists(colored_glb_path):
                raise FileNotFoundError(
                    f"GLB file was not created at {colored_glb_path}"
                )
            file_size = os.path.getsize(colored_glb_path)
            if file_size == 0:
                raise ValueError(
                    f"GLB file was created but is empty: {colored_glb_path}"
                )

            segmentation_summary["colored_mesh_path"] = colored_glb_path

        except Exception as e:
            print(
                f"[ingest_mesh_segment] Warning: Could not create colored mesh GLB: {e}",
                flush=True,
            )
            import traceback

            traceback.print_exc()
            segmentation_summary["colored_mesh_path"] = None

        # Save colored point cloud visualization
        try:
            # Use the same color palette function as mesh export for consistency
            unique_label_ids_pc = np.unique(labels)
            num_unique_labels_pc = len(unique_label_ids_pc)

            # Get color palette for the number of unique labels
            color_palette_pc = get_color_palette(num_unique_labels_pc)

            # Map label IDs to colors (same mapping logic as mesh)
            sorted_label_ids_pc = np.sort(unique_label_ids_pc)
            label_id_to_color_pc = {}
            for palette_idx, label_id in enumerate(sorted_label_ids_pc):
                label_id_int = int(label_id)
                # Use palette index to get color (ensures consistent colors)
                # Convert from [0,255] to [0,1] for point cloud (trimesh expects [0,1] for PointCloud colors)
                label_id_to_color_pc[label_id_int] = (
                    color_palette_pc[palette_idx] / 255.0
                )

            # Apply colors to point cloud
            colors = np.zeros((len(points), 3))
            for i, label_id in enumerate(labels):
                label_id_int = int(label_id)
                colors[i] = label_id_to_color_pc.get(
                    label_id_int,
                    np.array(
                        [0.533, 0.533, 0.533]
                    ),  # Default gray [0.533, 0.533, 0.533] ≈ [136, 136, 136]/255
                )

            pc = trimesh.PointCloud(vertices=points, colors=colors)
            viz_path = os.path.join(temp_dir, "segmentation_colored.ply")
            pc.export(viz_path)
            segmentation_summary["visualization_path"] = viz_path
        except Exception as e:
            print(f"[ingest_mesh_segment] Warning: Could not save visualization: {e}")
            segmentation_summary["visualization_path"] = None

        # Include vertex labels for coloring from PartTable (more reliable than raw labels)
        # PartTable.vertex_part_labels should match the mesh vertex count
        vertex_labels_for_frontend = (
            part_table.vertex_part_labels.tolist()
            if hasattr(part_table.vertex_part_labels, "tolist")
            else list(part_table.vertex_part_labels)
        )

        # Return segmentation results only (user will label parts, then call /ingest_mesh_label)
        # Include points and labels for point cloud visualization
        response_data = {
            "ok": True,
            "segmentation": segmentation_summary,
            "part_table": part_table_json,
            "vertex_labels": vertex_labels_for_frontend,  # Use PartTable vertex labels (matches mesh)
            "points": (
                points.tolist() if points is not None else None
            ),  # Point cloud coordinates for visualization
            "labels": (
                labels.tolist() if labels is not None else None
            ),  # Point labels for coloring
            "mesh_path": mesh_path,
            "temp_dir": temp_dir,
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback

        error_msg = f"Mesh segmentation error: {str(e)}"
        print(f"[ingest_mesh_segment] {error_msg}", flush=True)
        traceback.print_exc()
        return (
            jsonify(
                {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )


@bp.get("/api/mesh/download_colored_glb")
def download_colored_glb():
    """
    Download the colored GLB mesh file generated during segmentation.

    Query parameters:
    - temp_dir: (required) The temporary directory path where segmentation_colored.glb was saved

    Returns:
    - GLB file as binary response if found
    - JSON error if file not found or temp_dir invalid
    """
    from flask import send_file

    try:
        # Get temp_dir from query parameters
        temp_dir = request.args.get("temp_dir")
        if not temp_dir:
            return (
                jsonify({"ok": False, "error": "temp_dir parameter is required"}),
                400,
            )

        # Validate temp_dir path (security: ensure it's a directory path, not arbitrary file access)
        if not os.path.isdir(temp_dir):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Invalid temp_dir: {temp_dir} is not a directory",
                    }
                ),
                400,
            )

        # Construct path to colored GLB file
        colored_glb_path = os.path.join(temp_dir, "segmentation_colored.glb")

        # Check if file exists
        if not os.path.exists(colored_glb_path):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Colored GLB file not found at: {colored_glb_path}",
                    }
                ),
                404,
            )

        # Check if it's actually a file (not a directory)
        if not os.path.isfile(colored_glb_path):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Path exists but is not a file: {colored_glb_path}",
                    }
                ),
                400,
            )

        # Send file
        print(
            f"[download_colored_glb] Sending colored GLB: {colored_glb_path}",
            flush=True,
        )
        try:
            # Try newer Flask API first (download_name)
            return send_file(
                colored_glb_path,
                mimetype="model/gltf-binary",
                as_attachment=True,
                download_name="segmentation_colored.glb",
            )
        except TypeError:
            # Fallback for older Flask versions (attachment_filename)
            return send_file(
                colored_glb_path,
                mimetype="model/gltf-binary",
                as_attachment=True,
                attachment_filename="segmentation_colored.glb",
            )

    except Exception as e:
        import traceback

        error_msg = f"Error downloading colored GLB: {str(e)}"
        print(f"[download_colored_glb] {error_msg}", flush=True)
        traceback.print_exc()
        return (
            jsonify(
                {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )


@bp.post("/convert_mesh_to_glb")
def convert_mesh_to_glb():
    """
    Convert uploaded mesh file (STL/PLY/OBJ) to GLB format for display in viewer.
    Returns the GLB file directly as binary response (not JSON).
    """
    from flask import send_file, Response

    try:
        if "mesh" not in request.files:
            return jsonify({"ok": False, "error": "No mesh file provided"}), 400

        mesh_file = request.files["mesh"]
        if not mesh_file.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(mesh_file.filename)[1]
        ) as tmp:
            mesh_file.save(tmp.name)
            mesh_path = tmp.name

        try:
            # Load mesh
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Ensure mesh is valid
            if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                return jsonify({"ok": False, "error": "Invalid mesh: no vertices"}), 400

            # Export as GLB (ensure GLTF 2.0 format)
            import io

            glb_buffer = io.BytesIO()

            # Use trimesh's export with explicit GLTF 2.0 settings
            try:
                # Try exporting as GLB with GLTF 2.0 format
                mesh.export(file_obj=glb_buffer, file_type="glb")
            except Exception as e:
                print(
                    f"[convert_mesh] GLB export failed: {e}, trying alternative method"
                )
                # Fallback: export as GLTF then convert, or use scene export
                scene = trimesh.Scene([mesh])
                scene.export(file_obj=glb_buffer, file_type="glb")

            glb_buffer.seek(0)
            glb_data = glb_buffer.read()

            if len(glb_data) == 0:
                return (
                    jsonify({"ok": False, "error": "Failed to generate GLB file"}),
                    500,
                )

            # Verify GLB format (should start with "glTF" magic or be valid binary)
            # GLB files should be at least 12 bytes (header)
            if len(glb_data) < 12:
                return (
                    jsonify({"ok": False, "error": "Generated GLB file is too small"}),
                    500,
                )

            # Return GLB file directly as binary response
            return Response(
                glb_data,
                mimetype="model/gltf-binary",
                headers={
                    "Content-Disposition": f"attachment; filename=converted_mesh.glb",
                    "Content-Length": str(len(glb_data)),
                },
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(mesh_path)
            except:
                pass

    except Exception as e:
        import traceback

        error_msg = str(e)
        print(f"[convert_mesh_to_glb] Error: {error_msg}", flush=True)
        traceback.print_exc()
        return (
            jsonify(
                {"ok": False, "error": error_msg, "traceback": traceback.format_exc()}
            ),
            500,
        )


@bp.post("/ingest_mesh_label")
def ingest_mesh_label():
    """
    Step 2: Run VLM processing with user-provided part labels.
    Accepts:
    - mesh_path: path from step 1
    - temp_dir: temp directory from step 1
    - part_labels: JSON with user-assigned part names (from labeling UI)

    Returns:
    - category: object category
    - final_parameters: list of semantic parameters
    - raw_parameters: list of raw geometric parameters
    """
    from run import _INGEST_RESULT_CACHE

    # Debug: Check current MAX_CONTENT_LENGTH setting
    try:
        current_limit = current_app.config.get("MAX_CONTENT_LENGTH", "Not set")
        if isinstance(current_limit, (int, float)):
            print(
                f"[ingest_mesh_label] Current MAX_CONTENT_LENGTH: {current_limit} bytes ({current_limit / (1024*1024):.1f} MB)",
                flush=True,
            )
        else:
            print(
                f"[ingest_mesh_label] Current MAX_CONTENT_LENGTH: {current_limit}",
                flush=True,
            )
    except Exception as e:
        print(
            f"[ingest_mesh_label] Could not check MAX_CONTENT_LENGTH: {e}", flush=True
        )

    try:
        import json
        from pathlib import Path

        # Get data from request (can be JSON or FormData)
        mesh_path = None
        temp_dir = None
        part_labels_json = None
        segmentation_data = None

        # Check content length before parsing
        content_length = request.content_length
        max_length = current_app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)

        # Debug: Log request size
        if content_length:
            print(
                f"[ingest_mesh_label] Request size: {content_length / (1024*1024):.2f} MB",
                flush=True,
            )
            print(
                f"[ingest_mesh_label] Max allowed: {max_length / (1024*1024):.2f} MB",
                flush=True,
            )

        if content_length and content_length > max_length:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Request too large: {content_length / (1024*1024):.1f} MB exceeds limit of {max_length / (1024*1024):.1f} MB",
                    }
                ),
                413,
            )

        if request.is_json:
            # JSON request
            data = request.get_json()
            if not data:
                return jsonify({"ok": False, "error": "JSON body required"}), 400
            mesh_path = data.get("mesh_path")
            temp_dir = data.get("temp_dir")
            part_labels_json = data.get("part_labels")
            segmentation_data = data.get("segmentation_data")
        else:
            # FormData (for file uploads) - use try/except to handle size errors gracefully
            try:
                data = request.form.to_dict()
            except Exception as form_error:
                # If form parsing fails due to size, provide helpful error
                if "413" in str(form_error) or "RequestEntityTooLarge" in str(
                    type(form_error).__name__
                ):
                    return (
                        jsonify(
                            {
                                "ok": False,
                                "error": f"Upload too large. Current limit: {max_length / (1024*1024):.1f} MB. Try compressing images or reducing resolution.",
                                "content_length": content_length,
                                "max_length": max_length,
                            }
                        ),
                        413,
                    )
                raise

            mesh_path = data.get("mesh_path")
            temp_dir = data.get("temp_dir")

            # Parse part_labels if provided as JSON string
            part_labels_json_str = data.get("part_labels")
            if part_labels_json_str:
                try:
                    part_labels_json = json.loads(part_labels_json_str)
                except Exception as e:
                    print(
                        f"[ingest_mesh_label] Warning: Could not parse part_labels JSON: {e}",
                        flush=True,
                    )
                    part_labels_json = None

            # Parse segmentation_data if provided as JSON string
            segmentation_data_str = data.get("segmentation_data")
            if segmentation_data_str:
                try:
                    segmentation_data = json.loads(segmentation_data_str)
                except Exception as e:
                    print(
                        f"[ingest_mesh_label] Warning: Could not parse segmentation_data JSON: {e}",
                        flush=True,
                    )
                    segmentation_data = None

        # Get existing images (from frontend canvas snapshots or uploaded reference images)
        existing_images = []
        reference_image_path = (
            None  # Track reference image separately for classification
        )
        # Check for uploaded reference image
        if "reference_image" in request.files:
            ref_file = request.files["reference_image"]
            if ref_file and ref_file.filename:
                ref_path = os.path.join(temp_dir, "reference_image.png")
                ref_file.save(ref_path)
                existing_images.append(ref_path)
                reference_image_path = ref_path  # Store for prioritized classification
                print(
                    f"[ingest_mesh_label] ✓ Saved reference image from Step 1: {ref_path}",
                    flush=True,
                )
                print(
                    f"[ingest_mesh_label]   This image will be used for category classification",
                    flush=True,
                )
        # Check for canvas snapshot
        if "canvas_snapshot" in request.files:
            snapshot_file = request.files["canvas_snapshot"]
            if snapshot_file and snapshot_file.filename:
                snapshot_path = os.path.join(temp_dir, "canvas_snapshot.png")
                snapshot_file.save(snapshot_path)
                existing_images.append(snapshot_path)
                print(
                    f"[ingest_mesh_label] Saved canvas snapshot: {snapshot_path}",
                    flush=True,
                )
        # Also check in JSON data (if sent as base64 or paths)
        if data.get("reference_image_path"):
            ref_path = data.get("reference_image_path")
            if os.path.exists(ref_path):
                existing_images.append(ref_path)
                if (
                    not reference_image_path
                ):  # Only set if not already set from file upload
                    reference_image_path = ref_path
        if data.get("canvas_snapshot_path"):
            snapshot_path = data.get("canvas_snapshot_path")
            if os.path.exists(snapshot_path):
                existing_images.append(snapshot_path)

        if not mesh_path or not os.path.exists(mesh_path):
            return (
                jsonify({"ok": False, "error": "mesh_path required and must exist"}),
                400,
            )

        print(
            f"[ingest_mesh_label] Processing mesh with user labels: {mesh_path}",
            flush=True,
        )
        print(
            f"[ingest_mesh_label] NOTE: This step does NOT run segmentation - using data from step 1",
            flush=True,
        )

        # Import pipeline
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from app.services.vlm_client_finetuned import FinetunedVLMClient
        from meshml.semantics.vlm_client_ollama import OllamaVLMClient
        from meshml.semantics.vlm_client import DummyVLMClient
        from meshml.semantics.ingest_mesh import ingest_mesh_to_semantic_params
        from meshml.parts.parts import (
            build_part_table_from_segmentation,
            apply_labels_from_json,
            PartTable,
            PartInfo,
        )
        import torch
        import trimesh
        import numpy as np

        # Determine device for VLM (not for segmentation - we're not running segmentation)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reconstruct PartTable from step 1 data (NO segmentation needed)
        # If we have segmentation_data from step 1, use it directly
        if segmentation_data and segmentation_data.get("part_table"):
            # Reconstruct PartTable from JSON
            from meshml.parts.parts import PartTable, PartInfo

            part_table_dict = segmentation_data["part_table"]
            parts_dict = {}
            for part_entry in part_table_dict.get("parts", []):
                part_id = part_entry["part_id"]
                part_info = PartInfo(
                    part_id=part_id,
                    name=part_entry.get("name"),
                    description=part_entry.get("description"),
                    centroid=np.array(
                        part_entry.get("centroid", [0, 0, 0]), dtype=np.float32
                    ),
                    bbox_min=np.array(
                        part_entry.get("bbox_min", [0, 0, 0]), dtype=np.float32
                    ),
                    bbox_max=np.array(
                        part_entry.get("bbox_max", [0, 0, 0]), dtype=np.float32
                    ),
                    extents=np.array(
                        part_entry.get("extents", [1, 1, 1]), dtype=np.float32
                    ),
                    touches_ground=part_entry.get("touches_ground", False),
                    extra=part_entry.get("extra", {}),
                )
                parts_dict[part_id] = part_info

            vertex_labels = np.array(
                segmentation_data.get("vertex_labels", []), dtype=np.int32
            )
            part_table = PartTable(parts=parts_dict, vertex_part_labels=vertex_labels)
            print(
                f"[ingest_mesh_label] ✓ Reconstructed PartTable from step 1 data ({len(part_table.parts)} parts)",
                flush=True,
            )
        else:
            # Fallback: if segmentation_data not provided, we need to load mesh and reconstruct
            # But we should NOT run segmentation - this is a fallback only
            print(
                f"[ingest_mesh_label] Warning: segmentation_data not provided, attempting to reconstruct PartTable from mesh only",
                flush=True,
            )
            print(
                f"[ingest_mesh_label] NOTE: This should not happen - step 1 should provide segmentation_data",
                flush=True,
            )
            # Load mesh for geometry only (no segmentation)
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            vertices = np.array(mesh.vertices, dtype=np.float32)

            # Create minimal PartTable - this is not ideal but better than re-running segmentation
            # We'll create a single part for the whole mesh
            from meshml.parts.parts import PartInfo

            part_info = PartInfo(
                part_id=0,
                name=None,
                centroid=np.mean(vertices, axis=0).astype(np.float32),
                bbox_min=vertices.min(axis=0).astype(np.float32),
                bbox_max=vertices.max(axis=0).astype(np.float32),
                extents=(vertices.max(axis=0) - vertices.min(axis=0)).astype(
                    np.float32
                ),
            )
            vertex_labels = np.zeros(len(vertices), dtype=np.int32)
            part_table = PartTable(
                parts={0: part_info}, vertex_part_labels=vertex_labels
            )
            print(
                f"[ingest_mesh_label] ⚠ Created minimal PartTable (fallback - segmentation should have been provided)",
                flush=True,
            )

        # Apply user-provided labels
        if part_labels_json:
            part_table = apply_labels_from_json(part_table, part_labels_json)
            print(
                f"[ingest_mesh_label] Applied user labels to {len(part_labels_json.get('parts', []))} parts",
                flush=True,
            )
            # Debug: Print all user-provided part names
            for part_entry in part_labels_json.get("parts", []):
                part_id = part_entry.get("part_id")
                user_name = part_entry.get("name")
                if part_id is not None and user_name:
                    print(
                        f"[ingest_mesh_label]   Part {part_id}: user provided name = '{user_name}'",
                        flush=True,
                    )
            # Verify names are set in PartTable
            for part_id, part_info in part_table.parts.items():
                if part_info.name:
                    print(
                        f"[ingest_mesh_label]   ✓ PartTable part {part_id} has name: '{part_info.name}'",
                        flush=True,
                    )
                else:
                    print(
                        f"[ingest_mesh_label]   ⚠ PartTable part {part_id} has NO name (will use provisional)",
                        flush=True,
                    )

        # Initialize VLM (same logic as before)
        vlm = None
        vlm_model_info = None

        if device == "cpu":
            try:
                vlm = OllamaVLMClient()
                # Get Ollama model name from environment
                ollama_model = os.environ.get("OLLAMA_MODEL", "llava:latest")
                vlm_model_info = f"Ollama ({ollama_model})"
                print(
                    f"[ingest_mesh_label] ✓ Using Ollama VLM: {ollama_model} (fast on CPU)"
                )
            except Exception as e:
                print(f"[ingest_mesh_label] Warning: Could not use Ollama: {e}")
            if not vlm:
                try:
                    vlm = FinetunedVLMClient()
                    finetuned_path = os.environ.get(
                        "FINETUNED_MODEL_PATH",
                        "backend/checkpoints/onevision_lora_small/checkpoint-4",
                    )
                    vlm_model_info = f"Fine-tuned LlavaOnevision (base: llava-hf/llava-onevision-qwen2-7b-ov-hf, adapter: {finetuned_path})"
                    print(f"[ingest_mesh_label] ✓ Using fine-tuned VLM (slower on CPU)")
                    print(
                        f"[ingest_mesh_label]   Model: llava-hf/llava-onevision-qwen2-7b-ov-hf"
                    )
                    print(f"[ingest_mesh_label]   Adapter: {finetuned_path}")
                except Exception as e2:
                    print(
                        f"[ingest_mesh_label] Warning: Could not use fine-tuned VLM: {e2}"
                    )
            if not vlm:
                vlm = DummyVLMClient()
                vlm_model_info = "Dummy (testing only)"
                print(
                    "[ingest_mesh_label] ⚠ Using dummy VLM (for testing - no real model)"
                )
        else:
            try:
                vlm = FinetunedVLMClient()
                finetuned_path = os.environ.get(
                    "FINETUNED_MODEL_PATH",
                    "backend/checkpoints/onevision_lora_small/checkpoint-4",
                )
                vlm_model_info = f"Fine-tuned LlavaOnevision (base: llava-hf/llava-onevision-qwen2-7b-ov-hf, adapter: {finetuned_path})"
                print(
                    f"[ingest_mesh_label] ✓ Using fine-tuned VLM (pretrained model on GPU)"
                )
                print(
                    f"[ingest_mesh_label]   Model: llava-hf/llava-onevision-qwen2-7b-ov-hf"
                )
                print(f"[ingest_mesh_label]   Adapter: {finetuned_path}")
            except Exception as e:
                print(f"[ingest_mesh_label] Warning: Could not use fine-tuned VLM: {e}")
                try:
                    vlm = OllamaVLMClient()
                    ollama_model = os.environ.get("OLLAMA_MODEL", "llava:latest")
                    vlm_model_info = f"Ollama ({ollama_model})"
                    print(
                        f"[ingest_mesh_label] ✓ Using Ollama VLM (fallback): {ollama_model}"
                    )
                except Exception as e2:
                    print(f"[ingest_mesh_label] Warning: Could not use Ollama: {e2}")
                    vlm = DummyVLMClient()
                    vlm_model_info = "Dummy (testing only)"
                    print(
                        "[ingest_mesh_label] ⚠ Using dummy VLM (for testing - no real model)"
                    )

        # Store model info for later reference
        if vlm_model_info:
            print(
                f"[ingest_mesh_label] VLM Model Summary: {vlm_model_info}", flush=True
            )

        # Run ingestion pipeline with user-labeled PartTable
        render_dir = os.path.join(temp_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)

        print(f"[ingest_mesh_label] Running VLM-based semantic parameter extraction...")
        print(
            f"[ingest_mesh_label] NOTE: No segmentation will be run - using PartTable from step 1",
            flush=True,
        )

        # Extract points and labels from segmentation_data if available (for raw parameter extraction)
        # Note: If points/labels not provided, we'll sample from mesh (they're not needed for VLM)
        points = None
        labels = None
        if segmentation_data:
            if segmentation_data.get("points"):
                points = np.array(segmentation_data["points"], dtype=np.float32)
                print(
                    f"[ingest_mesh_label] Using points from segmentation_data: {len(points)} points",
                    flush=True,
                )
            if segmentation_data.get("labels"):
                labels = np.array(segmentation_data["labels"], dtype=np.int32)
                print(
                    f"[ingest_mesh_label] Using labels from segmentation_data: {len(labels)} labels",
                    flush=True,
                )

        # If points/labels not provided, we'll sample from mesh in ingest_mesh_to_semantic_params
        if points is None or labels is None:
            print(
                f"[ingest_mesh_label] Points/labels not in segmentation_data - will sample from mesh if needed",
                flush=True,
            )

        # Use environment variable or reasonable default (None = let function decide)
        ingest_result = ingest_mesh_to_semantic_params(
            mesh_path=mesh_path,
            vlm=vlm,
            model=None,  # No segmentation backend needed - we have PartTable from step 1
            render_output_dir=render_dir,
            num_points=None,  # None = use environment variable or default (5000 minimum)
            part_table=part_table,  # Provide PartTable to skip segmentation
            points=points,  # Provide points if available
            labels=labels,  # Provide labels if available
            existing_images=(
                existing_images if existing_images else None
            ),  # Use existing images if provided
            reference_image_path=reference_image_path,  # Pass reference image for prioritized classification
        )

        # Override the part_table with the user-labeled one
        if ingest_result and part_table:
            ingest_result.part_table = part_table

        # Cache result for /apply_mesh_params
        _INGEST_RESULT_CACHE[mesh_path] = ingest_result

        print(f"[ingest_mesh_label] ✓ Semantic parameter extraction complete!")
        print(f"[ingest_mesh_label]   Category: {ingest_result.category}")
        print(
            f"[ingest_mesh_label]   Parameters: {len(ingest_result.final_parameters)} semantic params"
        )

        # Build hierarchical structure: Category -> Parts -> Parameters
        def build_hierarchical_structure(ingest_result, part_table):
            """Build hierarchical output: Category -> Parts -> Parameters."""
            hierarchical = {
                "category": ingest_result.category,
                "category_confidence": ingest_result.pre_output.raw_response.get(
                    "category_confidence", 1.0
                ),
                "category_reasoning": ingest_result.pre_output.raw_response.get(
                    "category_reasoning", ""
                ),
                "parts": {},
            }

            # Group parameters by part
            if part_table:
                # Initialize parts structure
                for part_id, part_info in part_table.parts.items():
                    # CRITICAL: Prioritize user-provided name (part_info.name) over provisional
                    # This ensures user input from Step 3 is used
                    part_name = part_info.name
                    if not part_name:
                        # Fallback to provisional name if no user-provided name
                        part_name = (
                            part_info.extra.get("provisional_name")
                            if part_info.extra
                            else None
                        )
                    if not part_name:
                        # Last resort: use part_X format
                        part_name = f"part_{part_id}"

                    print(
                        f"[build_hierarchical] Initializing part {part_id} with name: '{part_name}' (user-provided: {bool(part_info.name)})",
                        flush=True,
                    )

                    hierarchical["parts"][part_name] = {
                        "part_id": part_id,
                        "description": part_info.description
                        or f"{part_name} component",
                        "geometry": {
                            "centroid": (
                                part_info.centroid.tolist()
                                if hasattr(part_info.centroid, "tolist")
                                else list(part_info.centroid)
                            ),
                            "extents": (
                                part_info.extents.tolist()
                                if hasattr(part_info.extents, "tolist")
                                else list(part_info.extents)
                            ),
                            "bbox_min": (
                                part_info.bbox_min.tolist()
                                if hasattr(part_info.bbox_min, "tolist")
                                else list(part_info.bbox_min)
                            ),
                            "bbox_max": (
                                part_info.bbox_max.tolist()
                                if hasattr(part_info.bbox_max, "tolist")
                                else list(part_info.bbox_max)
                            ),
                        },
                        "parameters": [],
                    }

                # Assign parameters to parts
                print(
                    f"[build_hierarchical] Total parameters: {len(ingest_result.final_parameters)}",
                    flush=True,
                )
                print(
                    f"[build_hierarchical] Total parts in hierarchical: {len(hierarchical['parts'])}",
                    flush=True,
                )

                # Create a reverse lookup: part_id -> part_name for faster matching
                part_id_to_name = {}
                for part_id, part_info in part_table.parts.items():
                    # CRITICAL: Prioritize user-provided name (part_info.name) over provisional
                    part_name = part_info.name
                    if not part_name:
                        part_name = (
                            part_info.extra.get("provisional_name")
                            if part_info.extra
                            else None
                        )
                    if not part_name:
                        part_name = f"part_{part_id}"
                    part_id_to_name[part_id] = part_name
                    print(
                        f"[build_hierarchical] Part ID {part_id} -> name: '{part_name}' (user-provided: {bool(part_info.name)})",
                        flush=True,
                    )

                unmatched_params = []
                for fp in ingest_result.final_parameters:
                    part_labels = getattr(fp, "part_labels", None) or []
                    print(
                        f"[build_hierarchical] Parameter {fp.id} ({fp.semantic_name}) has part_labels: {part_labels}",
                        flush=True,
                    )

                    matched = False
                    # Try to match via part_labels
                    for label in part_labels:
                        part_name = None
                        # Try to parse as part_id first (most reliable)
                        try:
                            # Handle both "part_0" and "0" formats
                            label_clean = label.replace("part_", "").strip()
                            part_id = int(label_clean)
                            if part_id in part_id_to_name:
                                part_name = part_id_to_name[part_id]
                                print(
                                    f"[build_hierarchical] Matched parameter {fp.id} to part_id {part_id} -> {part_name}",
                                    flush=True,
                                )
                        except (ValueError, AttributeError):
                            # Try to find by name
                            if part_table and hasattr(part_table, "get_part_by_name"):
                                part = part_table.get_part_by_name(label)
                                if part:
                                    part_name = (
                                        part.name
                                        or (
                                            part.extra.get("provisional_name")
                                            if part.extra
                                            else None
                                        )
                                        or f"part_{part.part_id}"
                                    )

                        if part_name and part_name in hierarchical["parts"]:
                            hierarchical["parts"][part_name]["parameters"].append(
                                {
                                    "id": fp.id,
                                    "semantic_name": fp.semantic_name,
                                    "name": fp.semantic_name,  # Alias
                                    "value": float(fp.value),
                                    "units": fp.units or "normalized",
                                    "description": fp.description,
                                    "confidence": (
                                        float(fp.confidence) if fp.confidence else 0.0
                                    ),
                                }
                            )
                            matched = True
                            print(
                                f"[build_hierarchical] ✓ Added parameter {fp.id} to part {part_name}",
                                flush=True,
                            )
                            break

                    if not matched:
                        unmatched_params.append(fp)
                        print(
                            f"[build_hierarchical] ⚠ Parameter {fp.id} ({fp.semantic_name}) could not be matched to any part",
                            flush=True,
                        )

                # If there are unmatched parameters, try to assign them based on raw_sources
                # or distribute them to parts that have no parameters
                if unmatched_params:
                    print(
                        f"[build_hierarchical] {len(unmatched_params)} unmatched parameters, attempting to assign...",
                        flush=True,
                    )
                    # Find parts with no parameters
                    parts_without_params = [
                        name
                        for name, part_data in hierarchical["parts"].items()
                        if len(part_data["parameters"]) == 0
                    ]

                    # Try to match based on raw parameter sources
                    for fp in unmatched_params:
                        # Try to extract part_id from raw_sources if available
                        assigned = False
                        if fp.raw_sources:
                            for source_id in fp.raw_sources:
                                # raw parameters might have part_labels
                                # Check if we can find the part from the raw parameter
                                # For now, assign to first part without parameters
                                if parts_without_params:
                                    part_name = parts_without_params.pop(0)
                                    hierarchical["parts"][part_name][
                                        "parameters"
                                    ].append(
                                        {
                                            "id": fp.id,
                                            "semantic_name": fp.semantic_name,
                                            "name": fp.semantic_name,
                                            "value": float(fp.value),
                                            "units": fp.units or "normalized",
                                            "description": fp.description,
                                            "confidence": (
                                                float(fp.confidence)
                                                if fp.confidence
                                                else 0.0
                                            ),
                                        }
                                    )
                                    print(
                                        f"[build_hierarchical] Assigned unmatched parameter {fp.id} to {part_name} (fallback)",
                                        flush=True,
                                    )
                                    assigned = True
                                    break

                        if not assigned and hierarchical["parts"]:
                            # Last resort: assign to first part
                            first_part_name = next(iter(hierarchical["parts"].keys()))
                            hierarchical["parts"][first_part_name]["parameters"].append(
                                {
                                    "id": fp.id,
                                    "semantic_name": fp.semantic_name,
                                    "name": fp.semantic_name,
                                    "value": float(fp.value),
                                    "units": fp.units or "normalized",
                                    "description": fp.description,
                                    "confidence": (
                                        float(fp.confidence) if fp.confidence else 0.0
                                    ),
                                }
                            )
                            print(
                                f"[build_hierarchical] Assigned unmatched parameter {fp.id} to {first_part_name} (last resort)",
                                flush=True,
                            )

            # If no part_table, create a flat structure with all parameters under a generic part
            if not part_table or not hierarchical["parts"]:
                hierarchical["parts"]["all_parts"] = {
                    "part_id": -1,
                    "description": "All parts combined",
                    "geometry": {},
                    "parameters": [],
                }
                for fp in ingest_result.final_parameters:
                    hierarchical["parts"]["all_parts"]["parameters"].append(
                        {
                            "id": fp.id,
                            "semantic_name": fp.semantic_name,
                            "name": fp.semantic_name,
                            "value": float(fp.value),
                            "units": fp.units or "normalized",
                            "description": fp.description,
                            "confidence": (
                                float(fp.confidence) if fp.confidence else 0.0
                            ),
                        }
                    )

            return hierarchical

        # Convert FinalParameter objects to dicts for JSON serialization (flat format for backward compatibility)
        def final_param_to_dict(fp):
            """Convert FinalParameter to dict."""
            part_labels = getattr(fp, "part_labels", None) or []
            # Map part IDs to part names if we have part_table
            if part_labels and ingest_result.part_table:
                named_labels = []
                for label in part_labels:
                    # Try to find part by name or ID
                    part = (
                        ingest_result.part_table.get_part_by_name(label)
                        if hasattr(ingest_result.part_table, "get_part_by_name")
                        else None
                    )
                    if not part:
                        # Try to parse as part_id
                        try:
                            part_id = int(label.replace("part_", ""))
                            part = ingest_result.part_table.parts.get(part_id)
                        except:
                            pass
                    if part:
                        named_labels.append(
                            part.name
                            or (
                                part.extra.get("provisional_name")
                                if part.extra
                                else None
                            )
                            or label
                        )
                    else:
                        named_labels.append(label)
                part_labels = named_labels

            return {
                "id": fp.id,
                "semantic_name": fp.semantic_name,
                "proposed_name": fp.semantic_name,  # Alias for compatibility
                "name": fp.semantic_name,  # Alias for compatibility
                "value": float(fp.value),
                "units": fp.units,
                "description": fp.description,
                "confidence": float(fp.confidence) if fp.confidence else 0.0,
                "part_labels": part_labels,  # Include part labels (mapped to names if possible)
            }

        def raw_param_to_dict(rp):
            """Convert RawParameter to dict."""
            return {
                "id": rp.id,
                "value": float(rp.value),
                "units": rp.units,
                "description": rp.description,
            }

        # Build hierarchical structure
        hierarchical_structure = build_hierarchical_structure(
            ingest_result, ingest_result.part_table
        )

        # Return results with both hierarchical and flat formats
        response_data = {
            "ok": True,
            "category": ingest_result.category,
            # Hierarchical structure: Category -> Parts -> Parameters
            "hierarchical": hierarchical_structure,
            # Flat format for backward compatibility
            "final_parameters": [
                final_param_to_dict(fp) for fp in ingest_result.final_parameters
            ],
            "proposed_parameters": [
                final_param_to_dict(fp) for fp in ingest_result.final_parameters
            ],  # Alias
            "raw_parameters": [
                raw_param_to_dict(rp) for rp in ingest_result.raw_parameters
            ],
            "mesh_path": mesh_path,
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        from werkzeug.exceptions import RequestEntityTooLarge

        # Handle 413 errors specifically with better messaging
        if (
            isinstance(e, RequestEntityTooLarge)
            or "413" in str(e)
            or "RequestEntityTooLarge" in str(type(e).__name__)
        ):
            max_length = current_app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)
            content_length = getattr(request, "content_length", None)
            error_msg = f"Upload too large. "
            if content_length:
                error_msg += f"Received: {content_length / (1024*1024):.1f} MB. "
            error_msg += f"Limit: {max_length / (1024*1024):.1f} MB. "
            error_msg += "Try compressing images or reducing canvas resolution."
            print(f"[ingest_mesh_label] {error_msg}", flush=True)
            return jsonify({"ok": False, "error": error_msg}), 413

        error_msg = f"Mesh labeling/VLM error: {str(e)}"
        print(f"[ingest_mesh_label] {error_msg}", flush=True)
        traceback.print_exc()
        return (
            jsonify(
                {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )


@bp.post("/ingest_mesh")
def ingest_mesh():
    """
    Legacy endpoint: Runs segmentation then VLM (auto-labels parts).
    For new workflow, use /ingest_mesh_segment then /ingest_mesh_label.
    """
    # For backward compatibility, call segment then label
    from run import ingest_mesh_segment, ingest_mesh_label

    # This is a simplified version - full implementation would chain the two endpoints
    return (
        jsonify(
            {
                "ok": False,
                "error": "Legacy endpoint. Please use /ingest_mesh_segment then /ingest_mesh_label",
                "workflow": [
                    "1. POST /ingest_mesh_segment with mesh file",
                    "2. User labels parts in UI",
                    "3. POST /ingest_mesh_label with mesh_path, temp_dir, and part_labels",
                ],
            }
        ),
        400,
    )


@bp.post("/modify_mesh_params")
def modify_mesh_params():
    """
    Use VLM to modify mesh parameters based on natural language instructions.

    Expects JSON: {
        "prompt": "natural language instruction",
        "mesh_path": "...",  # Optional, will use cached if not provided
        "current_parameters": [ ... ]  # Current parameter values
    }
    Returns: { "ok": True, "parameters": { "param_name": new_value, ... } }
    """
    from run import _INGEST_RESULT_CACHE

    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return (
                jsonify({"ok": False, "error": "Missing 'prompt' in request"}),
                400,
            )

        prompt = data.get("prompt", "").strip()
        if not prompt:
            return (
                jsonify({"ok": False, "error": "Prompt cannot be empty"}),
                400,
            )

        # Get mesh path from request or use cached one
        mesh_path = data.get("mesh_path")
        if not mesh_path:
            if _INGEST_RESULT_CACHE:
                mesh_path = list(_INGEST_RESULT_CACHE.keys())[-1]
            else:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "No mesh path provided and no cached ingestion result found.",
                        }
                    ),
                    400,
                )

        # Get current parameters
        current_params = data.get("current_parameters", [])
        if not current_params:
            ingest_result = _INGEST_RESULT_CACHE.get(mesh_path)
            if ingest_result:
                current_params = ingest_result.final_parameters

        # Build parameter context for VLM
        params_context = ""
        if current_params:
            params_list = []
            for p in current_params[:10]:  # Limit to first 10
                semantic_name = (
                    getattr(p, "semantic_name", None)
                    or getattr(p, "name", None)
                    or p.get("semantic_name")
                    or p.get("name", "")
                )
                value = getattr(p, "value", None) or p.get("value", 0)
                units = getattr(p, "units", None) or p.get("units", "m")
                params_list.append(f"- {semantic_name}: {value} {units}")
            params_context = "\n".join(params_list)

        # Build VLM prompt
        vlm_prompt = f"""You are modifying parameters for a 3D mesh object.

Current parameters:
{params_context}

User instruction: {prompt}

Based on the user's instruction, determine which parameters need to change and by how much.
Return ONLY valid JSON in this format:
{{
  "parameters": {{
    "parameter_name": new_value,
    ...
  }},
  "reasoning": "brief explanation of changes"
}}

Example:
{{
  "parameters": {{
    "wing_span": 2.5,
    "chord_length": 0.3
  }},
  "reasoning": "Increased wing span to 2.5m and chord length to 0.3m as requested"
}}"""

        # Call VLM
        from app.services.vlm_service import call_vlm

        response = call_vlm(vlm_prompt, None, expect_json=True)

        # Parse VLM response
        raw = response.get("raw", "")
        import json

        try:
            if isinstance(raw, str):
                parsed = json.loads(raw)
            else:
                parsed = raw

            if not isinstance(parsed, dict) or "parameters" not in parsed:
                raise ValueError("Invalid response format")

            modified_params = parsed.get("parameters", {})
            reasoning = parsed.get("reasoning", "")

            print(f"[modify_mesh_params] VLM reasoning: {reasoning}")
            print(f"[modify_mesh_params] Modified parameters: {modified_params}")

            return jsonify(
                {
                    "ok": True,
                    "parameters": modified_params,
                    "reasoning": reasoning,
                }
            )
        except Exception as e:
            print(f"[modify_mesh_params] Failed to parse VLM response: {e}")
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Failed to parse VLM response: {str(e)}",
                        "raw_response": raw[:500] if isinstance(raw, str) else str(raw),
                    }
                ),
                500,
            )

    except Exception as e:
        import traceback

        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


@bp.post("/apply_mesh_params")
def apply_mesh_params():
    """
    Apply parameter changes to deform a mesh using the full ParametricMeshDeformer.

    Expects JSON: {
        "parameters": { "param_name": value, ... },
        "mesh_path": "...",  # Optional, will use cached if not provided
        "enabled_parts": { part_id: bool, ... }  # Optional, for part add/remove
    }
    Returns: { "ok": True, "glb_path": "...", "message": "..." }
    """
    from run import _INGEST_RESULT_CACHE

    try:
        data = request.get_json()
        if not data or "parameters" not in data:
            return (
                jsonify({"ok": False, "error": "Missing 'parameters' in request"}),
                400,
            )

        parameters = data["parameters"]
        if not isinstance(parameters, dict):
            return (
                jsonify({"ok": False, "error": "'parameters' must be a dictionary"}),
                400,
            )

        # Get mesh path from request or use cached one
        mesh_path = data.get("mesh_path")
        if not mesh_path:
            # Try to find the most recent cached result
            if _INGEST_RESULT_CACHE:
                mesh_path = list(_INGEST_RESULT_CACHE.keys())[-1]
            else:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "No mesh path provided and no cached ingestion result found. Please run mesh ingestion first.",
                        }
                    ),
                    400,
                )

        if not os.path.exists(mesh_path):
            return (
                jsonify({"ok": False, "error": f"Mesh file not found: {mesh_path}"}),
                400,
            )

        # Import mesh deformation modules
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from meshml.mesh_deform import ParametricMeshDeformer, MeshData
        from meshml.semantics.ingest_mesh import (
            build_deformer_from_ingest_result,
            IngestResult,
        )

        # Get cached IngestResult
        ingest_result = _INGEST_RESULT_CACHE.get(mesh_path)
        if not ingest_result:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"No cached ingestion result for {mesh_path}. Please run /ingest_mesh_label first.",
                    }
                ),
                400,
            )

        # Build deformer from ingest result
        deformer = build_deformer_from_ingest_result(ingest_result)

        # Apply parameters
        print(f"[apply_mesh_params] Applying {len(parameters)} parameter changes...")
        deformed_mesh = deformer.deform(
            parameters, enabled_parts=data.get("enabled_parts")
        )

        # Export to GLB
        from run import ASSETS_DIR

        os.makedirs(ASSETS_DIR, exist_ok=True)
        glb_path = os.path.join(ASSETS_DIR, "deformed_mesh.glb")
        deformed_mesh.export(glb_path)

        print(f"[apply_mesh_params] ✓ Deformed mesh saved to {glb_path}")

        # Return URL-accessible path
        glb_url = "/assets/deformed_mesh.glb"

        return jsonify(
            {
                "ok": True,
                "glb_path": glb_url,  # Return URL path, not filesystem path
                "message": "Mesh parameters applied successfully",
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify(
                {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )
