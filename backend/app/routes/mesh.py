"""
Mesh processing routes blueprint
"""
from flask import Blueprint, request, jsonify
import os
import sys
import tempfile
import numpy as np
import trimesh

bp = Blueprint("mesh", __name__)

# Import from run.py for now - will be moved to service modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


@bp.post("/ingest_mesh_segment")
def ingest_mesh_segment():
    """
    Step 1: Run segmentation only (fast, ~1-5 seconds).
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
        from meshml.pointnet_seg.geometry import compute_part_statistics, compute_part_bounding_boxes
        from meshml.parts.parts import build_part_table_from_segmentation, part_table_to_labeling_json
        
        # Create segmentation backend
        print(f"[ingest_mesh_segment] Initializing segmentation backend...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
        print(f"[ingest_mesh_segment] Using segmentation backend: {backend_kind}")
        
        try:
            model = create_segmentation_backend(kind=backend_kind, device=device)
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": f"Failed to initialize segmentation backend '{backend_kind}': {str(e)}"
            }), 500
        
        # Run segmentation only (fast, ~1-5 seconds)
        print(f"[ingest_mesh_segment] Running part segmentation...")
        seg_result = model.segment(mesh_path, num_points=2048)
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
            "parts": []
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
                    if hasattr(v, 'tolist'):
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
            
            segmentation_summary["parts"].append({
                "id": int(label_id_int),
                "name": part_name,
                "point_count": int(count),
                "percentage": float(count / len(points) * 100),
                "bbox": bbox_data,
            })
        
        # Print part details
        print(f"[ingest_mesh_segment] Part breakdown:")
        for part in segmentation_summary["parts"]:
            print(f"[ingest_mesh_segment]   • Part {part['id']} ({part['name']}): {part['point_count']} points ({part['percentage']:.1f}%)")
        
        # Build PartTable for user labeling
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
        # Map point labels to vertex labels (approximate)
        vertex_labels = seg_result.labels
        if len(vertex_labels) != len(vertices):
            if len(vertex_labels) < len(vertices):
                vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
            else:
                vertex_labels = vertex_labels[:len(vertices)]
        
        # Build PartTable
        part_table = build_part_table_from_segmentation(
            vertices=vertices,
            part_labels=vertex_labels,
            ground_plane_z=None,
        )
        part_table_json = part_table_to_labeling_json(part_table)
        
        print(f"[ingest_mesh_segment] ✓ PartTable created with {len(part_table.parts)} parts for user labeling")
        
        # Save colored point cloud visualization
        try:
            colors = np.zeros((len(points), 3))
            for i, label_id in enumerate(labels):
                np.random.seed(int(label_id))
                color = np.random.rand(3)
                colors[i] = color
            pc = trimesh.PointCloud(vertices=points, colors=colors)
            viz_path = os.path.join(temp_dir, "segmentation_colored.ply")
            pc.export(viz_path)
            segmentation_summary["visualization_path"] = viz_path
        except Exception as e:
            print(f"[ingest_mesh_segment] Warning: Could not save visualization: {e}")
            segmentation_summary["visualization_path"] = None
        
        # Include vertex labels for coloring (convert to list for JSON)
        vertex_labels_list = vertex_labels.tolist() if hasattr(vertex_labels, 'tolist') else list(vertex_labels)
        
        # Return segmentation results only (user will label parts, then call /ingest_mesh_label)
        response_data = {
            "ok": True,
            "segmentation": segmentation_summary,
            "part_table": part_table_json,
            "vertex_labels": vertex_labels_list,
            "mesh_path": mesh_path,
            "temp_dir": temp_dir,
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Mesh segmentation error: {str(e)}"
        print(f"[ingest_mesh_segment] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@bp.post("/convert_mesh_to_glb")
def convert_mesh_to_glb():
    """
    Convert uploaded mesh file (STL/PLY/OBJ) to GLB format for display in viewer.
    Returns the GLB file directly as binary response (not JSON).
    """
    from flask import send_file, Response
    try:
        if 'mesh' not in request.files:
            return jsonify({"ok": False, "error": "No mesh file provided"}), 400
        
        mesh_file = request.files['mesh']
        if not mesh_file.filename:
            return jsonify({"ok": False, "error": "Empty filename"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(mesh_file.filename)[1]) as tmp:
            mesh_file.save(tmp.name)
            mesh_path = tmp.name
        
        try:
            # Load mesh
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Ensure mesh is valid
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                return jsonify({"ok": False, "error": "Invalid mesh: no vertices"}), 400
            
            # Export as GLB (ensure GLTF 2.0 format)
            import io
            glb_buffer = io.BytesIO()
            
            # Use trimesh's export with explicit GLTF 2.0 settings
            try:
                # Try exporting as GLB with GLTF 2.0 format
                mesh.export(file_obj=glb_buffer, file_type='glb')
            except Exception as e:
                print(f"[convert_mesh] GLB export failed: {e}, trying alternative method")
                # Fallback: export as GLTF then convert, or use scene export
                scene = trimesh.Scene([mesh])
                scene.export(file_obj=glb_buffer, file_type='glb')
            
            glb_buffer.seek(0)
            glb_data = glb_buffer.read()
            
            if len(glb_data) == 0:
                return jsonify({"ok": False, "error": "Failed to generate GLB file"}), 500
            
            # Verify GLB format (should start with "glTF" magic or be valid binary)
            # GLB files should be at least 12 bytes (header)
            if len(glb_data) < 12:
                return jsonify({"ok": False, "error": "Generated GLB file is too small"}), 500
            
            # Return GLB file directly as binary response
            return Response(
                glb_data,
                mimetype='model/gltf-binary',
                headers={
                    'Content-Disposition': f'attachment; filename=converted_mesh.glb',
                    'Content-Length': str(len(glb_data))
                }
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
        return jsonify({
            "ok": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }), 500


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
    try:
        import json
        from pathlib import Path

        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "error": "JSON body required"}), 400
        
        mesh_path = data.get("mesh_path")
        temp_dir = data.get("temp_dir")
        part_labels_json = data.get("part_labels")  # User-provided labels
        
        if not mesh_path or not os.path.exists(mesh_path):
            return jsonify({"ok": False, "error": "mesh_path required and must exist"}), 400
        
        print(f"[ingest_mesh_label] Processing mesh with user labels: {mesh_path}", flush=True)
        
        # Import pipeline
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from meshml.segmentation import create_segmentation_backend
        from meshml.semantics.vlm_client_finetuned import FinetunedVLMClient
        from meshml.semantics.vlm_client_ollama import OllamaVLMClient
        from meshml.semantics.vlm_client import DummyVLMClient
        from meshml.semantics.ingest_mesh import ingest_mesh_to_semantic_params
        from meshml.parts.parts import build_part_table_from_segmentation, apply_labels_from_json
        import torch
        
        # Initialize segmentation backend (reuse from step 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
        model = create_segmentation_backend(kind=backend_kind, device=device)
        
        # Re-run segmentation to get PartTable (or we could cache it from step 1)
        seg_result = model.segment(mesh_path, num_points=2048)
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
        vertex_labels = seg_result.labels
        if len(vertex_labels) != len(vertices):
            if len(vertex_labels) < len(vertices):
                vertex_labels = np.pad(vertex_labels, (0, len(vertices) - len(vertex_labels)), mode='edge')
            else:
                vertex_labels = vertex_labels[:len(vertices)]
        
        # Build PartTable
        part_table = build_part_table_from_segmentation(
            vertices=vertices,
            part_labels=vertex_labels,
            ground_plane_z=None,
        )
        
        # Apply user-provided labels
        if part_labels_json:
            part_table = apply_labels_from_json(part_table, part_labels_json)
            print(f"[ingest_mesh_label] Applied user labels to {len(part_labels_json.get('parts', []))} parts", flush=True)
        
        # Initialize VLM (same logic as before)
        vlm = None
        if device == "cpu":
            try:
                vlm = OllamaVLMClient()
                print("[ingest_mesh_label] Using Ollama VLM (fast on CPU)")
            except Exception as e:
                print(f"[ingest_mesh_label] Warning: Could not use Ollama: {e}")
            try:
                vlm = FinetunedVLMClient()
                print("[ingest_mesh_label] Using fine-tuned VLM (slower on CPU)")
            except Exception as e2:
                print(f"[ingest_mesh_label] Warning: Could not use fine-tuned VLM: {e2}")
                vlm = DummyVLMClient()
                print("[ingest_mesh_label] Using dummy VLM (for testing)")
        else:
            try:
                vlm = FinetunedVLMClient()
                print("[ingest_mesh_label] Using fine-tuned VLM (pretrained model on GPU)")
            except Exception as e:
                print(f"[ingest_mesh_label] Warning: Could not use fine-tuned VLM: {e}")
                try:
                    vlm = OllamaVLMClient()
                    print("[ingest_mesh_label] Using Ollama VLM (fallback)")
                except Exception as e2:
                    print(f"[ingest_mesh_label] Warning: Could not use Ollama: {e2}")
                    vlm = DummyVLMClient()
                    print("[ingest_mesh_label] Using dummy VLM (for testing)")
        
        # Run ingestion pipeline with user-labeled PartTable
        render_dir = os.path.join(temp_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)
        
        print(f"[ingest_mesh_label] Running VLM-based semantic parameter extraction...")
        ingest_result = ingest_mesh_to_semantic_params(
            mesh_path=mesh_path,
            part_table=part_table,
            vlm=vlm,
            render_dir=render_dir,
            device=device,
        )
        
        # Cache result for /apply_mesh_params
        _INGEST_RESULT_CACHE[mesh_path] = ingest_result
        
        print(f"[ingest_mesh_label] ✓ Semantic parameter extraction complete!")
        print(f"[ingest_mesh_label]   Category: {ingest_result.category}")
        print(f"[ingest_mesh_label]   Parameters: {len(ingest_result.final_parameters)} semantic params")
        
        # Return results
        response_data = {
            "ok": True,
            "category": ingest_result.category,
            "final_parameters": ingest_result.final_parameters,
            "raw_parameters": ingest_result.raw_parameters,
            "mesh_path": mesh_path,
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Mesh labeling/VLM error: {str(e)}"
        print(f"[ingest_mesh_label] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@bp.post("/ingest_mesh")
def ingest_mesh():
    """
    Legacy endpoint: Runs segmentation then VLM (auto-labels parts).
    For new workflow, use /ingest_mesh_segment then /ingest_mesh_label.
    """
    # For backward compatibility, call segment then label
    from run import ingest_mesh_segment, ingest_mesh_label
    # This is a simplified version - full implementation would chain the two endpoints
    return jsonify({
        "ok": False,
        "error": "Legacy endpoint. Please use /ingest_mesh_segment then /ingest_mesh_label",
        "workflow": [
            "1. POST /ingest_mesh_segment with mesh file",
            "2. User labels parts in UI",
            "3. POST /ingest_mesh_label with mesh_path, temp_dir, and part_labels"
        ]
    }), 400


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
            return jsonify({"ok": False, "error": "Missing 'parameters' in request"}), 400
        
        parameters = data["parameters"]
        if not isinstance(parameters, dict):
            return jsonify({"ok": False, "error": "'parameters' must be a dictionary"}), 400
        
        # Get mesh path from request or use cached one
        mesh_path = data.get("mesh_path")
        if not mesh_path:
            # Try to find the most recent cached result
            if _INGEST_RESULT_CACHE:
                mesh_path = list(_INGEST_RESULT_CACHE.keys())[-1]
            else:
                return jsonify({
                    "ok": False,
                    "error": "No mesh path provided and no cached ingestion result found. Please run mesh ingestion first."
                }), 400
        
        if not os.path.exists(mesh_path):
            return jsonify({
                "ok": False,
                "error": f"Mesh file not found: {mesh_path}"
            }), 400
        
        # Import mesh deformation modules
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from meshml.mesh_deform import ParametricMeshDeformer, MeshData
        from meshml.semantics.ingest_mesh import build_deformer_from_ingest_result, IngestResult
        
        # Get cached IngestResult
        ingest_result = _INGEST_RESULT_CACHE.get(mesh_path)
        if not ingest_result:
            return jsonify({
                "ok": False,
                "error": f"No cached ingestion result for {mesh_path}. Please run /ingest_mesh_label first."
            }), 400
        
        # Build deformer from ingest result
        deformer = build_deformer_from_ingest_result(ingest_result)
        
        # Apply parameters
        print(f"[apply_mesh_params] Applying {len(parameters)} parameter changes...")
        deformed_mesh = deformer.deform(parameters, enabled_parts=data.get("enabled_parts"))
        
        # Export to GLB
        from run import ASSETS_DIR
        os.makedirs(ASSETS_DIR, exist_ok=True)
        glb_path = os.path.join(ASSETS_DIR, "deformed_mesh.glb")
        deformed_mesh.export(glb_path)
        
        print(f"[apply_mesh_params] ✓ Deformed mesh saved to {glb_path}")
        
        return jsonify({
            "ok": True,
            "glb_path": glb_path,
            "message": "Mesh parameters applied successfully"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
