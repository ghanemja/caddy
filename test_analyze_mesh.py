#!/usr/bin/env python3
"""
Test script for mesh analysis (PointNet++ segmentation + VLM semantics).

This script replicates the /ingest_mesh endpoint logic but runs standalone,
so you can test without the GUI.
"""

import os
import sys
import tempfile
import torch
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from vlm_cad.pointnet_seg.model import load_pretrained_model
from vlm_cad.pointnet_seg.inference import segment_mesh
from vlm_cad.pointnet_seg.labels import get_category_from_flat_label
from vlm_cad.pointnet_seg.geometry import compute_part_statistics, compute_part_bounding_boxes
from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient
from vlm_cad.semantics.vlm_client_ollama import OllamaVLMClient
from vlm_cad.semantics.vlm_client import DummyVLMClient
from vlm_cad.semantics.ingest_mesh import ingest_mesh_to_semantic_params


def main():
    # Configuration
    checkpoint_path = os.path.join(
        parent_dir, "models", "pointnet2", "pointnet2_part_seg_msg.pth"
    )
    
    # Use the STL file from Downloads
    mesh_path = os.path.join(
        os.path.expanduser("~/Downloads"),
        "Curiosity Rover 3D Printed Model",
        "Simplified Curiosity Model (Small)",
        "STL Files",
        "body-small.STL"
    )
    
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        print("\nPlease update the mesh_path in the script or set MESH_PATH environment variable")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Mesh Analysis Test Script")
    print("=" * 60)
    print(f"Mesh: {mesh_path}")
    print(f"Device: {device}")
    print()
    
    # Load PointNet++ model
    print(f"Loading PointNet++ model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
    
    try:
        model = load_pretrained_model(
            checkpoint_path=checkpoint_path,
            num_classes=50,
            use_normals=True,
            device=device,
        )
        print("✓ PointNet++ model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize VLM client (same logic as GUI)
    vlm = None
    print("\nInitializing VLM client...")
    
    # On CPU, prefer Ollama (much faster)
    if device == "cpu":
        print("CPU detected - trying Ollama first for faster inference")
        ollama_available = False
        ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        try:
            import requests
            r = requests.get(f"{ollama_url}/api/tags", timeout=2)
            ollama_available = (r.status_code == 200)
        except:
            pass
        
        if ollama_available:
            try:
                vlm = OllamaVLMClient()
                print("✓ Using Ollama VLM (fast on CPU)")
            except Exception as e:
                print(f"⚠ Could not use Ollama: {e}")
        
        # Fallback to fine-tuned model if Ollama not available
        if vlm is None:
            try:
                vlm = FinetunedVLMClient()
                print("✓ Using fine-tuned VLM (slower on CPU, may take 2-5 minutes)")
            except Exception as e:
                print(f"⚠ Could not use fine-tuned VLM: {e}")
                vlm = DummyVLMClient()
                print("✓ Using dummy VLM (for testing)")
    else:
        # On GPU, prefer fine-tuned model
        try:
            vlm = FinetunedVLMClient()
            print("✓ Using fine-tuned VLM (pretrained model on GPU)")
        except Exception as e:
            print(f"⚠ Could not use fine-tuned VLM: {e}")
            # Fall back to Ollama
            ollama_available = False
            ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
            try:
                import requests
                r = requests.get(f"{ollama_url}/api/tags", timeout=2)
                ollama_available = (r.status_code == 200)
            except:
                pass
            
            if ollama_available:
                try:
                    vlm = OllamaVLMClient()
                    print("✓ Using Ollama VLM (fallback)")
                except Exception as e2:
                    print(f"⚠ Could not use Ollama: {e2}")
                    vlm = DummyVLMClient()
                    print("✓ Using dummy VLM (for testing)")
    
    # Create temp directory for renders
    temp_dir = tempfile.mkdtemp(prefix="mesh_ingest_")
    render_dir = os.path.join(temp_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print("Step 1: PointNet++ Segmentation (fast, ~1-5 seconds)")
    print(f"{'=' * 60}\n")
    
    # Run PointNet++ segmentation FIRST (fast, ~1-5 seconds)
    try:
        seg_result = segment_mesh(
            mesh_path,
            model,
            num_points=2048,
            return_logits=False,
        )
        points = seg_result["points"]
        labels = seg_result["labels"]
        unique_labels = np.unique(labels)
        
        print(f"✓ PointNet++ segmentation complete!")
        print(f"  Segmented into {len(unique_labels)} parts")
        print(f"  Point cloud: {len(points)} points")
        
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
        
        # Print part breakdown
        print(f"\nPart breakdown:")
        for label_id in unique_labels:
            label_id_int = int(label_id)
            part_name = part_label_names.get(label_id_int, f"part_{label_id_int}")
            count = np.sum(labels == label_id_int)
            percentage = count / len(points) * 100
            bbox = part_bboxes.get(label_id_int, {})
            
            print(f"  • Part {label_id_int} ({part_name}): {count} points ({percentage:.1f}%)")
            if bbox and "extent" in bbox:
                extent = bbox["extent"]
                print(f"    BBox: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")
        
        # Save colored point cloud visualization
        try:
            import trimesh
            colors = np.zeros((len(points), 3))
            for i, label_id in enumerate(labels):
                np.random.seed(int(label_id))
                color = np.random.rand(3)
                colors[i] = color
            
            pc = trimesh.PointCloud(vertices=points, colors=colors)
            viz_path = os.path.join(temp_dir, "segmentation_colored.ply")
            pc.export(viz_path)
            print(f"\n✓ Saved colored point cloud visualization: {viz_path}")
        except Exception as e:
            print(f"⚠ Could not save visualization: {e}")
        
    except Exception as e:
        print(f"✗ PointNet++ segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now run full ingestion pipeline (includes VLM calls)
    print(f"\n{'=' * 60}")
    print("Step 2: Full Ingestion Pipeline (VLM calls may take 30s-5min)")
    print(f"{'=' * 60}\n")
    
    try:
        result = ingest_mesh_to_semantic_params(
            mesh_path=mesh_path,
            vlm=vlm,
            model=model,
            render_output_dir=render_dir,
            num_points=2048,
        )
        
        # Display results
        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}\n")
        
        print(f"Category: {result.category}")
        print(f"\nRaw Parameters ({len(result.raw_parameters)}):")
        print("-" * 60)
        for param in result.raw_parameters[:10]:
            units_str = f" {param.units}" if param.units else ""
            print(f"  • {param.id}: {param.value:.4f}{units_str} - {param.description}")
        if len(result.raw_parameters) > 10:
            print(f"  ... and {len(result.raw_parameters) - 10} more")
        
        print(f"\nProposed Semantic Parameters ({len(result.proposed_parameters)}):")
        print("-" * 60)
        for param in result.proposed_parameters:
            units_str = f" {param.units}" if param.units else ""
            print(f"  • {param.semantic_name} = {param.value:.4f}{units_str}")
            print(f"    Description: {param.description}")
            print(f"    Confidence: {param.confidence:.2f}")
            if param.raw_sources:
                print(f"    Raw sources: {', '.join(param.raw_sources)}")
            print()
        
        print(f"\n✓ Pipeline complete!")
        print(f"\nOutput files:")
        print(f"  • Rendered images: {render_dir}")
        print(f"  • Colored point cloud: {os.path.join(temp_dir, 'segmentation_colored.ply')}")
        print(f"  • Temp directory: {temp_dir}")
        
    except Exception as e:
        print(f"\n✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

