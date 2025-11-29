"""
Example script for end-to-end mesh ingestion with VLM semantics.

This script demonstrates the full pipeline:
1. Render mesh views
2. Pre-VLM classification and candidate parameter generation
3. PointNet++ segmentation
4. Raw geometric parameter extraction
5. Post-VLM parameter refinement
"""

from pathlib import Path
import torch
import os

from vlm_cad.pointnet_seg.model import load_pretrained_model
from vlm_cad.semantics.vlm_client import DummyVLMClient
from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient
from vlm_cad.semantics.ingest_mesh import ingest_mesh_to_semantic_params


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
    render_dir = os.path.join(os.path.dirname(__file__), "renders")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Mesh Ingestion Pipeline with VLM Semantics")
    print("=" * 60)
    print(f"Mesh: {mesh_path}")
    print(f"Device: {device}")
    print()
    
    # Initialize VLM client
    # Try to use fine-tuned model, fall back to dummy if not available
    print("Initializing VLM client...")
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
        print("\nTo download:")
        print("1. Clone: git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git")
        print("2. Find the model in log/ directory (pointnet2_part_seg_msg.pth)")
        print(f"3. Copy it to: {checkpoint_path}")
        print("\nOr set POINTNET2_CHECKPOINT environment variable to your model path")
        print("="*60)
        print("\nCannot continue without the model. Exiting.")
        return
    
    # Create render directory
    os.makedirs(render_dir, exist_ok=True)
    
    # Run ingestion pipeline
    print(f"\n{'=' * 60}")
    print("Running ingestion pipeline...")
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
        print(f"\nFinal Semantic Parameters ({len(result.final_parameters)}):")
        print("-" * 60)
        
        for param in result.final_parameters:
            units_str = f" {param.units}" if param.units else ""
            print(f"  • {param.name} = {param.value:.4f}{units_str}")
            print(f"    Description: {param.description}")
            print(f"    Confidence: {param.confidence:.2f}")
            if param.raw_sources:
                print(f"    Raw sources: {', '.join(param.raw_sources)}")
            print()
        
        print(f"\nRaw Parameters ({len(result.raw_parameters)}):")
        print("-" * 60)
        for param in result.raw_parameters[:5]:  # Show first 5
            units_str = f" {param.units}" if param.units else ""
            print(f"  • {param.id} = {param.value:.4f}{units_str}")
        if len(result.raw_parameters) > 5:
            print(f"  ... and {len(result.raw_parameters) - 5} more")
        
        print(f"\n✓ Pipeline complete!")
        print(f"\nRendered images saved to: {render_dir}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

