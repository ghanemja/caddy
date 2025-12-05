#!/usr/bin/env python3
"""
Smoke test for vlm_cad package imports.
This verifies that all public APIs are importable.
"""

print("Testing vlm_cad.pointnet_seg imports...")
try:
    from vlm_cad.pointnet_seg import (
        load_pretrained_model,
        segment_mesh,
        PointNet2PartSegWrapper,
    )
    print("✓ pointnet_seg imports successful")
except ImportError as e:
    print(f"✗ pointnet_seg import failed: {e}")
    raise

print("\nTesting vlm_cad.semantics imports...")
try:
    from vlm_cad.semantics import (
        VLMClient,
        DummyVLMClient,
        RawParameter,
        CandidateParameter,
        FinalParameter,
        PreVLMOutput,
        PostVLMOutput,
        IngestResult,
        ingest_mesh_to_semantic_params,
    )
    print("✓ semantics imports successful")
except ImportError as e:
    print(f"✗ semantics import failed: {e}")
    raise

print("\n✓ All imports successful!")

