# VLM CAD: Semantic Geometry Analysis

This module provides PointNet++ part segmentation and VLM-powered semantic parameter extraction for 3D meshes.

## Overview

The `vlm_cad` package consists of two main modules:

1. **`pointnet_seg`**: PointNet++ part segmentation for ShapeNetPart
   - Model loading and inference
   - Mesh to point cloud conversion
   - Geometric parameter extraction

2. **`semantics`**: VLM-powered semantic parameter pipeline
   - Pre-VLM: Category classification and candidate parameter generation
   - Post-VLM: Parameter refinement and reconciliation
   - End-to-end mesh ingestion orchestrator

## Installation

The required dependencies are already in `requirements.txt`:
- `torch>=2.0.0` - PyTorch for model inference
- `trimesh` - Mesh I/O and point cloud sampling
- `pillow` - Image processing for mesh rendering
- `numpy` - Numerical operations

## Quick Start

### 1. Part Segmentation (Backend Abstraction)

```python
from vlm_cad.pointnet_seg.model import load_pretrained_model
from vlm_cad.pointnet_seg.inference import segment_mesh
from vlm_cad.pointnet_seg.geometry import compute_part_bounding_boxes

# Load model (download checkpoint from https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
model = load_pretrained_model(
    checkpoint_path="path/to/pointnet2_part_seg_msg.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Segment mesh
result = segment_mesh("mesh.obj", model, num_points=2048)
points = result["points"]
labels = result["labels"]

# Compute geometric properties
bboxes = compute_part_bounding_boxes(points, labels)
```

### 2. Full Ingestion Pipeline with Fine-tuned VLM

```python
from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient
from vlm_cad.semantics.ingest_mesh import ingest_mesh_to_semantic_params

# Initialize VLM client (uses your fine-tuned model from optim.py)
vlm = FinetunedVLMClient()

# Run full pipeline
result = ingest_mesh_to_semantic_params(
    mesh_path="mesh.obj",
    vlm=vlm,
    model=model,
    render_output_dir="renders/",
    num_points=2048,
)

# Access results
print(f"Category: {result.category}")
for param in result.final_parameters:
    print(f"{param.name} = {param.value} {param.units}")
```

### 3. Using Dummy VLM Client (for testing)

```python
from vlm_cad.semantics.vlm_client import DummyVLMClient

# Use dummy client if fine-tuned model is not available
vlm = DummyVLMClient()
```

## Module Structure

### `pointnet_seg/`

- **`model.py`**: PointNet++ model definition and weight loading
- **`mesh_io.py`**: Mesh loading and point cloud sampling
- **`inference.py`**: High-level segmentation API
- **`geometry.py`**: Geometric parameter extraction utilities
- **`labels.py`**: ShapeNetPart label mappings

### `semantics/`

- **`vlm_client.py`**: VLM client abstraction and dummy implementation
- **`vlm_client_finetuned.py`**: Client using your fine-tuned VLM from `optim.py`
- **`semantics_pre.py`**: Pre-VLM category classification
- **`semantics_post.py`**: Post-VLM parameter refinement
- **`ingest_mesh.py`**: End-to-end orchestration
- **`types.py`**: Shared dataclasses (RawParameter, FinalParameter, etc.)

## VLM Integration

### Using Your Fine-tuned VLM

The `FinetunedVLMClient` automatically uses your existing fine-tuned VLM from `cqparts_bucket/optim.py`:

```python
from vlm_cad.semantics.vlm_client_finetuned import FinetunedVLMClient

vlm = FinetunedVLMClient()  # Uses call_vlm() from optim.py
```

This client:
- Automatically loads your fine-tuned model (if not already loaded)
- Converts image file paths to base64 for the VLM
- Parses JSON responses from the VLM
- Handles errors gracefully

### Custom VLM Client

To use a different VLM, implement the `VLMClient` protocol:

```python
from vlm_cad.semantics.vlm_client import VLMClient, VLMMessage, VLMImage

class MyVLMClient:
    def complete_json(self, messages, images=None, schema_hint=None):
        # Call your VLM API here
        # Return a dict parsed from JSON response
        ...
```

## Examples

See `examples/` directory:
- `run_pointnet_seg.py`: Basic segmentation example
- `run_ingest_mesh.py`: Full ingestion pipeline example (uses FinetunedVLMClient by default)

## Pretrained Model

Download the PointNet++ pretrained checkpoint from:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The checkpoint should be a `.pth` file from the `log/` directory (e.g., `pointnet2_part_seg_msg.pth`).

## Notes

- Point clouds are normalized to unit sphere by default
- The model expects 6D input (XYZ + normals) by default
- ShapeNetPart has 50 part classes across 16 object categories
- All geometric parameters are in normalized coordinates unless specified
- The fine-tuned VLM client requires `optim.py` to be importable (i.e., run from the project root)
