# Segmentation Backends Guide

This document covers the segmentation backend system, how to use it, and how to complete the Hunyuan3D-Part integration.

## Quick Start

### Using PointNet++ (Default)

```python
from vlm_cad.segmentation import create_segmentation_backend

backend = create_segmentation_backend()  # Defaults to PointNet
result = backend.segment("mesh.obj")
print(f"Detected {result.num_parts} parts")
```

### Switching Backends

**Environment Variable:**
```bash
export SEGMENTATION_BACKEND=pointnet    # or hunyuan3d
python your_script.py
```

**In Code:**
```python
backend = create_segmentation_backend(kind="hunyuan3d")
```

## Available Backends

### 1. PointNet++ (Default)

**Status:** ✅ Fully functional

**Requirements:**
- PointNet++ checkpoint at `models/pointnet2/pointnet2_part_seg_msg.pth`
- Or set `POINTNET2_CHECKPOINT` environment variable

**Usage:**
```python
backend = create_segmentation_backend(kind="pointnet")
result = backend.segment("mesh.obj", num_points=2048)
```

### 2. Hunyuan3D-Part (P3-SAM)

**Status:** ⚠️ Structure ready, needs P3-SAM API implementation

**Requirements:**
- Hunyuan3D-Part repository: https://github.com/Tencent-Hunyuan/Hunyuan3D-Part
- Model from HuggingFace: `tencent/Hunyuan3D-Part`
- GPU recommended

**Installation:**
```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git
cd Hunyuan3D-Part/P3-SAM
pip install -r requirements.txt
```

**To Complete Implementation:**

1. **Study P3-SAM API** in the cloned repository
2. **Implement `_load_model()`** in `vlm_cad/segmentation/backends.py`:
   ```python
   # Load P3-SAM model from HuggingFace or checkpoint
   # Move to device and set to eval mode
   ```
3. **Implement `segment()`** method:
   ```python
   # Load mesh, run P3-SAM inference, return PartSegmentationResult
   ```
4. **Test** with sample meshes

See implementation TODOs in `vlm_cad/segmentation/backends.py` for details.

## Architecture

```
PartSegmentationBackend (Protocol)
├── segment(mesh_path) -> PartSegmentationResult
│
├── PointNetSegmentationBackend ✅
│   └── Wraps existing PointNet++ code
│
└── Hunyuan3DPartSegmentationBackend ⚠️
    └── P3-SAM integration (needs implementation)
```

## PartSegmentationResult

All backends return this unified result:

```python
@dataclass
class PartSegmentationResult:
    labels: np.ndarray          # [N] per-point/vertex labels
    points: Optional[np.ndarray] # [N, 3] point cloud
    vertex_labels: Optional[np.ndarray]  # [M] per-vertex labels
    face_labels: Optional[np.ndarray]    # [F] per-face labels
    logits: Optional[np.ndarray]         # [N, num_classes]
    num_parts: int
    num_points: int
    # ... additional metadata
```

## Migration from Direct PointNet

**Before:**
```python
from vlm_cad.pointnet_seg.model import load_pretrained_model
from vlm_cad.pointnet_seg.inference import segment_mesh

model = load_pretrained_model(...)
result = segment_mesh("mesh.obj", model)
labels = result["labels"]
```

**After:**
```python
from vlm_cad.segmentation import create_segmentation_backend

backend = create_segmentation_backend(kind="pointnet")
result = backend.segment("mesh.obj")
labels = result.labels
```

**Note:** Backward compatibility is maintained - existing code still works.

## Troubleshooting

### Backend Not Found
**Error:** `Unknown segmentation backend: X`

**Solution:** Use `pointnet` or `hunyuan3d` (case-insensitive)

### PointNet Model Not Found
**Error:** `PointNet++ model not found`

**Solution:**
1. Download from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
2. Place at `models/pointnet2/pointnet2_part_seg_msg.pth`
3. Or set `POINTNET2_CHECKPOINT` env var

### Hunyuan3D Not Implemented
**Error:** `NotImplementedError: Hunyuan3D-Part integration in progress`

**Solution:** Follow implementation steps above. The structure is ready in `vlm_cad/segmentation/backends.py`.

## Testing

```bash
# Test backends
python tests/test_segmentation_backends.py

# Test with specific backend
export SEGMENTATION_BACKEND=pointnet
python test_analyze_mesh.py
```

## Files

- **Abstraction:** `vlm_cad/segmentation/` - Core abstraction layer
- **PointNet Backend:** `vlm_cad/segmentation/backends.py` - PointNetSegmentationBackend
- **Hunyuan3D Backend:** `vlm_cad/segmentation/backends.py` - Hunyuan3DPartSegmentationBackend (needs implementation)
- **Tests:** `tests/test_segmentation_backends.py`

## References

- **PointNet++**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **Hunyuan3D-Part**: https://github.com/Tencent-Hunyuan/Hunyuan3D-Part
- **P3-SAM**: https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/tree/main/P3-SAM
- **HuggingFace**: https://huggingface.co/tencent/Hunyuan3D-Part

