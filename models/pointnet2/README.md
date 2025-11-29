# PointNet++ Pretrained Model

## Status: ✓ Model Installed

The pretrained PointNet++ part segmentation model is installed at:
```
models/pointnet2/pointnet2_part_seg_msg.pth
```

**Model Details:**
- **Type**: PointNet++ with Multi-Scale Grouping (MSG)
- **Task**: Part segmentation on ShapeNetPart
- **Size**: 20 MB
- **Source**: `Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth`

## Usage

The model is automatically loaded by the example scripts:
```bash
python examples/run_pointnet_seg.py
python examples/run_ingest_mesh.py
```

## Model Information

This model:
- Uses **normals** (6D input: XYZ + normal vectors)
- Supports **50 part classes** across 16 ShapeNet categories
- Trained on **ShapeNetPart** dataset
- Uses **multi-scale grouping** for better feature extraction

## Other Available Models

The repository also contains other checkpoints (not needed for this project):
- `pointnet2_msg_normals` - Classification with MSG and normals
- `pointnet2_ssg_wo_normals` - Classification with SSG without normals
- `pointnet2_sem_seg` - Semantic segmentation
- `pointnet_sem_seg` - PointNet (not PointNet++) semantic segmentation

