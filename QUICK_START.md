# Quick Start Guide

## Running the Mesh Analysis Pipeline

### Option 1: GUI (Recommended for Interactive Use)

1. **Start the server:**
   ```bash
   cd cqparts_bucket
   python optim.py
   ```

2. **Open the GUI:**
   - Open your browser to: `http://localhost:5160`
   - Click "Analyze Mesh" button
   - Upload a mesh file (OBJ, STL, PLY, GLB)
   - Wait for analysis to complete (~30 seconds to 5 minutes depending on VLM)

3. **What happens:**
   - Segmentation runs (PointNet++ or Hunyuan3D-Part, configurable via `SEGMENTATION_BACKEND` env var)
   - PartTable is automatically built with part metadata
   - VLM analyzes the mesh and proposes semantic parameters
   - Results include:
     - Category classification
     - Raw geometric parameters
     - Proposed semantic parameters (with names like "wing_span", "seat_height", etc.)
     - **PartTable JSON** (new!) - part metadata for labeling UI

### Option 2: Command Line (For Testing/Scripting)

1. **Test mesh analysis without GUI:**
   ```bash
   python test_analyze_mesh.py
   ```

2. **Export part labels for labeling UI:**
   ```bash
   python examples/export_part_labels.py
   ```
   This creates `examples/output/<mesh_name>_part_labels.json` that you can use in a labeling UI.

3. **Run full ingestion pipeline:**
   ```bash
   python examples/run_ingest_mesh.py
   ```

## Configuration

### Segmentation Backend

Choose between PointNet++ (default) or Hunyuan3D-Part:

```bash
# Use PointNet++ (default)
export SEGMENTATION_BACKEND=pointnet

# Use Hunyuan3D-Part (when implemented)
export SEGMENTATION_BACKEND=hunyuan3d
```

### VLM Model

The system automatically tries:
1. Fine-tuned model (if `USE_FINETUNED_MODEL=1` and model available)
2. Ollama (if running and `USE_FINETUNED_MODEL=0`)
3. Dummy VLM (fallback for testing)

## What's New: Part Metadata System

The pipeline now automatically:
- ✅ Builds **PartTable** from segmentation (geometry metadata per part)
- ✅ Includes part info in VLM prompts (so VLM can reference parts by name/geometry)
- ✅ Exports part labels as JSON for human labeling
- ✅ Supports per-part operations in mesh deformation (enable/disable, scale, offset)

### Using Part Labels

After running analysis, you can:
1. **View part metadata** in the JSON response (`part_table` field)
2. **Export for labeling** using `examples/export_part_labels.py`
3. **Apply human labels** using `apply_labels_from_json()` from `vlm_cad.parts`
4. **Use in deformation** - parts can be referenced by name or ID for operations

## Troubleshooting

- **"ModuleNotFoundError: No module named 'torch'"**: Activate conda environment: `conda activate vlm_optimizer`
- **"PointNet++ model not found"**: Download checkpoint and set `POINTNET2_CHECKPOINT` env var
- **VLM taking too long**: On CPU, it can take 5-10 minutes. Consider using Ollama or GPU.
- **Segmentation fails**: Check that mesh file is valid and `SEGMENTATION_BACKEND` is set correctly

