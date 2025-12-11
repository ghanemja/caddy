# Development Notes

This file contains technical notes, change logs, and development information.

## Table of Contents

1. [Segmentation Backends](#segmentation-backends)
2. [VLM Integration](#vlm-integration)
3. [Parameter System Changes](#parameter-system-changes)
4. [PointNet++ & VLM Audit](#pointnet--vlm-audit)
5. [VLM Training](#vlm-training)

---

## Segmentation Backends

See `SEGMENTATION_BACKENDS.md` for complete documentation on the segmentation backend abstraction layer.

**Quick Summary:**
- Abstraction layer supports multiple backends (PointNet++, Hunyuan3D-Part)
- Switch via `SEGMENTATION_BACKEND=pointnet|hunyuan3d` env var
- PointNet++ fully functional, Hunyuan3D-Part structure ready (needs P3-SAM API implementation)

---

## VLM Integration

### VLM Prompts Update

**System Prompts:**
- **Pre-VLM**: Expert 3D object analyst prompt that identifies category, parts, and proposes candidate parameters
- **Post-VLM**: Semantic naming prompt that proposes names for generic parameters (p1, p2, ...) based on category and geometry

**Output Formats:**
- Pre-VLM: `{category, parts, candidate_parameters}`
- Post-VLM: `{parameters: [{id, proposed_name, proposed_description, confidence}]}`

**Status:** ✅ Fully implemented and tested

### VLM Clients

- **FinetunedVLMClient**: Uses fine-tuned LLaVA model with LoRA adapter
- **OllamaVLMClient**: Uses Ollama server (faster on CPU)
- **DummyVLMClient**: Fallback for testing

**Configuration:**
```bash
export USE_FINETUNED_MODEL=1  # Use fine-tuned model
export FINETUNED_MODEL_PATH=/path/to/adapter
export OLLAMA_URL=http://127.0.0.1:11434
```

---

## Parameter System Changes

### Generic Parameter IDs (p1, p2, ...)

**Change:** Switched from semantic parameter names to generic IDs with VLM-based semantic naming.

**Flow:**
1. Geometry extraction → Creates `RawParameter` with generic IDs (p1, p2, p3, ...)
2. Pre-VLM → Classifies category and proposes candidate parameter names
3. Post-VLM → Proposes semantic names for generic parameters
4. Output → Both raw (p1, p2, ...) and proposed (wing_span, chord_length, ...) parameters

**Backward Compatibility:** ✅ Maintained
- `FinalParameter.name` property returns `semantic_name`
- `IngestResult.final_parameters` property returns `proposed_parameters`

**Files Changed:**
- `vlm_cad/semantics/types.py` - Added generic IDs to RawParameter, renamed FinalParameter.name to semantic_name
- `vlm_cad/semantics/ingest_mesh.py` - Updated to use generic IDs
- `vlm_cad/semantics/semantics_post.py` - Updated VLM prompt for semantic naming

---

## PointNet++ & VLM Audit

**Date:** 2025-01-29  
**Status:** ✅ Complete

### Module Structure

**PointNet++ Modules:**
- `vlm_cad/pointnet_seg/model.py` - Model definition and loading
- `vlm_cad/pointnet_seg/inference.py` - Segmentation functions
- `vlm_cad/pointnet_seg/geometry.py` - Geometry utilities
- `vlm_cad/pointnet_seg/labels.py` - ShapeNetPart label mappings

**Semantics Modules:**
- `vlm_cad/semantics/vlm_client*.py` - VLM client implementations
- `vlm_cad/semantics/types.py` - Data structures
- `vlm_cad/semantics/semantics_pre.py` - Pre-VLM classification
- `vlm_cad/semantics/semantics_post.py` - Post-VLM semantic naming
- `vlm_cad/semantics/ingest_mesh.py` - Main ingestion pipeline

### Known Issues

1. **Category Mapping**: Model uses flat label space (0-49) but expects 16-dim one-hot category vector. Currently uses index 0 as default.
2. **OpenMP Warning**: Environment has OpenMP library conflicts. Suppress with `KMP_DUPLICATE_LIB_OK=TRUE`.

### TODOs

- Validate ShapeNetPart label mapping
- Tune VLM prompts for better JSON reliability
- Implement proper category-to-index mapping
- Add more robust error handling

---

## VLM Training

### Training Configuration

**LoRA Parameters:**
- Rank (r): 4
- Alpha: 8
- Dropout: 0.05
- Target Modules: `["k_proj", "q_proj", "o_proj", "v_proj"]`

**Training Parameters:**
- Base Model: `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- Batch Size: 1
- Learning Rate: 2e-4
- Max Steps: 4 (test run - increase for production)

### Training Script

**Usage:**
```bash
python train_vlm.py \
  --dataset /path/to/dataset.json \
  --output_dir ./runs/onevision_lora_new \
  --checkpoint_dir ./runs/onevision_lora_small  # Load config from existing
```

**Dataset Format:**
```json
[
  {
    "image": "/path/to/image.jpg",
    "text": "User: What do you see?\nAssistant: I see a robot base..."
  }
]
```

**Checkpoints:**
- `runs/onevision_lora_small/adapter_model.safetensors` - LoRA weights
- `runs/onevision_lora_small/checkpoint-4/` - Training checkpoint

### Loading Fine-Tuned Model

```bash
export FINETUNED_MODEL_PATH=./runs/onevision_lora_small
python cqparts_bucket/optim.py
```

---

## Repository Migration

**Note:** This was a one-time task. See `MIGRATE_REPO.md` for details if needed.

**Summary:** Used `git filter-branch` to unify commit authors and create a new repository with clean history.

---

## Quick Reference

### Environment Variables

```bash
# Segmentation
export SEGMENTATION_BACKEND=pointnet  # or hunyuan3d
export POINTNET2_CHECKPOINT=/path/to/model.pth

# VLM
export USE_FINETUNED_MODEL=1
export FINETUNED_MODEL_PATH=/path/to/adapter
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llava:latest

# Server
export PORT=5160
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP conflicts (macOS)
```

### Key Files

- `cqparts_bucket/optim.py` - Main server
- `vlm_cad/segmentation/` - Segmentation backends
- `vlm_cad/semantics/` - VLM integration
- `vlm_cad/pointnet_seg/` - PointNet++ implementation
- `train_vlm.py` - VLM training script

