# Fine-Tuned VLM Model Integration

## Summary

The `optim.py` file has been modified to use your fine-tuned VLM model (LLaVA OneVision with LoRA adapter) instead of Ollama by default.

## Changes Made

### 1. **New Configuration Variables** (Lines 51-64)
- `USE_FINETUNED_MODEL`: Enable/disable fine-tuned model (default: enabled)
- `FINETUNED_MODEL_PATH`: Path to your LoRA adapter
- `_finetuned_model` and `_finetuned_processor`: Global variables to store loaded model

### 2. **Model Loading Function** (Lines 67-129)
- `load_finetuned_model()`: Loads the base model and LoRA adapter
- Automatically detects CUDA/CPU
- Uses fp16 on GPU for faster inference
- Falls back gracefully if libraries aren't installed

### 3. **Updated `call_vlm()` Function** (Lines 1505-1582)
- **First tries**: Fine-tuned model (if enabled and loaded)
- **Falls back to**: Ollama → LLAVA URL
- Handles image processing for the fine-tuned model
- Formats prompts in LLaVA OneVision conversation format

### 4. **Startup Integration** (Lines 3107-3111)
- Loads fine-tuned model during app startup
- Runs before Flask server starts

## How It Works

1. **On startup**: The fine-tuned model is loaded into memory
2. **On VLM request**: 
   - Images are decoded from base64
   - Prompt is formatted in chat template format
   - Model generates response using your fine-tuned weights
   - Falls back to Ollama if fine-tuned model fails

## Configuration

### Use Fine-Tuned Model (Default)
```bash
python optim.py
```

### Use Ollama Instead
```bash
USE_FINETUNED_MODEL=0 python optim.py
```

### Custom Model Path
```bash
FINETUNED_MODEL_PATH=/path/to/your/adapter python optim.py
```

## Requirements

The fine-tuned model requires additional dependencies:
```bash
pip install transformers peft torch pillow accelerate
```

## Model Details

- **Base Model**: `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- **Adapter**: LoRA (rank=4, alpha=8)
- **Location**: `/home/ec2-user/Documents/cad-optimizer/runs/onevision_lora_small`
- **Device**: Auto-detected (CUDA if available, else CPU)

## Performance

- **Generation**: Up to 2048 tokens for code, 1024 for JSON
- **Temperature**: 0.1 (slightly higher than Ollama's 0.0 for better diversity)
- **Top-p**: 0.9 (nucleus sampling)

## Fallback Behavior

If the fine-tuned model fails (missing dependencies, GPU OOM, etc.):
1. Error is logged with full traceback
2. System automatically falls back to Ollama
3. No interruption to user experience

## Testing

To test the integration:
```bash
cd /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
python optim.py
```

Look for these startup messages:
```
[startup] Loading fine-tuned VLM model...
[vlm] Loading fine-tuned model from /home/ec2-user/Documents/cad-optimizer/runs/onevision_lora_small...
[vlm] Loading base model: llava-hf/llava-onevision-qwen2-7b-ov-hf
[vlm] Using device: cuda
[vlm] Loading LoRA adapter from /home/ec2-user/Documents/cad-optimizer/runs/onevision_lora_small
[vlm] ✓ Fine-tuned model loaded successfully on cuda
```

## Advantages

1. **Domain-specific**: Your fine-tuned model is optimized for CAD code generation
2. **No external service**: Runs locally, no need for Ollama server
3. **Better control**: Direct access to generation parameters
4. **Consistent**: Same model behavior across runs
5. **Fallback**: Still works if fine-tuned model fails

## Notes

- First load may take 30-60 seconds (downloads base model if needed)
- Requires ~15-20GB GPU memory for full fp16 inference
- On CPU, inference will be significantly slower but still functional


