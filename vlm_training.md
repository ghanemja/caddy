# VLM Fine-Tuning Guide

1. **Model Checkpoints** (`runs/onevision_lora_small/`):
   - `adapter_model.safetensors` - The trained LoRA adapter weights (~13MB)
   - `adapter_config.json` - LoRA configuration
   - `training_args.bin` - Training arguments (can be loaded with transformers)
   - `checkpoint-4/` - Training checkpoint with optimizer states

2. **Training Script** (`train_vlm.py`):
   - Complete fine-tuning script with all hyperparameters from your checkpoint
   - Automatically loads configuration from existing checkpoint
   - Ready to use for fine-tuning new models

## Training Configuration 

### LoRA Parameters
- **Rank (r)**: 4
- **Alpha**: 8
- **Dropout**: 0.05
- **Target Modules**: `["k_proj", "q_proj", "o_proj", "v_proj"]`
- **Task Type**: CAUSAL_LM

### Training Parameters
- **Base Model**: `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- **Batch Size**: 1
- **Num Epochs**: 1
- **Learning Rate**: (default: 2e-4, can be adjusted)
- **Logging Steps**: 20
- **Save Steps**: 500
- **Eval Steps**: 500
- **Max Steps**: 4 (was used for quick test)

## How to Use the Training Script

### Basic Usage

```bash
python train_vlm.py \
  --dataset /path/to/your/dataset.json \
  --output_dir ./runs/onevision_lora_new
```

### Load Configuration from Existing Checkpoint

```bash
python train_vlm.py \
  --dataset /path/to/your/dataset.json \
  --output_dir ./runs/onevision_lora_new \
  --checkpoint_dir ./runs/onevision_lora_small
```

This will automatically load all the training arguments (batch size, epochs, LoRA config, etc.) from the previous checkpoint.

### Resume Training from Checkpoint

```bash
python train_vlm.py \
  --dataset /path/to/your/dataset.json \
  --output_dir ./runs/onevision_lora_small \
  --resume_from_checkpoint ./runs/onevision_lora_small/checkpoint-4
```

### Full Example with All Options

```bash
python train_vlm.py \
  --base_model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --dataset /path/to/training_data.json \
  --output_dir ./runs/onevision_lora_new \
  --checkpoint_dir ./runs/onevision_lora_small \
  --learning_rate 2e-4 \
  --warmup_steps 100 \
  --max_steps 1000 \
  --gradient_accumulation_steps 4 \
  --fp16
```

## Dataset Format

The training dataset should be a JSON file with the following structure:

```json
[
  {
    "image": "/path/to/image1.jpg",
    "text": "User: What do you see?\nAssistant: I see a robot base with wheels..."
  },
  {
    "image": "/path/to/image2.jpg",
    "text": "User: Generate code for this component\nAssistant: import cadquery as cq..."
  }
]
```

Or if using a HuggingFace dataset directory:

```
dataset/
  train/
    images/
      image1.jpg
      image2.jpg
    train.jsonl  # JSON Lines format
```

## Training Arguments Reference

All arguments you can customize:

- `--base_model`: Base model name (default: `llava-hf/llava-onevision-qwen2-7b-ov-hf`)
- `--dataset`: Path to training dataset (required)
- `--output_dir`: Where to save the fine-tuned adapter (default: `./runs/onevision_lora_small`)
- `--checkpoint_dir`: Load training config from existing checkpoint
- `--resume_from_checkpoint`: Resume training from specific checkpoint
- `--learning_rate`: Learning rate (default: 2e-4)
- `--warmup_steps`: Warmup steps (default: 100)
- `--max_steps`: Maximum training steps (optional)
- `--gradient_accumulation_steps`: Accumulate gradients (default: 4)
- `--fp16`: Use FP16 mixed precision
- `--bf16`: Use BF16 mixed precision

## Loading Your Fine-Tuned Model

After training, use it in `optim.py`:

```bash
export FINETUNED_MODEL_PATH=./runs/onevision_lora_new
python cqparts_bucket/optim.py
```

Or set it in the code:
```python
FINETUNED_MODEL_PATH = "./runs/onevision_lora_new"
```

## Notes

- The original training used very few steps (4) - this was likely a test run
- For real training, increase `--max_steps` or `--num_epochs`
- Training on CPU is possible but very slow - use GPU if available
- The script automatically loads LoRA configuration from your checkpoint's `adapter_config.json`


## Summary

1. Prepare your training dataset in JSON format
2. Review and customize `train_vlm.py` as needed
3. Run training with: `python train_vlm.py --dataset training_data.json`
4. Use the new adapter in `optim.py`

