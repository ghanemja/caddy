#!/usr/bin/env python3
"""
Fine-tuning script for LLaVA OneVision model with LoRA adapter.
Based on the training configuration from runs/onevision_lora_small/
"""

import os
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import json


def load_training_args_from_checkpoint(checkpoint_path: str) -> dict:
    """Load training arguments from the checkpoint."""
    try:
        # Try to load from trainer_state.json
        state_path = Path(checkpoint_path) / "checkpoint-4" / "trainer_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                print("Found trainer_state.json with training info:")
                print(f"  - Train batch size: {state.get('train_batch_size', 'N/A')}")
                print(f"  - Num train epochs: {state.get('num_train_epochs', 'N/A')}")
                print(f"  - Logging steps: {state.get('logging_steps', 'N/A')}")
                print(f"  - Save steps: {state.get('save_steps', 'N/A')}")
                print(f"  - Eval steps: {state.get('eval_steps', 'N/A')}")
        
        # Load adapter config for LoRA parameters
        adapter_config_path = Path(checkpoint_path) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
                print("\nLoRA Configuration from adapter_config.json:")
                print(f"  - r (rank): {adapter_config.get('r', 'N/A')}")
                print(f"  - lora_alpha: {adapter_config.get('lora_alpha', 'N/A')}")
                print(f"  - lora_dropout: {adapter_config.get('lora_dropout', 'N/A')}")
                print(f"  - target_modules: {adapter_config.get('target_modules', 'N/A')}")
                return {
                    "lora_r": adapter_config.get("r", 4),
                    "lora_alpha": adapter_config.get("lora_alpha", 8),
                    "lora_dropout": adapter_config.get("lora_dropout", 0.05),
                    "target_modules": adapter_config.get("target_modules", ["k_proj", "q_proj", "o_proj", "v_proj"]),
                    "train_batch_size": state.get("train_batch_size", 1),
                    "num_epochs": state.get("num_train_epochs", 1),
                    "logging_steps": state.get("logging_steps", 20),
                    "save_steps": state.get("save_steps", 500),
                    "eval_steps": state.get("eval_steps", 500),
                }
    except Exception as e:
        print(f"Warning: Could not load from checkpoint: {e}")
    
    # Return defaults based on what we found
    return {
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": ["k_proj", "q_proj", "o_proj", "v_proj"],
        "train_batch_size": 1,
        "num_epochs": 1,
        "logging_steps": 20,
        "save_steps": 500,
        "eval_steps": 500,
    }


def create_lora_config(checkpoint_config: dict):
    """Create LoRA configuration based on checkpoint."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=checkpoint_config["lora_r"],
        lora_alpha=checkpoint_config["lora_alpha"],
        lora_dropout=checkpoint_config["lora_dropout"],
        target_modules=checkpoint_config["target_modules"],
        bias="none",
    )


def prepare_dataset(dataset_path: str):
    """Prepare the training dataset.
    
    Expected format: JSON file with image paths and text prompts/responses.
    Each item should have:
    - "image": path to image file
    - "text": instruction/prompt + response
    """
    if dataset_path.endswith('.json'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    else:
        # Assume it's a directory with dataset files
        dataset = load_dataset(dataset_path, split='train')
    
    return dataset


def preprocess_function(examples, processor):
    """Preprocess images and texts for the model."""
    images = [Image.open(img_path).convert('RGB') for img_path in examples['image']]
    texts = examples['text']
    
    # Process with the model's processor
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA OneVision with LoRA")
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                       help="Base model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to training dataset (JSON file or dataset directory)")
    parser.add_argument("--output_dir", type=str, default="./runs/onevision_lora_small",
                       help="Output directory for the fine-tuned adapter")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                       help="Path to existing checkpoint to load training args from")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from this checkpoint")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16 mixed precision")
    
    args = parser.parse_args()
    
    # Load training config from checkpoint if provided
    checkpoint_config = {}
    if args.checkpoint_dir and Path(args.checkpoint_dir).exists():
        print(f"Loading training configuration from {args.checkpoint_dir}...")
        checkpoint_config = load_training_args_from_checkpoint(args.checkpoint_dir)
    else:
        # Use defaults
        checkpoint_config = {
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ["k_proj", "q_proj", "o_proj", "v_proj"],
            "train_batch_size": 1,
            "num_epochs": 1,
            "logging_steps": 20,
            "save_steps": 500,
            "eval_steps": 500,
        }
    
    print("\n" + "="*60)
    print("VLM Fine-tuning Configuration")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"LoRA rank (r): {checkpoint_config['lora_r']}")
    print(f"LoRA alpha: {checkpoint_config['lora_alpha']}")
    print(f"LoRA dropout: {checkpoint_config['lora_dropout']}")
    print(f"Target modules: {checkpoint_config['target_modules']}")
    print("="*60 + "\n")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Training on CPU will be very slow. Consider using a GPU.")
    
    # Load processor and model
    print(f"\nLoading processor and base model: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.fp16 and device == "cuda" else torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Apply LoRA
    print("Applying LoRA adapter...")
    lora_config = create_lora_config(checkpoint_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    from PIL import Image
    dataset = prepare_dataset(args.dataset)
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    # Note: You'll need to adapt this based on your dataset format
    def tokenize_function(examples):
        # This is a placeholder - adapt to your dataset format
        # Your dataset should have 'image' and 'text' columns
        images = [Image.open(img).convert('RGB') if isinstance(img, str) else img for img in examples['image']]
        texts = examples['text']
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=checkpoint_config.get("num_epochs", 1),
        per_device_train_batch_size=checkpoint_config.get("train_batch_size", 1),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=checkpoint_config.get("logging_steps", 20),
        save_steps=checkpoint_config.get("save_steps", 500),
        eval_steps=checkpoint_config.get("eval_steps", 500) if 'eval' in tokenized_dataset.column_names else None,
        save_total_limit=3,
        load_best_model_at_end=True if 'eval' in tokenized_dataset.column_names else False,
        fp16=args.fp16 and device == "cuda",
        bf16=args.bf16 and device == "cuda",
        max_steps=args.max_steps,
        report_to="tensorboard" if Path(args.output_dir).exists() else None,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=None,  # Processor handles collation
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    print("\nTraining completed!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

