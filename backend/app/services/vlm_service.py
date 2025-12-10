"""
VLM (Vision Language Model) Service
Handles VLM model loading and inference calls.
"""
from typing import Dict, Any, Optional, List
import os
from pathlib import Path


# VLM Configuration
USE_FINETUNED_MODEL = os.environ.get("USE_FINETUNED_MODEL", "1") == "1"
BACKEND_DIR = Path(__file__).parent.parent.parent
FINETUNED_MODEL_PATH = os.environ.get(
    "FINETUNED_MODEL_PATH",
    str(BACKEND_DIR / "checkpoints" / "onevision_lora_small" / "checkpoint-4")
)
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava:latest")
LLAVA_URL = os.environ.get("LLAVA_URL")

# Global model state
_finetuned_model = None
_finetuned_processor = None


def load_finetuned_model():
    """Load the fine-tuned VLM model with LoRA adapter."""
    global _finetuned_model, _finetuned_processor
    
    if not USE_FINETUNED_MODEL:
        print("[vlm] Fine-tuned model disabled, will use Ollama/LLAVA")
        return
    
    if _finetuned_model is not None:
        print("[vlm] Fine-tuned model already loaded")
        return
    
    try:
        print(f"[vlm] Loading fine-tuned model from {FINETUNED_MODEL_PATH}...")
        
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        from peft import PeftModel
        import torch
        
        base_model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        print(f"[vlm] Loading base model: {base_model_name}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[vlm] Using device: {device}")
        
        print(f"[vlm] Loading processor...")
        _finetuned_processor = AutoProcessor.from_pretrained(base_model_name)
        print(f"[vlm] ✓ Processor loaded")
        
        print(f"[vlm] Loading base model from cache (or downloading if first time)...")
        base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        print(f"[vlm] ✓ Base model loaded to {device}")
        
        print(f"[vlm] Loading LoRA adapter from {FINETUNED_MODEL_PATH}")
        _finetuned_model = PeftModel.from_pretrained(
            base_model,
            FINETUNED_MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        _finetuned_model.eval()
        print(f"[vlm] ✓ Fine-tuned model loaded successfully on {device}")
        
    except ImportError as e:
        print(f"[vlm] ✗ Failed to import required libraries: {e}")
        print("[vlm] Install with: pip install transformers peft torch pillow accelerate")
        _finetuned_model = None
        _finetuned_processor = None
    except Exception as e:
        print(f"[vlm] ✗ Failed to load fine-tuned model: {e}")
        import traceback
        traceback.print_exc()
        _finetuned_model = None
        _finetuned_processor = None


def get_finetuned_model():
    """Get the loaded fine-tuned model (lazy load if needed)."""
    if _finetuned_model is None and USE_FINETUNED_MODEL:
        load_finetuned_model()
    return _finetuned_model


def get_finetuned_processor():
    """Get the loaded processor (lazy load if needed)."""
    if _finetuned_processor is None and USE_FINETUNED_MODEL:
        load_finetuned_model()
    return _finetuned_processor


def call_vlm(
    final_prompt: str,
    image_data_urls: Optional[List[str] | str],
    *,
    expect_json: bool = True,
) -> Dict[str, Any]:
    """
    Call VLM with prompt and images.
    
    This function will be implemented to use either Ollama or fine-tuned model.
    For now, it's a placeholder that will be filled from run.py during migration.
    """
    # This will be implemented by importing from run.py initially
    # TODO: Full implementation here
    raise NotImplementedError("VLM calling will be implemented in next step")

