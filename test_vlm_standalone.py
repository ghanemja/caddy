#!/usr/bin/env python3
"""
Standalone test script for VLM model that doesn't require all optim.py dependencies.
This script only imports VLM-related code and can run even if CAD dependencies are missing.

Usage:
    # Activate conda environment first (if using conda):
    conda activate vlm_optimizer
    
    # Then run:
    python test_vlm_standalone.py
    python test_vlm_standalone.py --no-mesh
"""

import os
import sys

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
cqparts_bucket_path = os.path.join(script_dir, "cqparts_bucket")
if cqparts_bucket_path not in sys.path:
    sys.path.insert(0, cqparts_bucket_path)

# Change to cqparts_bucket for relative imports
os.chdir(cqparts_bucket_path)

# Now import only VLM-related parts
# We'll manually import what we need instead of importing all of optim.py

def test_vlm_standalone(include_mesh_analysis=True):
    """Standalone VLM test that doesn't require CAD dependencies."""
    print("=" * 80)
    print("Testing VLM Model (Standalone)")
    print("=" * 80)
    
    # Import only what we need for VLM
    try:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        from peft import PeftModel
        import torch
    except ImportError as e:
        print(f"[test] ✗ ERROR: Missing required VLM dependencies: {e}")
        print("[test] Install with: pip install transformers peft torch pillow accelerate")
        return False
    
    # Set up model path - should be relative to project root (optim/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir is optim/, so runs/ is at optim/runs/
    _default_model_path = os.path.join(script_dir, "runs", "onevision_lora_small", "checkpoint-4")
    FINETUNED_MODEL_PATH = os.environ.get("FINETUNED_MODEL_PATH", _default_model_path)
    
    print(f"\n[test] Model path: {FINETUNED_MODEL_PATH}")
    print(f"[test] Path exists: {os.path.exists(FINETUNED_MODEL_PATH)}")
    
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"[test] ✗ ERROR: Model path does not exist!")
        return False
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for f in required_files:
        file_path = os.path.join(FINETUNED_MODEL_PATH, f)
        if not os.path.exists(file_path):
            print(f"[test] ✗ ERROR: Missing required file: {f}")
            return False
        print(f"[test] ✓ Found {f}")
    
    # Load model
    print(f"\n[test] Loading model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[test] Using device: {device}")
        
        base_model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        print(f"[test] Loading processor...")
        processor = AutoProcessor.from_pretrained(base_model_name)
        print(f"[test] ✓ Processor loaded")
        
        print(f"[test] Loading base model...")
        base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        print(f"[test] ✓ Base model loaded")
        
        print(f"[test] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            FINETUNED_MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model.eval()
        print(f"[test] ✓ Model loaded successfully")
        
    except Exception as e:
        print(f"[test] ✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference
    print(f"\n[test] Testing inference...")
    try:
        from PIL import Image
        import io
        
        test_prompt = "What is 2+2? Answer with just the number."
        
        # Format as conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": test_prompt}
                ]
            }
        ]
        
        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=prompt_text,
            return_tensors="pt"
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        print(f"[test] Generating response...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
            )
        
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"[test] ✓ Inference successful!")
        print(f"[test] Response: {response[:200]}...")
        
    except Exception as e:
        print(f"[test] ✗ ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test mesh analysis if requested
    if include_mesh_analysis:
        print(f"\n" + "=" * 80)
        print("Testing Mesh Analysis Pipeline")
        print("=" * 80)
        print("[test] ⚠ Mesh analysis requires full optim.py - skipping for standalone test")
        print("[test] Run 'python cqparts_bucket/optim.py --test-vlm' for full mesh analysis test")
    
    return True


if __name__ == "__main__":
    include_mesh = "--no-mesh" not in sys.argv
    try:
        success = test_vlm_standalone(include_mesh_analysis=include_mesh)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[test] Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[test] ✗ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

