"""
VLM (Vision Language Model) Service
Handles VLM model loading and inference calls.
"""
from typing import Dict, Any, Optional, List
import os
import io
import base64
import json
import time
import threading
import requests
from pathlib import Path

# Import prompts from run.py (will be moved to vlm_prompts.py later)
# For now, import them to avoid circular dependencies
VLM_SYSTEM_PROMPT = None
VLM_CODEGEN_PROMPT = None


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


def stitch_images_side_by_side(img1_data_url: str, img2_data_url: str) -> str:
    """
    Combine two images side-by-side for models that only support one image.
    Returns a data URL of the combined image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Decode base64 images
        def decode_data_url(url):
            if url.startswith("data:"):
                header, data = url.split(",", 1)
            else:
                data = url
            return base64.b64decode(data)
        
        img1_bytes = decode_data_url(img1_data_url)
        img2_bytes = decode_data_url(img2_data_url)
        
        # Open images
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))
        
        # Resize to same height if needed (preserve aspect ratio)
        target_height = 1024  # Max height for combined image
        
        # Calculate new dimensions
        h1, h2 = img1.height, img2.height
        w1, w2 = img1.width, img2.width
        
        if h1 > target_height or h2 > target_height:
            scale1 = target_height / h1
            scale2 = target_height / h2
            img1 = img1.resize((int(w1 * scale1), target_height), Image.Resampling.LANCZOS)
            img2 = img2.resize((int(w2 * scale2), target_height), Image.Resampling.LANCZOS)
        elif h1 != h2:
            # Make same height
            target_h = min(h1, h2)
            if h1 > target_h:
                scale = target_h / h1
                img1 = img1.resize((int(w1 * scale), target_h), Image.Resampling.LANCZOS)
            if h2 > target_h:
                scale = target_h / h2
                img2 = img2.resize((int(w2 * scale), target_h), Image.Resampling.LANCZOS)
        
        # Create combined image
        combined_width = img1.width + img2.width
        combined_height = max(img1.height, img2.height)
        combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        
        # Paste images side by side
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))
        
        # Add labels
        try:
            draw = ImageDraw.Draw(combined)
            font_size = 30
            try:
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add text labels
            draw.text((10, 10), "REFERENCE (Target)", fill=(255, 0, 0), font=font)
            draw.text((img1.width + 10, 10), "CURRENT CAD", fill=(0, 0, 255), font=font)
        except Exception as e:
            print(f"[stitch] Could not add labels: {e}")
        
        # Convert to base64
        output = io.BytesIO()
        combined.save(output, format='JPEG', quality=85)
        output.seek(0)
        combined_b64 = base64.b64encode(output.read()).decode('ascii')
        
        print(f"[stitch] ✓ Combined {img1.width}x{img1.height} + {img2.width}x{img2.height} → {combined.width}x{combined.height}")
        
        return f"data:image/jpeg;base64,{combined_b64}"
        
    except ImportError:
        print("[stitch] ✗ PIL not available, cannot stitch images")
        print("[stitch] Install with: pip install Pillow")
        return img1_data_url  # Return first image as fallback
    except Exception as e:
        print(f"[stitch] ✗ Failed to stitch images: {e}")
        return img1_data_url  # Return first image as fallback


def call_vlm(
    final_prompt: str,
    image_data_urls: Optional[List[str] | str],
    *,
    expect_json: bool = True,
    vlm_system_prompt: Optional[str] = None,
    vlm_codegen_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call VLM with prompt and images.
    
    Args:
        final_prompt: The text prompt to send to the VLM
        image_data_urls: Image(s) as data URLs or base64 strings
        expect_json: Whether to expect JSON output (vs code)
        vlm_system_prompt: System prompt for JSON mode (if None, will import from run.py)
        vlm_codegen_prompt: System prompt for codegen mode (if None, will import from run.py)
    
    Returns:
        Dict with "provider" and "raw" response
    """
    # Import prompts if not provided (from run.py for now)
    global VLM_SYSTEM_PROMPT, VLM_CODEGEN_PROMPT
    if VLM_SYSTEM_PROMPT is None or VLM_CODEGEN_PROMPT is None:
        try:
            from run import VLM_SYSTEM_PROMPT as _sys, VLM_CODEGEN_PROMPT as _code
            VLM_SYSTEM_PROMPT = _sys
            VLM_CODEGEN_PROMPT = _code
        except ImportError:
            pass
    
    # Use provided prompts or fall back to imported ones
    system_prompt = vlm_system_prompt or (VLM_CODEGEN_PROMPT if not expect_json else VLM_SYSTEM_PROMPT)
    codegen_prompt = vlm_codegen_prompt or VLM_CODEGEN_PROMPT
    
    def _normalize(imgs):
        if not imgs:
            return None
        if isinstance(imgs, str):
            imgs = [imgs]
        out = []
        for u in imgs:
            if not u:
                continue
            out.append(u.split(",", 1)[1] if u.startswith("data:") else u)
        return out or None

    images_payload = _normalize(image_data_urls)
    err = None
    
    # Check if Ollama model exists (for fallback)
    ollama_model_available = False
    if OLLAMA_URL:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                models_data = r.json()
                model_names = [m.get("name", "") for m in models_data.get("models", [])]
                ollama_model_available = OLLAMA_MODEL in model_names
        except:
            pass
    
    # Try fine-tuned model first (user's pretrained model)
    # Load it if not already loaded
    if USE_FINETUNED_MODEL:
        if _finetuned_model is None or _finetuned_processor is None:
            print("[vlm] Loading fine-tuned model (pretrained)...")
            load_finetuned_model()
    
    # Use fine-tuned model if it's loaded
    if _finetuned_model is not None and _finetuned_processor is not None:
        try:
            print("[vlm] Using fine-tuned model...")
            import torch
            from PIL import Image
            
            # Prepare images
            images = []
            if images_payload:
                for img_b64 in images_payload:
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    images.append(img)
            
            # Prepare the conversation format
            current_system_prompt = codegen_prompt if not expect_json else system_prompt
            
            # Format as conversation (LLaVA OneVision format)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": current_system_prompt + "\n\n" + final_prompt}
                    ]
                }
            ]
            
            # Add images to the user message
            if images:
                for img in images:
                    conversation[0]["content"].insert(0, {"type": "image"})
            
            # Apply chat template and process
            prompt_text = _finetuned_processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = _finetuned_processor(
                images=images if images else None,
                text=prompt_text,
                return_tensors="pt"
            )
            
            # Move to same device as model
            device = next(_finetuned_model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # For code generation, use more tokens and very low temperature for faithful copying
            # Reduce max_tokens on CPU for faster generation
            if device == "cpu":
                max_tokens = 2048 if not expect_json else 512  # Reduced for CPU speed
                print(f"[vlm]   ⚠ CPU detected: Reduced max_tokens to {max_tokens} for faster generation")
            else:
                max_tokens = 6144 if not expect_json else 1024  # Full tokens on GPU
            temp = 0.01 if not expect_json else 0.1  # Very low temp for precise copying
            
            print(f"[vlm] Generating response...")
            print(f"[vlm]   Max tokens: {max_tokens}, Temperature: {temp}")
            print(f"[vlm]   Input shape: {inputs['input_ids'].shape if 'input_ids' in inputs else 'N/A'}")
            print(f"[vlm]   Device: {device}")
            print(f"[vlm]   Starting generation (this may take 30-120 seconds on CPU)...")
            
            start_time = time.time()
            generation_done = threading.Event()
            
            # Progress monitor thread
            def progress_monitor():
                """Print periodic status updates during generation."""
                check_interval = 10  # Check every 10 seconds
                warning_threshold = 120  # Warn if taking longer than 2 minutes
                while not generation_done.is_set():
                    elapsed = time.time() - start_time
                    if elapsed < warning_threshold:
                        time.sleep(check_interval)
                        if not generation_done.is_set():
                            print(f"[vlm]   Still generating... ({elapsed:.0f}s elapsed)", flush=True)
                    else:
                        # After warning threshold, check more frequently
                        time.sleep(5)
                        if not generation_done.is_set():
                            elapsed = time.time() - start_time
                            print(f"[vlm]   ⚠ Still generating after {elapsed:.0f}s (may take 2-5 minutes on CPU)...", flush=True)
            
            monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
            monitor_thread.start()
            
            try:
                with torch.no_grad():
                    output_ids = _finetuned_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        top_p=0.98,
                        do_sample=True if temp > 0 else False,
                        repetition_penalty=1.1,  # Prevent getting stuck in loops
                    )
            finally:
                generation_done.set()  # Signal that generation is complete
            
            elapsed = time.time() - start_time
            print(f"[vlm]   ✓ Generation completed in {elapsed:.1f} seconds", flush=True)
                
            print(f"[vlm] Generated with max_tokens={max_tokens}, temp={temp}")
            
            # Decode response
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            response = _finetuned_processor.decode(generated_ids, skip_special_tokens=True)
            
            print(f"[vlm] ✓ Got response from fine-tuned model: {len(response)} chars")
            return {"provider": "finetuned", "raw": response}
            
        except Exception as e:
            err = f"Fine-tuned model error: {e}"
            print(f"[vlm] ✗ Fine-tuned model failed: {err}")
            import traceback
            traceback.print_exc()
            # Fall back to Ollama
    
    # Handle models that only support 1 image (like llama3.2-vision)
    # If we have 2 images, stitch them side-by-side
    if images_payload and len(images_payload) > 1:
        model_name = OLLAMA_MODEL.lower()
        single_image_models = ["llama3.2-vision", "llama3.2", "llava:7b"]
        
        if any(m in model_name for m in single_image_models):
            print(f"[vlm] Model {OLLAMA_MODEL} supports only 1 image")
            print(f"[vlm] Stitching {len(images_payload)} images together...")
            
            # Reconstruct data URLs for stitching
            img1_url = f"data:image/jpeg;base64,{images_payload[0]}"
            img2_url = f"data:image/jpeg;base64,{images_payload[1]}"
            
            combined_url = stitch_images_side_by_side(img1_url, img2_url)
            
            # Extract just the base64 part
            if combined_url.startswith("data:"):
                images_payload = [combined_url.split(",", 1)[1]]
            else:
                images_payload = [combined_url]
            
            print(f"[vlm] ✓ Now sending 1 combined image")

    if OLLAMA_URL:
        try:
            # set different stop sequences depending on mode
            # For code generation: NO STOP SEQUENCES - let VLM finish naturally
            # The VLM often wants to explain first, then output code in fences
            # Stop sequences would cut it off prematurely
            if not expect_json:
                # Code generation mode - no stop sequences, let it finish
                stops = []  # Let VLM generate complete response
            else:
                # JSON mode - stop on summary or fences
                stops = ["```", "SUMMARY:"]

            current_system_prompt = codegen_prompt if not expect_json else system_prompt
            
            payload = {
                "model": OLLAMA_MODEL,
                "system": current_system_prompt,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.9,
                    # Reduce context for vision models - they often have smaller windows
                    "num_ctx": 4096 if "vision" in OLLAMA_MODEL.lower() else 8192,
                    "stop": stops,
                },
            }
            if expect_json:
                payload["format"] = "json"
            if images_payload:
                payload["images"] = images_payload

            # Code generation can be slow - use longer timeout
            timeout_seconds = 300 if not expect_json else 120  # 5 min for code, 2 min for JSON
            
            print(f"[vlm] Sending to Ollama (timeout: {timeout_seconds}s, context: {payload['options']['num_ctx']})")
            print(f"[vlm] Model: {OLLAMA_MODEL}")
            print(f"[vlm] Images: {len(images_payload) if images_payload else 0}")
            print(f"[vlm] System prompt length: {len(payload.get('system', ''))}")
            print(f"[vlm] User prompt length: {len(payload.get('prompt', ''))}")
            
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout_seconds)
            
            # Check for errors before raising
            if r.status_code != 200:
                error_detail = r.text
                print(f"[vlm] ✗ Ollama returned {r.status_code}")
                print(f"[vlm] Error response: {error_detail}")
                
                # Try to parse error message
                try:
                    error_json = r.json()
                    error_msg = error_json.get("error", error_detail)
                except:
                    error_msg = error_detail
                
                raise RuntimeError(f"Ollama error ({r.status_code}): {error_msg}")
            
            r.raise_for_status()
            response = r.json().get("response", "")
            print(f"[vlm] ✓ Got response: {len(response)} chars")
            return {"provider": "ollama", "raw": response}
        except requests.exceptions.RequestException as e:
            err = f"Ollama error: {e}"
            print(f"[vlm] ✗ Request failed: {err}")
        except Exception as e:
            err = f"Ollama error: {e}"
            print(f"[vlm] ✗ Unexpected error: {err}")
        # If Ollama fails, continue to try other options
    
    # Fallback: Try LLAVA_URL if configured
    if LLAVA_URL:
        try:
            current_system_prompt = codegen_prompt if not expect_json else system_prompt
            payload = {
                "prompt": current_system_prompt + "\n\n" + final_prompt
            }
            imgs = images_payload or []
            if imgs:
                payload["image"] = imgs[0]
            r = requests.post(LLAVA_URL, json=payload, timeout=120)
            r.raise_for_status()
            try:
                js = r.json()
                if isinstance(js, dict) and "response" in js:
                    return {"provider": "llava_url", "raw": js["response"]}
                return {"provider": "llava_url", "raw": json.dumps(js)}
            except Exception:
                return {"provider": "llava_url", "raw": r.text}
        except Exception as e:
            err = (err or "") + f" ; LLAVA_URL error: {e}"

    raise RuntimeError(err or "No VLM endpoint configured")
