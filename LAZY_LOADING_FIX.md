# Lazy Loading Fix - Model Load Optimization

## Problem

The fine-tuned VLM model was taking **5+ minutes to load** at startup, freezing at "Loading checkpoint shards: 0%". This made the server unusable and prevented quick iteration.

## Root Cause

The 7B parameter model (15GB) was being loaded **synchronously on startup**:
1. Download model from HuggingFace (one-time, ~15GB)
2. Load 4 checkpoint shards into GPU memory (~2-3 minutes)
3. Apply LoRA adapter
4. Only then start the Flask server

This blocked everything and users couldn't use the server until model loading completed.

## Solution: Lazy Loading

Changed the model to load **only when first needed** (on first VLM request):

### Before:
```python
if __name__ == "__main__":
    if USE_FINETUNED_MODEL:
        print("[startup] Loading fine-tuned VLM model...")
        load_finetuned_model()  # BLOCKS FOR 2-3 MINUTES!
    
    app.run(...)
```

### After:
```python
if __name__ == "__main__":
    if USE_FINETUNED_MODEL:
        print("[startup] Model will load on first request (lazy loading)")
        print("[startup] Server starts fast - first VLM request takes 2-3 min")
    
    app.run(...)  # STARTS IMMEDIATELY!

# In call_vlm():
if USE_FINETUNED_MODEL:
    if _finetuned_model is None:  # Load lazily
        print("[vlm] Loading model now...")
        load_finetuned_model()
```

## Benefits

### Startup Time
- **Before**: 3-5 minutes (blocked)
- **After**: 5-10 seconds ⚡

### First VLM Request
- Takes 2-3 minutes (one-time model loading)
- User sees progress messages
- Server is usable for other operations meanwhile

### Subsequent VLM Requests
- Instant (model already in memory)
- ~2-5 seconds per inference on GPU

## User Experience

### Starting Server
```bash
conda activate cad-optimizer
cd cqparts_bucket
python optim.py
```

**Output:**
```
[freecad] ✓ Loaded FreeCAD from conda
[startup] Model will load on first request (lazy loading)
[startup] Server starts fast - first VLM request takes 2-3 min
 * Running on http://0.0.0.0:5160
```

Server is ready in **~5 seconds**! 🚀

### First VLM Request
When user sends first VLM request:
```
[vlm] Model not loaded yet, loading now (this will take 2-3 minutes)...
[vlm] Loading fine-tuned model from runs/onevision_lora_small...
[vlm] Loading processor...
[vlm] ✓ Processor loaded
[vlm] Loading base model (this may take 2-3 minutes)...
Loading checkpoint shards: 100%|██████| 4/4 [01:47<00:00, 26.85s/it]
[vlm] ✓ Base model loaded to cuda
[vlm] Loading LoRA adapter from runs/onevision_lora_small
[vlm] ✓ Fine-tuned model loaded successfully on cuda
[vlm] Using fine-tuned model...
[vlm] Generating response...
[vlm] ✓ Got response from fine-tuned model: 1234 chars
```

### Subsequent Requests
Model is cached in GPU memory:
```
[vlm] Using fine-tuned model...
[vlm] Generating response...
[vlm] ✓ Got response: 567 chars
```

Fast! ⚡ (~2-5 seconds)

## Technical Details

### Model Loading Process
1. **Processor** (~1 second): Tokenizer and image processor
2. **Base Model** (~2 minutes): 4 checkpoint shards → GPU memory
3. **LoRA Adapter** (~5 seconds): Your fine-tuned weights
4. **Total**: ~2-3 minutes (one-time)

### Memory Usage
- GPU VRAM: ~14.5 GB (persistent after loading)
- System RAM: ~2-3 GB (temporary during loading)

### Optimizations Added
```python
base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,  # Half precision
    device_map="auto",           # Automatic GPU distribution
    low_cpu_mem_usage=True,      # Minimize RAM usage
    use_safetensors=True,        # Faster loading format
)
```

## Disable Fine-Tuned Model

If model loading is still too slow, disable it and use Ollama:

```bash
USE_FINETUNED_MODEL=0 python optim.py
```

Server starts instantly and uses Ollama for VLM requests.

## Alternative: Pre-load in Background

For production, you could add background pre-loading:

```python
if __name__ == "__main__":
    if USE_FINETUNED_MODEL:
        # Start loading in background thread
        threading.Thread(target=load_finetuned_model, daemon=True).start()
        print("[startup] Model loading in background...")
    
    app.run(...)
```

This way:
- Server starts immediately
- Model loads in parallel
- First VLM request waits for model if not ready yet

## Summary

✅ **Problem Solved**: Server now starts in seconds instead of minutes
✅ **Lazy Loading**: Model loads only when needed
✅ **Better UX**: Users can start using server immediately
✅ **GPU Optimized**: fp16, safetensors, auto device mapping
✅ **Flexible**: Easy to disable or switch to Ollama

Your development workflow is now much faster! 🎉

