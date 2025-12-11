import os
import sys
from app.core import BACKEND_DIR
from flask import Flask
from app.services.vlm_service import (
    USE_FINETUNED_MODEL,
    FINETUNED_MODEL_PATH,
    OLLAMA_URL,
    OLLAMA_MODEL,
    load_finetuned_model,
)
_finetuned_model = None  # Will be lazy-loaded
_finetuned_processor = None  # Will be lazy-loaded
from app.config import Config
# Configuration values from Config
TARGET_ALIASES = Config.TARGET_ALIASES
ACTION_ALIASES = Config.ACTION_ALIASES
PORT = Config.PORT
BASE_DIR = BACKEND_DIR  # backend/ directory
ASSETS_DIR = Config.ASSETS_DIR  # frontend/assets/
os.makedirs(ASSETS_DIR, exist_ok=True)
USE_CQPARTS = os.environ.get("USE_CQPARTS", "1") == "1"

app = Flask(
    __name__,
    template_folder=str(Config.TEMPLATES_FOLDER),
    static_folder=str(Config.STATIC_FOLDER),
    static_url_path="/static"
)

from app.routes import main, api, vlm, mesh, model, test
app.register_blueprint(main.bp)  # Routes: /, /debug, /demo/*, /static/*
app.register_blueprint(api.bp)  # Routes: /state, /apply, /params, /model.glb, etc.
app.register_blueprint(vlm.bp)  # Routes: /codegen, /vlm, /recommend
app.register_blueprint(mesh.bp, url_prefix="/api/mesh")  # Routes: /ingest_mesh_segment, etc.
app.register_blueprint(model.bp, url_prefix="/api/model")
app.register_blueprint(test.bp, url_prefix="/test")  # Routes: /test/rotate, /test/translate, /test/modify

# Legacy rover components removed - no longer used
# Component registration and rover building removed
# Models are now uploaded by users as CadQuery/Python files

from app.services.vlm.prompts_loader import get_system_prompt, get_codegen_prompt

# Backward compatibility: prompts for modules that import from run.py
VLM_SYSTEM_PROMPT = get_system_prompt()
VLM_CODEGEN_PROMPT = get_codegen_prompt()

if __name__ == "__main__":
    import sys
    
    # Check for test flag - run test BEFORE other imports that might fail
    if "--test-vlm" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "test-vlm"):
        from app.utils.test_vlm import test_vlm_model
        include_mesh = "--no-mesh" not in sys.argv
        try:
            test_vlm_model(include_mesh_analysis=include_mesh)
            sys.exit(0)
        except Exception as e:
            print(f"\n[test] ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # VLM setup - prefer Ollama if available, otherwise fine-tuned model
    # Check if Ollama is available and has the requested model
    ollama_available = False
    ollama_model_exists = False
    if OLLAMA_URL:
        try:
            import requests
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if r.status_code == 200:
                ollama_available = True
                # Check if the requested model exists
                models_data = r.json()
                model_names = [m.get("name", "") for m in models_data.get("models", [])]
                ollama_model_exists = OLLAMA_MODEL in model_names
                if ollama_model_exists:
                    print(f"[startup] ✓ Ollama is available (URL: {OLLAMA_URL}, Model: {OLLAMA_MODEL})")
                else:
                    print(f"[startup] ⚠ Ollama is available but model '{OLLAMA_MODEL}' not found")
                    print(f"[startup] Available models: {', '.join(model_names)}")
                    print(f"[startup] Will use fine-tuned model instead")
            else:
                print(f"[startup] ⚠ Ollama URL responded with {r.status_code}")
        except Exception as e:
            print(f"[startup] ⚠ Ollama not available: {e}")
            print("[startup] Make sure Ollama is running: ollama serve")
    
    # Only use Ollama if it's available AND has the requested model
    # Otherwise, fall back to fine-tuned model
    if not (ollama_available and ollama_model_exists) and USE_FINETUNED_MODEL:
        print(f"[startup] Preloading fine-tuned VLM model...")
        print(f"[startup] Model path: {FINETUNED_MODEL_PATH}")
        print(f"[startup] Model path exists: {os.path.exists(FINETUNED_MODEL_PATH)}")
        print("[startup] Loading model (base model will be loaded from cache if available)...")
        try:
            load_finetuned_model()
            if _finetuned_model is not None:
                print("[startup] ✓ VLM model preloaded and ready")
            else:
                print("[startup] ⚠ VLM model not loaded (will load on first use)")
        except Exception as e:
            print(f"[startup] ⚠ Could not preload VLM model: {e}")
            print("[startup] Model will be loaded on first use")
    elif ollama_available and ollama_model_exists:
        print(f"[startup] ✓ Using Ollama VLM (Model: {OLLAMA_MODEL})")
        if USE_FINETUNED_MODEL and os.path.exists(FINETUNED_MODEL_PATH):
            print(f"[startup] Fine-tuned model available at {FINETUNED_MODEL_PATH} but using Ollama")
    else:
        if ollama_available and not ollama_model_exists:
            print(f"[startup] ⚠ Ollama available but model '{OLLAMA_MODEL}' not found")
            print(f"[startup] To use Ollama: ollama pull {OLLAMA_MODEL}")
            if USE_FINETUNED_MODEL and os.path.exists(FINETUNED_MODEL_PATH):
                print(f"[startup] Will use fine-tuned model instead (lazy load on first use)")
        else:
            print("[startup] ⚠ No VLM configured - VLM features will not work")
    
    # Disabled warm build - don't build rover on startup
    # Wait for user to upload mesh or parametric model instead
    # threading.Thread(target=_warm_build, daemon=True).start()
    print("[startup] Skipping warm build - waiting for user to upload mesh or parametric model", flush=True)
    app.run(host="0.0.0.0", port=PORT, debug=False)
