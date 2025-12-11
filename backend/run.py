import os
import sys
from app.core import BACKEND_DIR
from flask import Flask

from app.config import Config
from app.routes import main, api, vlm, mesh, model, test
from app.services.vlm_service import FINETUNED_MODEL_PATH, load_finetuned_model
from app.services.vlm.prompts_loader import get_system_prompt, get_codegen_prompt

_finetuned_model = None  # Will be lazy-loaded
_finetuned_processor = None  # Will be lazy-loaded
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

app.register_blueprint(main.bp)  # Routes: /, /debug, /demo/*, /static/*
app.register_blueprint(api.bp)  # Routes: /state, /apply, /params, /model.glb, etc.
app.register_blueprint(vlm.bp)  # Routes: /codegen, /vlm, /recommend
app.register_blueprint(mesh.bp, url_prefix="/api/mesh")  # Routes: /ingest_mesh_segment, etc.
app.register_blueprint(model.bp, url_prefix="/api/model")
app.register_blueprint(test.bp, url_prefix="/test")  # Routes: /test/rotate, /test/translate, /test/modify

VLM_SYSTEM_PROMPT = get_system_prompt()
VLM_CODEGEN_PROMPT = get_codegen_prompt()

# Backward compatibility: Export state variables and functions for modules that import from run.py
from app.core.state import (
    STATE,
    CURRENT_PARAMS,
    CONTEXT,
    HISTORY,
    H_PTR,
    _INGEST_RESULT_CACHE,
    INIT_SNAPSHOT,
    PENDING_ADDS,
    HIDDEN_PREFIXES,
)

from app.services.state_service import (
    snapshot_global as _snapshot,
    ensure_initial_history_global as _ensure_initial_history,
    restore_global as _restore,
)

from app.utils.inspection import introspect_params_from_cls as _introspect_params_from_cls
from app.utils.request_parsing import parse_apply_request as _parse_apply_request
from app.services.cad_service import rebuild_and_save_glb as _rebuild_and_save_glb
from app.core.component_registry import COMPONENT_REGISTRY

if __name__ == "__main__":
    import sys
    
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
    
    print("[startup] Preloading fine-tuned VLM model...")
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

    app.run(host="0.0.0.0", port=PORT, debug=False)
