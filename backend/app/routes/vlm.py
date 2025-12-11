"""
VLM (Vision Language Model) routes blueprint
"""
from flask import Blueprint, request, jsonify
from app.services.vlm_service import call_vlm
import os
import sys

bp = Blueprint("vlm", __name__)

# Import from legacy optim.py for now
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)


# Note: Some VLM routes are handled by legacy route registration
# The /codegen and /recommend routes below are new implementations
# Legacy /vlm route is registered from optim.py

@bp.post("/codegen")
def codegen():
    """
    VLM-powered code generation endpoint.
    
    Accepts:
    - reference: image file (required) - target design
    - snapshot: image file (optional) - current CAD orthogonal views
    - prompt: text (optional) - user qualitative feedback/intent
    
    Returns:
    - Generated robot_base.py code
    - Saved to generated/robot_base_vlm.py
    - Also creates a backup in generated/robot_base_vlm_TIMESTAMP.py
    """
    from app.utils.helpers import data_url_from_upload as _data_url_from_upload
    from app.utils.codegen import (
        extract_python_module,
        normalize_generated_code_advanced as _normalize_generated_code_advanced
    )
    from app.utils.inspection import baseline_cqparts_source as _baseline_cqparts_source
    from app.services.vlm_service import build_codegen_prompt as _build_codegen_prompt
    from run import (
        _rebuild_and_save_glb,
        OLLAMA_MODEL
    )
    import time
    import requests
    
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        ref_url = _data_url_from_upload(request.files.get("reference"))
        snap_url = _data_url_from_upload(request.files.get("snapshot"))

        if not ref_url:
            return jsonify({"ok": False, "error": "reference image required"}), 400

        print(f"[codegen] Building prompt with user_text: {prompt_text[:100] if prompt_text else '(none)'}")
        
        # Quick pre-check: verify source extraction works
        max_source_chars = 10000 if "vision" in OLLAMA_MODEL.lower() else 15000
        baseline_test = _baseline_cqparts_source(max_chars=max_source_chars)
        print(f"[codegen] Using max_source_chars={max_source_chars} for model {OLLAMA_MODEL}")
        if len(baseline_test) < 1000:
            print(f"[codegen] ✗ ERROR: Source extraction failed! Only {len(baseline_test)} chars")
            return jsonify({
                "ok": False,
                "error": "Failed to extract robot_base.py source code",
                "source_length": len(baseline_test),
                "source_preview": baseline_test[:500],
                "help": "Check that robot_base.py exists in the same directory as run.py"
            }), 500
        
        final_prompt, images = _build_codegen_prompt(ref_url, snap_url, prompt_text)
        
        print(f"[codegen] Final prompt length: {len(final_prompt)} chars")
        print(f"[codegen] Calling VLM with {len(images)} image(s)...")
        # Ask the VLM for full Python code (NOT JSON)
        out = call_vlm(final_prompt, images, expect_json=False)
        raw_txt = out.get("raw", "")
        
        print(f"[codegen] Got {len(raw_txt)} chars from VLM")
        print(f"[codegen] Raw VLM output (first 500 chars):")
        print(raw_txt[:500])

        gen_dir = os.path.join(BASE_DIR, "generated")
        os.makedirs(gen_dir, exist_ok=True)

        # Extract valid Python code from VLM output
        try:
            code_txt = extract_python_module(raw_txt.strip())
            print(f"[codegen] ✓ Extracted {len(code_txt)} chars of Python code")
        except Exception as e:
            # Save the raw response for debugging
            reject_path = os.path.join(gen_dir, f"robot_base_vlm.reject_{int(time.time())}.txt")
            with open(reject_path, "w", encoding="utf-8") as rf:
                rf.write(raw_txt)
            print(f"[codegen] ✗ Failed to extract Python code. Saved to {reject_path}")
            print(f"[codegen] Error: {e}")
            
            return jsonify({
                "ok": False,
                "error": f"Could not extract valid Python from VLM output: {e}",
                "reject_path": reject_path,
                "raw_preview": raw_txt[:1000],
                "help": f"Check {reject_path} for full VLM output."
            }), 400

        # Apply automatic fixes/normalization
        code_txt = _normalize_generated_code_advanced(code_txt)

        # Validate code compiles (after normalization)
        try:
            compile(code_txt, "robot_base_vlm.py", "exec")
            print(f"[codegen] ✓ Normalized code compiles successfully")
        except SyntaxError as e:
            reject_path = os.path.join(gen_dir, f"robot_base_vlm.syntax_error_{int(time.time())}.py")
            with open(reject_path, "w", encoding="utf-8") as rf:
                rf.write(f"# SYNTAX ERROR\n# {e}\n\n")
                rf.write(code_txt)
            
            return jsonify({
                "ok": False,
                "error": f"Generated code has syntax error: {e}",
                "reject_path": reject_path
            }), 400
        
        # Save the generated code
        timestamp = int(time.time())
        backup_path = os.path.join(gen_dir, f"robot_base_vlm_{timestamp}.py")
        mod_path = os.path.join(gen_dir, "robot_base_vlm.py")
        
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        
        print(f"[codegen] ✓ Saved to {mod_path}")
        print(f"[codegen] ✓ Backup: {backup_path}")
        print(f"[codegen] Code: {len(code_txt)} chars, {len(code_txt.splitlines())} lines")

        # Try to rebuild the GLB with the new code
        try:
            print("[codegen] Attempting to rebuild GLB with new generated code...")
            _rebuild_and_save_glb(use_generated=True)  # Use the generated code!
            print("[codegen] ✓ GLB rebuild successful with generated code")
            glb_updated = True
        except Exception as e:
            print(f"[codegen] GLB rebuild failed: {e}")
            import traceback
            traceback.print_exc()
            glb_updated = False

        return jsonify({
            "ok": True,
            "code": code_txt,  # Include the generated code for display
            "module_path": mod_path,
            "backup_path": backup_path,
            "code_length": len(code_txt),
            "code_lines": len(code_txt.splitlines()),
            "glb_updated": glb_updated,
            "message": "Generated complete robot_base.py with modifications"
        })
        
    except requests.exceptions.Timeout as e:
        print(f"[codegen] ⏱️ TIMEOUT: VLM took too long (>5 minutes)")
        return jsonify({
            "ok": False,
            "error": "VLM request timed out after 5 minutes",
            "suggestions": [
                "Try a faster/smaller VLM model (e.g., codellama:7b instead of 13b)",
                "Reduce image resolution",
                "Simplify your prompt",
                "Check if Ollama is running and has GPU access",
                "Try without snapshot image (only reference)",
            ],
        }), 504
    except Exception as e:
        import traceback
        error_msg = f"Codegen endpoint error: {str(e)}"
        print(f"[codegen] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@bp.post("/recommend")
def recommend():
    """Get VLM recommendations for CAD changes."""
    from app.utils.helpers import data_url_from_upload as _data_url_from_upload
    from run import (
        _split_multi_json_and_summaries, call_vlm
    )
    from app.services.vlm.prompts_loader import get_system_prompt
    import json
    
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        classes = json.loads(request.form.get("classes") or "[]")
        if not isinstance(classes, list):
            classes = []

        ref_url = _data_url_from_upload(request.files.get("reference"))
        if not ref_url:
            return jsonify({"ok": False, "error": "no reference image"}), 400
        snapshot_url = _data_url_from_upload(request.files.get("snapshot"))

        # Build prompt
        from app.services.state_service import cad_state_json as _cad_state_json
        cad_state = _cad_state_json()
        grounding_lines = [
            "Goal: Compare the REFERENCE image (photo/render) to the CURRENT CAD and propose precise, conservative changes that align CAD to the image.",
            "",
            "You are given:",
            "1) REFERENCE image (index 0).",
        ]
        if snapshot_url:
            grounding_lines.append("2) CURRENT CAD SNAPSHOT image (index 1).")
        grounding_lines += [
            "3) CURRENT CAD STATE JSON (below):",
            json.dumps(cad_state, indent=2),
            "",
            "Known classes (from client):",
            *[f"- {c}" for c in classes],
            "",
        ]
        if prompt_text:
            grounding_lines += ["User prompt:", prompt_text]

        images = [ref_url, snapshot_url] if snapshot_url else [ref_url]
        final_prompt = f"{get_system_prompt()}\n\n---\n" + "\n".join(grounding_lines)

        provider_out = call_vlm(final_prompt, images)
        raw = provider_out.get("raw", "")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        raw = raw.strip()

        parsed_changes, summaries, kept_blocks = _split_multi_json_and_summaries(raw)

        return jsonify({
            "ok": True,
            "changes": parsed_changes or [],
            "summaries": summaries,
            "raw_blocks": kept_blocks,
            "provider": provider_out.get("provider", "unknown"),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/vlm")
def vlm():
    """VLM endpoint for parameter extraction."""
    from run import _data_url_from_upload
    from app.services.vlm.prompts_loader import get_system_prompt
    import json
    import re
    
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        selected = (request.form.get("selected_class") or "").strip() or None
        classes = json.loads(request.form.get("classes") or "[]")
        if not isinstance(classes, list):
            classes = []
        data_url = _data_url_from_upload(request.files.get("image"))
        grounding = ["Known component classes:", *[f"- {c}" for c in classes]]
        if selected:
            grounding.append(f"\nUser highlighted class: {selected}")
        grounding.append("\nUser prompt:\n" + prompt_text)
        final_prompt = f"{get_system_prompt()}\n\n---\n" + "\n".join(grounding)
        resp = call_vlm(final_prompt, data_url)
        raw = resp.get("raw", "")
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}\s*$", raw.strip())
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        return jsonify(
            {
                "ok": True,
                "provider": resp.get("provider"),
                "response": {"raw": raw, "json": parsed},
            }
        )
    except Exception as e:
        import traceback
        error_msg = f"VLM endpoint error: {str(e)}"
        print(f"[vlm endpoint] {error_msg}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500

