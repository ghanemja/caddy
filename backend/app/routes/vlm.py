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
    """VLM-powered code generation endpoint."""
    from optim import (
        _build_codegen_prompt, _data_url_from_upload,
        _baseline_cqparts_source, extract_python_module,
        normalize_generated_code, OLLAMA_MODEL
    )
    import time
    
    try:
        prompt_text = (request.form.get("prompt") or "").strip()
        ref_url = _data_url_from_upload(request.files.get("reference"))
        snap_url = _data_url_from_upload(request.files.get("snapshot"))

        if not ref_url:
            return jsonify({"ok": False, "error": "reference image required"}), 400

        # Quick pre-check
        max_source_chars = 10000 if "vision" in OLLAMA_MODEL.lower() else 15000
        baseline_test = _baseline_cqparts_source(max_chars=max_source_chars)
        if len(baseline_test) < 1000:
            return jsonify({
                "ok": False,
                "error": "Failed to extract robot_base.py source code",
                "source_length": len(baseline_test),
            }), 500
        
        final_prompt, images = _build_codegen_prompt(ref_url, snap_url, prompt_text)
        
        # Call VLM
        out = call_vlm(final_prompt, images, expect_json=False)
        raw_txt = out.get("raw", "")

        # Extract and normalize code
        gen_dir = os.path.join(BASE_DIR, "generated")
        os.makedirs(gen_dir, exist_ok=True)

        try:
            code_txt = extract_python_module(raw_txt.strip())
        except Exception as e:
            reject_path = os.path.join(gen_dir, f"robot_base_vlm.reject_{int(time.time())}.txt")
            with open(reject_path, "w", encoding="utf-8") as rf:
                rf.write(raw_txt)
            return jsonify({
                "ok": False,
                "error": f"Could not extract valid Python from VLM output: {e}",
                "reject_path": reject_path,
            }), 400

        code_txt = normalize_generated_code(code_txt)

        # Validate code
        try:
            compile(code_txt, "<generated>", "exec")
        except SyntaxError as e:
            return jsonify({
                "ok": False,
                "error": f"Generated code has syntax errors: {e}",
                "code_preview": code_txt[:500],
            }), 400

        # Save generated code
        output_path = os.path.join(gen_dir, "robot_base_vlm.py")
        backup_path = os.path.join(gen_dir, f"robot_base_vlm_{int(time.time())}.py")
        
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(code_txt)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code_txt)

        return jsonify({
            "ok": True,
            "output_path": output_path,
            "backup_path": backup_path,
            "code_length": len(code_txt),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/recommend")
def recommend():
    """Get VLM recommendations for CAD changes."""
    from optim import (
        _data_url_from_upload, _cad_state_json,
        _split_multi_json_and_summaries, VLM_SYSTEM_PROMPT, call_vlm
    )
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
        final_prompt = f"{VLM_SYSTEM_PROMPT}\n\n---\n" + "\n".join(grounding_lines)

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


@bp.post("")
def vlm_call():
    """Direct VLM call endpoint."""
    from optim import call_vlm
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    images = data.get("images", [])
    expect_json = data.get("expect_json", True)
    
    if not prompt:
        return jsonify({"ok": False, "error": "prompt required"}), 400
    
    try:
        result = call_vlm(prompt, images, expect_json=expect_json)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

