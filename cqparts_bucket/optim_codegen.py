#!/usr/bin/env python3
# send_to_vlm_codegen.py — feed your Rover source + image to a VLM and save pure Python output

import os, json, base64, requests, re, ast, argparse, sys
from typing import Optional

# --- VLM endpoint config (env overrides) ---
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava-llama3:latest")
LLAVA_URL = os.environ.get("LLAVA_URL")  # optional fallback


# --- Helper: load file into a string variable (your question) ---
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# --- Helper: load image -> base64 (for VLM vision inputs) ---
def img_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    # mime guess is optional; most VLMs accept raw base64 too
    ext = os.path.splitext(path)[1].lower()
    mime = (
        "image/png"
        if ext == ".png"
        else ("image/jpeg" if ext in (".jpg", ".jpeg") else "application/octet-stream")
    )
    return f"data:{mime};base64,{b64}"


# --- STRICT “code-only” generation prompt ---
VLM_CODEGEN_PROMPT = """
You are a CAD code generator that writes complete Python modules for CadQuery/cqparts.

GOAL:
Modify the baseline CAD source to produce a parametric rover assembly that matches the reference image.

OUTPUT:
- Output ONLY Python code. No explanations, no markdown, no backticks.
- The final line of your output MUST be exactly:
# END_OF_MODULE

REQUIREMENTS:
- Define:
    def build():
        return Rover()
- Syntactically valid Python 3.10; importable as a module.
- Keep existing class names/imports where possible.
- Place wheels using cqparts Mates with mid-plane joins and correct orientation.
- Use the longer side of the base for wheel placement.
- Avoid triple-quoted docstrings; use single-line comments (#).
- If unsure, make conservative, syntactically valid assumptions.

INPUTS:
<<<BASELINE_ROVER_PY>>>
{ROVER_SOURCE}
<<<END_BASELINE_ROVER_PY>>>

<<<CURRENT_STATE_JSON>>>
{CURRENT_STATE_JSON}
<<<END_CURRENT_STATE_JSON>>>

<<<REFERENCE_IMAGE_INFO>>>
{REFERENCE_IMAGE_HINT}
<<<END_REFERENCE_IMAGE_INFO>>>

INSTRUCTIONS:
- Read the inputs above.
- Produce a single complete Python module redefining Rover() accordingly.
- Print ONLY the Python code module; nothing else.
- End with the exact sentinel line.
""".strip()


# --- Minimal state + hint; customize as needed ---
def default_state_json() -> dict:
    return {
        "current_params": {
            "wheel_diameter": None,
            "wheel_width": None,
            "wheels_per_side": None,
            "axle_spacing_mm": None,
            "wheelbase_span_mm": None,
            "wheel_z_offset_mm": None,
            "rover_yaw_deg": None,
        },
        "context": {"terrain_mode": "flat"},
        "known_classes": ["rover", "wheel", "pan_tilt", "sensor_fork"],
    }


def reference_hint(text: Optional[str]) -> str:
    return (
        text
        or "Match overall wheel count and size; align wheelbase span; keep conservative changes."
    )


# --- Call VLM expecting pure code text (not JSON) ---
def call_vlm_code_only(final_prompt: str, images_data_urls: list[str]) -> str:
    images_payload = [
        u.split(",", 1)[1] if u.startswith("data:") else u
        for u in images_data_urls
        if u
    ]

    # Prefer Ollama if configured
    if OLLAMA_URL:
        payload = {
            "model": OLLAMA_MODEL,
            "system": VLM_CODEGEN_PROMPT,  # reinforce constraints in system
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_ctx": 4096,
                "stop": ["```", "# END_OF_MODULE"],  # stop right after code
            },
        }
        if images_payload:
            payload["images"] = images_payload
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "")

    # Fallback: a simple LLAVA_URL that takes {prompt,image}
    if LLAVA_URL:
        payload = {"prompt": VLM_CODEGEN_PROMPT + "\n\n" + final_prompt}
        if images_payload:
            payload["image"] = images_payload[0]
        r = requests.post(LLAVA_URL, json=payload, timeout=180)
        r.raise_for_status()
        try:
            js = r.json()
            return js.get("response", "") if isinstance(js, dict) else str(js)
        except Exception:
            return r.text

    raise RuntimeError("No VLM endpoint configured (set OLLAMA_URL or LLAVA_URL).")


# --- Extract the largest valid Python block from chatty output (safety) ---
def extract_python_module(text: str) -> str:
    if not text:
        raise ValueError("empty code output")
    t = text.replace("\r\n", "\n")
    # prefer fenced blocks
    fences = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", t, flags=re.I)
    candidates = fences + [t]  # also try the whole output

    def scrub(s: str) -> str:
        s = re.sub(r"^```(?:python)?\s*", "", s.strip(), flags=re.I)
        s = re.sub(r"\s*```$", "", s, flags=re.I).strip()
        s = re.split(r"\n(?:Explanation|Notes|Rationale|SUMMARY:)\b", s, maxsplit=1)[0]
        return s.strip()

    seen = set()
    for raw in candidates:
        if not raw:
            continue
        s = scrub(raw)
        if s in seen:
            continue
        seen.add(s)
        try:
            ast.parse(s)
            return s
        except Exception:
            # progressively trim to first code-like line
            lines = s.splitlines()
            codey = re.compile(r"^\s*(from|import|class|def|@|if\s+__name__)")
            for i in range(len(lines)):
                if codey.match(lines[i]):
                    chunk = "\n".join(lines[i:])
                    try:
                        ast.parse(chunk)
                        return chunk
                    except Exception:
                        pass
    raise ValueError("no valid Python block found in model output")


def main():
    ap = argparse.ArgumentParser(
        description="Send Rover source + image to VLM for code-only CAD generation."
    )
    ap.add_argument(
        "--rover-src",
        default="robot_base.py",
        help="Path to the Rover Python source (e.g., robot_base.py or your module).",
    )
    ap.add_argument(
        "--ref-image", required=True, help="Path to the reference image to match."
    )
    ap.add_argument("--snapshot", help="Optional path to a current CAD snapshot image.")
    ap.add_argument("--hint", help="Optional short text hint for the model.")
    ap.add_argument(
        "--out",
        default="generated/rover_vlm.py",
        help="Where to save the generated Python module.",
    )
    args = ap.parse_args()

    # 1) Put the Rover Python file into a variable (what you asked)
    rover_source = load_text(args.rover_src)

    # 2) Any current CAD state you want the model to consider
    state_json = default_state_json()
    current_state_json = json.dumps(state_json, indent=2)

    # 3) Optional human hint
    ref_hint = reference_hint(args.hint)

    # 4) Build the final prompt (fills the placeholders)
    final_prompt = VLM_CODEGEN_PROMPT.format(
        ROVER_SOURCE=rover_source,
        CURRENT_STATE_JSON=current_state_json,
        REFERENCE_IMAGE_HINT=ref_hint,
    )

    # 5) Images for the VLM (reference + optional snapshot)
    images = [img_to_data_url(args.ref_image)]
    if args.snapshot:
        images.append(img_to_data_url(args.snapshot))

    # 6) Call the VLM expecting PYTHON CODE ONLY
    raw = call_vlm_code_only(final_prompt, images)

    # 7) Trim after sentinel if present and extract a valid module
    end_ix = raw.find("# END_OF_MODULE")
    if end_ix != -1:
        raw = raw[:end_ix]
    code_txt = extract_python_module(raw.strip())

    # 8) Save the generated module
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code_txt)
        f.write("\n")  # keep POSIX newline
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
