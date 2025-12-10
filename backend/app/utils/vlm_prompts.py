"""
VLM prompt building utilities
"""
from typing import Optional, List, Tuple


VLM_SYSTEM_PROMPT = """You are a CAD parameter extraction assistant. Analyze the provided images and extract precise parameter changes in JSON format.

Output ONLY valid JSON, no markdown, no explanations. Format:
[
  {
    "target_component": "wheel",
    "action": "modify",
    "parameters": {"wheel_diameter": 100}
  }
]"""

VLM_CODEGEN_PROMPT = """You are a CAD code generation assistant. Given a baseline Python class and user instructions, generate modified code.

CRITICAL RULES:
1. Copy the ENTIRE baseline code exactly
2. Make ONLY the requested parameter changes
3. Change ONLY numbers inside PositiveFloat(...)
4. Keep everything else identical
5. NO markdown fences - output raw Python only
6. Start with: #!/usr/bin/env python3"""


def build_codegen_prompt(
    ref_url: Optional[str],
    snapshot_url: Optional[str],
    user_text: str = "",
    baseline_code: str = ""
) -> Tuple[str, List[str]]:
    """
    Build the complete prompt for VLM code generation.
    
    Returns:
        Tuple of (prompt_text, image_urls)
    """
    prompt_lines = [
        VLM_CODEGEN_PROMPT,
        "",
        "BASELINE CODE:",
        "```python",
        baseline_code,
        "```",
        "",
        "USER REQUEST:",
        user_text,
        "",
        "OUTPUT REQUIREMENTS:",
        "• NO markdown fences (```python or ```) - output raw Python only",
        "• NO explanations - just the code",
        "• Start with: #!/usr/bin/env python3",
        "• Copy every import, every class, every method from baseline",
        "• Make ONLY the requested changes",
        "",
        "START YOUR PYTHON CODE NOW:",
    ]
    
    prompt_text = "\n".join(prompt_lines)
    images = [url for url in [ref_url, snapshot_url] if url]
    
    return prompt_text, images

