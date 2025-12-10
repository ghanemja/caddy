"""
VLM prompt building utilities
"""
from typing import Optional, List, Tuple


# Import full prompts from run.py for now (will be moved here during refactoring)
# These are the full, detailed prompts used by the VLM
VLM_SYSTEM_PROMPT = None  # Will be imported from run.py
VLM_CODEGEN_PROMPT = None  # Will be imported from run.py


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

