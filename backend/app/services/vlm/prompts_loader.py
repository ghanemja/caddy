"""
VLM Prompts Loader

Loads prompts from text files in the prompts/ directory.
Prompts are stored as .txt files for easy editing without code changes.
"""
from pathlib import Path
from typing import Dict, Optional
import os

# Path to prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Cache for loaded prompts
_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(filename: str, cache: bool = True) -> str:
    """
    Load a prompt from a text file.
    
    Args:
        filename: Name of the prompt file (e.g., "system_prompt.txt")
        cache: Whether to cache the loaded prompt (default: True)
    
    Returns:
        Prompt text as a string
    
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    # Check cache first
    if cache and filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    
    # Load from file
    prompt_path = _PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Available prompts: {list(_PROMPTS_DIR.glob('*.txt'))}"
        )
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    
    # Cache if requested
    if cache:
        _PROMPT_CACHE[filename] = prompt_text
    
    return prompt_text


def get_system_prompt() -> str:
    """Get the system prompt for JSON-based VLM interactions."""
    return load_prompt("system_prompt.txt")


def get_codegen_prompt() -> str:
    """Get the code generation prompt for VLM codegen."""
    return load_prompt("codegen_prompt.txt")


def reload_prompts():
    """Reload all prompts from disk (clears cache)."""
    global _PROMPT_CACHE
    _PROMPT_CACHE.clear()


def list_available_prompts() -> list:
    """List all available prompt files."""
    return [f.name for f in _PROMPTS_DIR.glob("*.txt")]


# Module-level variables (loaded lazily on first access)
_VLM_SYSTEM_PROMPT: Optional[str] = None
_VLM_CODEGEN_PROMPT: Optional[str] = None


# Backward-compatible module-level constants
def _get_vlm_system_prompt() -> str:
    """Get system prompt (lazy-loaded)."""
    global _VLM_SYSTEM_PROMPT
    if _VLM_SYSTEM_PROMPT is None:
        _VLM_SYSTEM_PROMPT = get_system_prompt()
    return _VLM_SYSTEM_PROMPT


def _get_vlm_codegen_prompt() -> str:
    """Get codegen prompt (lazy-loaded)."""
    global _VLM_CODEGEN_PROMPT
    if _VLM_CODEGEN_PROMPT is None:
        _VLM_CODEGEN_PROMPT = get_codegen_prompt()
    return _VLM_CODEGEN_PROMPT


# Use properties for backward compatibility with existing code that expects constants
class _PromptModule:
    """Helper class to provide prompt constants that load lazily."""
    
    @property
    def VLM_SYSTEM_PROMPT(self) -> str:
        return _get_vlm_system_prompt()
    
    @property
    def VLM_CODEGEN_PROMPT(self) -> str:
        return _get_vlm_codegen_prompt()


# Create a module-level instance for backward compatibility
_prompt_module = _PromptModule()

# Export as module-level "constants" (actually lazy-loaded)
def __getattr__(name: str):
    """Allow accessing VLM_SYSTEM_PROMPT and VLM_CODEGEN_PROMPT as module attributes."""
    if name == "VLM_SYSTEM_PROMPT":
        return _get_vlm_system_prompt()
    elif name == "VLM_CODEGEN_PROMPT":
        return _get_vlm_codegen_prompt()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

