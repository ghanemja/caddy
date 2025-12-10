"""
VLM (Vision Language Model) Service
Handles VLM model loading and inference calls.
"""
from typing import Dict, Any, Optional, List
import sys
import os

# Import from legacy optim.py for now (will be migrated)
# BASE_DIR is now at root level (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BASE_DIR)

# These will be imported from optim.py initially
# TODO: Extract these functions into this service
def call_vlm(
    final_prompt: str,
    image_data_urls: Optional[List[str] | str],
    *,
    expect_json: bool = True,
) -> Dict[str, Any]:
    """
    Call VLM with prompt and images.
    
    Args:
        final_prompt: Text prompt for the VLM
        image_data_urls: Image data URLs (base64 encoded)
        expect_json: Whether to expect JSON response
        
    Returns:
        Dict with 'provider' and 'raw' response
    """
    # Import from legacy module
    from optim import call_vlm as _call_vlm
    return _call_vlm(final_prompt, image_data_urls, expect_json=expect_json)


def load_finetuned_model():
    """Load the fine-tuned VLM model (lazy loading)."""
    from optim import load_finetuned_model as _load_finetuned_model
    return _load_finetuned_model()

