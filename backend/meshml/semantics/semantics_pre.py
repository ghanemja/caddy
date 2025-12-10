"""
Pre-VLM semantics: category classification and candidate parameter generation.

This module uses a VLM to:
1. Classify the object into a ShapeNet-like category
2. Propose candidate semantic parameter names and descriptions
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json

from .vlm_client import VLMClient, VLMImage, VLMMessage
from .types import CandidateParameter


@dataclass
class PreVLMOutput:
    """Output from pre-VLM step."""
    category: str
    parts: List[str]  # Major semantic parts identified
    candidate_parameters: List[CandidateParameter]
    raw_response: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure parts list exists even if not provided."""
        if not hasattr(self, 'parts') or self.parts is None:
            self.parts = []


def infer_category_and_candidates(
    image_paths: List[str],
    vlm: VLMClient,
) -> PreVLMOutput:
    """
    Use the VLM to classify the object and propose candidate semantic parameters.
    
    Args:
        image_paths: list of paths to rendered images of the mesh
        vlm: VLM client instance
        
    Returns:
        PreVLMOutput with category and candidate parameters
    """
    # Build system prompt
    system_prompt = """You are an expert 3D object analyst. You will be given several rendered views
of a 3D mesh. Your tasks:

1. Identify the object's category among common 3D model classes:
   ["airplane", "car", "chair", "table", "lamp", "sofa", "motorbike", 
    "bag", "bed", "boat", "bottle", "cabinet", "monitor", "pistol",
    "rocket", "skateboard"].

2. Identify the major semantic parts likely present given the category
   (e.g., airplane → ["fuselage", "left_wing", "right_wing", "tail"]).

3. Propose a list of high-level semantic parameter names (no numbers yet).
   These describe dimensions humans care about:
   - For airplanes: wing_span, chord_length, tail_height, fuselage_length
   - For chairs: seat_height, back_height, leg_length, seat_width
   - For cars: wheelbase, wheel_diameter, body_length, roof_height
   - For unknown shapes: use generic names like major_extent, minor_extent, thickness

4. Output ONLY JSON using this exact structure:
{
  "category": "...",
  "parts": [...],
  "candidate_parameters": [
    {"name": "...", "description": "..."},
    ...
  ]
}"""

    # Build user message with schema hint
    schema_hint = """{
  "category": "string (e.g., Airplane, Chair, Car)",
  "candidate_parameters": [
    {
      "name": "string (e.g., wing_span, seat_height)",
      "description": "string (brief description)"
    }
  ]
}"""
    
    user_message = f"""Analyze the following {len(image_paths)} rendered view(s) of a 3D mesh.

Identify the object category, major parts, and propose semantic parameter names that would be useful for CAD modeling.

Return ONLY valid JSON matching the specified structure."""

    # Prepare images
    images = [VLMImage(path=path) for path in image_paths]
    
    # Build messages
    messages = [
        VLMMessage(role="system", content=system_prompt),
        VLMMessage(role="user", content=user_message),
    ]
    
    # Call VLM
    try:
        response = vlm.complete_json(
            messages=messages,
            images=images,
            schema_hint=schema_hint,
        )
    except Exception as e:
        print(f"[Pre-VLM] Error calling VLM: {e}")
        # Fallback response
        response = {
            "category": "unknown",
            "parts": [],
            "candidate_parameters": [],
        }
    
    # Parse response
    try:
        category = response.get("category", "unknown")
        if not isinstance(category, str):
            category = "unknown"
        
        candidate_params_raw = response.get("candidate_parameters", [])
        candidate_params = []
        for param in candidate_params_raw:
            if isinstance(param, dict):
                name = param.get("name", "")
                description = param.get("description", "")
                if name:
                    candidate_params.append(
                        CandidateParameter(name=name, description=description)
                    )
        
        return PreVLMOutput(
            category=category,
            parts=parts if isinstance(parts, list) else [],
            candidate_parameters=candidate_params,
            raw_response=response,
        )
    
    except Exception as e:
        print(f"[Pre-VLM] Error parsing response: {e}")
        # Fallback
        return PreVLMOutput(
            category="unknown",
            parts=[],
            candidate_parameters=[],
            raw_response=response,
        )

