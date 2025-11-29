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
    candidate_parameters: List[CandidateParameter]
    raw_response: Dict[str, Any]


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
    system_prompt = """You are a 3D geometry analysis assistant. Your task is to:
1. Classify 3D objects into ShapeNet-like categories (Airplane, Chair, Car, Table, Lamp, etc.)
2. Propose semantic parameter names and descriptions that would be useful for CAD modeling

Analyze the provided images of a 3D object and return a JSON response with:
- category: the object category (e.g., "Airplane", "Chair", "Car")
- candidate_parameters: a list of objects, each with:
  - name: a semantic parameter name (e.g., "wing_span", "seat_height")
  - description: a brief description of what this parameter measures

Focus on parameters that are:
- Semantically meaningful (e.g., "wing_span" not "bbox_x_length")
- Useful for CAD modeling and design
- Measurable from the geometry

Return only valid JSON."""

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
    
    user_message = f"""Analyze these images of a 3D object and classify it, then propose semantic parameters.

Expected JSON schema:
{schema_hint}

Return the JSON response now."""

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
            candidate_parameters=candidate_params,
            raw_response=response,
        )
    
    except Exception as e:
        print(f"[Pre-VLM] Error parsing response: {e}")
        # Fallback
        return PreVLMOutput(
            category="unknown",
            candidate_parameters=[],
            raw_response=response,
        )

