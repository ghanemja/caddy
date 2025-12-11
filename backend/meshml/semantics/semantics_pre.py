"""
Pre-VLM semantics: category classification and candidate parameter generation.

This module uses a VLM to:
1. Classify the object into a ShapeNet-like category
2. Propose candidate semantic parameter names and descriptions
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os

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
        if not hasattr(self, "parts") or self.parts is None:
            self.parts = []


def infer_category_and_candidates(
    image_paths: List[str],
    vlm: VLMClient,
    reference_image_path: Optional[str] = None,
) -> PreVLMOutput:
    """
    Use the VLM to classify the object and propose candidate semantic parameters.

    Args:
        image_paths: list of paths to rendered images of the mesh
        vlm: VLM client instance
        reference_image_path: Optional path to reference image from Step 1 (prioritized for classification)

    Returns:
        PreVLMOutput with category and candidate parameters
    """
    # Use ONLY the reference image from Step 1 for category classification
    images_for_classification = []
    if reference_image_path and os.path.exists(reference_image_path):
        # Use ONLY the reference image for category classification
        images_for_classification = [reference_image_path]
        print(
            f"[Pre-VLM] ✓ Using ONLY reference image from Step 1 for category classification: {reference_image_path}",
            flush=True,
        )
        print(
            f"[Pre-VLM]   File exists: {os.path.exists(reference_image_path)}, size: {os.path.getsize(reference_image_path) if os.path.exists(reference_image_path) else 0} bytes",
            flush=True,
        )
    else:
        # Fallback: use provided images if no reference image
        images_for_classification = image_paths.copy()
        print(
            f"[Pre-VLM] ⚠ No reference image found at {reference_image_path if reference_image_path else 'None'}, using {len(image_paths)} rendered view(s) for classification",
            flush=True,
        )

    system_prompt = """You are an expert at identifying objects from images. 
You will be given a reference image uploaded by the user showing a real object.

Your task:
Examine the image carefully and identify what object is shown. Look at:
- The overall shape and structure
- Distinctive features (wings, wheels, handles, tines, blades, etc.)
- What the object appears to be used for
- Common characteristics of similar objects

Identify the object's category by its common name (e.g., "rake", "airplane", "chair", "car", "shovel", "hammer", etc.). 
Use a single, specific word that best describes the object. Only use "unknown" if you truly cannot identify what the object is.

Provide your confidence level (0.0 to 1.0) and reasoning for why you identified it as this category.

2. Identify the major semantic parts likely present given the category
   (e.g., rake → ["handle", "head", "tines"]; airplane → ["fuselage", "left_wing", "right_wing", "tail"]).

3. Propose a list of high-level semantic parameter names (no numbers yet).
   These describe dimensions humans care about:
   - For airplanes: wing_span, chord_length, tail_height, fuselage_length
   - For chairs: seat_height, back_height, leg_length, seat_width
   - For cars: wheelbase, wheel_diameter, body_length, roof_height
   - For unknown shapes: use generic names like major_extent, minor_extent, thickness

4. Output ONLY JSON using this exact structure:
{{
  "category": "...",
  "category_confidence": 0.0-1.0,
  "category_reasoning": "brief explanation of why this category was chosen",
  "parts": [...],
  "candidate_parameters": [
    {{"name": "...", "description": "..."}},
    ...
  ]
}}"""

    # Build user message with schema hint
    schema_hint = """{{
  "category": "string (e.g., airplane, chair, car)",
  "category_confidence": 0.0-1.0,
  "category_reasoning": "string (brief explanation of why this category was chosen)",
  "parts": ["string array of major semantic parts"],
  "candidate_parameters": [
    {{
      "name": "string (e.g., wing_span, seat_height)",
      "description": "string (brief description)"
    }}
  ]
}}"""

    # Build user message for category classification
    if reference_image_path and os.path.exists(reference_image_path):
        user_message = """Examine this reference image uploaded by the user. 

Identify what object is shown in the image. Look for:
- Overall shape and proportions
- Distinctive features (wings, wheels, legs, handles, tines, etc.)
- What the object is used for
- Common characteristics

Be precise - if it's clearly a rake, say "rake". If it's clearly an airplane, say "airplane". Only use "unknown" if you truly cannot identify it.

Based on the category you identify, also suggest:
- Major semantic parts that would be present (e.g., for a rake: ["handle", "head", "tines"])
- Candidate parameter names that would be useful for CAD modeling (e.g., for a rake: handle_length, tine_count, tine_spacing)

Return ONLY valid JSON matching the specified structure."""
    else:
        # Fallback message if no reference image
        num_images = len(images_for_classification)
        user_message = f"""Analyze the following {num_images} rendered view(s) of a 3D mesh from different angles.

CAREFULLY examine all views to identify the object category. Look for:
- Overall shape and proportions
- Distinctive features (wings, wheels, legs, handles, etc.)
- Symmetry and structure
- Scale and relative dimensions

Be precise - if it's clearly an airplane, don't say "unknown". If it's ambiguous, explain why in category_reasoning.

Identify the object category, major parts, and propose semantic parameter names that would be useful for CAD modeling.

Return ONLY valid JSON matching the specified structure."""

    # Prepare images (use prioritized list)
    images = [VLMImage(path=path) for path in images_for_classification]

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

        # Normalize category to lowercase and clean up
        category = category.lower().strip()

        # Remove quotes if present
        if category.startswith('"') and category.endswith('"'):
            category = category[1:-1]
        if category.startswith("'") and category.endswith("'"):
            category = category[1:-1]

        # Extract just the first word if multiple words (e.g., "garden rake" -> "rake")
        # But keep compound words like "motorbike", "skateboard"
        category_words = category.split()
        if len(category_words) > 1:
            # Check if it's a compound word (common ones)
            compound_words = ["motorbike", "skateboard", "screwdriver", "airplane"]
            if category not in compound_words:
                # Take the most specific word (usually the last one)
                category = category_words[-1]

        # Only reject if it's truly empty or "unknown"
        if not category or category == "unknown":
            print(
                f"[Pre-VLM] Category is empty or 'unknown', checking reasoning...",
                flush=True,
            )
            reasoning = response.get("category_reasoning", "").lower()
            if reasoning:
                # Try to extract a category from reasoning
                common_categories = [
                    "rake",
                    "shovel",
                    "airplane",
                    "car",
                    "chair",
                    "table",
                    "lamp",
                    "sofa",
                    "motorbike",
                    "bag",
                    "bed",
                    "boat",
                    "bottle",
                    "cabinet",
                    "monitor",
                    "pistol",
                    "rocket",
                    "skateboard",
                    "hammer",
                    "wrench",
                    "pliers",
                    "drill",
                    "saw",
                    "knife",
                    "fork",
                    "spoon",
                    "cup",
                    "mug",
                    "bowl",
                ]
                for cat in common_categories:
                    if cat in reasoning:
                        print(
                            f"[Pre-VLM] Extracted '{cat}' from reasoning: '{reasoning[:100]}...'",
                            flush=True,
                        )
                        category = cat
                        break

            if not category or category == "unknown":
                print(
                    f"[Pre-VLM] Could not determine category, using 'unknown'",
                    flush=True,
                )
                category = "unknown"
        else:
            print(
                f"[Pre-VLM] Identified category: '{category}'",
                flush=True,
            )

        # Extract parts first
        parts = response.get("parts", [])
        if not isinstance(parts, list):
            parts = []

        # Extract confidence and reasoning if available
        category_confidence = response.get("category_confidence", 1.0)
        category_reasoning = response.get("category_reasoning", "")
        if category_reasoning:
            print(
                f"[Pre-VLM] Category: {category} (confidence: {category_confidence:.2f})"
            )
            print(f"[Pre-VLM] Reasoning: {category_reasoning}")

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
