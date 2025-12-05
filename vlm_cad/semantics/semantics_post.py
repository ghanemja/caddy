"""
Post-VLM semantics: parameter refinement and reconciliation.

This module uses a VLM to:
1. Map candidate semantic parameters to raw geometric parameters
2. Drop irrelevant or low-confidence parameters
3. Produce final semantic parameter list with values and confidences
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json

from .vlm_client import VLMClient, VLMImage, VLMMessage
from .types import RawParameter, FinalParameter
from .semantics_pre import PreVLMOutput


@dataclass
class PostVLMOutput:
    """Output from post-VLM step."""
    final_parameters: List[FinalParameter]
    raw_response: Dict[str, Any]


def refine_parameters_with_vlm(
    image_paths: List[str],
    pre_output: PreVLMOutput,
    raw_parameters: List[RawParameter],
    vlm: VLMClient,
    part_labels: Optional[List[str]] = None,
) -> PostVLMOutput:
    """
    Use the VLM to propose semantic names for generic raw parameters (p1, p2, ...).
    
    Args:
        image_paths: list of paths to rendered images (optional, for context)
        pre_output: output from pre-VLM step (category + candidate params)
        raw_parameters: raw geometric parameters with generic IDs (p1, p2, p3, ...)
        vlm: VLM client instance
        part_labels: optional list of detected part names from segmentation
        
    Returns:
        PostVLMOutput with proposed semantic parameters
    """
    # Build system prompt
    system_prompt = """You are assisting with naming geometric parameters for a 3D model.

You will be given:
- The object's predicted category and part list.
- A set of RAW geometric parameters, each having:
    id: "p1", "p2", ...
    value: numeric measurement
    units: e.g. "m" or "normalized"
    description: explanation of the geometry
- Candidate semantic names from a previous step.

Your tasks:

1. For each raw parameter (p1, p2, ...), propose a meaningful semantic name.
   - Use the object's category and parts to decide names.
   - Example for airplanes:
       p1 → wing_span
       p2 → chord_length
       p3 → fuselage_length
   - For furniture:
       p1 → seat_height
       p2 → backrest_height
   - If unsure, propose generic names:
       p1 → major_extent, p2 → minor_extent

2. For each proposed name, create:
   - A human-readable description
   - A confidence score between 0 and 1

3. Output ONLY JSON using this structure:
{
  "parameters": [
    {
      "id": "p1",
      "proposed_name": "wing_span",
      "proposed_description": "Distance between wing tips",
      "confidence": 0.94
    },
    {
      "id": "p2",
      "proposed_name": "chord_length",
      "proposed_description": "Distance from leading to trailing edge",
      "confidence": 0.89
    }
  ]
}"""

    # Serialize raw parameters with all available info
    raw_params_data = []
    for p in raw_parameters:
        param_dict = {
            "id": p.id,
            "value": p.value,
            "units": p.units or "normalized",
            "description": p.description,
        }
        if p.part_labels:
            param_dict["part_labels"] = p.part_labels
        raw_params_data.append(param_dict)
    
    raw_params_str = json.dumps(raw_params_data, indent=2)
    
    # Build part labels info
    part_info = ""
    if part_labels:
        part_info = f"\nDetected parts: {', '.join(part_labels)}"
    elif raw_parameters:
        # Extract unique part labels from raw parameters
        unique_parts = set()
        for p in raw_parameters:
            if p.part_labels:
                unique_parts.update(p.part_labels)
        if unique_parts:
            part_info = f"\nDetected parts: {', '.join(sorted(unique_parts))}"
    
    schema_hint = """{
  "parameters": [
    {
      "id": "string (e.g., p1, p2, p3)",
      "proposed_name": "string (semantic name like wing_span, chord_length)",
      "proposed_description": "string (human-readable description)",
      "confidence": float (0.0 to 1.0)
    }
  ]
}"""
    
    # Build candidate parameters context (if available)
    candidate_params_str = ""
    if pre_output.candidate_parameters:
        candidate_list = [f"- {p.name}: {p.description}" for p in pre_output.candidate_parameters[:10]]
        candidate_params_str = f"\n\nCandidate semantic names from previous step:\n" + "\n".join(candidate_list)
    
    user_message = f"""Object category: {pre_output.category}
Parts identified: {', '.join(pre_output.parts) if pre_output.parts else 'None specified'}{part_info}{candidate_params_str}

Raw geometric parameters (p1, p2, p3, ...):
{raw_params_str}

For each raw parameter, propose a meaningful semantic name based on:
- The object category ({pre_output.category})
- The identified parts: {', '.join(pre_output.parts) if pre_output.parts else 'None'}
- The geometric description of each parameter
- Candidate semantic names (if provided above)

Return ONLY valid JSON matching the specified structure."""

    # Prepare images (optional, for context)
    images = [VLMImage(path=path) for path in image_paths] if image_paths else None
    
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
        print(f"[Post-VLM] Error calling VLM: {e}")
        # Fallback: create default mappings
        return _fallback_parameter_mapping(pre_output, raw_parameters)
    
    # Parse response
    try:
        # VLM returns parameters array with proposed names
        proposed_params_raw = response.get("parameters", [])
        final_params = []
        
        # Create a lookup for raw parameters by ID
        raw_param_lookup = {p.id: p for p in raw_parameters}
        
        for param in proposed_params_raw:
            if isinstance(param, dict):
                param_id = param.get("id", "")
                proposed_name = param.get("proposed_name", "")
                proposed_description = param.get("proposed_description", "")
                confidence = param.get("confidence", 0.5)
                
                # Find the corresponding raw parameter
                raw_param = raw_param_lookup.get(param_id)
                if raw_param and proposed_name:
                    final_params.append(
                        FinalParameter(
                            id=param_id,
                            semantic_name=proposed_name,
                            value=raw_param.value,
                            units=raw_param.units,
                            description=proposed_description or raw_param.description,
                            confidence=float(confidence),
                            raw_sources=[param_id],  # Typically just the ID itself
                        )
                    )
        
        return PostVLMOutput(
            final_parameters=final_params,
            raw_response=response,
        )
    
    except Exception as e:
        print(f"[Post-VLM] Error parsing response: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        return _fallback_parameter_mapping(pre_output, raw_parameters)


def _fallback_parameter_mapping(
    pre_output: PreVLMOutput,
    raw_parameters: List[RawParameter],
) -> PostVLMOutput:
    """
    Fallback heuristic: create default parameter mappings with generic names.
    
    This is used when VLM response is malformed or unavailable.
    """
    final_params = []
    
    # Use first few candidate params as semantic names, or generate generic ones
    for i, raw in enumerate(raw_parameters[:10]):  # Limit to first 10
        if i < len(pre_output.candidate_parameters):
            semantic_name = pre_output.candidate_parameters[i].name
        else:
            semantic_name = f"param_{i+1}"  # Generic fallback
        
        final_params.append(
            FinalParameter(
                id=raw.id,
                semantic_name=semantic_name,
                value=raw.value,
                units=raw.units,
                description=raw.description,
                confidence=0.3,  # Low confidence for fallback
                raw_sources=[raw.id],
            )
        )
    
    return PostVLMOutput(
        final_parameters=final_params,
        raw_response={"fallback": True},
    )


def build_user_review_payload(final_parameters: List[FinalParameter]) -> Dict[str, Any]:
    """
    Build a JSON-ready structure for frontend user review/confirmation.
    
    Each item contains:
    - id (p1, p2, p3, ...)
    - proposed semantic name
    - proposed description
    - confidence
    - value + units
    
    Args:
        final_parameters: List of FinalParameter objects with proposed semantic names
        
    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        "parameters": [
            {
                "id": param.id,
                "proposed_name": param.semantic_name,
                "proposed_description": param.description,
                "value": param.value,
                "units": param.units or "normalized",
                "confidence": param.confidence,
            }
            for param in final_parameters
        ]
    }

