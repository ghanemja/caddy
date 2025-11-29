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
) -> PostVLMOutput:
    """
    Use the VLM to reconcile candidate semantic params with raw geometric params.
    
    Args:
        image_paths: list of paths to rendered images (optional, for context)
        pre_output: output from pre-VLM step (category + candidate params)
        raw_parameters: raw geometric parameters from PointNet++ segmentation
        vlm: VLM client instance
        
    Returns:
        PostVLMOutput with final semantic parameters
    """
    # Build system prompt
    system_prompt = """You are a 3D geometry parameter mapping assistant. Your task is to:
1. Map candidate semantic parameter names to raw geometric parameters
2. Compute semantic parameter values from raw parameters
3. Assign confidence scores (0.0 to 1.0) to each mapping
4. Drop parameters that are irrelevant or cannot be reliably computed

You will receive:
- The object category
- Candidate semantic parameters (names + descriptions)
- Raw geometric parameters (IDs + values + descriptions)

For each candidate parameter, determine:
- Which raw parameter(s) should be used to compute it
- The computed value
- Appropriate units
- A confidence score

Return a JSON response with final_parameters array."""

    # Serialize inputs for user message
    candidate_params_str = json.dumps([
        {"name": p.name, "description": p.description}
        for p in pre_output.candidate_parameters
    ], indent=2)
    
    raw_params_str = json.dumps([
        {
            "id": p.id,
            "value": p.value,
            "units": p.units,
            "description": p.description,
        }
        for p in raw_parameters
    ], indent=2)
    
    schema_hint = """{
  "final_parameters": [
    {
      "name": "string (semantic name)",
      "value": float,
      "units": "string or null",
      "description": "string",
      "confidence": float (0.0 to 1.0),
      "raw_sources": ["string (raw param IDs)"]
    }
  ]
}"""
    
    user_message = f"""Object category: {pre_output.category}

Candidate semantic parameters:
{candidate_params_str}

Raw geometric parameters:
{raw_params_str}

Map the candidate parameters to raw parameters and compute values.

Expected JSON schema:
{schema_hint}

Return the JSON response now."""

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
        final_params_raw = response.get("final_parameters", [])
        final_params = []
        
        for param in final_params_raw:
            if isinstance(param, dict):
                name = param.get("name", "")
                value = param.get("value", 0.0)
                units = param.get("units")
                description = param.get("description", "")
                confidence = param.get("confidence", 0.5)
                raw_sources = param.get("raw_sources", [])
                
                if name:
                    final_params.append(
                        FinalParameter(
                            name=name,
                            value=float(value),
                            units=units,
                            description=description,
                            confidence=float(confidence),
                            raw_sources=raw_sources if isinstance(raw_sources, list) else [],
                        )
                    )
        
        return PostVLMOutput(
            final_parameters=final_params,
            raw_response=response,
        )
    
    except Exception as e:
        print(f"[Post-VLM] Error parsing response: {e}")
        # Fallback
        return _fallback_parameter_mapping(pre_output, raw_parameters)


def _fallback_parameter_mapping(
    pre_output: PreVLMOutput,
    raw_parameters: List[RawParameter],
) -> PostVLMOutput:
    """
    Fallback heuristic: create default parameter mappings.
    
    This is used when VLM response is malformed or unavailable.
    """
    final_params = []
    
    # Map first few candidate params to first few raw params
    num_to_map = min(len(pre_output.candidate_parameters), len(raw_parameters))
    
    for i in range(num_to_map):
        candidate = pre_output.candidate_parameters[i]
        raw = raw_parameters[i]
        
        final_params.append(
            FinalParameter(
                name=candidate.name,
                value=raw.value,
                units=raw.units,
                description=candidate.description,
                confidence=0.5,  # Low confidence for fallback
                raw_sources=[raw.id],
            )
        )
    
    return PostVLMOutput(
        final_parameters=final_params,
        raw_response={"fallback": True},
    )

