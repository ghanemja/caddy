"""
Post-VLM semantics: parameter refinement and reconciliation.

This module uses a VLM to:
1. Map candidate semantic parameters to raw geometric parameters
2. Drop irrelevant or low-confidence parameters
3. Produce final semantic parameter list with values and confidences

IMPORTANT: This module does NOT use Hunyuan3D/P3-SAM.
- Hunyuan3D/P3-SAM is only used for segmentation (in backends.py)
- This module only uses:
  - VLM client (Ollama, fine-tuned model, or dummy)
  - Segmentation results (PartTable with user-provided names)
  - Geometry data (extents, centroids, shape hints)
  - Rendered mesh images (optional, for visual context)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json

from .vlm_client import VLMClient, VLMImage, VLMMessage
from .types import RawParameter, FinalParameter
from .semantics_pre import PreVLMOutput

if TYPE_CHECKING:
    from ..parts.parts import PartTable


@dataclass
class PostVLMOutput:
    """Output from post-VLM step."""

    final_parameters: List[FinalParameter]
    raw_response: Dict[str, Any]


def refine_parameters_per_part(
    image_paths: List[str],
    pre_output: PreVLMOutput,
    raw_parameters: List[RawParameter],
    vlm: VLMClient,
    part_table: Optional["PartTable"] = None,
) -> PostVLMOutput:
    """
    Refine parameters per-part, assigning semantic parameters based on each part's semantic name.

    This processes each part separately, using the part's semantic name (e.g., "wheel", "wing")
    and geometry to assign context-appropriate parameters (e.g., wheel → diameter, wing → span).

    Args:
        image_paths: list of paths to rendered images (optional, for context)
        pre_output: output from pre-VLM step (category + candidate params)
        raw_parameters: raw geometric parameters with generic IDs (p1, p2, p3, ...)
        vlm: VLM client instance
        part_table: PartTable with user-assigned semantic names for each part

    Returns:
        PostVLMOutput with proposed semantic parameters grouped by part
    """
    if not part_table:
        # Fallback to global processing if no part table
        return refine_parameters_with_vlm(
            image_paths, pre_output, raw_parameters, vlm, None, None
        )

    # Group raw parameters by part
    params_by_part: Dict[int, List[RawParameter]] = {}
    params_without_part = []

    print(
        f"[refine_parameters_per_part] Grouping {len(raw_parameters)} raw parameters by part...",
        flush=True,
    )

    for param in raw_parameters:
        if param.part_labels:
            for part_label in param.part_labels:
                # Extract part_id from label (e.g., "part_0" -> 0)
                try:
                    # Handle both "part_0" and "0" formats
                    label_clean = part_label.replace("part_", "").strip()
                    part_id = int(label_clean)
                    if part_id not in params_by_part:
                        params_by_part[part_id] = []
                    params_by_part[part_id].append(param)
                    print(
                        f"[refine_parameters_per_part] Assigned param {param.id} to part_id {part_id}",
                        flush=True,
                    )
                except (ValueError, AttributeError) as e:
                    print(
                        f"[refine_parameters_per_part] Could not parse part_label '{part_label}': {e}",
                        flush=True,
                    )
                    params_without_part.append(param)
        else:
            params_without_part.append(param)
            print(
                f"[refine_parameters_per_part] Param {param.id} has no part_labels",
                flush=True,
            )

    print(
        f"[refine_parameters_per_part] Grouped into {len(params_by_part)} parts: {list(params_by_part.keys())}",
        flush=True,
    )
    print(
        f"[refine_parameters_per_part] {len(params_without_part)} parameters without part assignment",
        flush=True,
    )

    # If no parameters are grouped by part, fall back to global processing
    if not params_by_part:
        print(
            f"[refine_parameters_per_part] No parameters grouped by part, falling back to global processing",
            flush=True,
        )
        return refine_parameters_with_vlm(
            image_paths, pre_output, raw_parameters, vlm, None, part_table
        )

    # Distribute parameters without part assignment to parts that have no parameters
    if params_without_part:
        parts_with_params = set(params_by_part.keys())
        all_part_ids = set(part_table.parts.keys()) if part_table else set()
        parts_without_params = all_part_ids - parts_with_params

        if parts_without_params:
            print(
                f"[refine_parameters_per_part] Distributing {len(params_without_part)} unassigned params to {len(parts_without_params)} parts without params",
                flush=True,
            )
            # Distribute evenly
            for i, param in enumerate(params_without_part):
                part_id = list(parts_without_params)[i % len(parts_without_params)]
                if part_id not in params_by_part:
                    params_by_part[part_id] = []
                params_by_part[part_id].append(param)
                print(
                    f"[refine_parameters_per_part] Assigned unassigned param {param.id} to part_id {part_id}",
                    flush=True,
                )

    all_final_params = []
    param_id_counter = 1

    # Process each part separately
    print(
        f"[refine_parameters_per_part] Processing {len(part_table.parts)} parts from PartTable",
        flush=True,
    )
    for part_id, part_info in part_table.parts.items():
        part_params = params_by_part.get(part_id, [])
        print(
            f"[refine_parameters_per_part] Part {part_id} has {len(part_params)} raw parameters",
            flush=True,
        )
        if not part_params:
            print(
                f"[refine_parameters_per_part] ⚠ Skipping part {part_id} - no raw parameters assigned",
                flush=True,
            )
            continue  # Skip parts with no parameters

        # Get part's semantic name (user-provided or provisional)
        provisional_name = (
            part_info.extra.get("provisional_name") if part_info.extra else None
        )
        part_name = part_info.name or provisional_name or f"part_{part_id}"

        # CRITICAL: Log which name is being used
        if part_info.name:
            print(
                f"[refine_parameters_per_part] ✓ Part {part_id} using USER-PROVIDED name: '{part_info.name}'",
                flush=True,
            )
        elif provisional_name:
            print(
                f"[refine_parameters_per_part] ⚠ Part {part_id} using provisional name: '{provisional_name}' (no user input)",
                flush=True,
            )
        else:
            print(
                f"[refine_parameters_per_part] ⚠ Part {part_id} using fallback name: '{part_name}' (no user input)",
                flush=True,
            )

        # Get part geometry info
        extents = (
            part_info.extents.tolist()
            if hasattr(part_info.extents, "tolist")
            else list(part_info.extents)
        )
        centroid = (
            part_info.centroid.tolist()
            if hasattr(part_info.centroid, "tolist")
            else list(part_info.centroid)
        )
        shape_hint = (
            part_info.extra.get("shape_hint", "unknown")
            if part_info.extra
            else "unknown"
        )

        # Call VLM for this specific part
        # Pass the explicit user-provided name if it exists
        part_final_params = _refine_parameters_for_single_part(
            image_paths=image_paths,
            pre_output=pre_output,
            raw_parameters=part_params,
            vlm=vlm,
            part_name=part_name,
            part_id=part_id,
            extents=extents,
            centroid=centroid,
            shape_hint=shape_hint,
            param_id_counter=param_id_counter,
            part_info_name=part_info.name,  # Pass explicit user-provided name
        )

        all_final_params.extend(part_final_params)
        param_id_counter += len(part_final_params)

    return PostVLMOutput(
        final_parameters=all_final_params,
        raw_response={"per_part_processing": True},
    )


def _refine_parameters_for_single_part(
    image_paths: List[str],
    pre_output: PreVLMOutput,
    raw_parameters: List[RawParameter],
    vlm: VLMClient,
    part_name: str,
    part_id: int,
    extents: List[float],
    centroid: List[float],
    shape_hint: str,
    param_id_counter: int,
    part_info_name: Optional[str] = None,  # Explicit user-provided name from PartInfo
) -> List[FinalParameter]:
    """
    Refine parameters for a single part based on its semantic name and geometry.

    VLM CONTEXT RECEIVED:
    - Object category (from pre-VLM step)
    - Part semantic name (user-provided from input boxes)
    - Part ID
    - Geometry: extents [x, y, z], centroid [x, y, z], shape hint
    - Raw geometric parameters (p1, p2, p3, ...) with values and descriptions
    - Rendered mesh images (optional, for visual context)

    NOTE: Hunyuan3D/P3-SAM is NOT used here - only segmentation results are used.

    Args:
        image_paths: rendered images for context
        pre_output: pre-VLM output with category
        raw_parameters: raw parameters for this part only
        vlm: VLM client (Ollama, fine-tuned, or dummy - NOT Hunyuan3D)
        part_name: semantic name of the part (e.g., "wheel", "wing") - FROM USER INPUT
        part_id: integer ID of the part
        extents: [x, y, z] extents of the part (from segmentation geometry)
        centroid: [x, y, z] centroid of the part (from segmentation geometry)
        shape_hint: shape classification (e.g., "long_thin", "flat_plate") (from segmentation)
        param_id_counter: starting counter for parameter IDs

    Returns:
        List of FinalParameter objects for this part
    """
    system_prompt = f"""You are assisting with assigning semantic parameters to a 3D part based on its semantic name and geometry.

You will be given:
- The object's category: {pre_output.category}
- A specific part with:
  - Semantic name: "{part_name}"
  - Part ID: {part_id}
  - Geometry: extents {extents}, centroid {centroid}, shape hint: {shape_hint}
- Raw geometric parameters for this part (p1, p2, p3, ...) with values and descriptions

Your task:
Assign contextually appropriate semantic parameter names based on:
1. The part's semantic name (e.g., "wheel" → diameter, radius, width; "wing" → span, chord, thickness)
2. The part's geometry (extents, shape)
3. Common parameters for that part type in the object category

Examples:
- For a "wheel" part: diameter, radius, width, hub_diameter
- For a "wing" part: span, chord_length, thickness, sweep_angle
- For a "fuselage" part: length, width, height, nose_radius
- For a "seat" part: width, depth, height, backrest_height
- For a "leg" part: length, diameter, taper_ratio

Return ONLY valid JSON with this structure:
{{
  "parameters": [
    {{
      "id": "p1",
      "proposed_name": "diameter",
      "proposed_description": "Wheel diameter",
      "confidence": 0.95
    }}
  ]
}}"""

    # Serialize raw parameters for this part
    raw_params_data = []
    for i, p in enumerate(raw_parameters):
        param_dict = {
            "id": p.id,
            "value": p.value,
            "units": p.units or "normalized",
            "description": p.description,
        }
        raw_params_data.append(param_dict)

    raw_params_str = json.dumps(raw_params_data, indent=2)

    # Check if part_name is user-provided (not just "part_X")
    # CRITICAL: Use part_info_name if provided - that's the explicit user-provided name
    if part_info_name:
        # User explicitly provided a name - use it and mark as user-provided
        part_name = part_info_name
        is_user_provided = True
        print(
            f"[_refine_parameters_for_single_part] ✓ Using USER-PROVIDED name: '{part_name}' for part {part_id}",
            flush=True,
        )
    else:
        # Check if part_name looks user-provided (not just "part_X")
        is_user_provided = (
            part_name
            and not part_name.startswith("part_")
            and part_name != f"part_{part_id}"
        )
        print(
            f"[_refine_parameters_for_single_part] Processing part {part_id} with name: '{part_name}'",
            flush=True,
        )
        print(
            f"[_refine_parameters_for_single_part]   Is user-provided: {is_user_provided} (part_info_name was None)",
            flush=True,
        )

    # Build conditional strings to avoid backslash issues in f-strings
    name_label = (
        "(USER-PROVIDED LABEL - USE THIS!)"
        if is_user_provided
        else "(provisional/auto-generated)"
    )
    critical_note = ""
    user_naming_note = ""
    example_section = ""
    user_naming_question = ""

    if is_user_provided:
        critical_note = f'CRITICAL: The user has explicitly labeled this part as "{part_name}". You MUST use this semantic name to determine appropriate parameters.\n'
        user_naming_note = "(this is what the USER called it - respect their naming!)"
        example_section = f'Since the user labeled this as "{part_name}", think about what parameters are relevant for a {part_name} in a {pre_output.category}. For example:\n'

        part_lower = part_name.lower()
        if "handle" in part_lower:
            example_section += (
                "- If it's a handle: handle_length, handle_diameter, grip_width\n"
            )
        if "tine" in part_lower:
            example_section += (
                "- If it's a tine: tine_length, tine_width, tine_spacing\n"
            )
        if "wheel" in part_lower:
            example_section += (
                "- If it's a wheel: wheel_diameter, tire_width, hub_diameter\n"
            )
        if "wing" in part_lower:
            example_section += (
                "- If it's a wing: wing_span, wing_chord, wing_thickness\n"
            )

        user_naming_question = f'- The user specifically named it "{part_name}" - what does that tell you about what parameters matter?\n'

    user_message = f"""Object category: {pre_output.category}

Part information:
- Semantic name: "{part_name}" {name_label}
- Part ID: {part_id}
- Geometry extents (x, y, z): {extents}
- Centroid (x, y, z): {centroid}
- Shape classification: {shape_hint}

Raw geometric parameters for this {part_name} part:
{raw_params_str}

Task:
{critical_note}Analyze what "{part_name}" means in the context of a {pre_output.category}. Based on:
1. The semantic meaning of "{part_name}" {user_naming_note}
2. The actual geometry values (extents: {extents}, shape: {shape_hint})
3. What parameters would be meaningful for describing or modifying this type of part

CRITICAL: Assign semantic parameter names that are SPECIFIC and MEANINGFUL for a "{part_name}" in a {pre_output.category}.

{example_section}Think about:
- What is this part's FUNCTION in the {pre_output.category}?
- What dimensions would someone want to MODIFY or SPECIFY?
- What makes this part unique or important?
{user_naming_question}Return ONLY valid JSON matching the specified structure."""

    schema_hint = """{
  "parameters": [
    {
      "id": "string (e.g., p1, p2, p3)",
      "proposed_name": "string (semantic name like diameter, span, width)",
      "proposed_description": "string (human-readable description)",
      "confidence": float (0.0 to 1.0)
    }
  ]
}"""

    images = [VLMImage(path=path) for path in image_paths] if image_paths else None

    messages = [
        VLMMessage(role="system", content=system_prompt),
        VLMMessage(role="user", content=user_message),
    ]

    # DEBUG: Print what context the VLM is receiving
    print(f"\n[VLM Context Debug] Part: {part_name} (ID: {part_id})")
    print(f"[VLM Context Debug] Object category: {pre_output.category}")
    print(
        f"[VLM Context Debug] Geometry - extents: {extents}, centroid: {centroid}, shape: {shape_hint}"
    )
    print(f"[VLM Context Debug] Raw parameters count: {len(raw_parameters)}")
    print(f"[VLM Context Debug] Images provided: {len(images) if images else 0}")
    print(
        f"[VLM Context Debug] User message preview (first 500 chars): {user_message[:500]}..."
    )
    print(
        f"[VLM Context Debug] Note: Hunyuan3D/P3-SAM is NOT used in this VLM step - only segmentation results are used\n"
    )

    try:
        response = vlm.complete_json(
            messages=messages,
            images=images,
            schema_hint=schema_hint,
        )
    except Exception as e:
        print(f"[Post-VLM] Error calling VLM for part {part_name}: {e}")
        # Fallback: create generic parameter names
        return _fallback_parameters_for_part(
            raw_parameters, part_name, param_id_counter
        )

    # Parse response
    try:
        proposed_params_raw = response.get("parameters", [])
        final_params = []

        raw_param_lookup = {p.id: p for p in raw_parameters}

        for param in proposed_params_raw:
            if isinstance(param, dict):
                param_id = param.get("id", "")
                proposed_name = param.get("proposed_name", "")
                proposed_description = param.get("proposed_description", "")
                confidence = param.get("confidence", 0.5)

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
                            raw_sources=[param_id],
                            part_labels=[f"part_{part_id}"],
                        )
                    )

        return final_params

    except Exception as e:
        print(f"[Post-VLM] Error parsing response for part {part_name}: {e}")
        import traceback

        traceback.print_exc()
        return _fallback_parameters_for_part(
            raw_parameters, part_name, param_id_counter
        )


def _fallback_parameters_for_part(
    raw_parameters: List[RawParameter],
    part_name: str,
    param_id_counter: int,
) -> List[FinalParameter]:
    """Fallback: create generic parameter names for a part when VLM fails."""
    final_params = []

    # Use generic names based on the part name and parameter descriptions
    # This is a fallback - ideally the VLM should handle this
    for i, raw in enumerate(raw_parameters):
        # Try to extract meaningful name from description
        desc_lower = raw.description.lower()
        if "extent" in desc_lower or "span" in desc_lower:
            if "x" in desc_lower or "axis" in desc_lower:
                semantic_name = f"{part_name}_length"
            elif "y" in desc_lower:
                semantic_name = f"{part_name}_width"
            elif "z" in desc_lower:
                semantic_name = f"{part_name}_height"
            else:
                semantic_name = f"{part_name}_extent_{i+1}"
        elif "center" in desc_lower or "coordinate" in desc_lower:
            if "x" in desc_lower:
                semantic_name = f"{part_name}_center_x"
            elif "y" in desc_lower:
                semantic_name = f"{part_name}_center_y"
            elif "z" in desc_lower:
                semantic_name = f"{part_name}_center_z"
            else:
                semantic_name = f"{part_name}_position_{i+1}"
        else:
            # Generic fallback
            semantic_name = f"{part_name}_param_{i+1}"

        final_params.append(
            FinalParameter(
                id=raw.id,
                semantic_name=semantic_name,
                value=raw.value,
                units=raw.units,
                description=raw.description,
                confidence=0.3,  # Low confidence for fallback
                raw_sources=[raw.id],
                part_labels=raw.part_labels or [],
            )
        )

    return final_params


def refine_parameters_with_vlm(
    image_paths: List[str],
    pre_output: PreVLMOutput,
    raw_parameters: List[RawParameter],
    vlm: VLMClient,
    part_labels: Optional[List[str]] = None,
    part_table: Optional["PartTable"] = None,
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

    # Build part labels info - prioritize user-provided labels from PartTable
    part_info = ""
    if part_table:
        # Use user-provided labels from PartTable
        named_parts = part_table.get_named_parts()
        if named_parts:
            part_names = [
                f"{name} (part_id: {info.part_id})"
                for name, info in named_parts.items()
            ]
            part_info = f"\nUser-labeled parts: {', '.join(part_names)}"
        else:
            # Fallback to part IDs if no names assigned
            part_ids = [str(pid) for pid in part_table.get_part_ids()]
            part_info = f"\nDetected parts (IDs): {', '.join(part_ids)}"
    elif part_labels:
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
        candidate_list = [
            f"- {p.name}: {p.description}" for p in pre_output.candidate_parameters[:10]
        ]
        candidate_params_str = (
            f"\n\nCandidate semantic names from previous step:\n"
            + "\n".join(candidate_list)
        )

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
                            part_labels=(
                                raw_param.part_labels if raw_param.part_labels else []
                            ),  # Copy part labels from raw parameter
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
                part_labels=(
                    raw.part_labels if raw.part_labels else []
                ),  # Copy part labels from raw parameter
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
