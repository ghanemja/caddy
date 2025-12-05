# Changes: Generic Parameter IDs (p1, p2, ...) + VLM Semantic Name Proposal

## Summary

Updated the semantic ingestion pipeline to use generic parameter IDs (p1, p2, p3, ...) instead of semantic names, and added VLM-based semantic name proposal functionality.

## Changes Made

### 1. Updated `RawParameter` (`vlm_cad/semantics/types.py`)
- ✅ Changed `id` field to use generic identifiers: "p1", "p2", "p3", ...
- ✅ Added optional `part_labels` field for part context
- ✅ Updated docstring to reflect generic IDs

### 2. Updated `FinalParameter` (`vlm_cad/semantics/types.py`)
- ✅ Added `id` field (generic identifier: "p1", "p2", ...)
- ✅ Renamed `name` to `semantic_name` (proposed semantic name)
- ✅ Added `@property name` for backward compatibility
- ✅ Updated docstring

### 3. Updated `_extract_raw_parameters()` (`vlm_cad/semantics/ingest_mesh.py`)
- ✅ Changed all parameter IDs from semantic names (e.g., "bbox_x_length") to generic IDs (p1, p2, p3, ...)
- ✅ Added `part_labels` to per-part parameters
- ✅ Parameters are numbered sequentially in order of generation

### 4. Updated `refine_parameters_with_vlm()` (`vlm_cad/semantics/semantics_post.py`)
- ✅ Changed from mapping candidate params to raw params → proposing semantic names for generic params
- ✅ Updated system prompt to instruct VLM to propose semantic names based on:
  - Object category
  - Geometric descriptions
  - Part structure
- ✅ Updated JSON schema to expect:
  ```json
  {
    "parameters": [
      {
        "id": "p1",
        "proposed_name": "wing_span",
        "proposed_description": "...",
        "confidence": 0.95
      }
    ]
  }
  ```
- ✅ Added `part_labels` parameter to function signature
- ✅ Updated response parsing to create `FinalParameter` with both `id` and `semantic_name`

### 5. Added `build_user_review_payload()` (`vlm_cad/semantics/semantics_post.py`)
- ✅ New function to build JSON-ready structure for frontend user review
- ✅ Returns structure with: id, proposed_name, proposed_description, value, units, confidence

### 6. Updated `IngestResult` (`vlm_cad/semantics/ingest_mesh.py`)
- ✅ Renamed `final_parameters` to `proposed_parameters` (more accurate)
- ✅ Added `@property final_parameters` for backward compatibility
- ✅ Updated field order: `raw_parameters` first, then `proposed_parameters`

### 7. Updated `ingest_mesh_to_semantic_params()` (`vlm_cad/semantics/ingest_mesh.py`)
- ✅ Extracts part labels from segmentation
- ✅ Passes part labels to `refine_parameters_with_vlm()`
- ✅ Returns `IngestResult` with `proposed_parameters` instead of `final_parameters`

### 8. Updated `DummyVLMClient` (`vlm_cad/semantics/vlm_client.py`)
- ✅ Updated `_dummy_post_vlm_response()` to return new structure:
  ```json
  {
    "parameters": [
      {"id": "p1", "proposed_name": "wing_span", ...}
    ]
  }
  ```

### 9. Updated API endpoint (`cqparts_bucket/optim.py`)
- ✅ Updated `/ingest_mesh` endpoint to return both `raw_parameters` and `proposed_parameters`
- ✅ Added backward compatibility: also returns `final_parameters` (alias for `proposed_parameters`)
- ✅ Updated parameter serialization to include `id` and `semantic_name`
- ✅ Added `part_labels` to raw parameter response

### 10. Updated example script (`examples/run_ingest_mesh.py`)
- ✅ Updated to display both raw parameters and proposed parameters
- ✅ Shows mapping: `p1 → wing_span = 2.01 m`
- ✅ Uses backward-compatible `.name` property

### 11. Updated exports (`vlm_cad/semantics/__init__.py`)
- ✅ Added `build_user_review_payload` to exports

## Backward Compatibility

✅ **Maintained** - All changes include backward compatibility:
- `FinalParameter.name` property returns `semantic_name`
- `IngestResult.final_parameters` property returns `proposed_parameters`
- API endpoint still returns `final_parameters` field (in addition to `proposed_parameters`)

## New Flow

1. **Geometry extraction** → Creates `RawParameter` objects with generic IDs (p1, p2, p3, ...)
2. **Pre-VLM** → Classifies object category and proposes candidate parameter names (unchanged)
3. **Post-VLM** → Proposes semantic names for generic parameters based on:
   - Category (airplane, chair, car, etc.)
   - Geometric descriptions
   - Part structure
4. **Output** → Returns both:
   - `raw_parameters`: Generic parameters (p1, p2, ...) with values and descriptions
   - `proposed_parameters`: Proposed semantic names (wing_span, chord_length, etc.) with confidence scores

## User Confirmation Flow (Ready for Frontend)

The `build_user_review_payload()` function prepares data for user review:

```python
payload = build_user_review_payload(result.proposed_parameters)
# Returns:
# {
#   "parameters": [
#     {
#       "id": "p1",
#       "proposed_name": "wing_span",
#       "proposed_description": "...",
#       "value": 2.01,
#       "units": "m",
#       "confidence": 0.95
#     }
#   ]
# }
```

Frontend can display:
- p1 → wing_span = 2.01 m (confidence: 0.95) [Accept] [Edit]

## Testing

✅ All type definitions tested and working
✅ Backward compatibility verified
✅ No linter errors

## Next Steps (Frontend)

1. Display raw parameters (p1, p2, ...) with their geometric descriptions
2. Display proposed semantic names with confidence scores
3. Allow user to accept/edit/reject proposals
4. Send confirmed mappings back to backend for CAD optimization

