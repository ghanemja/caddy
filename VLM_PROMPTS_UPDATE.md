# VLM Prompts Update Summary

## Changes Made

### 1. Updated `semantics_pre.py`

#### System Prompt
✅ **Replaced** placeholder system message with the specified expert 3D object analyst prompt that:
- Lists all 16 common 3D model categories
- Instructs to identify major semantic parts
- Provides examples for different categories (airplanes, chairs, cars)
- Specifies exact JSON output structure with `category`, `parts`, and `candidate_parameters`

#### User Message
✅ **Updated** to be more concise and direct:
- Mentions number of rendered views
- Instructs to identify category, parts, and propose semantic parameters
- Emphasizes "ONLY valid JSON"

#### Schema Hint
✅ **Updated** to include `parts` field in the expected JSON structure

#### PreVLMOutput Dataclass
✅ **Added** `parts: List[str]` field to store identified parts
✅ **Added** `__post_init__` to ensure parts list exists even if not provided

#### Response Parsing
✅ **Updated** to extract and store `parts` from VLM response
✅ **Normalized** category to lowercase for consistency
✅ **Updated** fallback responses to include empty `parts` list

### 2. Updated `semantics_post.py`

#### System Prompt
✅ **Replaced** placeholder with the specified prompt that:
- Explains the task: naming geometric parameters for 3D models
- Lists what will be given (category, parts, raw parameters)
- Provides detailed examples for airplanes and furniture
- Shows exact JSON output structure with examples

#### User Message
✅ **Enhanced** to include:
- Object category and identified parts from pre-VLM step
- Candidate semantic names from previous step (if available)
- Detected parts from segmentation
- Clear instructions to propose names based on category, parts, and descriptions

#### Schema Hint
✅ **Already matches** the specified structure with:
- `parameters` array
- Each parameter has: `id`, `proposed_name`, `proposed_description`, `confidence`

### 3. Updated `ingest_mesh.py`

✅ **Added** `identified_parts` to `IngestResult.extra` to store parts from pre-VLM
✅ **Ensured** all necessary fields are accessible:
- `raw_parameters` (p1, p2, p3, ...)
- `proposed_parameters` (semantic naming candidates)
- `category` + `identified_parts` (from pre-VLM)

### 4. Updated `vlm_client.py` (DummyVLMClient)

✅ **Updated** `_dummy_pre_vlm_response()` to return new structure:
- Includes `parts` field
- Uses lowercase category ("airplane" not "Airplane")
- Matches the new JSON schema

## Verification

### ✅ Prompts Inserted
- Both `semantics_pre.py` and `semantics_post.py` now contain the full specified prompts
- No placeholders remain

### ✅ VLM Calls Updated
- `infer_category_and_candidates()` uses the new system prompt
- `refine_parameters_with_vlm()` uses the new system prompt
- Both functions construct messages with the full prompt text

### ✅ JSON Schemas Match
- Pre-VLM: `{category, parts, candidate_parameters}`
- Post-VLM: `{parameters: [{id, proposed_name, proposed_description, confidence}]}`

### ✅ IngestResult Includes All Fields
- `raw_parameters`: Generic parameters (p1, p2, p3, ...)
- `proposed_parameters`: Semantic naming candidates
- `category`: From pre-VLM
- `extra.identified_parts`: Parts identified by pre-VLM

## Testing

✅ No linter errors
✅ All dataclass structures updated
✅ Backward compatibility maintained (PreVLMOutput.parts defaults to empty list)

## Next Steps

The prompts are now ready for use with actual VLM calls. The system will:
1. Pre-VLM: Classify category, identify parts, propose candidate parameter names
2. Post-VLM: Propose semantic names for generic parameters (p1, p2, ...) based on category, parts, and geometry

