# Code Generation UI Update

## Summary

Updated the Code Generation section to match other UI sections and automatically update the 3D rendering when code is generated.

## Changes Made

### 1. HTML Template (`templates/partials/_gen.html`)

**Before:**
- Used `<section>` tag (inconsistent)
- Simple `<div class="row">` layout
- No status indicator
- No code display area
- Minimal styling

**After:**
- Matches other sections with `<div class="section">` structure
- Proper `<header>` with title and collapse button
- `<div class="section-body">` for content
- Status pill showing generation state
- Code display textarea (readonly, monospace)
- Copy button for generated code
- File path indicator
- Consistent styling with other panels

### 2. JavaScript (`static/js/app.js`)

**Added Features:**
- Display generated code in textarea
- Status updates during generation:
  - "Generating code..." (orange)
  - "Code generated successfully" (green)
  - "Code generation failed" (red)
- Loading indicator integration
- Console logging for progress
- Copy to clipboard functionality
- Better error handling

**Flow:**
1. User clicks "Generate Code"
2. Status updates to "Generating code..."
3. Loading line appears
4. VLM processes request
5. Generated code displayed in textarea
6. Status updates to success/failure
7. 3D model automatically refreshes with new code
8. Console logs show progress

### 3. Backend (`optim.py`)

**Modified `/codegen` Endpoint:**
- Now includes `"code": code_txt` in JSON response
- Frontend can display the generated code immediately
- Code is still saved to file system as before

## User Experience

### Before:
```
[Code Generation Section]
- Reference image: [file input]
- Optional snapshot: [file input]
- [Generate Code] button
→ No feedback except alert/console errors
→ No way to see generated code
→ Had to manually check file system
```

### After:
```
[Code Generation Section]
Status: [No code generated yet]

- Reference image: [file input]
- Optional snapshot: [file input]  
- [Generate Code] button

[After generation:]
Status: [Code generated successfully] ✓

Generated Code:         [Copy]
┌────────────────────────────────┐
│ import cadquery as cq          │
│ ...                            │
│ class Rover(...):              │
│     ...                        │
└────────────────────────────────┘
Code saved to: generated/robot_base_vlm.py

✓ Console logs show:
  - "Generating code from VLM"
  - "✓ Code generated successfully"  
  - "Rebuilding 3D model..."
  - "✓ 3D model updated with new code"
```

## Features

### ✅ Status Indicators
- Pill badge showing current state
- Color-coded: gray → orange → green/red
- Clear feedback to user

### ✅ Code Display
- Readonly textarea with monospace font
- Scrollable for long code
- Syntax-highlighted appearance (gray background)
- Shows exactly what was generated

### ✅ Copy to Clipboard
- One-click copy button
- Temporary "Copied!" confirmation
- Error handling if clipboard unavailable

### ✅ Auto-Update 3D Model
- Automatically calls `refreshModel()` after generation
- Reloads GLB file
- Updates parameters hint
- User sees changes immediately

### ✅ Console Integration
- Progress messages in console panel
- Error messages visible
- Loading indicator in console

### ✅ Consistent Formatting
- Matches Recommendations, Components, VLM Prompt sections
- Collapsible with expand/collapse button
- Same header style
- Same spacing and layout

## Files Modified

1. `/cqparts_bucket/templates/partials/_gen.html` - UI structure
2. `/cqparts_bucket/static/js/app.js` - Frontend logic  
3. `/cqparts_bucket/optim.py` - Backend response

## Technical Details

### HTML Structure
```html
<div class="section" id="codegen">
  <header>
    <h3>Code Generation</h3>
    <button class="toggle" ...>Collapse</button>
  </header>
  <div class="section-body">
    <div class="pill" id="codegenStatus">...</div>
    <!-- inputs -->
    <div id="codegenOutput">
      <textarea id="codegenText" readonly>...</textarea>
      <button id="copyCode">Copy</button>
    </div>
  </div>
</div>
```

### JavaScript Flow
```javascript
btn.addEventListener('click', async () => {
  // 1. Validate input
  // 2. Update status → "Generating..."
  // 3. Show loading indicator
  // 4. Fetch /codegen
  // 5. Display code in textarea
  // 6. Update status → "Success" 
  // 7. Refresh 3D model
  // 8. Log to console
  // 9. Handle errors
});
```

### API Response
```json
{
  "ok": true,
  "code": "import cadquery...",  // NEW: full generated code
  "module_path": "generated/robot_base_vlm.py",
  "backup_path": "generated/robot_base_vlm_1234567.py",
  "code_length": 5432,
  "glb_updated": true,
  "message": "Generated robot_base.py..."
}
```

## Benefits

1. **Immediate Feedback** - User sees code right away
2. **Visual Updates** - 3D model refreshes automatically
3. **Better UX** - Clear status, progress messages
4. **Easy Copy** - One-click code copying
5. **Consistent** - Matches rest of UI
6. **Professional** - Polished appearance

## Testing

To test the changes:

1. Start server: `python optim.py`
2. Open browser: `http://localhost:5160`
3. Upload reference image
4. (Optional) Add prompt in VLM Prompt section
5. Click "Generate Code" in Code Generation section
6. Observe:
   - Status changes
   - Loading indicator
   - Console messages
   - Generated code appears
   - 3D model updates
   - Can copy code

## Future Enhancements

Possible additions:
- Syntax highlighting in textarea
- Download code button
- Code diff viewer (show changes from previous)
- Inline error highlighting
- Apply button to rebuild without regenerating
- History of generated codes

## Compatibility

- Works with existing VLM setup (Ollama or fine-tuned)
- Backward compatible with existing code
- No breaking changes to API
- Preserves all file-saving behavior

