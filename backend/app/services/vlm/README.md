# VLM Services

This directory contains all VLM (Vision Language Model) related services, clients, and prompts.

## Structure

```
vlm/
├── __init__.py              # Package exports
├── prompts_loader.py        # Loads prompts from text files
├── prompts/                 # Prompt text files (easy to edit!)
│   ├── __init__.py
│   ├── system_prompt.txt    # JSON-based VLM interactions
│   └── codegen_prompt.txt   # Code generation prompts
└── README.md                # This file
```

## Usage

### Loading Prompts

```python
from app.services.vlm.prompts_loader import get_system_prompt, get_codegen_prompt

system_prompt = get_system_prompt()
codegen_prompt = get_codegen_prompt()
```

### Editing Prompts

Simply edit the `.txt` files in the `prompts/` directory. Changes will be loaded automatically (prompts are cached by default but can be reloaded).

To reload prompts programmatically:
```python
from app.services.vlm.prompts_loader import reload_prompts
reload_prompts()
```

### Adding New Prompts

1. Create a new `.txt` file in `prompts/` directory
2. Use `load_prompt("filename.txt")` to load it

## Best Practices

- **Keep prompts in `.txt` files** - easier to edit than JSON for multi-line text
- **Use descriptive filenames** - e.g., `system_prompt.txt`, `codegen_prompt.txt`
- **Document prompt purpose** - add comments at the top of each prompt file
- **Version control prompts** - prompts should be committed to git for reproducibility

## Related Files

- `app/services/vlm_service.py` - Main VLM service (model loading, inference)
- `app/services/vlm_client_finetuned.py` - Fine-tuned model client
- `app/routes/vlm.py` - VLM API routes

