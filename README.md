# CAD Optimizer

A web-based CAD viewer and editor that uses Vision Language Models (VLM) for code generation and optimization. Built with CadQuery, cqparts, Flask, and PyTorch.

## Features

- Interactive 3D CAD model viewer (GLB format)
- VLM-powered code generation and modification
- Real-time GLB model building from cqparts assemblies
- Support for fine-tuned models and Ollama integration
- Mesh analysis with PointNet++ segmentation
- Semantic parameter extraction from 3D meshes

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the container
docker build -t cad-optimizer .

# Run the container
docker run -p 5160:5160 \
  -v $(pwd)/frontend/assets:/app/frontend/assets \
  -v $(pwd)/backend/checkpoints:/app/backend/checkpoints \
  -v $(pwd)/backend/runs:/app/backend/runs \
  cad-optimizer
```

Then open http://localhost:5160 in your browser.

### Option 2: Local Installation

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Start the server (from root)
./start_server.sh
# or
cd backend && python run.py
```

## Installation

### Requirements

- Python 3.10+
- FreeCAD (for legacy cqparts compatibility)
- CUDA-capable GPU (optional, for faster VLM inference)

### Step-by-Step Installation

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FreeCAD:**
   ```bash
   # Via conda (recommended)
   conda install -c conda-forge freecad
   
   # Or via system package manager
   # Ubuntu/Debian: sudo apt-get install freecad
   # macOS: brew install freecad
   ```

3. **Configure VLM Model (Optional):**
   
   **Option A: Fine-tuned Model (Default)**
   ```bash
   export USE_FINETUNED_MODEL=1
   export FINETUNED_MODEL_PATH=./runs/onevision_lora_small/checkpoint-4
   ```
   
   **Option B: Ollama (Faster on CPU)**
   ```bash
   export USE_FINETUNED_MODEL=0
   export OLLAMA_URL=http://127.0.0.1:11434
   export OLLAMA_MODEL=llava:latest
   ```

### Required Models

**PointNet++ Model (for mesh segmentation):**
```bash
# Download and place at:
backend/checkpoints/pointnet2/pointnet2_part_seg_msg.pth

# Or set environment variable:
export POINTNET2_CHECKPOINT=/path/to/pointnet2_part_seg_msg.pth
```

**VLM Model (optional):**
- Default location: `backend/runs/onevision_lora_small/checkpoint-4/`
- Or set `FINETUNED_MODEL_PATH` environment variable

## Project Structure

```
.
├── backend/                # Backend application
│   ├── app/               # Flask application
│   │   ├── routes/        # Route blueprints
│   │   ├── services/      # Business logic
│   │   ├── models/cad/    # CAD component models
│   │   └── config.py      # Configuration
│   ├── meshml/            # Mesh ML analysis (PointNet++, VLM)
│   ├── cqparts_bucket/    # cqparts libraries
│   ├── checkpoints/       # ML model checkpoints
│   ├── runs/              # Training runs
│   ├── tests/             # Backend tests
│   ├── run.py             # Main entry point
│   ├── optim.py           # Legacy server (preserved)
│   └── requirements.txt   # Python dependencies
├── frontend/               # Frontend assets
│   ├── static/            # CSS, JS assets
│   └── templates/         # HTML templates
├── assets/                 # Generated assets (GLB files)
├── Dockerfile              # Docker configuration
├── start_server.sh         # Startup script
└── README.md               # This file
```

## Usage

### Running the Server

```bash
# Option 1: Startup script (recommended)
./start_server.sh

# Option 2: Direct Python
python run.py

# Option 3: Legacy (still works)
python optim.py
```

### API Endpoints

- `GET /` - Main viewer
- `GET /debug` - Debug viewer
- `GET /api/state` - Get application state
- `POST /api/apply` - Apply parameter changes
- `POST /api/vlm/codegen` - VLM code generation
- `POST /api/vlm/recommend` - VLM recommendations
- `GET /api/model/glb` - Get GLB model
- `POST /api/mesh/ingest` - Ingest and analyze mesh

### Testing

```bash
# Run all tests
pytest tests/

# Test specific functionality
python tests/test_vlm.py
python tests/test_analyze_mesh.py
python tests/test_segmentation.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Flask server port | `5160` |
| `USE_FINETUNED_MODEL` | Use fine-tuned VLM (1) or Ollama (0) | `1` |
| `FINETUNED_MODEL_PATH` | Path to fine-tuned VLM checkpoint | `runs/onevision_lora_small/checkpoint-4` |
| `OLLAMA_URL` | Ollama server URL | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llava:latest` |
| `POINTNET2_CHECKPOINT` | Path to PointNet++ model | `checkpoints/pointnet2/pointnet2_part_seg_msg.pth` |
| `SEGMENTATION_BACKEND` | Segmentation backend (`pointnet` or `hunyuan3d`) | `pointnet` |
| `KMP_DUPLICATE_LIB_OK` | Fix OpenMP conflicts (macOS) | Not set |

## Development

### Application Structure

The application follows Flask best practices:

- **Routes**: Organized into blueprints (`app/routes/`)
- **Services**: Business logic separated (`app/services/`)
- **Models**: CAD components in `app/models/cad/`
- **Configuration**: Centralized in `app/config.py`

### Adding New Functionality

**New Route:**
```python
# app/routes/api.py
@bp.post("/new-endpoint")
def new_endpoint():
    from app.services.cad_service import build_rover_glb
    return jsonify({"ok": True})
```

**New Service:**
```python
# app/services/new_service.py
def do_something():
    # Business logic here
    return result
```

**New Model:**
```python
# app/models/cad/new_component.py
class NewComponent:
    # Component definition
```

### Segmentation Backends

The system supports multiple segmentation backends:

- **PointNet++** (default): Fully functional
- **Hunyuan3D-Part**: Structure ready, needs P3-SAM API implementation

Switch backends:
```bash
export SEGMENTATION_BACKEND=pointnet  # or hunyuan3d
```

## Troubleshooting

Main viewer — http://localhost:5160/
Debug viewer — http://localhost:5160/debug
API state — http://localhost:5160/api/state
GLB model — http://localhost:5160/api/model/glb


### OpenMP Error (macOS)
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### FreeCAD Not Found
FreeCAD is optional but recommended. Install via conda:
```bash
conda install -c conda-forge freecad
```

### CUDA Issues
If PyTorch doesn't detect CUDA:
1. Check CUDA version: `nvidia-smi`
2. Install matching PyTorch version from [pytorch.org](https://pytorch.org)

### VLM Model Issues
- **Slow on CPU**: Use Ollama (10-30 seconds vs 2-5 minutes)
- **Model not found**: Check `FINETUNED_MODEL_PATH` environment variable
- **Ollama not working**: Ensure Ollama server is running: `ollama serve`

## Docker

### Building

```bash
docker build -t cad-optimizer .
```

### Running

```bash
docker run -p 5160:5160 \
  -v $(pwd)/frontend/assets:/app/frontend/assets \
  -v $(pwd)/backend/checkpoints:/app/backend/checkpoints \
  -v $(pwd)/backend/runs:/app/backend/runs \
  cad-optimizer
```

### Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  cad-optimizer:
    build: .
    ports:
      - "5160:5160"
    volumes:
      - ./frontend/assets:/app/frontend/assets
      - ./backend/checkpoints:/app/backend/checkpoints
      - ./backend/runs:/app/backend/runs
    environment:
      - PORT=5160
      - USE_FINETUNED_MODEL=0
      - OLLAMA_URL=http://ollama:11434
```

## System Requirements

### Minimum
- **OS**: macOS, Linux, or Windows (WSL)
- **Python**: 3.10+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space

### Recommended
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16GB+
- **CPU**: Multi-core processor

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## References

- **PointNet++**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **Hunyuan3D-Part**: https://github.com/Tencent-Hunyuan/Hunyuan3D-Part
- **CadQuery**: https://github.com/CadQuery/cadquery
- **cqparts**: https://github.com/cqparts/cqparts
