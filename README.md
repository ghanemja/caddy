# CAD Optimizer

A web-based CAD viewer and editor that uses Vision Language Models (VLM) for code generation and optimization. Built with CadQuery, cqparts, Flask, and PyTorch.

## Features

- Interactive 3D CAD model viewer (GLB format)
- VLM-powered code generation and modification
- Real-time GLB model building from cqparts assemblies
- Support for fine-tuned models and Ollama integration

## Requirements

- Python 3.10+
- FreeCAD (for legacy cqparts compatibility)
- CUDA-capable GPU (optional, for faster VLM inference)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FreeCAD

FreeCAD is typically installed via conda:

```bash
conda install -c conda-forge freecad
```

Or via system package manager:

```bash
# Ubuntu/Debian
sudo apt-get install freecad

# macOS
brew install freecad
```

### 3. Configure VLM Model (Optional)

The application supports fine-tuned VLM models or Ollama. By default, it attempts to use a fine-tuned model.

**Option A: Use Fine-tuned Model (Default)**

Set environment variables:

```bash
export USE_FINETUNED_MODEL=1
export FINETUNED_MODEL_PATH=/path/to/your/adapter
```

**Option B: Use Ollama**

```bash
export USE_FINETUNED_MODEL=0
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llava-llama3:latest
```

Then start Ollama:

```bash
ollama serve
```

## Usage

### Start the Server

Navigate to the `cqparts_bucket` directory and run:

```bash
cd cqparts_bucket
python optim.py
```

The server will start on `http://0.0.0.0:5160` by default.

You can customize the port:

```bash
PORT=8080 python optim.py
```

### Access the Web Interface

Open your browser and navigate to:

```
http://localhost:5160
```

Or if accessing remotely:

```
http://<your-server-ip>:5160
```

## Project Structure

```
optim/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── cqparts_bucket/          # Main application directory
    ├── optim.py             # Main server file (Flask app)
    ├── robot_base.py        # Rover assembly definition
    ├── electronics.py       # Electronics component
    ├── pan_tilt.py          # Pan-tilt mechanism
    ├── wheel.py             # Wheel component
    ├── sensor_fork.py       # Sensor fork component
    ├── motor_mount.py       # Motor mount component
    ├── partref.py           # Part reference utility
    ├── manufacture.py       # Manufacturing base classes
    ├── cqparts/             # cqparts library
    ├── cqparts_motors/      # Motor components library
    ├── templates/           # HTML templates
    ├── static/              # Static assets (CSS, JS)
    └── assets/              # Model assets (GLB files)
```

## Main File

The primary file is `cqparts_bucket/optim.py`, which contains:

- Flask web server
- GLB model building pipeline
- VLM integration for code generation
- REST API endpoints

## Dependencies

See `requirements.txt` for the full list of Python dependencies. Key packages include:

- **cadquery**: CAD modeling library
- **cqparts**: Component-based CAD framework
- **flask**: Web framework
- **trimesh**: 3D mesh processing
- **transformers**: Hugging Face transformers (for VLM)
- **peft**: Parameter-efficient fine-tuning
- **torch**: PyTorch (for VLM inference)

## Configuration

### Environment Variables

- `USE_FINETUNED_MODEL`: Enable/disable fine-tuned model (default: "1")
- `FINETUNED_MODEL_PATH`: Path to fine-tuned model adapter
- `OLLAMA_URL`: Ollama server URL (default: "http://127.0.0.1:11434")
- `OLLAMA_MODEL`: Ollama model name (default: "llava-llama3:latest")
- `PORT`: Server port (default: 5160)

### VLM Model Configuration

The fine-tuned model uses:
- Base model: `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- LoRA adapter: Loaded from `FINETUNED_MODEL_PATH`

First-time model loading takes 2-3 minutes. Subsequent requests are faster.

## API Endpoints

- `GET /`: Main viewer interface
- `GET /debug`: Debug viewer
- `POST /api/vlm/codegen`: Generate/modify CAD code using VLM
- `GET /api/model/glb`: Get current GLB model
- `POST /api/model/rebuild`: Rebuild GLB from code

## Troubleshooting

### FreeCAD Import Error

If you see `ModuleNotFoundError: No module named 'FreeCAD'`:

1. Install FreeCAD: `conda install -c conda-forge freecad`
2. Verify installation: `python -c "import FreeCAD; print(FreeCAD.Version())"`

### VLM Model Loading Issues

If the fine-tuned model fails to load:

1. Check model path: `echo $FINETUNED_MODEL_PATH`
2. Verify model files exist at the path
3. Fall back to Ollama: `USE_FINETUNED_MODEL=0 python optim.py`

### GPU Not Detected

The application will automatically use CPU if GPU is not available. For GPU support:

1. Install CUDA-compatible PyTorch
2. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## License

This project uses components from cqparts, which is Apache 2.0 licensed.

## Author

Originally created by Simon Kirkby. Extended with VLM integration and web interface.

