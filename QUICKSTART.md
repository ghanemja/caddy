# CAD Optimizer Quick Start Guide

## ✅ Environment Created Successfully!

Your conda environment `cad-optimizer` is ready with:
- ✅ Python 3.10.16
- ✅ CadQuery 2.4.0 (with OCP 7.7.2)
- ✅ PyTorch 2.4.0 (CPU-only)
- ✅ Transformers 4.57.0
- ✅ PEFT 0.17.1 (LoRA fine-tuning)
- ✅ Flask 3.1.2
- ✅ Trimesh 4.8.3
- ✅ All project dependencies

## 🚀 Start Using It Now

### 1. Activate the Environment
```bash
conda activate cad-optimizer
```

### 2. Navigate to Project
```bash
cd /home/ec2-user/Documents/cad-optimizer/cqparts_bucket
```

### 3. Run the Server
```bash
python optim.py
```

The server will start on: **http://0.0.0.0:5160**

### 4. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5160
```

Or if accessing remotely:
```
http://<your-server-ip>:5160
```

## 🎯 What You Can Do

### Option 1: Use Fine-Tuned Model (Default)
```bash
# This uses your fine-tuned LLaVA OneVision model
python optim.py
```

**First run will download the base model (~15GB)**:
- Location: `~/.cache/huggingface/hub/`
- Time: 5-10 minutes depending on internet speed
- Disk space: ~20GB total

### Option 2: Use Ollama Instead
```bash
# Disable fine-tuned model and use Ollama
USE_FINETUNED_MODEL=0 python optim.py
```

Make sure Ollama is running:
```bash
ollama serve
```

### Option 3: Custom Model Path
```bash
# Use a different fine-tuned model
FINETUNED_MODEL_PATH=/path/to/your/adapter python optim.py
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Fine-tuned model
export USE_FINETUNED_MODEL=1  # 1=enabled, 0=disabled
export FINETUNED_MODEL_PATH=/path/to/adapter

# Ollama (fallback)
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llava-llama3:latest

# Server
export PORT=5160
```

### Performance Tips

**CPU Mode (Current Setup):**
- First inference will be slow (30-60s)
- Subsequent inferences: 10-20s
- Consider using Ollama for faster responses

**For GPU Support:**
```bash
# Recreate environment with GPU support
conda env remove -n cad-optimizer
conda env create -f environment.yml  # GPU version
```

## 📝 Deactivate When Done

```bash
conda deactivate
```

## 🐛 Troubleshooting

### Model Download Issues

If base model download fails:
```bash
# Set HuggingFace cache to location with more space
export HF_HOME=/path/to/large/disk/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

### Slow Inference on CPU

For faster performance:
1. **Disable fine-tuned model**: `USE_FINETUNED_MODEL=0 python optim.py`
2. **Use Ollama**: Start `ollama serve` and run with Ollama
3. **Get GPU support**: Recreate with `environment.yml`

### "ModuleNotFoundError"

Make sure environment is activated:
```bash
conda activate cad-optimizer
python optim.py
```

### Port Already in Use

Change the port:
```bash
PORT=8080 python optim.py
```

## 📚 More Information

- [Environment Setup Details](ENVIRONMENT_SETUP.md)
- [Fine-Tuned Model Integration](FINETUNED_MODEL_INTEGRATION.md)
- [CadQuery Documentation](https://cadquery.readthedocs.io/)

## 🎉 You're All Set!

Your environment is configured and ready to use. The system will:
1. Load your fine-tuned VLM model on startup (if enabled)
2. Serve a web interface for CAD model editing
3. Generate parametric CAD code using vision+language AI
4. Export models as GLB files for 3D viewing

Have fun building! 🚀


