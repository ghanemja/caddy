# VLM → CadQuery (via cadquery2web) Optimizer

This service lets a Vision-Language Model (e.g., LLaVA via Ollama) *write CadQuery code* to morph a model toward a target photo.  
We delegate all CAD execution & viewing to **cadquery2web** (Docker). The agent only orchestrates and scores.

## Prereqs
- Docker (for cadquery2web)
- Python 3.10+
- Ollama running a vision model (e.g., `llava-llama3:latest`)

## 1) Run cadquery2web
```bash
git clone https://github.com/30hours/cadquery2web
cd cadquery2web
sudo docker compose up -d
# UI: http://localhost:49157
