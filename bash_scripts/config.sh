export OLLAMA_URL=http://localhost:11434
export VLM_MODEL=llava-llama3:latest
mkdir -p data
# (optional) a clear rover side image:
python optim.py
# open http://localhost:5152/
