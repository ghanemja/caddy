# caddy in a container: the way to run it on macOS (NVIDIA publishes usd-brep wheels for Linux and Windows only).
# Works natively on Apple Silicon (linux/arm64 wheel) and on Intel/AMD (linux/amd64).
FROM python:3.12-slim
WORKDIR /app
COPY packages packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY caddy caddy
COPY data data
COPY docs/runs docs/runs
EXPOSE 8000
CMD ["python", "-m", "caddy", "--host", "0.0.0.0", "--port", "8000"]
