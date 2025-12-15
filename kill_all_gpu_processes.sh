#!/bin/bash
# Aggressively kill ALL Python processes using GPU (no confirmation)

set -e

echo "=========================================="
echo "Killing ALL GPU-Intensive Python Processes"
echo "=========================================="

# Get all Python processes using GPU
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | \
    grep -i python | awk -F', ' '{print $1}' || echo "")

if [ -z "$GPU_PIDS" ]; then
    echo "No Python GPU processes found."
    exit 0
fi

echo "Found Python GPU processes:"
for pid in $GPU_PIDS; do
    if ps -p "$pid" > /dev/null 2>&1; then
        PROC_INFO=$(ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || echo "")
        echo "  PID $pid: $PROC_INFO"
    fi
done

echo ""
echo "Killing processes..."

for pid in $GPU_PIDS; do
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "  Killing PID $pid..."
        # Try graceful shutdown first
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -KILL "$pid" 2>/dev/null || true
            echo "    ✓ Killed (forced)"
        else
            echo "    ✓ Killed (graceful)"
        fi
    else
        echo "  PID $pid: Already terminated"
    fi
done

# Also kill any python run.py processes that might be hanging
PYTHON_RUN_PIDS=$(ps aux | grep -E "python.*run\.py" | grep -v grep | awk '{print $2}' || echo "")
if [ -n "$PYTHON_RUN_PIDS" ]; then
    echo ""
    echo "Found additional python run.py processes:"
    for pid in $PYTHON_RUN_PIDS; do
        echo "  Killing PID $pid..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "Waiting for processes to terminate..."
sleep 3

echo ""
echo "=========================================="
echo "Verification:"
echo "=========================================="

# Check remaining GPU processes
REMAINING=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | grep -i python || echo "")
if [ -z "$REMAINING" ]; then
    echo "✓ All Python GPU processes killed!"
else
    echo "⚠ Some Python processes may still be running:"
    echo "$REMAINING"
fi

# Show GPU status
echo ""
echo "Current GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  Used:  %.2f GB\n  Total: %.2f GB\n  Free:  %.2f GB\n", $1/1024, $2/1024, $3/1024}' || \
    echo "  Could not query GPU status"

echo ""
echo "✓ Done! You can now restart the server with ./start_server.sh"
