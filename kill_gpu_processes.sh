#!/bin/bash
# Kill all GPU-intensive processes (especially Python processes using GPU)

set -e

echo "=========================================="
echo "Finding GPU Processes"
echo "=========================================="

# Get all processes using GPU
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "")

if [ -z "$GPU_PROCS" ]; then
    echo "No GPU processes found or nvidia-smi not available"
    exit 0
fi

# Parse and display processes
echo "GPU processes found:"
echo ""
TOTAL_MEM=0
declare -a PIDS_TO_KILL

while IFS=', ' read -r pid name mem_mb; do
    # Skip empty lines
    [ -z "$pid" ] && continue
    
    mem_gb=$(echo "scale=2; $mem_mb/1024" | bc)
    TOTAL_MEM=$(echo "scale=2; $TOTAL_MEM + $mem_mb/1024" | bc)
    
    echo "  PID $pid ($name): ${mem_gb} GB"
    
    # Kill Python processes using significant GPU memory (>100 MB)
    if [[ "$name" == *"python"* ]] && [ "$mem_mb" -gt 100 ]; then
        PIDS_TO_KILL+=("$pid")
    fi
done <<< "$GPU_PROCS"

echo ""
echo "Total GPU memory used: ${TOTAL_MEM} GB"

if [ ${#PIDS_TO_KILL[@]} -eq 0 ]; then
    echo ""
    echo "No Python GPU processes found to kill."
    exit 0
fi

echo ""
echo "=========================================="
echo "Processes to Kill:"
echo "=========================================="
for pid in "${PIDS_TO_KILL[@]}"; do
    # Get process details
    PROC_INFO=$(ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || echo "")
    if [ -n "$PROC_INFO" ]; then
        echo "  PID $pid: $PROC_INFO"
    else
        echo "  PID $pid: (process not found)"
    fi
done

echo ""
echo "=========================================="
read -p "Kill these processes? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Killing processes..."
KILLED_COUNT=0
FAILED_COUNT=0

for pid in "${PIDS_TO_KILL[@]}"; do
    if kill -TERM "$pid" 2>/dev/null; then
        echo "  ✓ Sent TERM signal to PID $pid"
        KILLED_COUNT=$((KILLED_COUNT + 1))
        
        # Wait a bit, then force kill if still running
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            echo "    → Process still running, sending SIGKILL..."
            kill -KILL "$pid" 2>/dev/null && echo "    ✓ Killed PID $pid" || echo "    ✗ Failed to kill PID $pid"
        fi
    else
        echo "  ✗ Failed to kill PID $pid (may not exist or permission denied)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "  Killed: $KILLED_COUNT processes"
[ $FAILED_COUNT -gt 0 ] && echo "  Failed: $FAILED_COUNT processes"
echo ""

# Wait a moment for processes to fully terminate
sleep 2

# Show current GPU status
echo "Current GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  Used: %.2f GB / %.2f GB (%.1f%% free)\n", $1/1024, $2/1024, (($2-$1)/$2)*100}' || \
    echo "  Could not query GPU status"

echo ""
echo "✓ Done!"
