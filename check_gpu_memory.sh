#!/bin/bash
# Quick script to check GPU memory usage and processes

echo "=========================================="
echo "GPU Memory Usage"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "GPU %s (%s):\n  Used:  %.2f GB / %.2f GB\n  Free:  %.2f GB\n", $1, $2, $3/1024, $4/1024, $5/1024}'

echo ""
echo "=========================================="
echo "Processes Using GPU"
echo "=========================================="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "PID %s (%s): %.2f GB\n", $1, $2, $3/1024}' || echo "No processes found or nvidia-smi not available"

echo ""
echo "=========================================="
echo "Python Processes"
echo "=========================================="
ps aux | grep -E "python.*run\.py|python.*freecad" | grep -v grep | awk '{printf "PID %s: %s %s\n", $2, $11, $12}'

echo ""
echo "=========================================="
echo "To Clear GPU Memory:"
echo "=========================================="
echo ""
echo "Option 1: Kill ALL GPU-intensive processes (RECOMMENDED):"
echo "  ./kill_gpu_processes.sh"
echo ""
echo "Option 2: Kill specific process manually:"
CURRENT_PID=$(ps aux | grep -E "python.*run\.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$CURRENT_PID" ]; then
    echo "  kill $CURRENT_PID  # Your server process"
else
    echo "  kill <PID>  # Replace <PID> with the process ID from above"
fi
echo ""
echo "Option 3: Use the API endpoint (if server is running):"
echo "  curl -X POST http://localhost:5000/api/mesh/clear_gpu_memory"
echo ""
echo "Option 4: Restart the server (after killing processes):"
echo "  ./start_server.sh"
echo ""
echo ""
echo "=========================================="
echo "For Detailed Analysis:"
echo "=========================================="
echo "  python diagnose_gpu_memory.py  # Shows detailed PyTorch memory breakdown"
echo ""
echo "=========================================="
echo "Understanding GPU Memory Usage:"
echo "=========================================="
echo "The P3-SAM model initialization uses ~15-20 GB based on:"
echo "  - P3SAM_POINT_NUM (default: 20000) - creates point buffers"
echo "  - P3SAM_PROMPT_NUM (default: 100) - creates prompt buffers"
echo ""
echo "Once loaded, the model stays in memory (~18-19 GB)."
echo "Inference needs additional memory (~1-3 GB) on top of this."
echo ""
echo "To reduce memory:"
echo "  1. Kill and restart with lower P3SAM_POINT_NUM/P3SAM_PROMPT_NUM"
echo "  2. See start_server.sh for ultra-low memory settings"
echo "  3. See explain_gpu_usage.md for detailed explanation"
echo ""
