#!/bin/bash
# Quick kill script - kills the main server process using GPU

PID=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | grep -i python | awk -F', ' '{print $1}' | head -1)

if [ -z "$PID" ]; then
    echo "No Python GPU processes found."
    exit 0
fi

echo "Killing Python GPU process PID $PID..."
kill -TERM "$PID" 2>/dev/null
sleep 2
if ps -p "$PID" > /dev/null 2>&1; then
    kill -KILL "$PID" 2>/dev/null
    echo "✓ Force killed PID $PID"
else
    echo "✓ Killed PID $PID"
fi

echo "GPU memory after kill:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  Used: %.2f GB, Free: %.2f GB\n", $1/1024, $2/1024}'
