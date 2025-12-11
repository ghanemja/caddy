#!/usr/bin/env python3
"""
Diagnose what's consuming GPU memory.
Shows detailed breakdown of PyTorch tensors and models in memory.
"""

import subprocess
import sys


def check_nvidia_smi():
    """Get GPU memory info from nvidia-smi."""
    print("=" * 70)
    print("GPU Memory Usage (nvidia-smi)")
    print("=" * 70)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(", ")]
            if len(parts) >= 5:
                idx, name, used, total, free = parts
                used_gb = float(used) / 1024
                total_gb = float(total) / 1024
                free_gb = float(free) / 1024
                print(f"GPU {idx} ({name}):")
                print(
                    f"  Used:  {used_gb:.2f} GB / {total_gb:.2f} GB ({used_gb/total_gb*100:.1f}%)"
                )
                print(f"  Free:  {free_gb:.2f} GB")
    except Exception as e:
        print(f"Error: {e}")


def check_processes():
    """Get processes using GPU."""
    print("\n" + "=" * 70)
    print("Processes Using GPU")
    print("=" * 70)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            total = 0
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(", ")]
                if len(parts) >= 3:
                    pid, name, mem_mb = parts
                    mem_gb = float(mem_mb) / 1024
                    total += mem_gb
                    print(f"  PID {pid:>8} ({name:30s}): {mem_gb:7.2f} GB")
            print(f"\n  Total GPU memory used by processes: {total:.2f} GB")
        else:
            print("  No processes found")
    except Exception as e:
        print(f"Error: {e}")


def check_pytorch_memory():
    """Get detailed PyTorch memory breakdown if available."""
    print("\n" + "=" * 70)
    print("PyTorch Memory Breakdown (if torch available)")
    print("=" * 70)
    try:
        import torch

        if not torch.cuda.is_available():
            print("  CUDA not available")
            return

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free_memory = total_memory - reserved

            print(f"  Total:      {total_memory:.2f} GB")
            print(
                f"  Allocated:  {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)"
            )
            print(f"  Reserved:   {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
            print(
                f"  Free:       {free_memory:.2f} GB ({free_memory/total_memory*100:.1f}%)"
            )

            # Get memory summary
            try:
                summary = torch.cuda.memory_summary(device=i, abbreviated=True)
                print(f"\n  Memory Summary:")
                for line in summary.split("\n")[:20]:  # First 20 lines
                    if line.strip():
                        print(f"    {line}")
            except Exception as e:
                print(f"  Could not get detailed summary: {e}")

            # Try to get largest tensors (requires recent PyTorch)
            try:
                snapshot = torch.cuda.memory_snapshot()
                if snapshot:
                    print(f"\n  Note: Use torch.profiler for detailed tensor analysis")
            except:
                pass

    except ImportError:
        print("  PyTorch not available")
    except Exception as e:
        print(f"  Error: {e}")


def check_model_memory():
    """Check if we can identify what models are loaded."""
    print("\n" + "=" * 70)
    print("PyTorch Memory Breakdown")
    print("=" * 70)
    try:
        import torch

        if not torch.cuda.is_available():
            print("  CUDA not available")
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1024**3

        print(f"  Total GPU Memory:      {total_memory:.2f} GB")
        print(
            f"  PyTorch Reserved:      {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)"
        )
        print(
            f"  PyTorch Allocated:     {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)"
        )
        print(f"  Allocator Overhead:    {reserved - allocated:.2f} GB")
        print(f"  Free Memory:           {total_memory - reserved:.2f} GB")

        print(f"\n  Why PyTorch Uses So Much Memory:")
        print(
            f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        print(f"  1. Model Weights (~12-16 GB):")
        print(f"     • P3-SAM base model: ~8-10 GB")
        print(f"     • Sonata sub-model: ~4-6 GB")
        print(f"     • These are neural network parameters (cannot be reduced)")
        print(f"")
        print(f"  2. Pre-allocated Buffers (~3-6 GB):")
        print(f"     • Point cloud buffers: ~2-4 GB (controlled by P3SAM_POINT_NUM)")
        print(f"     • Prompt buffers: ~1-2 GB (controlled by P3SAM_PROMPT_NUM)")
        print(f"     • Created when model loads, stay in memory")
        print(f"")
        print(f"  3. Allocator Overhead (~700 MB):")
        print(f"     • PyTorch reserves extra memory for efficiency")
        print(f"     • Reduces allocation/deallocation overhead")
        print(f"     • Normal behavior - helps performance")

        # Estimate breakdown based on current usage
        weights_est = min(allocated * 0.7, 16)  # Up to 70% or 16GB max
        buffers_est = allocated - weights_est
        overhead = reserved - allocated

        print(f"\n  Estimated Current Breakdown:")
        print(f"    Model Weights:        ~{weights_est:.2f} GB (fixed - can't reduce)")
        print(
            f"    Buffers:              ~{buffers_est:.2f} GB (can reduce via env vars)"
        )
        print(f"    Allocator Overhead:   ~{overhead:.2f} GB (normal PyTorch behavior)")
        print(f"    ────────────────────────────────────────────────────────────────")
        print(f"    TOTAL:                ~{reserved:.2f} GB")

        print(f"\n  Recommendation:")
        if allocated > 18:
            print(f"    ⚠⚠ VERY HIGH memory usage ({allocated:.2f} GB)")
            print(
                f"       Current settings: P3SAM_POINT_NUM=20000, P3SAM_PROMPT_NUM=100"
            )
            print(f"       Action: Uncomment ultra-low settings in start_server.sh")
            print(f"                P3SAM_POINT_NUM=15000, P3SAM_PROMPT_NUM=75")
        elif allocated > 15:
            print(f"    ⚠ High memory usage ({allocated:.2f} GB)")
            print(f"       Current settings should reduce this after restart")
            print(
                f"       Expected: ~12-15 GB with P3SAM_POINT_NUM=20000, P3SAM_PROMPT_NUM=100"
            )
        elif allocated > 12:
            print(f"    ✓ Moderate memory usage ({allocated:.2f} GB)")
            print(
                f"       Consider: Lower P3SAM_POINT_NUM/P3SAM_PROMPT_NUM if OOM occurs"
            )
        else:
            print(f"    ✓✓ Memory usage is reasonable ({allocated:.2f} GB)")

    except ImportError:
        print("  PyTorch not available")
    except Exception as e:
        print(f"  Error: {e}")


def check_environment_variables():
    """Show relevant environment variables."""
    print("\n" + "=" * 70)
    print("Environment Variables")
    print("=" * 70)
    import os

    relevant_vars = [
        "P3SAM_POINT_NUM",
        "P3SAM_PROMPT_NUM",
        "P3SAM_INFERENCE_POINT_NUM",
        "P3SAM_INFERENCE_PROMPT_NUM",
        "P3SAM_PROMPT_BS",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]

    for var in relevant_vars:
        value = os.environ.get(var, "(not set)")
        print(f"  {var:30s} = {value}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GPU Memory Diagnosis")
    print("=" * 70)

    check_nvidia_smi()
    check_processes()
    check_pytorch_memory()
    check_model_memory()
    check_environment_variables()

    print("\n" + "=" * 70)
    print("Summary & Recommendations")
    print("=" * 70)
    print(
        """
The main consumer of GPU memory is the P3-SAM model loaded in your Python server.

Current situation:
  - Model loaded: ~18.69 GB
  - Free memory: ~1.55 GB  
  - Inference needs: ~1.53 GB more

Solutions:
  1. Reduce model initialization memory:
     export P3SAM_POINT_NUM=20000      # Reduce from 30000 (default)
     export P3SAM_PROMPT_NUM=100       # Reduce from 150 (default)
     
  2. Unload model between inferences (requires code change)
  
  3. Use even lower inference parameters:
     export P3SAM_INFERENCE_POINT_NUM=15000
     export P3SAM_INFERENCE_PROMPT_NUM=75
     export P3SAM_PROMPT_BS=4

  4. Kill and restart server to clear memory:
     ./kill_all_gpu_processes.sh && ./start_server.sh
    """
    )
    print("=" * 70 + "\n")
