#!/usr/bin/env python3
"""
Utility script to check and clear GPU memory usage.
Run this to see what's using GPU memory and optionally clear it.
"""

import sys
import os

# Add backend to path before importing anything
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(script_dir))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import subprocess


def check_gpu_processes():
    """Check what processes are using the GPU."""
    print("=" * 60)
    print("GPU Memory Usage by Process")
    print("=" * 60)

    try:
        # Use nvidia-smi to get process info
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
            print(result.stdout)
        else:
            print("No GPU processes found (or nvidia-smi not available)")
    except FileNotFoundError:
        print("nvidia-smi not found. Install NVIDIA drivers to use this feature.")

    # Check PyTorch memory if available
    try:
        import torch

        if torch.cuda.is_available():
            print("\n" + "=" * 60)
            print("PyTorch GPU Memory Status")
            print("=" * 60)
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                print(f"GPU {i}:")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Max allocated: {max_allocated:.2f} GB")

                # Get detailed memory summary
                print(f"\nDetailed memory breakdown:")
                print(torch.cuda.memory_summary(device=i, abbreviated=False))
    except ImportError:
        print("PyTorch not available")


def clear_gpu_memory():
    """Clear GPU memory by clearing PyTorch cache."""
    try:
        import torch

        if torch.cuda.is_available():
            print("\n" + "=" * 60)
            print("Clearing GPU Memory...")
            print("=" * 60)

            before_allocated = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            after_allocated = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3

            print(
                f"Before: {before_allocated:.2f} GB allocated, {before_reserved:.2f} GB reserved"
            )
            print(
                f"After:  {after_allocated:.2f} GB allocated, {after_reserved:.2f} GB reserved"
            )
            print(f"Freed:  {before_reserved - after_reserved:.2f} GB")
            print("\n✓ GPU cache cleared!")

            # Note: This only clears cached memory, not memory held by loaded models
            # To fully clear, you need to delete model objects and run garbage collection
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            print("✓ Ran garbage collection")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not available - cannot clear GPU memory")


def kill_gpu_processes(pids=None):
    """Kill specific GPU processes."""
    if pids is None:
        pids = []

    if not pids:
        print("No PIDs specified. Use --kill <pid1> <pid2> ... to kill processes.")
        print(
            "\nTo find PIDs, run this script without arguments and check nvidia-smi output."
        )
        return

    print(f"\nKilling processes: {pids}")
    for pid in pids:
        try:
            os.kill(int(pid), 9)  # SIGKILL
            print(f"✓ Killed process {pid}")
        except ProcessLookupError:
            print(f"✗ Process {pid} not found")
        except PermissionError:
            print(f"✗ Permission denied to kill process {pid} (try running with sudo)")
        except Exception as e:
            print(f"✗ Error killing process {pid}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check and clear GPU memory")
    parser.add_argument("--clear", action="store_true", help="Clear PyTorch GPU cache")
    parser.add_argument("--kill", nargs="+", type=int, help="Kill processes by PID")
    args = parser.parse_args()

    check_gpu_processes()

    if args.clear:
        clear_gpu_memory()

    if args.kill:
        kill_gpu_processes(args.kill)

    if not args.clear and not args.kill:
        print("\n" + "=" * 60)
        print("Usage:")
        print("  python clear_gpu_memory.py --clear    # Clear PyTorch GPU cache")
        print("  python clear_gpu_memory.py --kill <pid1> <pid2>  # Kill GPU processes")
        print("=" * 60)
