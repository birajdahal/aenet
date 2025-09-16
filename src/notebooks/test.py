import os
import sys
import traceback
import torch
import torch.multiprocessing as mp

def train_one(rank):
    try:
        # Show what each child actually sees
        print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[rank {rank}] torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"[rank {rank}] device_count(child)={torch.cuda.device_count()}")

        # If CUDA isn't visible/available in this child, this will tell us before set_device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available inside child process")

        # Map rank→device (rank is always 0..nprocs-1; beware CUDA_VISIBLE_DEVICES remaps)
        torch.cuda.set_device(rank)
        dev = torch.device(f"cuda:{rank}")
        # Touch the device to force init (flushes hidden init errors)
        torch.randn(1, device=dev)
        print(f"[rank {rank}] OK on {dev} ({torch.cuda.get_device_name(dev)})")

    except Exception:
        # Print full traceback from the child so we actually see the error
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print(f"[parent] torch version: {torch.__version__}")
    print(f"[parent] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[parent] torch.cuda.is_available()={torch.cuda.is_available()}")
    n = torch.cuda.device_count()
    print(f"[parent] device_count(parent)={n}")
    if n == 0:
        raise RuntimeError("No GPUs visible to the parent process")

    # Keep nprocs small for debugging (change to n when fixed)
    mp.spawn(train_one, nprocs=min(n, 2), join=True)
