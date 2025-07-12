import os

import torch
import torch.distributed as dist

from DSV.models.parallel.sparse_kv_gather import (
    test_sparse_kv_gather, test_sparse_kv_gather_backward)

# to test the correctness of the sparse gather kv forward and backward 

def init_distributed(rank,world_size):
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    return world_size, rank

def clean_up():
    dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc-per-node=4 test_sparse_gather_kv_correctness.py
    world_size, rank = init_distributed(int(os.environ.get("RANK")),int(os.environ.get("WORLD_SIZE")))

    try:
        test_sparse_kv_gather(B=1,H=16,S=128000,D=128)

        test_sparse_kv_gather_backward(B=1,H=16,S=128000,D=128)

    except Exception as e:
        print(f"Error: {e}")
        raise e

    clean_up()