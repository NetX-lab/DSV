import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

if __name__ == "__main__":
    # init the distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    B, H, total_S, D = 1, 16, 128000, 128
    device = "cuda"

    # 创建测试数据
    Q = torch.randn(B, H, total_S, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, total_S, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, total_S, D, device=device, dtype=torch.bfloat16)

    for i in tqdm(range(1000000000)):
        print(f"Running {i} times")
        start_time = time.time()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=False, scale=1 / math.sqrt(D)
            )

        torch.cuda.synchronize()
        end_time = time.time()
        del x
        print(f"Rank {rank} Time taken: {end_time - start_time} seconds")
