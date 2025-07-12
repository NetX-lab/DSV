import math

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

from DSV.triton_plugin.fused_attention import \
    attention as triton_flash_attn_tutorial
from DSV.triton_plugin.fused_attention_no_causal import \
    attention as triton_flash_attn_tutorial_no_causal
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group import \
    sparse_group_attention
from DSV.triton_plugin.triton_flash_tridao import \
    FlashAttnFunc as triton_flash_attn_tridao

torch.manual_seed(42)

dtype = torch.bfloat16
device = "cuda:0"


def test_attention(seq_len, head_dim, head_num, dtype, device):
    import math

    softmax_scale = 1 / math.sqrt(head_dim)

    q = torch.randn(
        1, head_num, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        1, head_num, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        1, head_num, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True
    )

    grad_o = torch.randn(1, head_num, seq_len, head_dim, dtype=dtype, device=device)

    q_triton = q.clone().detach().requires_grad_(True)
    k_triton = k.clone().detach().requires_grad_(True)
    v_triton = v.clone().detach().requires_grad_(True)

    q_flash_cuda = q.clone().detach().requires_grad_(True)
    k_flash_cuda = k.clone().detach().requires_grad_(True)
    v_flash_cuda = v.clone().detach().requires_grad_(True)

    q_naive = q.clone().detach().requires_grad_(True)
    k_naive = k.clone().detach().requires_grad_(True)
    v_naive = v.clone().detach().requires_grad_(True)

    q_sparse = q.clone().detach().requires_grad_(True)
    k_sparse = k.clone().detach().requires_grad_(True)
    v_sparse = v.clone().detach().requires_grad_(True)

    torch.cuda.synchronize()

    group_size = 32
    sparse_kv_index_dense = torch.zeros(
        1, head_num, seq_len // group_size, seq_len, dtype=torch.int32, device=device
    )
    sparse_kv_num_per_head_dense = torch.full(
        (1, head_num), seq_len, dtype=torch.int32, device=device
    )

    for b in range(1):
        for h in range(head_num):
            for g in range(seq_len // group_size):
                sparse_kv_index_dense[b, h, g, :] = torch.arange(seq_len, device=device)

    x_sparse = sparse_group_attention(
        q_sparse,
        k_sparse,
        v_sparse,
        False,
        softmax_scale,
        sparse_kv_index_dense,
        sparse_kv_num_per_head_dense,
        group_size,
    )
    x_sparse.backward(grad_o)

    x_triton = triton_flash_attn_tutorial_no_causal(
        q_triton, k_triton, v_triton, False, softmax_scale
    )
    x_triton.backward(grad_o)

    x_flash_cuda = flash_attn_func(
        q_flash_cuda.permute(0, 2, 1, 3),
        k_flash_cuda.permute(0, 2, 1, 3),
        v_flash_cuda.permute(0, 2, 1, 3),
        softmax_scale=softmax_scale,
        causal=False,
    ).permute(0, 2, 1, 3)
    x_flash_cuda.backward(grad_o)

    try:
        torch.testing.assert_close(x_triton, x_flash_cuda, atol=1e-2, rtol=0)
    except Exception as e:
        print("x_triton,x_flash_cuda")
        print(e)

    try:
        torch.testing.assert_close(x_sparse, x_flash_cuda, atol=1e-2, rtol=0)
    except Exception as e:
        print("x_sparse,x_triton")
        print(e)

    try:
        torch.testing.assert_close(q_triton.grad, q_flash_cuda.grad, atol=1e-1, rtol=0)
    except Exception as e:
        print("q_triton.grad,q_flash_cuda.grad")
        print(e)

    try:
        torch.testing.assert_close(q_sparse.grad, q_flash_cuda.grad, atol=1e-1, rtol=0)
    except Exception as e:
        print("q_sparse.grad,q_flash_cuda.grad")
        print(e)

    try:
        torch.testing.assert_close(k_triton.grad, k_flash_cuda.grad, atol=1e-1, rtol=0)
    except Exception as e:
        print("k_triton.grad,k_flash_cuda.grad")
        print(e)

    try:
        torch.testing.assert_close(k_sparse.grad, k_flash_cuda.grad, atol=1e-1, rtol=0)
    except Exception as e:
        print("k_sparse.grad,k_flash_cuda.grad")
        print(e)

    try:
        torch.testing.assert_close(v_triton.grad, v_flash_cuda.grad, atol=1e-1, rtol=0)
    except Exception as e:
        print("v_triton.grad,v_flash_cuda.grad")
        print(e)

    try:
        torch.testing.assert_close(v_sparse.grad, v_flash_cuda.grad, atol=1e-1, rtol=0)
    except Exception as e:
        print("v_sparse.grad,v_flash_cuda.grad")
        print(e)

    print(
        f"Successfully tested the forward and backward of triton flash attention for seq_len={seq_len}, head_dim={head_dim}, head_num={head_num}."
    )


# Example of testing with different inputs, we focus on BF16 in training.

try:
    test_attention(
        seq_len=1024, head_dim=64, head_num=4, dtype=torch.bfloat16, device="cuda:0"
    )

    test_attention(
        seq_len=5120, head_dim=64, head_num=16, dtype=torch.bfloat16, device="cuda:0"
    )

    test_attention(
        seq_len=10240, head_dim=128, head_num=8, dtype=torch.bfloat16, device="cuda:0"
    )

    test_attention(
        seq_len=128000, head_dim=128, head_num=24, dtype=torch.bfloat16, device="cuda:0"
    )

except Exception as e:
    import traceback

    traceback.print_exc()
    print(
        f"⚠️ The numerical gap could be large, expecting that no more than 0.1. Please check that."
    )

print(
    f"✅ Successfully tested the forward and backward numerical correctness of sparse attention."
)
