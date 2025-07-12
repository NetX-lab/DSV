
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from DSV.triton_plugin.fused_attention_no_causal import \
    attention as full_attention
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group import \
    sparse_group_attention

import argparse

torch.manual_seed(42)  # Reproducible results for evaluation


def prepare_data(
    batch_size, seq_len, head_num, head_dim, low_rank_dim, dtype=torch.bfloat16
):
    low_rank_proj = torch.randn(
        head_dim * head_num,
        low_rank_dim * head_num * 2,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    query_proj = torch.randn(
        head_num * head_dim,
        head_num * head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    key_proj = torch.randn(
        head_num * head_dim,
        head_num * head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    value_proj = torch.randn(
        head_num * head_dim,
        head_num * head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    input_ = torch.randn(
        batch_size,
        seq_len,
        head_num * head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    query = (
        torch.matmul(input_, query_proj)
        .contiguous()
        .view(batch_size, seq_len, head_num, head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    key = (
        torch.matmul(input_, key_proj)
        .contiguous()
        .view(batch_size, seq_len, head_num, head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    value = (
        torch.matmul(input_, value_proj)
        .contiguous()
        .view(batch_size, seq_len, head_num, head_dim)
        .transpose(1, 2)
        .contiguous()
    )

    low_rank_query_value = torch.matmul(input_, low_rank_proj).chunk(2, dim=-1)
    low_rank_query = (
        low_rank_query_value[0]
        .contiguous()
        .view(batch_size, seq_len, head_num, low_rank_dim)
        .transpose(1, 2)
        .contiguous()
    )
    low_rank_value = (
        low_rank_query_value[1]
        .contiguous()
        .view(batch_size, seq_len, head_num, low_rank_dim)
        .transpose(1, 2)
        .contiguous()
    )

    return query, key, value, low_rank_query, low_rank_value


def extract_middle_tokens(tensor, num_groups, seq_len_per_group):
    group_indices = torch.arange(num_groups, device=tensor.device, dtype=torch.long)
    start_positions = group_indices * seq_len_per_group
    middle_positions = start_positions + seq_len_per_group // 2
    return tensor[:, :, middle_positions, :]  # [B, H, num_groups, D]


def get_sparse_kv_num_per_head(sparsity_per_head, kv_len, multiple_of=256):
    res = [
        int(((1 - s) * kv_len + multiple_of - 1) // multiple_of * multiple_of)
        for s in sparsity_per_head
    ]
    return res


def compute_mismatch_ratio(output_reference, output_sparse, atol=0.1, rtol=0.1):
    abs_diff = torch.abs(output_reference - output_sparse)
    rel_diff = torch.abs(output_sparse) * rtol + atol
    abs_diff_mask = abs_diff <= rel_diff
    ratio_pass = torch.sum(abs_diff_mask) / abs_diff_mask.numel()
    return round((1 - ratio_pass.item()) * 100, 2)


def sparse_attention_w_low_rank_correctness_and_performance_benchmark(
    batch_size,
    seq_len,
    head_num,
    head_dim,
    low_rank_dim,
    sparsity_per_head,
    warm_up=False,
    group_size=32,
):
    print(f"Step 1: Preparing attention matrices...") if not warm_up else None
    query, key, value, low_rank_query, low_rank_value = prepare_data(
        batch_size, seq_len, head_num, head_dim, low_rank_dim
    )
    B, H, S, D = query.shape

    print(
        f"Step 2: Low-rank approximation for sparse KV selection..."
    ) if not warm_up else None

    # Setup timing events for low-rank operations
    event_low_rank_begin = torch.cuda.Event(enable_timing=True)
    event_low_rank_end = torch.cuda.Event(enable_timing=True)

    # Calculate how many KV pairs to keep per head
    sparse_kv_num_per_head = get_sparse_kv_num_per_head(sparsity_per_head, S)
    seq_len_per_group = group_size
    maximum_sparse_kv_num = max(sparse_kv_num_per_head)
    num_groups = S // group_size
    low_rank_query_compressed = extract_middle_tokens(
        low_rank_query, num_groups, seq_len_per_group
    )

    # Start timing low-rank operations
    event_low_rank_begin.record()
    # Low-rank approximation: compressed_query × low_rank_value^T → attention scores
    low_rank_qk = torch.matmul(
        low_rank_query_compressed, low_rank_value.transpose(-2, -1)
    )
    # Select top-k KV indices based on low-rank attention scores
    sparse_kv_index = torch.zeros(
        B, H, num_groups, maximum_sparse_kv_num, device="cuda", dtype=torch.int32
    )
    for h in range(H):
        sparse_kv_num = sparse_kv_num_per_head[h]
        _, topk_indices = torch.topk(
            low_rank_qk[:, h, :, :], sparse_kv_num, dim=-1, sorted=False
        )
        sparse_kv_index[:, h, :, :sparse_kv_num] = topk_indices

    # End timing low-rank operations
    event_low_rank_end.record()

    sparse_kv_num_per_head_tensor = (
        torch.tensor(sparse_kv_num_per_head, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .repeat(B, 1)
    )

    print(
        f"Step 3: Running sparse attention with performance measurement..."
    ) if not warm_up else None

    # Prepare tensors for sparse attention (fresh copies with gradients)
    query_sparse = query.detach().clone().requires_grad_(True)
    key_sparse = key.detach().clone().requires_grad_(True)
    value_sparse = value.detach().clone().requires_grad_(True)

    # Setup timing events for precise GPU timing
    sparse_forward_start = torch.cuda.Event(enable_timing=True)
    sparse_forward_end = torch.cuda.Event(enable_timing=True)
    sparse_backward_start = torch.cuda.Event(enable_timing=True)
    sparse_backward_end = torch.cuda.Event(enable_timing=True)

    # Sparse attention forward pass timing
    sparse_forward_start.record()
    output_sparse = sparse_group_attention(
        query_sparse,
        key_sparse,
        value_sparse,
        False,
        1.0 / (D**0.5),
        sparse_kv_index,
        sparse_kv_num_per_head_tensor,
        group_size,
    )
    sparse_forward_end.record()

    # Sparse attention backward pass timing
    output_grad = torch.randn_like(output_sparse).contiguous()
    sparse_backward_start.record()
    output_sparse.backward(output_grad.clone())
    sparse_backward_end.record()

    print(
        f"Step 4: Running full attention baseline for comparison..."
    ) if not warm_up else None

    # Prepare tensors for full attention (fresh copies with gradients)
    query_full = query.detach().clone().requires_grad_(True)
    key_full = key.detach().clone().requires_grad_(True)
    value_full = value.detach().clone().requires_grad_(True)

    # Setup timing events for full attention
    full_forward_start = torch.cuda.Event(enable_timing=True)
    full_forward_end = torch.cuda.Event(enable_timing=True)
    full_backward_start = torch.cuda.Event(enable_timing=True)
    full_backward_end = torch.cuda.Event(enable_timing=True)

    # Full attention forward pass timing
    full_forward_start.record()
    output_full = full_attention(
        query_full, key_full, value_full, False, 1.0 / (D**0.5)
    )
    full_forward_end.record()

    # Full attention backward pass timing
    full_backward_start.record()
    output_full.backward(output_grad.clone())
    full_backward_end.record()

    print(
        f"Step 5: Validating correctness against masked full attention baseline..."
    ) if not warm_up else None

    correct_check_pass = True
    failed_ratio = 0.0

    try:
        mask = torch.zeros(B, H, num_groups, S, device="cuda", dtype=torch.bool)
        for b in range(B):
            for h in range(H):
                sparse_kv_num = sparse_kv_num_per_head_tensor[b, h].item()
                sparse_kv_index_h = sparse_kv_index[b, h, :, :sparse_kv_num]
                for g in range(sparse_kv_index_h.shape[0]):
                    mask[b, h, g, sparse_kv_index_h[g]] = True

        mask = mask.repeat_interleave(group_size, dim=-2)

        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            output_reference = F.scaled_dot_product_attention(
                query_ref,
                key_ref,
                value_ref,
                attn_mask=mask,
                is_causal=False,
                dropout_p=0.0,
                scale=1.0 / (D**0.5),
            )

        failed_ratio = compute_mismatch_ratio(output_reference, output_sparse)

        if failed_ratio > 0:
            correct_check_pass = False
            print(
                f" ❌ Failed to pass the correctness check, failed_ratio: {failed_ratio}%"
            )
        else:
            correct_check_pass = True
            print(f" ✅ Passed the correctness check, failed_ratio: {failed_ratio}%")

    except torch.cuda.OutOfMemoryError:
        print(
            f"⚠️  Memory insufficient for full correctness check - bypassing"
        ) if not warm_up else None
        correct_check_pass = None

    torch.cuda.synchronize()

    # Calculate timing results
    low_rank_time = event_low_rank_begin.elapsed_time(event_low_rank_end)
    sparse_forward_time = sparse_forward_start.elapsed_time(sparse_forward_end)
    sparse_backward_time = sparse_backward_start.elapsed_time(sparse_backward_end)
    sparse_total_time = sparse_forward_time + sparse_backward_time
    sparse_total_with_lowrank = low_rank_time + sparse_total_time

    full_forward_time = full_forward_start.elapsed_time(full_forward_end)
    full_backward_time = full_backward_start.elapsed_time(full_backward_end)
    full_total_time = full_forward_time + full_backward_time

    # Calculate speedup ratios
    forward_speedup = (
        full_forward_time / sparse_forward_time
        if sparse_forward_time > 0
        else float("inf")
    )
    backward_speedup = (
        full_backward_time / sparse_backward_time
        if sparse_backward_time > 0
        else float("inf")
    )
    total_speedup = (
        full_total_time / sparse_total_time if sparse_total_time > 0 else float("inf")
    )
    total_speedup_with_lowrank = (
        full_total_time / sparse_total_with_lowrank
        if sparse_total_with_lowrank > 0
        else float("inf")
    )

    if not warm_up:
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"  Performance Metrics:")
        print(f"    Low-rank KV Selection:")
        print(f"      - Time:              {low_rank_time:.2f} ms")
        print(f"    Forward Pass:")
        print(f"      - Sparse Attention:  {sparse_forward_time:.2f} ms")
        print(f"      - Full Attention:    {full_forward_time:.2f} ms")
        print(f"      - Speedup:           {forward_speedup:.2f}x")
        print(f"    Backward Pass:")
        print(f"      - Sparse Attention:  {sparse_backward_time:.2f} ms")
        print(f"      - Full Attention:    {full_backward_time:.2f} ms")
        print(f"      - Speedup:           {backward_speedup:.2f}x")
        print(f"    Total Time:")
        print(f"      - Sparse (excl. low-rank): {sparse_total_time:.2f} ms")
        print(f"      - Sparse (incl. low-rank): {sparse_total_with_lowrank:.2f} ms")
        print(f"      - Full Attention:          {full_total_time:.2f} ms")
        print(f"      - Speedup (excl. low-rank): {total_speedup:.2f}x")
        print(f"      - Speedup (incl. low-rank): {total_speedup_with_lowrank:.2f}x")

        print(f"  Correctness Metrics:")
        if correct_check_pass is True:
            print(f"    ✅ PASSED - Numerical error: {failed_ratio}% (within tolerance)")
        elif correct_check_pass is False:
            print(
                f"    ❌ FAILED - Numerical error: {failed_ratio}% (exceeds tolerance)"
            )
        else:
            print(f"    ⚠️  SKIPPED - Memory constraints prevented full validation")

        print(f"  Configuration:")

        if len(set(sparsity_per_head)) == 1:
            print(
                f"    - Sparsity (uniform): {sparsity_per_head[0]*100:.1f}% (keeping {(1-sparsity_per_head[0])*100:.1f}% of KV pairs)"
            )
        else:
            min_sparsity = min(sparsity_per_head)
            max_sparsity = max(sparsity_per_head)
            avg_sparsity = sum(sparsity_per_head) / len(sparsity_per_head)
            print(
                f"    - Sparsity (per head): {[f'{s*100:.1f}%' for s in sparsity_per_head]}"
            )
            print(
                f"    - Sparsity range: {min_sparsity*100:.1f}% - {max_sparsity*100:.1f}% (avg: {avg_sparsity*100:.1f}%)"
            )
            print(f"    - Avg KV retention: {(1-avg_sparsity)*100:.1f}%")
        print(f"    - Selected KV per head: {sparse_kv_num_per_head}")

    # Return timing data for visualization
    return {
        "low_rank_time": low_rank_time,
        "sparse_forward_time": sparse_forward_time,
        "sparse_backward_time": sparse_backward_time,
        "full_forward_time": full_forward_time,
        "full_backward_time": full_backward_time,
        "correct_check_pass": correct_check_pass,
        "failed_ratio": failed_ratio,
    }


def plot_performance_comparison(
    timing_data_list, seq_len, head_num, sparsity_per_head, group_size
):
    low_rank_times = [d["low_rank_time"] for d in timing_data_list]
    sparse_forward_times = [d["sparse_forward_time"] for d in timing_data_list]
    sparse_backward_times = [d["sparse_backward_time"] for d in timing_data_list]
    full_forward_times = [d["full_forward_time"] for d in timing_data_list]
    full_backward_times = [d["full_backward_time"] for d in timing_data_list]

    mean_low_rank_time = np.mean(low_rank_times)
    mean_sparse_forward_time = np.mean(sparse_forward_times)
    mean_sparse_backward_time = np.mean(sparse_backward_times)
    mean_full_forward_time = np.mean(full_forward_times)
    mean_full_backward_time = np.mean(full_backward_times)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    x_forward = np.array([0, 1])
    x_backward = np.array([3, 4])
    width = 0.6

    bars_full_forward = ax.bar(
        x_forward[0],
        mean_full_forward_time,
        width,
        label="Full Attention",
        color="#1f77b4",
        alpha=0.8,
    )

    bars_sparse_forward = ax.bar(
        x_forward[1],
        mean_sparse_forward_time,
        width,
        label="Sparse Attention",
        color="#ff7f0e",
        alpha=0.8,
    )

    bars_low_rank = ax.bar(
        x_forward[1],
        mean_low_rank_time,
        width,
        bottom=mean_sparse_forward_time,
        label="Low-rank Selection",
        color="#2ca02c",
        alpha=0.8,
    )

    bars_full_backward = ax.bar(
        x_backward[0], mean_full_backward_time, width, color="#1f77b4", alpha=0.8
    )

    bars_sparse_backward = ax.bar(
        x_backward[1], mean_sparse_backward_time, width, color="#ff7f0e", alpha=0.8
    )

    ax.set_ylabel("Time (ms)", fontsize=14, fontweight="bold")

    if len(set(sparsity_per_head)) == 1:
        sparsity_desc = f"{sparsity_per_head[0]*100:.0f}%"
    else:
        avg_sparsity = sum(sparsity_per_head) / len(sparsity_per_head)
        min_sparsity = min(sparsity_per_head)
        max_sparsity = max(sparsity_per_head)
        sparsity_desc = f"{min_sparsity*100:.0f}%-{max_sparsity*100:.0f}% (avg: {avg_sparsity*100:.0f}%)"

    ax.set_title(
        "Sparse vs Full Attention Performance Comparison\n"
        + f"Sequence Length: {seq_len:,}, Heads: {head_num}, Sparsity: {sparsity_desc}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(
        ["Forward Pass", "Backward Pass"], fontsize=13, fontweight="bold"
    )

    def add_value_labels(bars, values, offset=0):
        for bar, value in zip(bars, values):
            height = bar.get_height() + offset
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}ms",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

    add_value_labels(bars_full_forward, [mean_full_forward_time])
    add_value_labels(bars_sparse_forward, [mean_sparse_forward_time])
    add_value_labels(bars_low_rank, [mean_low_rank_time], mean_sparse_forward_time)
    add_value_labels(bars_full_backward, [mean_full_backward_time])
    add_value_labels(bars_sparse_backward, [mean_sparse_backward_time])

    forward_speedup = mean_full_forward_time / (
        mean_sparse_forward_time + mean_low_rank_time
    )
    backward_speedup = mean_full_backward_time / mean_sparse_backward_time

    ax.text(
        0.5,
        max(mean_full_forward_time, mean_sparse_forward_time + mean_low_rank_time)
        * 1.3,
        f"Speedup: {forward_speedup:.2f}x",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    ax.text(
        3.5,
        max(mean_full_backward_time, mean_sparse_backward_time) * 1.1,
        f"Speedup: {backward_speedup:.2f}x",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    max_height = max(
        mean_full_forward_time,
        mean_full_backward_time,
        mean_sparse_forward_time + mean_low_rank_time,
        mean_sparse_backward_time,
    )
    ax.set_ylim(0, max_height * 1.5)

    total_sparse_time = (
        mean_sparse_forward_time + mean_low_rank_time + mean_sparse_backward_time
    )
    total_full_time = mean_full_forward_time + mean_full_backward_time
    overall_speedup = total_full_time / total_sparse_time

    summary_text = (
        f"Performance Summary:\n" f"Overall Speedup: {overall_speedup:.2f}x\n"
    )

    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()

    import datetime

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(
        f"sparse_attention_performance_breakdown_{current_time}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(
        f"\nPerformance visualization saved as 'sparse_attention_performance_breakdown_{current_time}.pdf'"
    )


def sparse_attention_speedup_vs_full_attention(
    batch_size=1,
    seq_len=32000,
    head_num=2,
    head_dim=128,
    low_rank_dim=16,
    uniform_sparsity_to_test=0.9,
    dtype=torch.bfloat16,
    group_size=32,
):
    warm_up_num = 10
    test_num = 20

    query, key, value, _, _ = prepare_data(
        batch_size, seq_len, head_num, head_dim, low_rank_dim, dtype
    )

    num_groups = seq_len // group_size
    sparsity_per_head = [uniform_sparsity_to_test] * head_num
    sparse_kv_num_per_head = get_sparse_kv_num_per_head(sparsity_per_head, seq_len)
    sparse_kv_index = torch.zeros(
        batch_size,
        head_num,
        num_groups,
        max(sparse_kv_num_per_head),
        device="cuda",
        dtype=torch.int32,
    )
    # randomly select sparse kv index
    for b in range(batch_size):
        for h in range(head_num):
            for g in range(num_groups):
                sparse_kv_num = sparse_kv_num_per_head[h]
                sparse_kv_index[b, h, g, :sparse_kv_num] = torch.randperm(
                    seq_len, device="cuda", dtype=torch.int32
                )[:sparse_kv_num]

    sparse_kv_num_per_head_tensor = (
        torch.tensor(sparse_kv_num_per_head, device="cuda", dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    output_grad = torch.randn_like(query).contiguous()

    for i in range(warm_up_num):
        output_sparse = sparse_group_attention(
            query,
            key,
            value,
            False,
            1.0 / (head_dim**0.5),
            sparse_kv_index,
            sparse_kv_num_per_head_tensor,
            group_size,
        )
        output_sparse.backward(output_grad.clone(), retain_graph=True)

    sparse_forward_start = torch.cuda.Event(enable_timing=True)
    sparse_forward_end = torch.cuda.Event(enable_timing=True)
    sparse_backward_start = torch.cuda.Event(enable_timing=True)
    sparse_backward_end = torch.cuda.Event(enable_timing=True)

    sparse_forward_start.record()
    for i in range(test_num):
        output_sparse = sparse_group_attention(
            query,
            key,
            value,
            False,
            1.0 / (head_dim**0.5),
            sparse_kv_index,
            sparse_kv_num_per_head_tensor,
            group_size,
        )
    sparse_forward_end.record()

    sparse_backward_start.record()
    for i in range(test_num):
        output_sparse.backward(output_grad.clone(), retain_graph=True)
    sparse_backward_end.record()

    full_forward_start = torch.cuda.Event(enable_timing=True)
    full_forward_end = torch.cuda.Event(enable_timing=True)
    full_backward_start = torch.cuda.Event(enable_timing=True)
    full_backward_end = torch.cuda.Event(enable_timing=True)

    for i in range(warm_up_num):
        output_full = full_attention(query, key, value, False, 1.0 / (head_dim**0.5))
        output_full.backward(output_grad.clone(), retain_graph=True)

    full_forward_start.record()
    for i in range(test_num):
        output_full = full_attention(query, key, value, False, 1.0 / (head_dim**0.5))
    full_forward_end.record()

    full_backward_start.record()
    for i in range(test_num):
        output_full.backward(output_grad.clone(), retain_graph=True)
    full_backward_end.record()

    torch.cuda.synchronize()

    sparse_forward_time = (
        sparse_forward_start.elapsed_time(sparse_forward_end) / test_num
    )
    sparse_backward_time = (
        sparse_backward_start.elapsed_time(sparse_backward_end) / test_num
    )

    full_forward_time = full_forward_start.elapsed_time(full_forward_end) / test_num
    full_backward_time = full_backward_start.elapsed_time(full_backward_end) / test_num

    return {
        "sparse_forward_time": sparse_forward_time,
        "sparse_backward_time": sparse_backward_time,
        "full_forward_time": full_forward_time,
        "full_backward_time": full_backward_time,
    }


def test_sparse_attention_low_rank_vs_full_attention(
    batch_size=1,
    seq_len=32000,
    head_num=2,
    head_dim=128,
    low_rank_dim=16,
    sparsity_per_head=[0.93],
    group_size=32,
):
    print("=" * 80)
    print(f"Sparse Attention Kernel Evaluation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Sequence Length: {seq_len:,}")
    print(f"   • Attention Heads: {head_num}")
    print(f"   • Head Dimension: {head_dim}")
    print(f"   • Low-rank Dimension: {low_rank_dim}")
    print(f"   • Group Size: {group_size}")
    print(f"   • Sparsity Per Head: {sparsity_per_head}")
    print("=" * 80)

    print(f"\nWarming up GPU kernels...")
    for i in range(10):
        sparse_attention_w_low_rank_correctness_and_performance_benchmark(
            batch_size,
            seq_len,
            head_num,
            head_dim,
            low_rank_dim,
            sparsity_per_head=sparsity_per_head,
            warm_up=True,
            group_size=group_size,
        )

    print(f"\nRunning evaluation benchmarks...")
    timing_data_list = []

    real_run_num = 3
    for i in range(real_run_num):
        print(f"\n{'='*40} RUN {i+1}/{real_run_num} {'='*40}")
        timing_data = sparse_attention_w_low_rank_correctness_and_performance_benchmark(
            batch_size,
            seq_len,
            head_num,
            head_dim,
            low_rank_dim,
            sparsity_per_head=sparsity_per_head,
            warm_up=False,
            group_size=group_size,
        )
        timing_data_list.append(timing_data)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    print(f"\nGenerating performance visualization...")
    plot_performance_comparison(
        timing_data_list, seq_len, head_num, sparsity_per_head, group_size
    )


def test_sparse_attention_speedup_with_sparsity(
    batch_size=1,
    seq_len=32000,
    head_num=2,
    head_dim=128,
    low_rank_dim=16,
    uniform_sparsity_to_test=[0.8, 0.85, 0.9, 0.95, 0.98],
    dtype=torch.bfloat16,
    group_size=32,
):
    forward_speedup_list = []
    backward_speedup_list = []
    for sparsity in uniform_sparsity_to_test:
        print(f"Testing sparsity: {sparsity}")
        timing_data = sparse_attention_speedup_vs_full_attention(
            batch_size,
            seq_len,
            head_num,
            head_dim,
            low_rank_dim,
            sparsity,
            dtype,
            group_size,
        )
        print(
            f"Sparse Attention Forward Time: {timing_data['sparse_forward_time']:.2f} ms, Backward Time: {timing_data['sparse_backward_time']:.2f} ms"
        )
        print(
            f"Full Attention Forward Time: {timing_data['full_forward_time']:.2f} ms, Backward Time: {timing_data['full_backward_time']:.2f} ms"
        )

        forward_speedup = (
            timing_data["full_forward_time"] / timing_data["sparse_forward_time"]
        )
        backward_speedup = (
            timing_data["full_backward_time"] / timing_data["sparse_backward_time"]
        )
        forward_speedup_list.append(forward_speedup)
        backward_speedup_list.append(backward_speedup)

    print(f"Forward Speedup: {forward_speedup_list}")
    print(f"Backward Speedup: {backward_speedup_list}")

    plt.figure(figsize=(4, 3))
    # large marksize and wider line
    plt.plot(
        uniform_sparsity_to_test,
        forward_speedup_list,
        marker="o",
        label="Forward Speedup",
        markersize=10,
        linewidth=2,
    )
    plt.plot(
        uniform_sparsity_to_test,
        backward_speedup_list,
        marker="o",
        label="Backward Speedup",
        markersize=10,
        linewidth=2,
    )
    plt.xlabel("Sparsity", fontsize=14)
    plt.ylabel("Speedup", fontsize=14)
    plt.title("Speedup vs Sparsity", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    import datetime

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(
        f"sparse_attention_speedup_with_different_sparsity_{current_time}.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", type=str, default="breakdown", choices=["breakdown", "speedup"])
    args = parser.parse_args()

    if args.test_type == "speedup":
        test_sparse_attention_speedup_with_sparsity(
            batch_size=1,
            seq_len=32000,
            head_num=16,
            head_dim=128,
            low_rank_dim=16,
            uniform_sparsity_to_test=[0.8, 0.85, 0.9, 0.95, 0.98],
        )


    elif args.test_type == "breakdown":
        test_sparse_attention_low_rank_vs_full_attention(
            batch_size=1,
            seq_len=64000,
            head_num=4, # use a small head number to enable the correctness test
            head_dim=128,
            low_rank_dim=16,
            sparsity_per_head=[0.92]*4,
    )
