"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import triton
import triton.language as tl

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [32,64, 128]\
#     for BN in [32, 64]\
#     for s in ([1,3,5])\
#     # 3,4,7
#     for w in [4, 8]\
# ]

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [32]
    for BN in [128, 256]
    for s in ([3, 5])  # 3,4,7
    for w in [4]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def sparse_group_attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #
    KV_INDEX,
    TOPK_PER_HEAD,
    stride_kvz,
    stride_kvh,
    stride_kvm,
    stride_kvk,
    stride_topk_per_headz,
    stride_topk_per_headh,
    HEAD_DIM: tl.constexpr,  #
    QUERY_GROUP_SIZE: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    """
    Sparse Query Group Attention Forward Pass Kernel

    This kernel implements forward pass for sparse attention where queries are grouped
    and each group shares the same sparse KV indices.
    """
    # tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )

    topk_per_head_offset = (
        off_z.to(tl.int64) * stride_topk_per_headz
        + off_h.to(tl.int64) * stride_topk_per_headh
    )

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    k_base_ptr = K + qvk_offset
    v_base_ptr = V + qvk_offset

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    topk = tl.load(TOPK_PER_HEAD + topk_per_head_offset)

    lo, hi = 0, topk  # non-causal

    # loop over k, v and update accumulator
    hi = (topk + BLOCK_N - 1) // BLOCK_N * BLOCK_N

    head_dim_offset = tl.arange(0, HEAD_DIM)

    # Calculate group index for this block
    # Assumes queries are pre-arranged so that tokens in the same group are contiguous
    group_idx = (start_m * BLOCK_M) // QUERY_GROUP_SIZE

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        kv_index_offset = (
            kvindex_base_ptr + group_idx * stride_kvm + (start_n + offs_n) * stride_kvk
        )

        # Create mask for valid indices first
        # valid_mask = offs_n < topk
        mask = (start_n + offs_n) < topk
        kv_index = tl.load(kv_index_offset, mask=mask)

        # -- compute qk ----
        # K is stored as [N_CTX, HEAD_DIM], but we load it as transposed [HEAD_DIM, BLOCK_N]
        # by swapping the stride order to get the transpose directly
        k = tl.load(
            k_base_ptr
            + head_dim_offset[:, None] * stride_kk  # Head dimension stride
            + kv_index[None, :] * stride_kn,  # Token position stride
            mask=mask[None, :],
        )  # Shape: [HEAD_DIM, BLOCK_N] (transposed during loading)

        qk = tl.dot(
            q, k
        )  # q: [BLOCK_M, HEAD_DIM], k: [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N]

        # Apply mask to qk to zero out invalid positions
        # qk = tl.where(valid_mask[None, :], qk, float("-inf"))

        # only non-causal here
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        # V is stored as [N_CTX, HEAD_DIM], and we load it normally
        v = tl.load(
            v_base_ptr
            + kv_index[:, None] * stride_vk  # Token position stride
            + head_dim_offset[None, :] * stride_vn,  # Head dimension stride
            mask=mask[:, None],
        )  # Shape: [BLOCK_N, HEAD_DIM]

        p = p.to(v.dtype)
        acc = tl.dot(
            p, v, acc
        )  # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, HEAD_DIM] -> [BLOCK_M, HEAD_DIM]

        # update m_i and l_i
        m_i = m_ij
        # Loop variable start_n is automatically updated by range() function

    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    m_i += tl.math.log2(l_i)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)


@triton.jit
def sparse_group_attn_bwd_delta_preprocess(
    O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """
    Sparse Query Group Attention Backward Delta Preprocessing Kernel

    Computes delta = sum(O * dO) for each token, required for backward pass.
    """
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def sparse_group_attn_bwd_dkdv(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    KV_INDEX,
    TOPK_PER_HEAD,
    stride_kvz,
    stride_kvh,
    stride_kvm,
    stride_kvk,
    stride_topk_per_headz,
    stride_topk_per_headh,
    H,
    N_CTX,
    QUERY_GROUP_SIZE: tl.constexpr,  # 添加QUERY_GROUP_SIZE参数
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Sparse Query Group Attention Backward Pass Kernel for dK and dV

    Computes gradients for keys and values in grouped sparse attention.
    Uses atomic operations to handle shared KV indices across query groups.
    """
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    off_z = bhid // H
    off_h = bhid % H

    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )

    topk_per_head_offset = (
        off_z.to(tl.int64) * stride_topk_per_headz
        + off_h.to(tl.int64) * stride_topk_per_headh
    )
    topk = tl.load(TOPK_PER_HEAD + topk_per_head_offset)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    num_steps = (topk + BLOCK_N1 - 1) // BLOCK_N1
    start_m = pid * BLOCK_M1

    offset_m = start_m + tl.arange(0, BLOCK_M1)
    offset_k = tl.arange(0, HEAD_DIM)
    offset_n = tl.arange(0, BLOCK_N1)

    # Fix: group index calculation, keep consistent with forward passtion, keep consistent with forward pass
    # start_m is already pid * BLOCK_M1, so just divide by QUERY_GROUP_SIZE
    group_idx = start_m // QUERY_GROUP_SIZE

    # Base offset for this group
    kv_index_base_offset = kvindex_base_ptr + group_idx * stride_kvm

    # Load qT (query transpose): Q is [N_CTX, HEAD_DIM], we need qT: [HEAD_DIM, BLOCK_M1]
    qT_ptrs = Q + offset_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    qT = tl.load(qT_ptrs)  # Shape: [HEAD_DIM, BLOCK_M1]

    # Load other data normally
    do_ptrs = DO + offset_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    do = tl.load(do_ptrs)  # Shape: [BLOCK_M1, HEAD_DIM]
    m = tl.load(M + offset_m)  # Shape: [BLOCK_M1]
    Di = tl.load(D + offset_m)  # Shape: [BLOCK_M1]

    # Create base offset array for this block
    offset_n_base = tl.arange(0, BLOCK_N1)

    for blk_idx in range(num_steps):
        # Calculate current offset for this iteration
        current_offset_n = blk_idx * BLOCK_N1 + offset_n_base
        kv_index_offset = kv_index_base_offset + current_offset_n * stride_kvk

        kv_index = tl.load(kv_index_offset)  # [BLOCK_N]
        # Create mask for valid indices
        # valid_mask = current_offset_n < topk

        # Load K and V normally: both are stored as [N_CTX, HEAD_DIM], load as [BLOCK_N, HEAD_DIM]
        k = tl.load(
            K + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d,
            # mask=valid_mask[:, None],  # Apply mask to avoid invalid memory access
            # other=0.0
        )  # Shape: [BLOCK_N, HEAD_DIM]

        v = tl.load(
            V + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d,
            # mask=valid_mask[:, None],  # Apply mask to avoid invalid memory access
            # other=0.0
        )  # Shape: [BLOCK_N, HEAD_DIM]

        # TODO:
        qkT = tl.dot(
            k, qT
        )  # k: [BLOCK_N, HEAD_DIM], qT: [HEAD_DIM, BLOCK_M] -> [BLOCK_N, BLOCK_M]
        pT = tl.math.exp2(qkT - m[None, :])

        # Apply mask to pT to zero out invalid positions
        # pT = tl.where(valid_mask[:, None], pT, 0.0)

        ppT = pT.to(do.dtype)
        dv = tl.dot(
            ppT, do
        )  # ppT: [BLOCK_N, BLOCK_M], do: [BLOCK_M, HEAD_DIM] -> [BLOCK_N, HEAD_DIM]

        dpT = tl.dot(v, tl.trans(do)).to(
            tl.float32
        )  # v: [BLOCK_N, HEAD_DIM], do.T: [HEAD_DIM, BLOCK_M] -> [BLOCK_N, BLOCK_M]
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk = tl.dot(
            dsT, tl.trans(qT)
        )  # dsT: [BLOCK_N, BLOCK_M], qT.T: [BLOCK_M, HEAD_DIM] -> [BLOCK_N, HEAD_DIM]

        # Store gradients normally - both DK and DV are stored as [N_CTX, HEAD_DIM]
        # Note: For demo purposes, we use regular store instead of atomic_add
        # This may cause minor data races but should work for demonstration

        # Convert to the target data type
        dv_final = dv.to(tl.float32)

        dk_final = (dk * sm_scale).to(tl.float32)

        # Apply mask to ensure only valid gradients are stored
        # dv_masked = tl.where(valid_mask[:, None], dv_final, 0.0)
        # dk_masked = tl.where(valid_mask[:, None], dk_final, 0.0)

        # Use regular store operations (no atomic operations)
        dv_ptrs = DV + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d
        dk_ptrs = DK + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d

        # Store with mask to avoid storing to invalid locations
        # tl.store(dv_ptrs, dv_final)#, mask=valid_mask[:, None])
        # tl.store(dk_ptrs, dk_final)#, mask=valid_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_final, sem="relaxed")
        tl.atomic_add(dk_ptrs, dk_final, sem="relaxed")


@triton.jit
def sparse_group_attn_bwd_dq(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    KV_INDEX,
    TOPK_PER_HEAD,
    stride_kvz,
    stride_kvh,
    stride_kvm,
    stride_kvk,
    stride_topk_per_headz,
    stride_topk_per_headh,
    H,
    N_CTX,
    QUERY_GROUP_SIZE: tl.constexpr,  # Add QUERY_GROUP_SIZE parameter
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Sparse Query Group Attention Backward Pass Kernel for dQ

    Computes gradients for queries in grouped sparse attention.
    Each query group shares the same sparse KV indices.
    """
    LN2: tl.constexpr = 0.6931471824645996
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    off_z = bhid // H
    off_h = bhid % H

    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )

    topk_per_head_offset = (
        off_z.to(tl.int64) * stride_topk_per_headz
        + off_h.to(tl.int64) * stride_topk_per_headh
    )
    topk = tl.load(TOPK_PER_HEAD + topk_per_head_offset)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    start_m = pid * BLOCK_M2

    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    Di = tl.load(D + offs_m)
    offs_n = tl.arange(0, BLOCK_N2)

    # Fix: group index calculation, keep consistent with forward passtion, keep consistent with forward pass
    # start_m is already pid * BLOCK_M2, so just divide by QUERY_GROUP_SIZE
    group_idx = start_m // QUERY_GROUP_SIZE

    # Base offset for this group
    kv_index_base_offset = kvindex_base_ptr + group_idx * stride_kvm

    num_steps = (topk + BLOCK_N2 - 1) // BLOCK_N2

    # Create base offset array for this block
    offs_n_base = tl.arange(0, BLOCK_N2)

    for blk_idx in range(num_steps):
        # Calculate current offset for this iteration
        current_offs_n = blk_idx * BLOCK_N2 + offs_n_base
        kv_index_offset = kv_index_base_offset + current_offs_n * stride_kvk

        kv_index = tl.load(kv_index_offset)  # , mask=(current_offs_n < topk))
        # Create mask for valid indices
        # valid_mask = current_offs_n < topk

        # Load K transpose directly: K is stored as [N_CTX, HEAD_DIM], but we load it as kT: [HEAD_DIM, BLOCK_N]
        # by swapping stride order to get the transpose directly during loading
        kT = tl.load(
            K + offs_k[:, None] * stride_d + kv_index[None, :] * stride_tok,
            # mask=valid_mask[None, :],  # Apply mask to avoid invalid memory access
            # other=0.0
        )  # Shape: [HEAD_DIM, BLOCK_N] (transposed during loading)

        # Load V transpose directly: V is stored as [N_CTX, HEAD_DIM], but we load it as vT: [HEAD_DIM, BLOCK_N]
        # by swapping stride order to get the transpose directly during loading
        vT = tl.load(
            V + kv_index[None, :] * stride_tok + offs_k[:, None] * stride_d,
            # mask=valid_mask[None, :],  # Apply mask to avoid invalid memory access
            # other=0.0
        )  # Shape: [HEAD_DIM, BLOCK_N] (transposed during loading)

        qk = tl.dot(
            q, kT
        )  # q: [BLOCK_M, HEAD_DIM], kT: [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N]

        # Apply mask to qk to zero out invalid positions
        # qk = tl.where(valid_mask[None, :], qk, float("-inf"))

        p = tl.math.exp2(qk - m)

        dp = tl.dot(do, vT).to(
            tl.float32
        )  # do: [BLOCK_M, HEAD_DIM], vT: [HEAD_DIM, BLOCK_N] -> [BLOCK_M, BLOCK_N]

        # Apply mask to dp to zero out invalid positions
        # dp = tl.where(valid_mask[None, :], dp, 0.0)

        ds = p * (dp - Di[:, None])
        ds = ds.to(kT.dtype)

        dq += tl.dot(
            ds, tl.trans(kT)
        )  # ds: [BLOCK_M, BLOCK_N], kT.T: [BLOCK_N, HEAD_DIM] -> [BLOCK_M, HEAD_DIM]

        # Removed the incorrect offset updates - now using calculated offsets per iteration

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# Sparse Query Group Attention class with friendly names
class SparseGroupAttention(torch.autograd.Function):
    """
    Sparse Query Group Attention Implementation

    This implementation groups queries and allows each group to share the same
    sparse KV indices, reducing memory usage and improving efficiency for
    video processing applications.
    """

    @staticmethod
    def forward(
        ctx, q, k, v, causal, sm_scale, kv_index, topk_per_head, query_group_size
    ):
        """
        Forward pass for grouped sparse attention.

        Args:
            q: [B, H, S, D] - queries (should be pre-arranged by groups)
            k: [B, H, S, D] - keys
            v: [B, H, S, D] - values
            causal: bool - whether to use causal attention
            sm_scale: float - scaling factor
            kv_index: [B, H, num_groups, topk] - KV indices for each group
            topk_per_head: [B, H] - number of top-k values per head
            query_group_size: int - size of each query group
        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # Forward pass block size (from autotune configs)
        BLOCK_M = 32  # This is the BLOCK_M used in forward pass (from configs)

        # Assert: QUERY_GROUP_SIZE must be divisible by BLOCK_M for correct group indexing
        assert query_group_size % BLOCK_M == 0, (
            f"QUERY_GROUP_SIZE ({query_group_size}) must be divisible by BLOCK_M ({BLOCK_M}) "
            f"for correct group indexing in forward pass. "
            f"Please adjust your 3D group dimensions so that "
            f"group_f_size * group_h_size * group_w_size is divisible by {BLOCK_M}."
        )

        o = torch.empty_like(q)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        kv_index = kv_index.contiguous()
        topk_per_head = topk_per_head.contiguous()

        stage = 1  # non-causal

        extra_kern_args = {}

        grid = lambda args: (
            triton.cdiv(q.shape[2], args["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        sparse_group_attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q.shape[0],
            q.shape[1],  #
            N_CTX=q.shape[2],  #
            KV_INDEX=kv_index,  #
            TOPK_PER_HEAD=topk_per_head,  #
            stride_kvz=kv_index.stride(0),
            stride_kvh=kv_index.stride(1),
            stride_kvm=kv_index.stride(2),
            stride_kvk=kv_index.stride(3),
            stride_topk_per_headz=topk_per_head.stride(0),
            stride_topk_per_headh=topk_per_head.stride(1),
            HEAD_DIM=HEAD_DIM_K,  #
            QUERY_GROUP_SIZE=query_group_size,  #
            STAGE=stage,  #
        )

        ctx.save_for_backward(q, k, v, o, M, kv_index, topk_per_head)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.query_group_size = query_group_size

        return o.contiguous()

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, kv_index, topk_per_head = ctx.saved_tensors

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        assert do.is_contiguous()

        assert q.stride() == o.stride() == do.stride()
        assert k.stride() == v.stride()

        dq = torch.empty_like(q)
        dk = torch.zeros_like(
            k, dtype=torch.float32
        )  # Initialize to zero for non-atomic operations
        dv = torch.zeros_like(
            v, dtype=torch.float32
        )  # Initialize to zero for non-atomic operations

        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape[:4]
        PRE_BLOCK = 128

        if HEAD_DIM == 256:
            NUM_WARPS, NUM_STAGES = 4, 2
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 32, 128
        else:
            NUM_WARPS, NUM_STAGES = 4, 3
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 32, 256

        # Assert: QUERY_GROUP_SIZE must be divisible by BLOCK_M1 and BLOCK_M2 for correct group indexing
        assert ctx.query_group_size % BLOCK_M1 == 0, (
            f"QUERY_GROUP_SIZE ({ctx.query_group_size}) must be divisible by BLOCK_M1 ({BLOCK_M1}) "
            f"for correct group indexing in backward pass. "
            f"Please adjust your 3D group dimensions so that "
            f"group_f_size * group_h_size * group_w_size is divisible by {BLOCK_M1}."
        )
        assert ctx.query_group_size % BLOCK_M2 == 0, (
            f"QUERY_GROUP_SIZE ({ctx.query_group_size}) must be divisible by BLOCK_M2 ({BLOCK_M2}) "
            f"for correct group indexing in backward pass. "
            f"Please adjust your 3D group dimensions so that "
            f"group_f_size * group_h_size * group_w_size is divisible by {BLOCK_M2}."
        )

        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        sparse_group_attn_bwd_delta_preprocess[pre_grid](
            o,
            do,  #
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,  #
        )

        # print(f"execute delta success")

        grid = (N_CTX // BLOCK_M1, 1, BATCH * N_HEAD)
        sparse_group_attn_bwd_dkdv[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            kv_index,
            topk_per_head,
            kv_index.stride(0),
            kv_index.stride(1),
            kv_index.stride(2),
            kv_index.stride(3),
            topk_per_head.stride(0),
            topk_per_head.stride(1),
            N_HEAD,
            N_CTX,  #
            QUERY_GROUP_SIZE=ctx.query_group_size,  # 添加QUERY_GROUP_SIZE参数
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
        )

        # print(f"execute dkdv success")

        grid_dq_sparse = (N_CTX // BLOCK_M2, 1, BATCH * N_HEAD)
        sparse_group_attn_bwd_dq[grid_dq_sparse](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            kv_index,
            topk_per_head,
            kv_index.stride(0),
            kv_index.stride(1),
            kv_index.stride(2),
            kv_index.stride(3),
            topk_per_head.stride(0),
            topk_per_head.stride(1),
            N_HEAD,
            N_CTX,  #
            QUERY_GROUP_SIZE=ctx.query_group_size,  # Add QUERY_GROUP_SIZE parameter
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
        )
        # print(f"execute dq success")

        dk = dk.to(k.dtype)  # ✅ 只进行类型转换，不scaling
        dv = dv.to(v.dtype)

        dq = dq.contiguous()
        dk = dk.contiguous()
        dv = dv.contiguous()

        return dq, dk, dv, None, None, None, None, None


# Convenience function for 3D video attention
def video_attention_3d(
    q,
    k,
    v,
    causal,
    sm_scale,
    kv_index,
    topk_per_head,
    video_f_size,
    video_h_size,
    video_w_size,
    group_f_size,
    group_h_size,
    group_w_size,
):
    """
    Convenience function for 3D video attention with automatic token reordering.

    Args:
        q, k, v: [B, H, S, D] - input tensors
        causal: bool - whether to use causal attention
        sm_scale: float - scaling factor
        kv_index: [B, H, num_groups, topk] - KV indices (arranged for 3D groups)
        topk_per_head: [B, H] - number of top-k values per head
        video_f_size, video_h_size, video_w_size: 3D video dimensions
        group_f_size, group_h_size, group_w_size: 3D group dimensions

    Returns:
        output: [B, H, S, D] - attention output in original order
    """
    query_group_size = group_f_size * group_h_size * group_w_size

    # Assert: 3D group dimensions must result in a group size that works with kernel block sizes
    BLOCK_M = 32  # Standard block size used in kernels
    assert query_group_size % BLOCK_M == 0, (
        f"3D group size ({group_f_size} × {group_h_size} × {group_w_size} = {query_group_size}) "
        f"must be divisible by kernel block size ({BLOCK_M}). "
        f"Please choose group dimensions such that their product is divisible by {BLOCK_M}. "
        f"Examples: 32×1×1, 16×2×1, 8×4×1, 4×4×2, 2×4×4, etc."
    )

    # Verify video dimensions are compatible with group dimensions
    assert (
        video_f_size % group_f_size == 0
    ), f"video_f_size ({video_f_size}) must be divisible by group_f_size ({group_f_size})"
    assert (
        video_h_size % group_h_size == 0
    ), f"video_h_size ({video_h_size}) must be divisible by group_h_size ({group_h_size})"
    assert (
        video_w_size % group_w_size == 0
    ), f"video_w_size ({video_w_size}) must be divisible by group_w_size ({group_w_size})"

    # Rearrange query tokens to group 3D spatial neighbors
    q_rearranged, inverse_indices = rearrange_query_tokens_3d(
        q,
        video_f_size,
        video_h_size,
        video_w_size,
        group_f_size,
        group_h_size,
        group_w_size,
    )

    # Apply attention with grouped queries
    o_rearranged = SparseGroupAttention.apply(
        q_rearranged, k, v, causal, sm_scale, kv_index, topk_per_head, query_group_size
    )

    # Restore original token order
    o_restored = restore_output_order(o_rearranged, inverse_indices)

    return o_restored


# Backward compatibility aliases
_attention = SparseGroupAttention  # Keep old name for backward compatibility
attention = SparseGroupAttention.apply  # Keep old name for backward compatibility
sparse_group_attention = SparseGroupAttention.apply  # New friendly name
