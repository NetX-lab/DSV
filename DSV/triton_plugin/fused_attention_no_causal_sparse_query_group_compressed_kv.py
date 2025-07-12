
import torch
import triton
import triton.language as tl

# Tuning configurations
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [32]
    for BN in [128, 256]
    for s in ([3, 5])
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
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    KV_INDEX,
    TOPK_PER_HEAD,
    stride_kvz,
    stride_kvh,
    stride_kvm,
    stride_kvk,
    stride_topk_per_headz,
    stride_topk_per_headh,
    TOPK_INDEX_TO_PACKED_INDEX,
    stride_tipi_z,
    stride_tipi_h,
    stride_tipi_s,
    MAX_ACTIVATED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    """Sparse attention forward pass with compressed KV storage."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )

    tipi_base_ptr = (
        TOPK_INDEX_TO_PACKED_INDEX
        + off_z.to(tl.int64) * stride_tipi_z
        + off_h.to(tl.int64) * stride_tipi_h
    )

    topk_per_head_offset = (
        off_z.to(tl.int64) * stride_topk_per_headz
        + off_h.to(tl.int64) * stride_topk_per_headh
    )

    # Block pointers
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

    # Initialize accumulators
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    q = tl.load(Q_block_ptr)

    topk = tl.load(TOPK_PER_HEAD + topk_per_head_offset)
    lo, hi = 0, topk
    hi = (topk + BLOCK_N - 1) // BLOCK_N * BLOCK_N

    head_dim_offset = tl.arange(0, HEAD_DIM)
    group_idx = (start_m * BLOCK_M) // QUERY_GROUP_SIZE

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        kv_index_offset = (
            kvindex_base_ptr + group_idx * stride_kvm + (start_n + offs_n) * stride_kvk
        )

        mask = (start_n + offs_n) < topk
        kv_index = tl.load(kv_index_offset)

        # Convert global indices to packed indices
        tipi_offset = tipi_base_ptr + kv_index * stride_tipi_s
        kv_index_packed = tl.load(tipi_offset)

        valid_packed_mask = (kv_index_packed >= 0) & (kv_index_packed < MAX_ACTIVATED)
        final_mask = mask & valid_packed_mask

        # Load K with transpose
        k = tl.load(
            k_base_ptr
            + head_dim_offset[:, None] * stride_kk
            + kv_index_packed[None, :] * stride_kn,
            mask=final_mask[None, :],
        )

        qk = tl.dot(q, k)
        qk = tl.where(final_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # Update accumulators
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V
        v = tl.load(
            v_base_ptr
            + kv_index_packed[:, None] * stride_vk
            + head_dim_offset[None, :] * stride_vn,
            mask=final_mask[:, None],
        )

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    # Store output
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    m_i += tl.math.log2(l_i)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)


@triton.jit
def sparse_group_attn_bwd_delta_preprocess(
    O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """Precompute delta = sum(O * dO) for backward pass."""
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)

    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

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
    TOPK_INDEX_TO_PACKED_INDEX,
    stride_tipi_z,
    stride_tipi_h,
    stride_tipi_s,
    MAX_ACTIVATED: tl.constexpr,
    H,
    N_CTX,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Backward pass for dK and dV computation."""
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    off_z = bhid // H
    off_h = bhid % H

    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )

    tipi_base_ptr = (
        TOPK_INDEX_TO_PACKED_INDEX
        + off_z.to(tl.int64) * stride_tipi_z
        + off_h.to(tl.int64) * stride_tipi_h
    )

    topk_per_head_offset = (
        off_z.to(tl.int64) * stride_topk_per_headz
        + off_h.to(tl.int64) * stride_topk_per_headh
    )
    topk = tl.load(TOPK_PER_HEAD + topk_per_head_offset)

    # Offset pointers for batch/head
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

    group_idx = start_m // QUERY_GROUP_SIZE
    kv_index_base_offset = kvindex_base_ptr + group_idx * stride_kvm

    # Load query transpose: [HEAD_DIM, BLOCK_M1]
    qT_ptrs = Q + offset_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    qT = tl.load(qT_ptrs)

    # Load other data
    do_ptrs = DO + offset_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    do = tl.load(do_ptrs)
    m = tl.load(M + offset_m)
    Di = tl.load(D + offset_m)

    offset_n_base = tl.arange(0, BLOCK_N1)

    for blk_idx in range(num_steps):
        current_offset_n = blk_idx * BLOCK_N1 + offset_n_base
        kv_index_offset = kv_index_base_offset + current_offset_n * stride_kvk

        topk_mask = current_offset_n < topk
        kv_index = tl.load(kv_index_offset, mask=topk_mask, other=-1)

        # Convert global to packed indices
        tipi_offset = tipi_base_ptr + kv_index * stride_tipi_s
        kv_index_packed = tl.load(tipi_offset, mask=topk_mask, other=-1)

        valid_packed_mask = (kv_index_packed >= 0) & (kv_index_packed < MAX_ACTIVATED)
        final_mask = topk_mask & valid_packed_mask

        # Load K and V
        k = tl.load(
            K + kv_index_packed[:, None] * stride_tok + offset_k[None, :] * stride_d,
            mask=final_mask[:, None],
            other=0.0,
        )

        v = tl.load(
            V + kv_index_packed[:, None] * stride_tok + offset_k[None, :] * stride_d,
            mask=final_mask[:, None],
            other=0.0,
        )

        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        pT = tl.where(final_mask[:, None], pT, 0.0)

        ppT = pT.to(do.dtype)
        dv = tl.dot(ppT, do)

        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk = tl.dot(dsT, tl.trans(qT))

        # Store gradients
        dv_final = dv.to(tl.float32)
        dk_final = (dk * sm_scale).to(tl.float32)

        dv_masked = tl.where(final_mask[:, None], dv_final, 0.0)
        dk_masked = tl.where(final_mask[:, None], dk_final, 0.0)

        dv_ptrs = (
            DV + kv_index_packed[:, None] * stride_tok + offset_k[None, :] * stride_d
        )
        dk_ptrs = (
            DK + kv_index_packed[:, None] * stride_tok + offset_k[None, :] * stride_d
        )

        tl.atomic_add(dv_ptrs, dv_masked, mask=final_mask[:, None], sem="relaxed")
        tl.atomic_add(dk_ptrs, dk_masked, mask=final_mask[:, None], sem="relaxed")


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
    TOPK_INDEX_TO_PACKED_INDEX,
    stride_tipi_z,
    stride_tipi_h,
    stride_tipi_s,
    MAX_ACTIVATED: tl.constexpr,
    H,
    N_CTX,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Backward pass for dQ computation."""
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

    tipi_base_ptr = (
        TOPK_INDEX_TO_PACKED_INDEX
        + off_z.to(tl.int64) * stride_tipi_z
        + off_h.to(tl.int64) * stride_tipi_h
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

    group_idx = start_m // QUERY_GROUP_SIZE
    kv_index_base_offset = kvindex_base_ptr + group_idx * stride_kvm

    num_steps = (topk + BLOCK_N2 - 1) // BLOCK_N2
    offs_n_base = tl.arange(0, BLOCK_N2)

    for blk_idx in range(num_steps):
        current_offs_n = blk_idx * BLOCK_N2 + offs_n_base
        kv_index_offset = kv_index_base_offset + current_offs_n * stride_kvk

        topk_mask = current_offs_n < topk
        kv_index = tl.load(kv_index_offset, mask=topk_mask, other=-1)

        # Convert global to packed indices
        tipi_offset = tipi_base_ptr + kv_index * stride_tipi_s
        kv_index_packed = tl.load(tipi_offset, mask=topk_mask, other=-1)

        valid_packed_mask = (kv_index_packed >= 0) & (kv_index_packed < MAX_ACTIVATED)
        final_mask = topk_mask & valid_packed_mask

        # Load K and V transposed
        kT = tl.load(
            K + offs_k[:, None] * stride_d + kv_index_packed[None, :] * stride_tok,
            mask=final_mask[None, :],
            other=0.0,
        )

        vT = tl.load(
            V + kv_index_packed[None, :] * stride_tok + offs_k[:, None] * stride_d,
            mask=final_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, kT)
        qk = tl.where(final_mask[None, :], qk, float("-inf"))

        p = tl.math.exp2(qk - m)
        dp = tl.dot(do, vT).to(tl.float32)
        dp = tl.where(final_mask[None, :], dp, 0.0)

        ds = p * (dp - Di[:, None])
        ds = ds.to(kT.dtype)

        dq += tl.dot(ds, tl.trans(kT))

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class CompressedSparseGroupAttention(torch.autograd.Function):
    """
    Compressed Sparse Group Attention Implementation

    Groups queries and allows each group to share the same sparse KV indices.
    Uses compressed KV storage for memory efficiency.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        causal,
        sm_scale,
        kv_index,
        topk_per_head,
        topk_index_to_packed_index,
        query_group_size,
    ):
        """Forward pass for grouped sparse attention with compressed KV storage."""
        # Validate inputs
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        assert (
            kv_index.dtype == torch.int32
        ), f"kv_index must be int32, got {kv_index.dtype}"
        assert (
            topk_index_to_packed_index.dtype == torch.int32
        ), f"topk_index_to_packed_index must be int32, got {topk_index_to_packed_index.dtype}"
        assert (
            topk_per_head.dtype == torch.int32
        ), f"topk_per_head must be int32, got {topk_per_head.dtype}"

        batch_size, num_heads = q.shape[0], q.shape[1]
        assert (
            topk_index_to_packed_index.shape[0] == batch_size
        ), f"Batch size mismatch: topk_index_to_packed_index.shape[0]={topk_index_to_packed_index.shape[0]}, expected {batch_size}"
        assert (
            topk_index_to_packed_index.shape[1] == num_heads
        ), f"Head num mismatch: topk_index_to_packed_index.shape[1]={topk_index_to_packed_index.shape[1]}, expected {num_heads}"
        assert (
            kv_index.shape[0] == batch_size and kv_index.shape[1] == num_heads
        ), f"KV index shape mismatch"
        assert (
            topk_per_head.shape[0] == batch_size and topk_per_head.shape[1] == num_heads
        ), f"Top-k per head shape mismatch"

        # Block size constraints
        BLOCK_M = 32
        assert query_group_size % BLOCK_M == 0, (
            f"QUERY_GROUP_SIZE ({query_group_size}) must be divisible by BLOCK_M ({BLOCK_M}) "
            f"for correct group indexing in forward pass."
        )

        o = torch.empty_like(q)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        kv_index = kv_index.contiguous()
        topk_per_head = topk_per_head.contiguous()
        topk_index_to_packed_index = topk_index_to_packed_index.contiguous()

        max_activated = k.shape[2]
        stage = 1  # non-causal

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
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            N_CTX=q.shape[2],
            KV_INDEX=kv_index,
            TOPK_PER_HEAD=topk_per_head,
            stride_kvz=kv_index.stride(0),
            stride_kvh=kv_index.stride(1),
            stride_kvm=kv_index.stride(2),
            stride_kvk=kv_index.stride(3),
            stride_topk_per_headz=topk_per_head.stride(0),
            stride_topk_per_headh=topk_per_head.stride(1),
            TOPK_INDEX_TO_PACKED_INDEX=topk_index_to_packed_index,
            stride_tipi_z=topk_index_to_packed_index.stride(0),
            stride_tipi_h=topk_index_to_packed_index.stride(1),
            stride_tipi_s=topk_index_to_packed_index.stride(2),
            MAX_ACTIVATED=max_activated,
            HEAD_DIM=HEAD_DIM_K,
            QUERY_GROUP_SIZE=query_group_size,
            STAGE=stage,
        )

        ctx.save_for_backward(
            q, k, v, o, M, kv_index, topk_per_head, topk_index_to_packed_index
        )
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.query_group_size = query_group_size

        return o.contiguous()

    @staticmethod
    def backward(ctx, do):
        (
            q,
            k,
            v,
            o,
            M,
            kv_index,
            topk_per_head,
            topk_index_to_packed_index,
        ) = ctx.saved_tensors

        max_activated = k.shape[2]

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        assert do.is_contiguous()
        assert q.stride() == o.stride() == do.stride()

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape[:4]
        PRE_BLOCK = 128

        if HEAD_DIM == 256:
            NUM_WARPS, NUM_STAGES = 4, 2
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 32, 128
        else:
            NUM_WARPS, NUM_STAGES = 4, 3
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 32, 256

        # Block size constraints for backward pass
        assert (
            ctx.query_group_size % BLOCK_M1 == 0
        ), f"QUERY_GROUP_SIZE ({ctx.query_group_size}) must be divisible by BLOCK_M1 ({BLOCK_M1})"
        assert (
            ctx.query_group_size % BLOCK_M2 == 0
        ), f"QUERY_GROUP_SIZE ({ctx.query_group_size}) must be divisible by BLOCK_M2 ({BLOCK_M2})"

        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k * (ctx.sm_scale * RCP_LN2)

        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        sparse_group_attn_bwd_delta_preprocess[pre_grid](
            o,
            do,
            delta,
            BATCH,
            N_HEAD,
            N_CTX,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        grid = (N_CTX // BLOCK_M1, 1, BATCH * N_HEAD)
        sparse_group_attn_bwd_dkdv[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            kv_index,
            topk_per_head,
            kv_index.stride(0),
            kv_index.stride(1),
            kv_index.stride(2),
            kv_index.stride(3),
            topk_per_head.stride(0),
            topk_per_head.stride(1),
            topk_index_to_packed_index,
            topk_index_to_packed_index.stride(0),
            topk_index_to_packed_index.stride(1),
            topk_index_to_packed_index.stride(2),
            max_activated,
            N_HEAD,
            N_CTX,
            QUERY_GROUP_SIZE=ctx.query_group_size,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        grid_dq_sparse = (N_CTX // BLOCK_M2, 1, BATCH * N_HEAD)
        sparse_group_attn_bwd_dq[grid_dq_sparse](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            kv_index,
            topk_per_head,
            kv_index.stride(0),
            kv_index.stride(1),
            kv_index.stride(2),
            kv_index.stride(3),
            topk_per_head.stride(0),
            topk_per_head.stride(1),
            topk_index_to_packed_index,
            topk_index_to_packed_index.stride(0),
            topk_index_to_packed_index.stride(1),
            topk_index_to_packed_index.stride(2),
            max_activated,
            N_HEAD,
            N_CTX,
            QUERY_GROUP_SIZE=ctx.query_group_size,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

        dq = dq.contiguous()
        dk = dk.contiguous()
        dv = dv.contiguous()

        return dq, dk, dv, None, None, None, None, None, None

# Aliases
_attention = CompressedSparseGroupAttention
attention = CompressedSparseGroupAttention.apply
compressed_sparse_group_attention = CompressedSparseGroupAttention.apply


if __name__ == "__main__":
    import torch
    from DSV.models.parallel.sparse_kv_gather import SparseKVGather

    torch.manual_seed(42)
    device = torch.device("cuda")
    sparse_kv_gather = SparseKVGather(0, None)

    # Test configuration
    head_num = 4
    world_size = 8
    batch_size = 1
    total_seq_len = 256000
    local_seq_len = total_seq_len // world_size
    group_size = 32
    group_num = local_seq_len // group_size
    head_dim = 128
    query_group_size = 32
    sparsity_per_head = [0.96, 0.95, 0.94, 0.92]
    top_k_per_head = [
        (int(total_seq_len * (1 - sparsity)) + 256 - 1) // 256 * 256
        for sparsity in sparsity_per_head
    ]

    print(f"Testing CompressedSparseGroupAttention")
    print(f"Config: B={batch_size}, H={head_num}, S={total_seq_len}, D={head_dim}")
    print(f"Query group size: {query_group_size}, Groups: {group_num}")
    print(f"Top-k per head: {top_k_per_head}")
    print(f"WARNING: Any 'operation scheduled before its operands' error messages can be safely ignored.")

    # Generate test data
    topk_indices = torch.full(
        (batch_size, head_num, group_num, max(top_k_per_head)),
        -1, device=device, dtype=torch.int32,
    )

    torch.manual_seed(123)
    for h in range(head_num):
        this_top_k = top_k_per_head[h]
        for g in range(group_num):
            random_perm = torch.randperm(total_seq_len, device=device)
            topk_indices[:, h, g, :this_top_k] = random_perm[:this_top_k]

    top_k_per_head_tensor = torch.tensor(
        top_k_per_head, device=device, dtype=torch.int32
    )

    # Generate K, V data
    K_full = torch.randn(
        batch_size,
        head_num,
        total_seq_len,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=False,
    )
    V_full = torch.randn(
        batch_size,
        head_num,
        total_seq_len,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=False,
    )

    # Generate query Q
    Q = torch.randn(
        batch_size,
        head_num,
        local_seq_len,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )

    print(f"Original K shape: {K_full.shape}")
    print(f"Original V shape: {V_full.shape}")
    print(f"Query Q shape: {Q.shape}")

    (
        activated_mask,
        topk_index_to_packed_index,
    ) = sparse_kv_gather.get_activated_indices_from_topk_indices(
        topk_indices, top_k_per_head_tensor, total_seq_len
    )

    max_activated = activated_mask.sum(dim=-1).max().item()

    # Compress K, V data
    K_compressed = torch.zeros(
        batch_size,
        head_num,
        max_activated,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=False,
    )
    V_compressed = torch.zeros(
        batch_size,
        head_num,
        max_activated,
        head_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=False,
    )

    for b in range(batch_size):
        for h in range(head_num):
            activated_positions = torch.where(activated_mask[b, h])[0]
            num_activated = len(activated_positions)

            if num_activated > 0:
                K_compressed[b, h, :num_activated] = K_full[b, h, activated_positions]
                V_compressed[b, h, :num_activated] = V_full[b, h, activated_positions]

    print(f"Compressed K shape: {K_compressed.shape}")
    print(f"Compressed V shape: {V_compressed.shape}")
    print(f"Actual max activated: {max_activated}")

    sm_scale = 1.0 / (head_dim**0.5)
    causal = False

    # Ensure contiguous tensors
    Q = Q.contiguous()
    K_compressed = K_compressed.contiguous()
    V_compressed = V_compressed.contiguous()
    topk_indices = topk_indices.contiguous()
    top_k_per_head_tensor = top_k_per_head_tensor.contiguous()
    topk_index_to_packed_index = topk_index_to_packed_index.contiguous()

    top_k_per_head_tensor = (
        torch.tensor(top_k_per_head, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    begin_event.record()

    output = CompressedSparseGroupAttention.apply(
        Q,  # [B, H, S, D] queries
        K_compressed,  # [B, H, max_activated, D] compressed keys
        V_compressed,  # [B, H, max_activated, D] compressed values
        causal,  # causal attention flag
        sm_scale,  # scaling factor
        topk_indices,  # [B, H, num_groups, topk] KV indices
        top_k_per_head_tensor,  # [B, H] top-k per head
        topk_index_to_packed_index,  # [B, H, total_seq_len] global to packed mapping
        query_group_size,  # query group size
    )

    end_event.record()
    torch.cuda.synchronize()
    forward_time = begin_event.elapsed_time(end_event)

    loss = output.sum()

    begin_event.record()

    # Backward pass
    loss.backward()

    end_event.record()
    torch.cuda.synchronize()
    backward_time = begin_event.elapsed_time(end_event)

    n_warmup = 10
    n_test = 100

    Q.grad = None
    K_compressed.grad = None
    V_compressed.grad = None

    # Warmup
    for _ in range(n_warmup):
        output = CompressedSparseGroupAttention.apply(
            Q,
            K_compressed,
            V_compressed,
            causal,
            sm_scale,
            topk_indices,
            top_k_per_head_tensor,
            topk_index_to_packed_index,
            query_group_size,
        )
        loss = output.sum()
        loss.backward()
        Q.grad = None
        K_compressed.grad = None
        V_compressed.grad = None

    # Forward performance test
    torch.cuda.synchronize()
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    begin_event.record()

    for _ in range(n_test):
        output = CompressedSparseGroupAttention.apply(
            Q,
            K_compressed,
            V_compressed,
            causal,
            sm_scale,
            topk_indices,
            top_k_per_head_tensor,
            topk_index_to_packed_index,
            query_group_size,
        )

    end_event.record()
    torch.cuda.synchronize()
    avg_fwd_time = begin_event.elapsed_time(end_event) / n_test
    print(f"Average forward time: {avg_fwd_time:.3f} ms")

    # Backward performance test
    grad = torch.randn_like(output)
    begin_event.record()
    for _ in range(n_test):
        output.backward(grad, retain_graph=True)
        Q.grad = None
        K_compressed.grad = None
        V_compressed.grad = None

    end_event.record()
    torch.cuda.synchronize()
    avg_bwd_time = begin_event.elapsed_time(end_event) / n_test
    print(f"Average backward time: {avg_bwd_time:.3f} ms")

    avg_time = avg_fwd_time + avg_bwd_time
    print(f"Average forward+backward time: {avg_time:.3f} ms")
