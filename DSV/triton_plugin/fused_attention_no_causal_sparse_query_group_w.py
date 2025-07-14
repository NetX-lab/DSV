

import torch
import triton
import triton.language as tl

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
def attn_fwd(
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
    GROUP_MASK,  
    UNIFIED_WINDOW_SIZE, 
    stride_kvz,
    stride_kvh,
    stride_kvm,
    stride_kvk,
    stride_maskz,
    stride_maskh,
    stride_maskm,
    stride_maskk,
    HEAD_DIM: tl.constexpr,  
    QUERY_GROUP_SIZE: tl.constexpr,  
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,  
    STAGE: tl.constexpr,  
):
   
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh


    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )
    
    groupmask_base_ptr = (
        GROUP_MASK + off_z.to(tl.int64) * stride_maskz + off_h.to(tl.int64) * stride_maskh
    )

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

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    lo, hi = 0, UNIFIED_WINDOW_SIZE
    hi = (UNIFIED_WINDOW_SIZE + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    head_dim_offset = tl.arange(0, HEAD_DIM)
    group_idx = (start_m * BLOCK_M) // QUERY_GROUP_SIZE

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        kv_index_offset = (
            kvindex_base_ptr + group_idx * stride_kvm + (start_n + offs_n) * stride_kvk
        )
        
        group_mask_offset = (
            groupmask_base_ptr + group_idx * stride_maskm + (start_n + offs_n) * stride_maskk
        )

        window_pos_mask = (start_n + offs_n) < UNIFIED_WINDOW_SIZE
        
        kv_index = tl.load(kv_index_offset, mask=window_pos_mask, other=0)
        group_mask = tl.load(group_mask_offset, mask=window_pos_mask, other=0)
        
        final_mask = window_pos_mask & (group_mask != 0)

        k = tl.load(
            k_base_ptr
            + head_dim_offset[:, None] * stride_kk
            + kv_index[None, :] * stride_kn,
            mask=final_mask[None, :],
            other=0.0
        )

        qk = tl.dot(q, k)
        
        qk = tl.where(final_mask[None, :], qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Load V
        v = tl.load(
            v_base_ptr
            + kv_index[:, None] * stride_vk
            + head_dim_offset[None, :] * stride_vn,
            mask=final_mask[:, None],
            other=0.0
        )

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    m_i += tl.math.log2(l_i)
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)


@triton.jit
def attn_bwd_delta_preprocess(
    O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
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
def attn_bwd_dkdv(
    Q, K, V, sm_scale, DO, DQ, DK, DV, M, D,
    stride_z, stride_h, stride_tok, stride_d,
    KV_INDEX, GROUP_MASK, UNIFIED_WINDOW_SIZE,
    stride_kvz, stride_kvh, stride_kvm, stride_kvk,
    stride_maskz, stride_maskh, stride_maskm, stride_maskk,
    H, N_CTX,
    QUERY_GROUP_SIZE: tl.constexpr, 
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    off_z = bhid // H
    off_h = bhid % H

    kvindex_base_ptr = (
        KV_INDEX + off_z.to(tl.int64) * stride_kvz + off_h.to(tl.int64) * stride_kvh
    )
    
    groupmask_base_ptr = (
        GROUP_MASK + off_z.to(tl.int64) * stride_maskz + off_h.to(tl.int64) * stride_maskh
    )

    Q += adj
    K += adj
    V += adj
    DO += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)
    num_steps = (UNIFIED_WINDOW_SIZE + BLOCK_N1 - 1) // BLOCK_N1
    start_m = pid * BLOCK_M1

    offset_m = start_m + tl.arange(0, BLOCK_M1)
    offset_k = tl.arange(0, HEAD_DIM)
    offset_n = tl.arange(0, BLOCK_N1)

    group_idx = start_m // QUERY_GROUP_SIZE
    kv_index_base_offset = kvindex_base_ptr + group_idx * stride_kvm
    group_mask_base_offset = groupmask_base_ptr + group_idx * stride_maskm

    qT_ptrs = Q + offset_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    qT = tl.load(qT_ptrs)

    do_ptrs = DO + offset_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    do = tl.load(do_ptrs)
    m = tl.load(M + offset_m)
    Di = tl.load(D + offset_m)

    offset_n_base = tl.arange(0, BLOCK_N1)

    for blk_idx in range(num_steps):
        current_offset_n = blk_idx * BLOCK_N1 + offset_n_base
        kv_index_offset = kv_index_base_offset + current_offset_n * stride_kvk
        group_mask_offset = group_mask_base_offset + current_offset_n * stride_maskk

        window_pos_mask = current_offset_n < UNIFIED_WINDOW_SIZE
        kv_index = tl.load(kv_index_offset, mask=window_pos_mask, other=0)
        group_mask_raw = tl.load(group_mask_offset, mask=window_pos_mask, other=0)
        
        # Convert group_mask to boolean type explicitly
        group_mask = group_mask_raw != 0
        
        final_mask = window_pos_mask & group_mask

        k = tl.load(
            K + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d,
            mask=final_mask[:, None],
            other=0.0
        )

        v = tl.load(
            V + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d,
            mask=final_mask[:, None],
            other=0.0
        )

        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])

        ppT = pT.to(do.dtype)
        dv = tl.dot(ppT, do)

        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk = tl.dot(dsT, tl.trans(qT))

        dv_final = dv.to(tl.float32)
        dk_final = (dk * sm_scale).to(tl.float32)

        # Apply final_mask when storing gradients
        dv_ptrs = DV + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d
        dk_ptrs = DK + kv_index[:, None] * stride_tok + offset_k[None, :] * stride_d

        # Only accumulate gradients for valid positions
        tl.atomic_add(dv_ptrs, tl.where(final_mask[:, None], dv_final, 0.0), sem="relaxed")
        tl.atomic_add(dk_ptrs, tl.where(final_mask[:, None], dk_final, 0.0), sem="relaxed")


@triton.jit
def attn_bwd_dq(
    Q, K, V, sm_scale, DO, DQ, M, D,
    stride_z, stride_h, stride_tok, stride_d,
    KV_INDEX, GROUP_MASK, UNIFIED_WINDOW_SIZE,
    stride_kvz, stride_kvh, stride_kvm, stride_kvk,
    stride_maskz, stride_maskh, stride_maskm, stride_maskk,
    H, N_CTX,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
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
    
    groupmask_base_ptr = (
        GROUP_MASK + off_z.to(tl.int64) * stride_maskz + off_h.to(tl.int64) * stride_maskh
    )

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
    group_mask_base_offset = groupmask_base_ptr + group_idx * stride_maskm

    num_steps = (UNIFIED_WINDOW_SIZE + BLOCK_N2 - 1) // BLOCK_N2
    offs_n_base = tl.arange(0, BLOCK_N2)

    for blk_idx in range(num_steps):
        current_offs_n = blk_idx * BLOCK_N2 + offs_n_base
        kv_index_offset = kv_index_base_offset + current_offs_n * stride_kvk
        group_mask_offset = group_mask_base_offset + current_offs_n * stride_maskk

        window_pos_mask = current_offs_n < UNIFIED_WINDOW_SIZE
        kv_index = tl.load(kv_index_offset, mask=window_pos_mask, other=0)
        group_mask_raw = tl.load(group_mask_offset, mask=window_pos_mask, other=0)
        
        # Convert group_mask to boolean type explicitly  
        group_mask = group_mask_raw != 0
        
        final_mask = window_pos_mask & group_mask

        kT = tl.load(
            K + offs_k[:, None] * stride_d + kv_index[None, :] * stride_tok,
            mask=final_mask[None, :],
            other=0.0
        )

        vT = tl.load(
            V + kv_index[None, :] * stride_tok + offs_k[:, None] * stride_d,
            mask=final_mask[None, :],
            other=0.0
        )

        qk = tl.dot(q, kT)
        qk = tl.where(final_mask[None, :], qk, float("-inf"))
        p = tl.math.exp2(qk - m)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(kT.dtype)

        dq += tl.dot(ds, tl.trans(kT))

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class WindowAttn(torch.autograd.Function):
    """Attn Implementation"""

    @staticmethod
    def forward(
        ctx, q, k, v, causal, sm_scale, kv_index, group_mask, unified_window_size, query_group_size
    ):

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        BLOCK_M = 32
        assert query_group_size % BLOCK_M == 0, (
            f"QUERY_GROUP_SIZE ({query_group_size}) must be divisible by BLOCK_M ({BLOCK_M})"
        )

        o = torch.empty_like(q)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        kv_index = kv_index.contiguous()
        group_mask = group_mask.contiguous()

        stage = 1


        grid = lambda args: (
            triton.cdiv(q.shape[2], args["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            N_CTX=q.shape[2],
            KV_INDEX=kv_index,
            GROUP_MASK=group_mask,
            UNIFIED_WINDOW_SIZE=unified_window_size,
            stride_kvz=kv_index.stride(0),
            stride_kvh=kv_index.stride(1),
            stride_kvm=kv_index.stride(2),
            stride_kvk=kv_index.stride(3),
            stride_maskz=group_mask.stride(0),
            stride_maskh=group_mask.stride(1),
            stride_maskm=group_mask.stride(2),
            stride_maskk=group_mask.stride(3),
            HEAD_DIM=HEAD_DIM_K,
            QUERY_GROUP_SIZE=query_group_size,
            STAGE=stage,
        )

        ctx.save_for_backward(q, k, v, o, M, kv_index, group_mask)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.query_group_size = query_group_size
        ctx.unified_window_size = unified_window_size

        return o.contiguous()

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, kv_index, group_mask = ctx.saved_tensors

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        kv_index = kv_index.contiguous()
        group_mask = group_mask.contiguous()

        assert do.is_contiguous()
        assert q.stride() == o.stride() == do.stride()
        assert k.stride() == v.stride()

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
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 32, 128

        assert ctx.query_group_size % BLOCK_M1 == 0
        assert ctx.query_group_size % BLOCK_M2 == 0

        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        attn_bwd_delta_preprocess[pre_grid](
            o, do, delta,
            BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        #print(f"q shape: {q.shape}, arg_k shape: {arg_k.shape}, v shape: {v.shape}, kv index shape: {kv_index.shape}, group mask shape: {group_mask.shape}")

        grid = (N_CTX // BLOCK_M1, 1, BATCH * N_HEAD)
        attn_bwd_dkdv[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            kv_index, group_mask, ctx.unified_window_size,
            kv_index.stride(0), kv_index.stride(1),
            kv_index.stride(2), kv_index.stride(3),
            group_mask.stride(0), group_mask.stride(1),
            group_mask.stride(2), group_mask.stride(3),
            N_HEAD, N_CTX,
            QUERY_GROUP_SIZE=ctx.query_group_size,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        grid_dq_window = (N_CTX // BLOCK_M2, 1, BATCH * N_HEAD)
        attn_bwd_dq[grid_dq_window](
            q, arg_k, v, ctx.sm_scale, do, dq,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            kv_index, group_mask, ctx.unified_window_size,
            kv_index.stride(0), kv_index.stride(1),
            kv_index.stride(2), kv_index.stride(3),
            group_mask.stride(0), group_mask.stride(1),
            group_mask.stride(2), group_mask.stride(3),
            N_HEAD, N_CTX,
            QUERY_GROUP_SIZE=ctx.query_group_size,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

        dq = dq.contiguous()
        dk = dk.contiguous()
        dv = dv.contiguous()

        return dq, dk, dv, None, None, None, None, None, None




_attention = WindowAttn 
attention = WindowAttn.apply 
sparse_window_attention = WindowAttn.apply
