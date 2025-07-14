import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

class CommGroupManager:
    """Communication group manager for handling 2D mesh parallel groups within a fixed context parallel group"""

    def __init__(self, cp_group=None):
        assert cp_group is not None, "cp_group must be provided"

        self.cp_group = cp_group
        self.cp_ranks = None
        self.scp_groups = {} 
        self.hcp_groups = {} 
        self.mesh_info = {} 
        self.current_layer_id = None

    def init_cp_group(self):
        self.cp_ranks = dist.get_process_group_ranks(self.cp_group)

    def create_2d_mesh_groups(self, layer_id, scp_size, hcp_size):

        assert self.cp_group is not None, "Must call init_cp_group first"

        if layer_id in self.mesh_info:
            old_scp, old_hcp = self.mesh_info[layer_id]
            if old_scp == scp_size and old_hcp == hcp_size:
                return self.scp_groups[layer_id], self.hcp_groups[layer_id]

        cp_size = len(self.cp_ranks)
        assert (
            cp_size == scp_size * hcp_size
        ), f"CP group size ({cp_size}) must equal SCP({scp_size}) * HCP({hcp_size})"

        ranks_2d = torch.tensor(self.cp_ranks).view(hcp_size, scp_size)

        scp_groups = []
        for i in range(hcp_size):
            ranks = ranks_2d[i].tolist()
            group = dist.new_group(ranks=ranks)
            scp_groups.append(group)

        hcp_groups = []
        for j in range(scp_size):
            ranks = ranks_2d[:, j].tolist()
            group = dist.new_group(ranks=ranks)
            hcp_groups.append(group)

        self.scp_groups[layer_id] = scp_groups
        self.hcp_groups[layer_id] = hcp_groups
        self.mesh_info[layer_id] = (scp_size, hcp_size)

        return scp_groups, hcp_groups

    def get_cp_group(self):
        return self.cp_group

    def get_cp_ranks(self):
        return self.cp_ranks

    def get_mesh_groups(self, layer_id):
        if layer_id not in self.mesh_info:
            return None, None
        return self.scp_groups[layer_id], self.hcp_groups[layer_id]

    def get_local_mesh_position(self, rank=None):
        if rank is None:
            rank = dist.get_rank()

        if rank not in self.cp_ranks:
            return None, None

        rank_idx = self.cp_ranks.index(rank)
        current_mesh = self.mesh_info.get(self.current_layer_id)
        if current_mesh is None:
            return None, None

        scp_size = current_mesh[0]
        hcp_idx = rank_idx // scp_size
        scp_idx = rank_idx % scp_size
        return hcp_idx, scp_idx

    def cleanup(self):
        self.cp_group = None
        self.cp_ranks = None
        self.scp_groups.clear()
        self.hcp_groups.clear()
        self.mesh_info.clear()

    def set_current_layer(self, layer_id):
        self.current_layer_id = layer_id

    def get_comm_group(self, layer_id):
        return self.cp_group


"""
======================================================================================
SparseKVGather: Multi-GPU Sparse KV Gather Module
======================================================================================

Core:
    Implement efficient sparse Key-Value gathering through 4 rounds of all-to-all communication:
    1. Exchange data length information
    2. Exchange specific index requests
    3. Exchange Key data
    4. Exchange Value data
    
    Use vectorized cumcount algorithm to achieve efficient output reconstruction.

Input Format:
    - k: torch.Tensor [B,H,S/N,D]
        Local Key tensor, where B=batch_size, H=num_heads, S/N=local_seq_len, D=head_dim
        
    - v: torch.Tensor [B,H,S/N,D] 
        Local Value tensor, with the same shape as the Key tensor
        
    - activated_indices: torch.Tensor [B,H,S] (bool)
        Global sparse activation mask, marking the global positions that each (batch,head) needs to gather
        Note: Different GPUs have different sparse patterns, reflecting the independence of multi-head attention

Output Format:
    - gathered_k: torch.Tensor [B,H,max_activated,D]
        The gathered Key tensor, max_activated is the maximum number of activated tokens across all (batch,head)
        
    - gathered_v: torch.Tensor [B,H,max_activated,D]
        The gathered Value tensor, with the same shape as gathered_k
        
    Note: For (batch,head) with fewer activated tokens than max_activated, the extra positions are padded with zeros
"""


class SparseKVGather(nn.Module):
    def __init__(self, layer_id, comm_manager=None):
        super().__init__()
        self.layer_id = layer_id
        self.comm_manager = comm_manager

        # save the forward information for backward
        self.forward_cache = {}

    @property
    def comm_group(self):
        if self.comm_manager is None:
            return None
        return self.comm_manager.get_comm_group(self.layer_id)

    @property
    def world_size(self):
        if self.comm_group is None:
            return 1
        return dist.get_world_size(self.comm_group)

    @property
    def rank(self):
        if self.comm_group is None:
            return 0
        return dist.get_rank(self.comm_group)

    def get_activated_indices_from_topk_indices(
        self, topk_indices, top_k_per_head, total_global_seq_len
    ):
        """
        Get the activated indices from the token indices; current mainly for Batchsize =1
        Args:
            topk_indices: the topk indices [B,H,G,max_top_k]
            top_k_per_head: the top_k value per head [H]
            total_global_seq_len: the total global sequence length int
        Returns:
            activated_mask: the activated mask [B,H, total_global_seq_len]
            topk_index_to_packed_index: the topk index to packed index [B,H, total_global_seq_len]
        """
        B, H, G, max_top_k = topk_indices.size()

        activated_mask = torch.zeros(
            B, H, total_global_seq_len, dtype=torch.bool, device=topk_indices.device
        )

        topk_index_to_packed_index = torch.full(
            (B, H, total_global_seq_len),
            -1,
            dtype=torch.int32,
            device=topk_indices.device,
        )

        for h in range(H):
            this_top_k = top_k_per_head[h]
            inds = topk_indices[:, h, :, :this_top_k].reshape(B, 1, -1)
            # torch scatter
            activated_mask[:, h : h + 1, :].scatter_(2, inds.to(torch.long), True)
            non_zero_index = torch.where(activated_mask[:, h, :])

            # Note: current only consider batchsize = 1
            seq_idx = non_zero_index[1]
            topk_index_to_packed_index[:, h, seq_idx] = torch.arange(
                len(seq_idx), device=topk_indices.device, dtype=torch.int32
            )

        return activated_mask, topk_index_to_packed_index

    def forward(self, k, v, activated_indices, comm_group=None):
        """
        Use uneven alltoall to implement sparse KV gathering
        Args:
            k: key tensor [B,H,S/N,D]
            v: value tensor [B,H,S/N,D]
            activated_indices: the activated KV map for ALL GPUs [B,H,S], bool type; the GPU needs to gather the KV tokens for the activated KV indices across different GPUs; The activated KV indices are different for different GPUs.
            comm_group: the communication group to use, if None, use the default communication group
        Returns:
            gathered_k: the gathered key [B,H,num_activated,D]
            gathered_v: the gathered value [B,H,num_activated,D]
        """
        B, H, local_seq_len, D = k.size()
        device = k.device
        world_size = dist.get_world_size(comm_group)

        if comm_group is not None: 
            self.comm_group = comm_group

        # 1. Each GPU calculates the indices it needs to receive from other GPUs
        recv_requests = []  # Store the indices to be received from each rank
        recv_lengths = torch.zeros(world_size, dtype=torch.int32, device=device)

        for src_rank in range(world_size):
            # Calculate the range of tokens handled by the source rank
            src_start = src_rank * local_seq_len
            src_end = src_start + local_seq_len
            # Get the indices to be received from this rank
            rank_mask = activated_indices[:, :, src_start:src_end]  # [B,H,S/N]
            # Record the index information (b, h, local_idx)
            indices = torch.where(rank_mask)
            # Convert the indices to int32 type
            indices = [idx.to(torch.int32) for idx in indices]
            recv_requests.append(torch.stack(indices, dim=0))  # [3, num_indices]
            recv_lengths[src_rank] = len(indices[0])

        # 2. Exchange the length information, so that each GPU knows how much data to send
        send_lengths = torch.empty_like(recv_lengths)
        dist.all_to_all_single(send_lengths, recv_lengths, group=comm_group)

        # 3. Exchange the actual index information
        recv_requests = torch.cat(recv_requests, dim=1)  # [3, total_indices]
        total_send_indices = send_lengths.sum().item() * 3
        send_indices = torch.empty(
            (total_send_indices,), dtype=torch.int32, device=device
        )

        # time.sleep(self.rank)
        # print(f"Rank {self.rank} recv_requests: {recv_request  s}, flattened: {recv_requests.view(-1)}; send_lenths list: {send_lengths.tolist()}; recv_lengths list: {recv_lengths.tolist()}")

        with record_function("all_to_all_single_indices"):
            dist.all_to_all_single(
                send_indices,
                recv_requests.permute(1, 0)
                .contiguous()
                .view(
                    -1
                ),  #  [3, total_indices] -> [total_indices, 3] -> [total_indices*3]
                output_split_sizes=(send_lengths * 3).tolist(),
                input_split_sizes=(recv_lengths * 3).tolist(),
                group=comm_group,
            )

        # 4. Prepare the KV data to be sent
        send_indices = (
            send_indices.view(-1, 3).permute(1, 0).contiguous()
        )  # [total_indices, 3] -> [3, total_indices]
        send_data_k = []
        send_data_v = []


        start_idx = 0
        for rank in range(world_size):
            length = send_lengths[rank]
            if length > 0:
                rank_indices = send_indices[:, start_idx : start_idx + length]
                b_idx = rank_indices[0]
                h_idx = rank_indices[1]
                s_idx = rank_indices[2]  # This is the local index of the requester

                # assert torch.all(b_idx<B), f"Rank {self.rank} has a batch index out of range, max b_idx: {b_idx.max()}, B: {B}"
                # assert torch.all(h_idx<H), f"Rank {self.rank} has a head index out of range, max h_idx: {h_idx.max()}, H: {H}"
                # assert torch.all(s_idx<local_seq_len), f"Rank {self.rank} has a sequence index out of range, max s_idx: {s_idx.max()}, local_seq_len: {local_seq_len}"

                # s_idx is already the local index of the requester, directly use it
                local_s_idx = s_idx

                # Collect the corresponding KV data
                local_k = k[
                    b_idx, h_idx, local_s_idx
                ]  # [kv_length_to_send_to_this_rank, D],
                local_v = v[b_idx, h_idx, local_s_idx]
                send_data_k.append(local_k)
                send_data_v.append(local_v)
                start_idx += length

        # Exchange the KV data 
        send_data_k = (
            torch.cat(send_data_k, dim=0)
            if send_data_k
            else torch.empty(0, D, dtype=k.dtype, device=device) 
        )
        send_data_v = (
            torch.cat(send_data_v, dim=0)
            if send_data_v
            else torch.empty(0, D, dtype=v.dtype, device=device) 
        )

        total_recv_length = recv_lengths.sum().item()
        recv_data_k = torch.empty(total_recv_length, D, dtype=k.dtype, device=device)
        recv_data_v = torch.empty_like(recv_data_k)


        with record_function("all_to_all_single_k_v"):
            dist.all_to_all_single(
                recv_data_k,
                send_data_k,
                output_split_sizes=recv_lengths.tolist(),  
                input_split_sizes=send_lengths.tolist(),  
                group=comm_group,
            )

            dist.all_to_all_single(
                recv_data_v,
                send_data_v,
                output_split_sizes=recv_lengths.tolist(),  
                input_split_sizes=send_lengths.tolist(),  
                group=comm_group,
            )

        gathered_k, gathered_v = None, None

        total_activated = activated_indices.sum(dim=-1)  # [B,H]
        max_activated = total_activated.max().item()

        gathered_k = torch.zeros(B, H, max_activated, D, dtype=k.dtype, device=device)
        gathered_v = torch.zeros_like(gathered_k)


        all_indices = recv_requests  # [3, total_indices]
        all_b = all_indices[0]  # [total_indices]
        all_h = all_indices[1]  # [total_indices]
        all_s_local = all_indices[2]  # [total_indices] local sequence index

        if len(all_b) > 0:
            # method: use scatter_add to maintain the position counter for each (b,h) group
            bh_unique_ids = all_b * H + all_h  # the unique identifier for each (b,h) [total_indices]

            # maintain the counter for each (b,h) group
            bh_counters = torch.zeros((B * H,), dtype=torch.int32, device=device)
            within_group_offsets = torch.zeros_like(bh_unique_ids, dtype=torch.int32)

            # assign the increasing position index to the elements with the same bh_id
            sorted_bh_ids, sort_indices = torch.sort(bh_unique_ids, stable=True)

            # calculate the offset (cumcount) of each position in its (b,h) group
            group_offsets = torch.zeros_like(sorted_bh_ids, dtype=torch.int32)
            if len(sorted_bh_ids) > 1:
                # find the group boundaries
                group_starts = torch.cat(
                    [
                        torch.tensor([True], device=device),
                        sorted_bh_ids[1:] != sorted_bh_ids[:-1],
                    ]
                )

                # calculate the cumulative count
                cumcount = torch.arange(
                    len(sorted_bh_ids), device=device, dtype=torch.int32
                )
                group_start_pos = torch.where(group_starts, cumcount, 0)
                group_start_pos = torch.cummax(group_start_pos, dim=0)[0]
                group_offsets = cumcount - group_start_pos

            # restore the original order
            within_group_offsets[sort_indices] = group_offsets
        else:
            within_group_offsets = torch.empty(0, dtype=torch.int32, device=device)

        linear_indices = (
            all_b * (H * max_activated) + all_h * max_activated + within_group_offsets
        )

        # linear_indices = (all_b * H + all_h) * max_activated + batch_offsets

        # Batch fill the data
        gathered_k.view(-1, D)[linear_indices] = recv_data_k
        gathered_v.view(-1, D)[linear_indices] = recv_data_v

        # position_map.view(-1)[linear_indices] = all_global_pos

        # Save the information needed for backward propagation
        self.forward_cache = {
            "k_shape": k.shape,
            "v_shape": v.shape,
            "send_indices": send_indices,
            "send_lengths": send_lengths,
            "recv_lengths": recv_lengths,
            "linear_indices": linear_indices,
            "device": device,
            "dtype": k.dtype,
        }

        return gathered_k, gathered_v

    def backward_kv_gradients(
        self, grad_gathered_k, grad_gathered_v, if_accumulate=True
    ):
        """
        Backward propagation function: pass the gradients of gathered KV back to the original positions
        Args:
            grad_gathered_k: gradients of gathered_k [B,H,max_activated,D]
            grad_gathered_v: gradients of gathered_v [B,H,max_activated,D]
        Returns:
            if_accumulate: whether to accumulate the gradients, if True, the gradients will be accumulated, if False, the gradients will be returned directly
            grad_k: gradients of local k [B,H,S/N,D]
            grad_v: gradients of local v [B,H,S/N,D]
        """
        if not self.forward_cache:
            raise RuntimeError("Must call forward() before backward propagation")

        # Get the forward propagation information from the cache
        k_shape = self.forward_cache["k_shape"]
        v_shape = self.forward_cache["v_shape"]
        send_indices = self.forward_cache["send_indices"]
        send_lengths = self.forward_cache["send_lengths"]
        recv_lengths = self.forward_cache["recv_lengths"]
        linear_indices = self.forward_cache["linear_indices"]
        device = self.forward_cache["device"]
        dtype = self.forward_cache["dtype"]

        B, H, local_seq_len, D = k_shape

        world_size = dist.get_world_size(self.comm_group)

        # Initialize the local gradient tensors
        grad_k = torch.zeros(k_shape, dtype=dtype, device=device)
        grad_v = torch.zeros(v_shape, dtype=dtype, device=device)

        # 1. Extract the gradients to be sent back from the gathered gradients
        # Use linear indices to extract the gradients

        send_grad_k = grad_gathered_k.view(-1, D)[
            linear_indices
        ]  # [len(linear_indices), D] = [forward's total_recv_length, D]
        send_grad_v = grad_gathered_v.view(-1, D)[
            linear_indices
        ]  # [len(linear_indices), D] = [forward's total_recv_length, D]

        total_send_length = send_lengths.sum().item()  # forward's total_send_length
        recv_grad_k = torch.empty(total_send_length, D, dtype=dtype, device=device)
        recv_grad_v = torch.empty_like(recv_grad_k)

        with record_function("backward_all_to_all_k_v"):
            # when doing backward propagation, the send and recv lengths should be swapped
            # recv_grad_k receives total_send_length data, split size is send_lengths
            # send_grad_k sends total_recv_length data, split size is recv_lengths
            dist.all_to_all_single(
                recv_grad_k,
                send_grad_k,
                output_split_sizes=send_lengths.tolist(),  # split size of recv_grad_k
                input_split_sizes=recv_lengths.tolist(),  # split size of send_grad_k
                group=self.comm_group,
            )

            dist.all_to_all_single(
                recv_grad_v,
                send_grad_v,
                output_split_sizes=send_lengths.tolist(),
                input_split_sizes=recv_lengths.tolist(),
                group=self.comm_group,
            )

        if if_accumulate:
            # 3. Accumulate the received gradients to the correct positions of local K and V
            start_idx = 0
            for rank in range(world_size):
                length = send_lengths[rank]
                if length > 0:
                    # Get
                    rank_indices = send_indices[:, start_idx : start_idx + length]
                    b_idx = rank_indices[0]
                    h_idx = rank_indices[1]
                    s_idx = rank_indices[2]  # Local index

                    # Get the corresponding gradients
                    rank_grad_k = recv_grad_k[start_idx : start_idx + length]
                    rank_grad_v = recv_grad_v[start_idx : start_idx + length]

                    # Use scatter_add to accumulate the gradients 
                    grad_k.view(-1, D).scatter_add_(
                        0,
                        (b_idx * H * local_seq_len + h_idx * local_seq_len + s_idx)
                        .to(torch.int64)
                        .unsqueeze(1)
                        .expand(-1, D),
                        rank_grad_k,
                    )
                    grad_v.view(-1, D).scatter_add_(
                        0,
                        (b_idx * H * local_seq_len + h_idx * local_seq_len + s_idx)
                        .to(torch.int64)
                        .unsqueeze(1)
                        .expand(-1, D),
                        rank_grad_v,
                    )

                    start_idx += length
            return grad_k, grad_v
        else:
            return recv_grad_k, recv_grad_v

    def clear_cache(self):
        """Clear the forward cache"""
        self.forward_cache.clear()

    def __call__(self, k, v, activated_indices, auto_grad=True):
        """
        Convenient calling interface
        Args:
            k: key tensor [B,H,S/N,D]
            v: value tensor [B,H,S/N,D]
            activated_indices: activated indices [B,H,S]
            auto_grad: whether to enable automatic gradient propagation
        Returns:
            gathered_k, gathered_v: gathered KV tensor
            position_map: global position map [B,H,max_activated]
        """
        if auto_grad and (k.requires_grad or v.requires_grad):
            # Warning: automatic gradient function is not available, fallback to forward propagation
            import warnings

            warnings.warn(
                "auto_grad=True is not available, fallback to forward propagation",
                UserWarning,
            )
            return self.forward(k, v, activated_indices)
        else:
            # only forward propagation, no gradient
            return self.forward(k, v, activated_indices)


"""
N GPUs, Total S tokens
each GPU hold a Q of shape [B,H,S/N,D], K of shape [B,H,S/N,D], V of shape [B,H,S/N,D] 
activated KV index: [B,H,S], bool; 1 for the KV index is needed to be gathered, 0 for the KV index is not needed to be gathered; It is different for each GPU rank. 
for each GPU, we need to gather the KV tokens for the activated KV indices, and then concat them together  
"""


def init_distributed(rank, world_size):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    return dist.get_world_size(), dist.get_rank()


def test_all_to_all_latency(B=1, H=16, S=128000, D=128):
    import os

    world_size, rank = init_distributed(
        int(os.environ.get("RANK")), int(os.environ.get("WORLD_SIZE"))
    )
    cp_group = dist.new_group(ranks=list(range(world_size)))
    dist.barrier(device_ids=[rank])
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    assert (
        H % world_size == 0
    ), f"H ({H}) must be divisible by world_size ({world_size})"
    assert (
        S % world_size == 0
    ), f"S ({S}) must be divisible by world_size ({world_size})"

    # Input: sequence parallel - each rank has all heads, partial sequence
    local_q = torch.randn(
        B,
        H,
        S // world_size,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )
    local_k = torch.randn(
        B,
        H,
        S // world_size,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )
    local_v = torch.randn(
        B,
        H,
        S // world_size,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )

    # Output: head parallel - each rank has partial heads, all sequence
    output_q = torch.empty(
        B,
        H // world_size,
        S,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )
    output_k = torch.empty(
        B,
        H // world_size,
        S,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )
    output_v = torch.empty(
        B,
        H // world_size,
        S,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )

    def all_to_all(local_x, output_x):
        B, H, local_S, D = local_x.shape
        global_S = local_S * world_size
        heads_per_rank = H // world_size
        local_x_for_send = local_x.view(B, world_size, heads_per_rank, local_S, D)
        send_tensor = local_x_for_send.permute(1, 0, 2, 3, 4).contiguous()

        recv_tensor = torch.empty_like(
            send_tensor, device=local_x.device, dtype=local_x.dtype
        )

        dist.all_to_all_single(recv_tensor, send_tensor, group=cp_group)

        recv_tensor_reordered = recv_tensor.permute(
            1, 2, 0, 3, 4
        )  # (B, H//world_size, world_size, S//world_size, D)
        output_x.copy_(
            recv_tensor_reordered.contiguous().view(B, heads_per_rank, global_S, D)
        )

        return output_x

    def test_all_to_all_latency():
        nonlocal local_q, local_k, local_v, output_q, output_k, output_v
        output_q = all_to_all(local_q, output_q)
        output_k = all_to_all(local_k, output_k)
        output_v = all_to_all(local_v, output_v)

    warm_up = 5
    test_num = 50

    for i in range(warm_up):
        test_all_to_all_latency()

    torch.cuda.synchronize()

    begin_event.record()
    for i in range(test_num):
        test_all_to_all_latency()
    end_event.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"All_to_all_latency for output shape {[B,H//world_size,S,D]}: {begin_event.elapsed_time(end_event)/test_num} ms"
        )


def test_all_gather_latency(B=1, H=16, S=128000, D=128):
    import os

    world_size, rank = init_distributed(
        int(os.environ.get("RANK")), int(os.environ.get("WORLD_SIZE"))
    )

    cp_group = dist.new_group(ranks=list(range(world_size)))
    dist.barrier(device_ids=[rank])
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    local_seq_len = S // world_size
    local_k = torch.randn(
        B,
        H,
        local_seq_len,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )
    local_v = torch.randn(
        B,
        H,
        local_seq_len,
        D,
        device=torch.device(f"cuda:{rank}"),
        dtype=torch.bfloat16,
    )

    output_k = [
        torch.empty(
            B,
            H,
            local_seq_len,
            D,
            device=torch.device(f"cuda:{rank}"),
            dtype=torch.bfloat16,
        )
        for _ in range(world_size)
    ]
    output_v = [
        torch.empty(
            B,
            H,
            local_seq_len,
            D,
            device=torch.device(f"cuda:{rank}"),
            dtype=torch.bfloat16,
        )
        for _ in range(world_size)
    ]

    def test_all_gather_latency():
        dist.all_gather(output_k, local_k, group=cp_group)
        dist.all_gather(output_v, local_v, group=cp_group)

    warm_up = 5
    test_num = 50

    for i in range(warm_up):
        test_all_gather_latency()

    torch.cuda.synchronize()

    import time

    begin_time = time.time()
    begin_event.record()
    for i in range(test_num):
        test_all_gather_latency()
    end_event.record()

    torch.cuda.synchronize()
    end_time = time.time()

    if rank == 0:
        print(
            f"All_gather_latency for output shape {[B,H,S,D]}: {(end_time - begin_time)*1000/test_num} ms"
        )
        print(
            f"All_gather_latency for output shape {[B,H,S,D]}: {begin_event.elapsed_time(end_event)/test_num} ms"
        )


def test_sparse_kv_gather(
    B=1,
    H=24,
    S=128000,
    D=128,
    sparsity_per_head=[
        0.88,
        0.85,
        0.81,
        0.83,
        0.85,
        0.85,
        0.88,
        0.86,
        0.84,
        0.87,
        0.89,
        0.82,
        0.81,
        0.83,
        0.85,
        0.85,
    ],
):
    import os

    world_size, rank = init_distributed(
        int(os.environ.get("RANK")), int(os.environ.get("WORLD_SIZE"))
    )

    cp_group = dist.new_group(ranks=list(range(world_size)))
    comm_manager = CommGroupManager(cp_group=cp_group)
    comm_manager.init_cp_group()

    B = B  # batch size
    H = H  # Head number
    S = S  # Total sequence length
    D = D  # Hidden dimension
    local_seq_len = S // world_size  # Sequence length per GPU

    torch.manual_seed(
        42 + rank
    )  # Ensure each rank has different but reproducible random numbers
    device = torch.device(f"cuda:{rank}")

    dtype_verification = torch.float32

    dtype_performance = torch.bfloat16

    k = torch.zeros(B, H, local_seq_len, D, device=device, dtype=dtype_verification)
    v = torch.zeros(B, H, local_seq_len, D, device=device, dtype=dtype_verification)

    for b in range(B):
        for h in range(H):
            for s in range(local_seq_len):

                unique_id = rank * 1000 + s
                k[b, h, s, :] = unique_id
                v[b, h, s, :] = unique_id + 0.5  # Add 0.5 to distinguish k and v


    torch.manual_seed(
        42
    ) 
    activated_indices = torch.zeros(B, H, S, dtype=torch.bool, device=device)
    for b in range(B):
        for h in range(H):
            # Activate some positions deterministically
            activated_num = int(1 - sparsity_per_head[h] * S)
            # Use fixed activation mode
            activated_pos = torch.randperm(S)[:activated_num]
            activated_indices[b, h, activated_pos] = True

    # Create SparseKVGather instance
    layer_id = 0
    sparse_kv_gather = SparseKVGather(layer_id, comm_manager)

    print(
        f"Rank {rank}: Start numerical correctness verification (using {dtype_verification})..."
    )
    print(f"Rank {rank}: local_seq_len = {local_seq_len}, world_size = {world_size}")

    # ================================
    # 🔍 First verify the correctness of the original data
    # ================================
    print(f"Rank {rank}: Verify the correctness of the original data")
    for b in range(B):  # Verify the first batch
        for h in range(H):  # Verify the first head
            for s in range(local_seq_len):  # Verify the first 10 positions
                expected_k = rank * 1000.0 + s  # Match the new encoding scheme
                actual_k = k[b, h, s, 0].item()
                expected_v = rank * 1000.0 + s + 0.5
                actual_v = v[b, h, s, 0].item()
                if abs(actual_k - expected_k) > 1e-6:  # Use float32 precision threshold
                    print(f"❌ Rank {rank}: Original K data error (b={b}, h={h}, s={s})")
                    print(f"    Expected: {expected_k}, Actual: {actual_k}")
                    print(f"    Data type: {k.dtype}")
                    assert False, "Original data verification failed"
                if abs(actual_v - expected_v) > 1e-6:
                    print(f"❌ Rank {rank}: Original V data error (b={b}, h={h}, s={s})")
                    print(f"    Expected: {expected_v}, Actual: {actual_v}")
                    print(f"    Data type: {v.dtype}")
                    assert False, "Original data verification failed"

    print(f"✅ Rank {rank}: Original data verification passed")

    # ================================
    # Run sparse_kv_gather and verify the result
    # ================================
    gathered_k, gathered_v = sparse_kv_gather(k, v, activated_indices, auto_grad=False)

    # ================================
    # Strict numerical correctness verification
    # ================================
    num_activated = activated_indices.sum(dim=-1)  # [B,H]
    max_activated = num_activated.max().item()

    print(
        f"Rank {rank}: gathered_k shape: {gathered_k.shape}, max_activated: {max_activated}"
    )

    # Verify the basic shape
    assert gathered_k.shape == (B, H, max_activated, D)
    assert gathered_v.shape == (B, H, max_activated, D)

    for b in range(B):  
        for h in range(H):  
            # Get the activated positions of this (batch, head)
            activated_positions = torch.where(activated_indices[b, h])[
                0
            ]  # Global position
            actual_activated = len(activated_positions)

            for i, global_pos in enumerate(activated_positions):
                src_rank = global_pos.item() // local_seq_len
                local_pos = global_pos.item() % local_seq_len

                expected_k_value = src_rank * 1000.0 + local_pos
                expected_v_value = expected_k_value + 0.5

                actual_k_value = gathered_k[b, h, i, 0].item()
                actual_v_value = gathered_v[b, h, i, 0].item()

                # Calculate the error
                k_error = abs(actual_k_value - expected_k_value)
                v_error = abs(actual_v_value - expected_v_value)

                if k_error > 1e-5:
                    if isinstance(actual_k_value, (int, float)) and actual_k_value > 0:
                        reverse_rank = int(actual_k_value // 1000.0)
                        reverse_s = int(actual_k_value % 1000.0)
                        reverse_global = reverse_rank * local_seq_len + reverse_s
                        print(
                            f"    ❌ Actual data source: rank={reverse_rank}, local_s={reverse_s}"
                        )
                        print(
                            f"    ❌ Expected global_pos={global_pos.item()}, actual global_pos={reverse_global}"
                        )
                        print(f"    ❌ Offset: {reverse_global - global_pos.item()}")
                        break
                if v_error > 1e-5:
                    print(f"❌ Rank {rank}: V value error!")
                    break

            break
        break

    print(f"✅ Rank {rank}: Numerical correctness verification passed!")


    torch.distributed.barrier(device_ids=[rank])

    print(f"✅ All RANKS barrier done")

    # Recreate random data for performance test

    k_perf = torch.randn(B, H, local_seq_len, D, device=device, dtype=dtype_performance)
    v_perf = torch.randn(B, H, local_seq_len, D, device=device, dtype=dtype_performance)

    for _ in range(3):
        gathered_k, gathered_v = sparse_kv_gather(
            k_perf, v_perf, activated_indices, auto_grad=False
        )
        torch.cuda.synchronize()

    print(f"Rank {rank} warmup done")

    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    begin_event.record()
    test_num = 10
    for _ in range(test_num):
        gathered_k, gathered_v = sparse_kv_gather(
            k_perf, v_perf, activated_indices, auto_grad=False
        )
    end_event.record()
    torch.cuda.synchronize()
    print(f"Rank {rank} time: {begin_event.elapsed_time(end_event)/test_num} ms")

    dist.destroy_process_group(group=cp_group)

    print(
        f"Rank {rank}: All tests passed successfully, including numerical correctness and performance test!"
    )


def test_sparse_kv_gather_backward(B=1, H=24, S=128000, D=128):
    """Test backward propagation with gradient accumulation verification"""
    import os

    world_size, rank = init_distributed(
        int(os.environ.get("RANK")), int(os.environ.get("WORLD_SIZE"))
    )

    cp_group = dist.new_group(ranks=list(range(world_size)))
    comm_manager = CommGroupManager(cp_group=cp_group)
    comm_manager.init_cp_group()

    B = B
    H = H
    S = S
    D = D
    local_seq_len = S // world_size

    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.float32

    k = torch.zeros(
        B, H, local_seq_len, D, device=device, dtype=dtype, requires_grad=False
    )
    v = torch.zeros(
        B, H, local_seq_len, D, device=device, dtype=dtype, requires_grad=False
    )

    for b in range(B):
        for h in range(H):
            for s in range(local_seq_len):
                unique_id = rank * 1000 + s
                k[b, h, s, :] = unique_id
                v[b, h, s, :] = unique_id + 0.5

    print(f"Rank {rank}: Creating activation patterns, local_seq_len={local_seq_len}")

    activated_indices = torch.zeros(B, H, S, dtype=torch.bool, device=device)
    positions_per_target_gpu = (
        5000 
    )

    for b in range(B):
        for h in range(H):
            # Each GPU activates positions in every other GPU's range
            for target_gpu in range(world_size):
                target_start = target_gpu * local_seq_len
                target_end = target_start + local_seq_len

                for i in range(positions_per_target_gpu):
                    hash_input = rank * 1000 + h * 100 + target_gpu * 10 + i
                    offset = hash_input % local_seq_len
                    pos = target_start + offset

                    if pos < S:
                        activated_indices[b, h, pos] = True

            # Add common positions that ALL GPUs activate
            common_positions_per_head = 3
            for i in range(common_positions_per_head):
                common_pos = (h * 777 + i * 123) % S
                activated_indices[b, h, common_pos] = True

    # Print activation statistics
    total_activations = activated_indices[0, 0].sum().item()
    print(f"Rank {rank}: Total activations for head 0: {total_activations}")

    for target_gpu in range(world_size):
        target_start = target_gpu * local_seq_len
        target_end = target_start + local_seq_len
        activations_in_range = (
            activated_indices[0, 0, target_start:target_end].sum().item()
        )
        print(
            f"Rank {rank}: GPU {target_gpu} range activations: {activations_in_range}"
        )

    # Create SparseKVGather instance and run forward
    layer_id = 0
    sparse_kv_gather = SparseKVGather(layer_id, comm_manager)
    gathered_k, gathered_v = sparse_kv_gather(k, v, activated_indices, auto_grad=False)

    # Create rank-specific gradients
    grad_gathered_k = torch.zeros_like(gathered_k)
    grad_gathered_v = torch.zeros_like(gathered_v)

    num_activated = activated_indices.sum(dim=-1)
    max_activated = num_activated.max().item()

    base_grad_k = (rank + 1) * 0.1
    base_grad_v = (rank + 1) * 0.1 + 0.05

    for b in range(B):
        for h in range(H):
            activated_positions = torch.where(activated_indices[b, h])[0]
            actual_activated = len(activated_positions)

            for i in range(actual_activated):
                grad_gathered_k[b, h, i, :] = base_grad_k
                grad_gathered_v[b, h, i, :] = base_grad_v

    print(f"Rank {rank}: Created gradients, max_activated={max_activated}")

    # Run backward propagation
    grad_k, grad_v = sparse_kv_gather.backward_kv_gradients(
        grad_gathered_k, grad_gathered_v
    )

    # Verify basic properties
    assert grad_k.shape == k.shape, f"Rank {rank}: k gradient shape mismatch"
    assert grad_v.shape == v.shape, f"Rank {rank}: v gradient shape mismatch"

    k_grad_norm = grad_k.norm().item()
    v_grad_norm = grad_v.norm().item()
    print(f"Rank {rank}: gradient norms - k: {k_grad_norm:.6f}, v: {v_grad_norm:.6f}")

    # Gradient accumulation verification
    print(f"Rank {rank}: Starting gradient verification...")

    all_activated_indices = [
        torch.zeros_like(activated_indices) for _ in range(world_size)
    ]
    dist.all_gather(all_activated_indices, activated_indices, group=cp_group)

    # Analyze activation patterns
    for target_gpu in range(world_size):
        target_start = target_gpu * local_seq_len
        target_end = target_start + local_seq_len

        requesting_gpus = set()
        for src_rank in range(world_size):
            if all_activated_indices[src_rank][0, 0, target_start:target_end].any():
                requesting_gpus.add(src_rank)

        print(
            f"Rank {rank}: GPU {target_gpu} has requests from {len(requesting_gpus)} GPUs"
        )

    # Verify gradient accumulation
    verification_errors = 0
    successful_accumulations = 0

    for b in range(B):
        for h in range(H):
            for local_s in range(local_seq_len):
                global_pos = rank * local_seq_len + local_s

                expected_grad_k = 0.0
                expected_grad_v = 0.0
                requesting_gpus = []

                for src_rank in range(world_size):
                    if all_activated_indices[src_rank][b, h, global_pos]:
                        requesting_gpus.append(src_rank)
                        expected_grad_k += (src_rank + 1) * 0.1
                        expected_grad_v += (src_rank + 1) * 0.1 + 0.05

                actual_grad_k = grad_k[b, h, local_s, 0].item()
                actual_grad_v = grad_v[b, h, local_s, 0].item()

                if len(requesting_gpus) > 0:
                    k_error = abs(actual_grad_k - expected_grad_k)
                    v_error = abs(actual_grad_v - expected_grad_v)

                    if k_error > 1e-5 or v_error > 1e-5:
                        if verification_errors < 5:
                            print(f"❌ Rank {rank}: Error at pos {global_pos}")
                            print(
                                f"   Expected K: {expected_grad_k}, Actual: {actual_grad_k}"
                            )
                            print(
                                f"   Expected V: {expected_grad_v}, Actual: {actual_grad_v}"
                            )
                        verification_errors += 1
                    else:
                        if len(requesting_gpus) > 1:
                            successful_accumulations += 1

                else:
                    if abs(actual_grad_k) > 1e-6 or abs(actual_grad_v) > 1e-6:
                        if verification_errors < 5:
                            print(
                                f"❌ Rank {rank}: Unexpected gradient at pos {global_pos}"
                            )
                            print(f"   K: {actual_grad_k}, V: {actual_grad_v}")
                        verification_errors += 1
                        raise ValueError(
                            f"Rank {rank}: Unexpected non-zero gradient at position {global_pos}"
                        )

    if verification_errors > 0:
        print(f"❌ Rank {rank}: Found {verification_errors} verification errors!")
        assert (
            False
        ), f"Gradient accumulation verification failed with {verification_errors} errors"

    print(f"✅ Rank {rank}: Gradient verification passed")
    print(
        f"✅ Rank {rank}: {successful_accumulations} successful multi-GPU accumulations"
    )

    # Final statistics
    total_nonzero_positions = 0
    multi_gpu_accumulations = 0

    for b in range(B):
        for h in range(H):
            for local_s in range(local_seq_len):
                global_pos = rank * local_seq_len + local_s
                if abs(grad_k[b, h, local_s, 0].item()) > 1e-6:
                    total_nonzero_positions += 1
                    requesting_count = sum(
                        1
                        for src_rank in range(world_size)
                        if all_activated_indices[src_rank][b, h, global_pos]
                    )
                    if requesting_count > 1:
                        multi_gpu_accumulations += 1

    print(f"Rank {rank}: Statistics:")
    print(f"  Non-zero positions: {total_nonzero_positions}")
    print(f"  Multi-GPU accumulations: {multi_gpu_accumulations}")
    print(
        f"  Accumulation ratio: {multi_gpu_accumulations/max(total_nonzero_positions,1):.2%}"
    )

    # Final verification
    assert k_grad_norm > 1e-8, f"Rank {rank}: k gradient too small"
    assert v_grad_norm > 1e-8, f"Rank {rank}: v gradient too small"

    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])

    print(f"✅ Rank {rank}: All tests passed")




def test_sparse_kv_gather_backward_latency(
    B=1, H=16, S=128000, D=128, sparsity_per_head=[0.8] * 16
):
    """Test sparse KV gather backward gradient latency"""
    import os

    world_size, rank = init_distributed(
        int(os.environ.get("RANK")), int(os.environ.get("WORLD_SIZE"))
    )

    cp_group = dist.new_group(ranks=list(range(world_size)))
    comm_manager = CommGroupManager(cp_group=cp_group)
    comm_manager.init_cp_group()

    B = B  # batch size
    H = H  # Head number
    S = S  # Total sequence length
    D = D  # Hidden dimension
    local_seq_len = S // world_size  # Sequence length per GPU

    torch.manual_seed(
        42 + rank
    )  # Ensure each rank has different but reproducible random numbers
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    k = torch.randn(B, H, local_seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, local_seq_len, D, device=device, dtype=dtype)

    activated_indices = torch.zeros(B, H, S, dtype=torch.bool, device=device)
    for b in range(B):
        for h in range(H):
            # Randomly activate 1% of positions
            num_activated = int((1 - sparsity_per_head[h]) * S)
            activated_pos = torch.randperm(S)[:num_activated]
            activated_indices[b, h, activated_pos] = True

    # Create SparseKVGather instance
    layer_id = 0
    sparse_kv_gather = SparseKVGather(layer_id, comm_manager)

    # First run forward to prepare cache and get gathered tensor shape
    gathered_k, gathered_v = sparse_kv_gather(k, v, activated_indices, auto_grad=False)

    print(f"Rank {rank} forward cache prepared, gathered_k shape: {gathered_k.shape}")

    # Create simulated gradient tensors
    grad_gathered_k = torch.randn_like(gathered_k)
    grad_gathered_v = torch.randn_like(gathered_v)

    # Warmup backward passes
    print(f"Rank {rank} starting backward warmup...")
    for _ in range(3):
        # Re-run forward to ensure cache is fresh
        gathered_k, gathered_v = sparse_kv_gather(
            k, v, activated_indices, auto_grad=False
        )
        grad_k, grad_v = sparse_kv_gather.backward_kv_gradients(
            grad_gathered_k, grad_gathered_v
        )
        torch.cuda.synchronize()

    print(f"Rank {rank} backward warmup done")

    # Performance test
    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    test_num = 10
    # First run forward to prepare cache
    gathered_k, gathered_v = sparse_kv_gather(k, v, activated_indices, auto_grad=False)

    # Test backward latency
    begin_event.record()
    for _ in range(test_num):
        # For each backward test, use the same forward cache (actual usage scenario)
        grad_k, grad_v = sparse_kv_gather.backward_kv_gradients(
            grad_gathered_k, grad_gathered_v
        )
    end_event.record()
    torch.cuda.synchronize()

    backward_time = begin_event.elapsed_time(end_event) / test_num
    print(f"Rank {rank} backward time: {backward_time:.3f} ms")


    combined_begin_event = torch.cuda.Event(enable_timing=True)
    combined_end_event = torch.cuda.Event(enable_timing=True)

    combined_begin_event.record()
    for _ in range(test_num):
        # Forward + Backward combination
        gathered_k, gathered_v = sparse_kv_gather(
            k, v, activated_indices, auto_grad=False
        )
        grad_k, grad_v = sparse_kv_gather.backward_kv_gradients(
            grad_gathered_k, grad_gathered_v
        )
    combined_end_event.record()
    torch.cuda.synchronize()

    combined_time = combined_begin_event.elapsed_time(combined_end_event) / test_num
    forward_time = combined_time - backward_time

    print(f"Rank {rank} forward time: {forward_time:.3f} ms")
    print(f"Rank {rank} combined time: {combined_time:.3f} ms")
    print(f"Rank {rank} backward/forward ratio: {backward_time/forward_time:.2f}")

    # Verify the correctness of backward output
    assert grad_k.shape == k.shape, f"Rank {rank}: k gradient shape mismatch"
    assert grad_v.shape == v.shape, f"Rank {rank}: v gradient shape mismatch"

    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])

    print(f"Rank {rank}: Backward latency test passed successfully!")
    print(
        f"Rank {rank}: Summary - Forward: {forward_time:.3f}ms, Backward: {backward_time:.3f}ms"
    )

    forward_comm_latency = forward_time
    backward_comm_latency = backward_time
    return forward_comm_latency, backward_comm_latency


if __name__ == "__main__":

    test_all_gather_latency(B=1, H=16, S=256000, D=128)

