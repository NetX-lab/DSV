# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0


from typing import Any, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module


def all_to_all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


def solve_optimal_head_allocation(sparsity_per_head, world_size):
    head_num = len(sparsity_per_head)

    head_burden_pairs = [(i, 1.0 - sparsity_per_head[i]) for i in range(head_num)]
    head_burden_pairs.sort(key=lambda x: x[1], reverse=True)

    group_loads = [0.0] * world_size
    group_heads = [[] for _ in range(world_size)]

    for head_idx, burden in head_burden_pairs:
        min_load_group = min(range(world_size), key=lambda i: group_loads[i])
        group_heads[min_load_group].append(head_idx)
        group_loads[min_load_group] += burden

    reallocated_head_idx_list = []
    for group_idx in range(world_size):
        reallocated_head_idx_list.extend(group_heads[group_idx])

    reallocated_head_num_list = [len(group_heads[i]) for i in range(world_size)]

    return reallocated_head_idx_list, reallocated_head_num_list


def all_to_all_balanced_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
    reallocated_head_idx_list: List[int] = None,
    reallocated_head_num_list: List[int] = None,
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.Tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 2
        gather_idx (int): default 1
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all
        reallocated_head_idx_list: List[int], the list of head indices reallocated globally
        reallocated_head_num_list: List[int], the list of head counts for each rank

    Returns:
        torch.Tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)

    if scatter_idx == 2 and gather_idx == 1:
        # 从 sequence 并行（均匀）转换到 head 并行（不均匀）
        # input: (bs, seqlen/P, hc, hs) -> output: (bs, seqlen, local_head_count, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        local_head_count = reallocated_head_num_list[group_rank]

        # 1. 重排heads并转置: (bs, seqlen/P, hc, hs) -> (hc, seqlen/P, bs, hs)
        input_t = input[:, :, reallocated_head_idx_list, :].transpose(0, 2).contiguous()

        # 2. 计算split sizes (核心：支持不均匀分布)
        # 发送：当前rank把heads按目标rank需求分割
        input_split_sizes = [
            head_count * shard_seqlen * bs * hs
            for head_count in reallocated_head_num_list
        ]
        # 接收：当前rank从每个rank接收local_head_count个heads
        output_split_sizes = [
            local_head_count * shard_seqlen * bs * hs
        ] * seq_world_size

        # 3. 直接flatten输入数据 (因为heads已经按顺序排列)
        input_flat = input_t.view(-1)

        # 准备输出buffer
        total_output_size = sum(output_split_sizes)
        output_flat = torch.empty(
            total_output_size, device=input.device, dtype=input.dtype
        )

        if seq_world_size > 1:
            dist.all_to_all_single(
                output_flat,
                input_flat,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
            if use_sync:
                torch.cuda.synchronize()
        else:
            output_flat = input_flat

        # 4. 重组输出: 来自各rank的sequence片段拼接
        # FIXED: Correct reshaping logic for seq->head conversion
        # output_flat contains data from each rank in order:
        # [rank0_data, rank1_data, ..., rankN_data]
        # where each rank_data contains local_head_count heads with shard_seqlen sequence

        # Reshape to separate data from different source ranks
        output_reshaped = output_flat.reshape(
            seq_world_size, local_head_count, shard_seqlen, bs, hs
        )

        # Transpose to reorganize: (seq_world_size, local_head_count, shard_seqlen, bs, hs)
        # -> (local_head_count, seq_world_size, shard_seqlen, bs, hs)
        output_transposed = output_reshaped.transpose(0, 1).contiguous()

        # Reshape to concatenate sequence fragments: (local_head_count, seqlen, bs, hs)
        output = output_transposed.reshape(local_head_count, seqlen, bs, hs)

        # 5. 最终格式: (bs, seqlen, local_head_count, hs)
        output = output.transpose(0, 2).contiguous()

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # 从 head 并行（不均匀）转换到 sequence 并行（均匀）
        # input: (bs, seqlen, local_head_count, hs) -> output: (bs, seqlen/P, hc, hs)

        world_size = dist.get_world_size(group)
        device = input.device

        bs, seqlen, local_head_count, hs = input.shape  # 统一使用hs
        shard_seqlen = seqlen // world_size
        total_head_count = sum(reallocated_head_num_list)

        # 1. 重排输入为all_to_all格式
        input_t = input.permute(
            2, 1, 0, 3
        ).contiguous()  # [local_head_count, seqlen, bs, hs]
        input_t = input_t.reshape(local_head_count, world_size, shard_seqlen, bs, hs)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        input_flat = input_t.view(-1)  # 使用view而不是reshape

        # 2. 计算split sizes
        send_size_per_rank = local_head_count * shard_seqlen * bs * hs
        input_split_sizes = [send_size_per_rank] * world_size
        output_split_sizes = [
            head_num * shard_seqlen * bs * hs for head_num in reallocated_head_num_list
        ]

        # 3. 准备输出buffer
        total_recv_size = sum(output_split_sizes)
        output_flat = torch.empty(total_recv_size, device=device, dtype=input.dtype)

        # 4. all_to_all通信
        if world_size > 1:
            dist.all_to_all_single(
                output_flat,
                input_flat,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
            if use_sync:
                torch.cuda.synchronize()  # 统一使用cuda sync
        else:
            output_flat = input_flat

        # 5. 重组输出 - FIXED
        # output_flat contains data from each rank in order:
        # [rank0_data, rank1_data, ..., rankN_data]
        # where rankK_data contains reallocated_head_num_list[k] heads

        # First reshape to separate data from different ranks
        chunks = []
        offset = 0
        for rank_idx in range(world_size):
            head_num = reallocated_head_num_list[rank_idx]
            chunk_size = head_num * shard_seqlen * bs * hs
            chunk = output_flat[offset : offset + chunk_size]
            chunk = chunk.reshape(head_num, shard_seqlen, bs, hs)
            chunks.append(chunk)
            offset += chunk_size

        # Concatenate all chunks to get (total_head_count, shard_seqlen, bs, hs)
        output_concat = torch.cat(
            chunks, dim=0
        )  # (total_head_count, shard_seqlen, bs, hs)
        output = output_concat.permute(
            2, 1, 0, 3
        )  # (bs, shard_seqlen, total_head_count, hs)

        # 6. 用argsort恢复原始head顺序
        original_head_idx_list = torch.argsort(
            torch.tensor(reallocated_head_idx_list, device=device)
        )
        output = output[:, :, original_head_idx_list, :]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return all_to_all_4D(
            input, scatter_idx, gather_idx, group=group, use_sync=use_sync
        )

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
            ),
            None,
            None,
            None,
        )


if __name__ == "__main__":
    import os

    def setup_distributed():
        """Initialize distributed environment"""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(
                os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())
            )
        else:
            print(
                "No distributed environment variables detected, using single process mode"
            )
            rank = 0
            world_size = 1
            local_rank = 0

        if world_size > 1:
            # Set CUDA device before initializing process group
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for distributed training")

            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            # Initialize process group
            dist.init_process_group(
                backend="nccl", init_method="env://", world_size=world_size, rank=rank
            )

            print(
                f"Rank {rank}/{world_size} initialized, using device: {device} (local_rank: {local_rank})"
            )
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Single process mode, using device: {device}")

        return rank, world_size, local_rank, device

    def test_all_to_all_balanced_4D(rank, world_size, device, group=None):
        """Test all_to_all_balanced_4D function with uneven head distribution"""
        print(f"[Rank {rank}] === Testing all_to_all_balanced_4D ===")

        # Test parameters
        bs, total_seqlen, total_heads, hs = 1, 128 * world_size, 16, 96
        shard_seqlen = total_seqlen // world_size

        # Generate uneven head allocation based on actual world_size
        # Ensure total heads are distributed unevenly across ranks
        base_heads = total_heads // world_size
        extra_heads = total_heads % world_size

        # Start with even distribution, then make it uneven
        reallocated_head_num_list = []
        for i in range(world_size):
            head_count = base_heads + (1 if i < extra_heads else 0)
            reallocated_head_num_list.append(head_count)

        # Now redistribute some heads to make it uneven while keeping total constant
        if world_size > 1:
            # Move some heads from later ranks to earlier ranks
            for i in range(world_size // 2):
                source_idx = world_size - 1 - i
                target_idx = i
                if reallocated_head_num_list[source_idx] > 1:
                    # Move 1 head from source to target
                    reallocated_head_num_list[source_idx] -= 1
                    reallocated_head_num_list[target_idx] += 1

        assert (
            sum(reallocated_head_num_list) == total_heads
        ), f"Total head count mismatch: {sum(reallocated_head_num_list)} vs {total_heads}"

        # Generate head index allocation: create a permuted order
        reallocated_head_idx_list = list(range(total_heads))
        # Create some permutation for testing
        if world_size > 1:
            import random

            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(reallocated_head_idx_list)

        local_head_count = reallocated_head_num_list[rank]

        print(f"[Rank {rank}] Head allocation: {reallocated_head_num_list}")
        print(f"[Rank {rank}] Local head count: {local_head_count}")
        if rank == 0:
            print(f"[Rank {rank}] Head index list: {reallocated_head_idx_list}")

        try:
            # Test 1: Sequence parallel -> Head parallel (uneven distribution)
            # Input shape: (bs, seqlen/P, total_heads, hs)
            input_tensor = torch.randn(bs, shard_seqlen, total_heads, hs, device=device)
            print(f"[Rank {rank}] Input shape: {input_tensor.shape}")

            # Forward pass: seq parallel -> head parallel
            output1 = all_to_all_balanced_4D(
                input_tensor,
                scatter_idx=2,
                gather_idx=1,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )

            if rank == 0:
                print(
                    f"[Rank {rank}] Balanced output shape (seq->head): {output1.shape}"
                )
            expected_shape = (bs, total_seqlen, local_head_count, hs)
            assert (
                output1.shape == expected_shape
            ), f"Output shape mismatch: {output1.shape} vs {expected_shape}"

            # Test 2: Head parallel -> Sequence parallel (reverse operation)
            # Input shape: (bs, total_seqlen, local_head_count, hs)
            output2 = all_to_all_balanced_4D(
                output1,
                scatter_idx=1,
                gather_idx=2,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )
            if rank == 0:
                print(
                    f"[Rank {rank}] Balanced output shape (head->seq): {output2.shape}"
                )

            assert (
                output2.shape == input_tensor.shape
            ), f"Round-trip shape mismatch: {output2.shape} vs {input_tensor.shape}"

            # Test 3: Advanced data content verification
            print(f"[Rank {rank}] Starting advanced data content verification...")

            # Create simple predictable pattern: unique value per position
            # Pattern: batch_id * 1000000 + global_seq_pos * 1000 + head_id
            pattern_input = torch.zeros(
                bs, shard_seqlen, total_heads, hs, device=device
            )

            for b in range(bs):
                for s in range(shard_seqlen):
                    for h in range(total_heads):
                        # Create unique identifier based on global position
                        global_seq_pos = rank * shard_seqlen + s
                        # Simple pattern: easier to verify
                        value = b * 1000000 + global_seq_pos * 1000 + h
                        pattern_input[
                            b, s, h, :
                        ] = value  # Same value across feature dimension

            print(f"[Rank {rank}] Created pattern input with unique identifiers")

            advanced_verification_passed = True

            # Forward pass: seq parallel -> head parallel
            pattern_output1 = all_to_all_balanced_4D(
                pattern_input,
                scatter_idx=2,
                gather_idx=1,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )

            # Verify the data content in head parallel format
            print(f"[Rank {rank}] Verifying data content after seq->head conversion...")

            # CORRECTED UNDERSTANDING:
            # reallocated_head_idx_list is used to REORDER the input heads BEFORE all_to_all
            # The function does: input[:, :, reallocated_head_idx_list, :]
            # Then distributes consecutive chunks of these REORDERED heads to ranks

            # Get the heads assigned to this rank in the REORDERED sequence
            head_start_idx = sum(reallocated_head_num_list[:rank])
            head_end_idx = head_start_idx + local_head_count

            # These are the positions in the REORDERED list that this rank gets
            my_reordered_positions = list(range(head_start_idx, head_end_idx))

            print(f"[Rank {rank}] My reordered positions: {my_reordered_positions}")
            print(
                f"[Rank {rank}] Full reallocated_head_idx_list: {reallocated_head_idx_list}"
            )

            # For debugging: show what heads we should receive
            print(f"[Rank {rank}] Heads I should receive (by reordered position):")
            for i, pos in enumerate(my_reordered_positions):
                original_head = reallocated_head_idx_list[pos]
                print(
                    f"[Rank {rank}]   local_h={i} <- reordered_pos={pos} <- original_head={original_head}"
                )

            verification_passed = True
            max_error = 0.0
            sample_checks = 0

            for b in range(bs):
                for s in range(
                    total_seqlen
                ):  # Only check first 10 sequence positions for efficiency
                    for local_h in range(local_head_count):
                        # The data at output[b, s, local_h] should come from:
                        # - sequence position s
                        # - reordered head position (head_start_idx + local_h)
                        # - which corresponds to original head reallocated_head_idx_list[head_start_idx + local_h]

                        reordered_pos = head_start_idx + local_h
                        original_head_id = reallocated_head_idx_list[reordered_pos]

                        # The original data was created as:
                        # value = b * 1000000 + s * 1000 + original_head_id
                        expected_value = b * 1000000 + s * 1000 + original_head_id
                        actual_value = pattern_output1[b, s, local_h, 0].item()

                        error = abs(actual_value - expected_value)
                        max_error = max(max_error, error)
                        sample_checks += 1

                        if error > 1e-6:
                            # Decode the actual value to understand what we got
                            actual_b = int(actual_value) // 1000000
                            remaining = int(actual_value) % 1000000
                            actual_s = remaining // 1000
                            actual_h = remaining % 1000

                            print(
                                f"[Rank {rank}] Data mismatch at output[{b},{s},{local_h}]: "
                            )
                            print(
                                f"[Rank {rank}]   Expected: b={b}, s={s}, head_id={original_head_id} (reordered_pos={reordered_pos}) -> {expected_value}"
                            )
                            print(
                                f"[Rank {rank}]   Got: b={actual_b}, s={actual_s}, head_id={actual_h} -> {actual_value}"
                            )
                            verification_passed = False
                            break
                    if not verification_passed:
                        break
                if not verification_passed:
                    break

            if verification_passed:
                print(
                    f"[Rank {rank}] ✓ Data content verification passed (max error: {max_error}, checked {sample_checks} samples)"
                )
            else:
                advanced_verification_passed = False
                print(
                    f"\033[91m[Rank {rank}] ✗ Data content verification failed\033[0m"
                )

            # Reverse pass: head parallel -> seq parallel
            pattern_output2 = all_to_all_balanced_4D(
                pattern_output1,
                scatter_idx=1,
                gather_idx=2,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )

            # Verify round-trip correctness
            round_trip_verification_passed = True
            round_trip_error = torch.abs(pattern_input - pattern_output2).max().item()
            if round_trip_error < 1e-6:
                print(
                    f"[Rank {rank}] ✓ Round-trip data integrity verified (max error: {round_trip_error})"
                )
            else:
                round_trip_verification_passed = False
                print(
                    f"\033[91m[Rank {rank}] ✗ Round-trip data integrity failed (max error: {round_trip_error})\033[0m"
                )

            if advanced_verification_passed and round_trip_verification_passed:
                # print green
                print(
                    f"\033[92m[Rank {rank}] ✓ all_to_all_balanced_4D test passed\033[0m"
                )
            else:
                print(
                    f"\033[91m[Rank {rank}] ✗ all_to_all_balanced_4D test failed\033[0m"
                )

        except Exception as e:
            print(
                f"\033[91m[Rank {rank}] ✗ all_to_all_balanced_4D test failed: {e}\033[0m"
            )
            import traceback

            traceback.print_exc()

    # Main test execution
    print("Starting all_to_all_balanced_4D distributed test...")

    try:
        rank, world_size, local_rank, device = setup_distributed()

        # Get or create process group
        if world_size > 1:
            group = dist.group.WORLD  # Use default world process group
        else:
            group = None

        # Synchronize all processes with explicit device specification
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        print(f"[Rank {rank}] Starting test with world size: {world_size}")

        # Run the test
        test_all_to_all_balanced_4D(rank, world_size, device, group)

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        print(f"[Rank {rank}] Test completed successfully!")

        # Clean up distributed environment
        if world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
        if world_size > 1:
            try:
                dist.destroy_process_group()
            except:
                pass
