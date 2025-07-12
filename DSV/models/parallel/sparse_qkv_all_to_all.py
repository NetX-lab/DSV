from typing import Any, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module


def setup_distributed():
    """Initialize distributed environment for multi-GPU testing"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for distributed training")

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


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
    All-to-all communication for QKV with uneven head distribution.

    Args:
        input: Input tensor to be redistributed
        scatter_idx: Dimension to scatter (2 for seq->head, 1 for head->seq)
        gather_idx: Dimension to gather (1 for seq->head, 2 for head->seq)
        group: Process group for communication
        use_sync: Whether to synchronize after communication
        reallocated_head_idx_list: Global head reordering indices
        reallocated_head_num_list: Head count per rank

    Returns:
        Redistributed tensor with new sharding pattern
    """
    assert input.dim() == 4, f"Input must be 4D tensor, got {input.dim()}D"

    seq_world_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)

    if scatter_idx == 2 and gather_idx == 1:
        # Sequence parallel -> Head parallel transformation
        # (bs, seqlen/P, total_heads, hs) -> (bs, seqlen, local_heads, hs)
        bs, shard_seqlen, hc, hs = input.shape
        local_head_count = reallocated_head_num_list[group_rank]

        # Step 1: Reorder heads and transpose for all-to-all
        input_t = input[:, :, reallocated_head_idx_list, :].transpose(0, 2).contiguous()

        # Step 2: Calculate split sizes for uneven distribution
        input_split_sizes = [
            head_count * shard_seqlen * bs * hs
            for head_count in reallocated_head_num_list
        ]
        output_split_sizes = [
            local_head_count * shard_seqlen * bs * hs
        ] * seq_world_size

        # Step 3: Prepare data and perform all-to-all communication
        input_flat = input_t.view(-1)
        output_flat = torch.empty(
            sum(output_split_sizes), device=input.device, dtype=input.dtype
        )

        if seq_world_size > 1:
            dist.all_to_all_single(
                output_flat,
                input_flat,
                output_split_sizes,
                input_split_sizes,
                group=group,
            )
            if use_sync:
                torch.cuda.synchronize()
        else:
            output_flat = input_flat

        # Step 4: Reassemble sequence fragments from all ranks
        chunks = []
        offset = 0
        for src_rank in range(seq_world_size):
            chunk_size = local_head_count * shard_seqlen * bs * hs
            chunk_data = output_flat[offset : offset + chunk_size]
            chunk_reshaped = chunk_data.reshape(local_head_count, shard_seqlen, bs, hs)
            chunks.append(chunk_reshaped)
            offset += chunk_size

        # Step 5: Concatenate and transpose to final format
        output_concat = torch.cat(chunks, dim=1)  # (local_heads, seqlen, bs, hs)
        output = output_concat.permute(
            2, 1, 0, 3
        ).contiguous()  # (bs, seqlen, local_heads, hs)
        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # Head parallel -> Sequence parallel transformation
        # (bs, seqlen, local_heads, hs) -> (bs, seqlen/P, total_heads, hs)
        world_size = dist.get_world_size(group)
        bs, seqlen, local_head_count, hs = input.shape
        shard_seqlen = seqlen // world_size
        total_head_count = sum(reallocated_head_num_list)

        # Step 1: Rearrange input for all-to-all
        input_t = input.permute(
            2, 1, 0, 3
        ).contiguous()  # (local_heads, seqlen, bs, hs)
        input_t = input_t.reshape(local_head_count, world_size, shard_seqlen, bs, hs)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        input_flat = input_t.view(-1)

        # Step 2: Calculate split sizes
        input_split_sizes = [local_head_count * shard_seqlen * bs * hs] * world_size
        output_split_sizes = [
            head_num * shard_seqlen * bs * hs for head_num in reallocated_head_num_list
        ]

        # Step 3: Perform all-to-all communication
        output_flat = torch.empty(
            sum(output_split_sizes), device=input.device, dtype=input.dtype
        )

        if world_size > 1:
            dist.all_to_all_single(
                output_flat,
                input_flat,
                output_split_sizes,
                input_split_sizes,
                group=group,
            )
            if use_sync:
                torch.cuda.synchronize()
        else:
            output_flat = input_flat

        # Step 4: Reassemble heads from all ranks
        chunks = []
        offset = 0
        for rank_idx in range(world_size):
            head_num = reallocated_head_num_list[rank_idx]
            chunk_size = head_num * shard_seqlen * bs * hs
            chunk = output_flat[offset : offset + chunk_size]
            chunk = chunk.reshape(head_num, shard_seqlen, bs, hs)
            chunks.append(chunk)
            offset += chunk_size

        # Step 5: Restore original head order
        output_concat = torch.cat(chunks, dim=0)  # (total_heads, shard_seqlen, bs, hs)
        output = output_concat.permute(
            2, 1, 0, 3
        )  # (bs, shard_seqlen, total_heads, hs)

        original_head_idx_list = torch.argsort(
            torch.tensor(reallocated_head_idx_list, device=input.device)
        )
        output = output[:, :, original_head_idx_list, :]
        return output

    else:
        raise ValueError(
            f"Invalid scatter_idx={scatter_idx}, gather_idx={gather_idx}. Must be (2,1) or (1,2)."
        )


class SeqAllToAllBalanced4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
        reallocated_head_idx_list: List[int] = None,
        reallocated_head_num_list: List[int] = None,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        ctx.reallocated_head_idx_list = reallocated_head_idx_list
        ctx.reallocated_head_num_list = reallocated_head_num_list

        return all_to_all_balanced_4D(
            input,
            scatter_idx,
            gather_idx,
            group=group,
            use_sync=use_sync,
            reallocated_head_idx_list=reallocated_head_idx_list,
            reallocated_head_num_list=reallocated_head_num_list,
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[None, Tensor, None, None, None]:
        return (
            None,
            SeqAllToAllBalanced4D.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.reallocated_head_idx_list,
                ctx.reallocated_head_num_list,
                ctx.use_sync,
            ),
            None,
            None,
            None,
        )


if __name__ == "__main__":
    import os

    def test_all_to_all_balanced_4D(
        rank, world_size, device, group=None, B=1, H=24, S=128000, D=128
    ):
        """Test all_to_all_balanced_4D function with uneven head distribution"""
        if rank == 0:
            print(f"Testing all_to_all_balanced_4D with {world_size} ranks...")

        # Test parameters
        bs, total_seqlen, total_heads, hs = B, S, H, D
        shard_seqlen = total_seqlen // world_size

        # Generate uneven head allocation
        base_heads = total_heads // world_size
        extra_heads = total_heads % world_size
        reallocated_head_num_list = [
            base_heads + (1 if i < extra_heads else 0) for i in range(world_size)
        ]

        # Create imbalance by moving heads between ranks
        if world_size > 1:
            for i in range(world_size // 2):
                source_idx = world_size - 1 - i
                target_idx = i
                if reallocated_head_num_list[source_idx] > 1:
                    reallocated_head_num_list[source_idx] -= 1
                    reallocated_head_num_list[target_idx] += 1

        assert sum(reallocated_head_num_list) == total_heads

        # Create shuffled head indices for realistic reordering
        reallocated_head_idx_list = list(range(total_heads))
        if world_size > 1:
            import random

            random.seed(42)
            random.shuffle(reallocated_head_idx_list)

        local_head_count = reallocated_head_num_list[rank]
        if rank == 0:
            print(f"Head allocation: {reallocated_head_num_list}")

        try:
            # Test 1: Basic shape transformation
            input_tensor = torch.randn(
                bs, shard_seqlen, total_heads, hs, device=device, dtype=torch.float32
            )

            # Forward: seq parallel -> head parallel
            output1 = all_to_all_balanced_4D(
                input_tensor,
                scatter_idx=2,
                gather_idx=1,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )
            expected_shape = (bs, total_seqlen, local_head_count, hs)
            assert (
                output1.shape == expected_shape
            ), f"Shape mismatch: {output1.shape} vs {expected_shape}"

            # Backward: head parallel -> seq parallel
            output2 = all_to_all_balanced_4D(
                output1,
                scatter_idx=1,
                gather_idx=2,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )
            assert output2.shape == input_tensor.shape, f"Round-trip shape mismatch"

            # Test 2: Data content verification
            pattern_input = torch.zeros(
                bs, shard_seqlen, total_heads, hs, device=device
            )

            # Create unique data pattern for verification
            for b in range(bs):
                for s in range(shard_seqlen):
                    for h in range(total_heads):
                        global_seq_pos = rank * shard_seqlen + s
                        value = b * 10000000.0 + global_seq_pos * 100.0 + h
                        pattern_input[b, s, h, :] = value

            # Forward transformation
            pattern_output1 = all_to_all_balanced_4D(
                pattern_input,
                scatter_idx=2,
                gather_idx=1,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )

            # Verify data content
            head_start_idx = sum(reallocated_head_num_list[:rank])
            verification_passed = True
            max_error = 0.0
            sample_checks = 0

            for b in range(bs):
                for s in range(total_seqlen):
                    for local_h in range(local_head_count):
                        reordered_pos = head_start_idx + local_h
                        expected_head_id = reallocated_head_idx_list[reordered_pos]
                        expected_value = b * 10000000.0 + s * 100.0 + expected_head_id
                        actual_value = pattern_output1[b, s, local_h, 0].item()

                        error = abs(actual_value - expected_value)
                        max_error = max(max_error, error)
                        sample_checks += 1

                        if error > 1e-6:
                            verification_passed = False
                            # Only rank 0 prints detailed error info
                            if rank == 0:
                                actual_b = int(actual_value) // 10000000
                                remaining = int(actual_value) % 10000000
                                actual_s = remaining // 100
                                actual_h = remaining % 100
                                print(
                                    f"\033[91mData verification failed at [{b},{s},{local_h}]:\033[0m"
                                )
                                print(
                                    f"  Expected: head_id={expected_head_id} -> {expected_value}"
                                )
                                print(f"  Got: head_id={actual_h} -> {actual_value}")
                            break
                    if not verification_passed:
                        break
                if not verification_passed:
                    break

            # Test 3: Round-trip verification
            pattern_output2 = all_to_all_balanced_4D(
                pattern_output1,
                scatter_idx=1,
                gather_idx=2,
                group=group,
                reallocated_head_idx_list=reallocated_head_idx_list,
                reallocated_head_num_list=reallocated_head_num_list,
            )

            round_trip_error = torch.abs(pattern_input - pattern_output2).max().item()
            round_trip_passed = round_trip_error < 1e-6

            if verification_passed and round_trip_passed:
                print(
                    f"[Rank {rank}] ✓ Passed (data_err={max_error:.2e}, rt_err={round_trip_error:.2e})"
                )
            else:
                print(f"[Rank {rank}] ✗ Failed", end="")
                if not verification_passed:
                    print(f" (data_err={max_error:.2e})", end="")
                if not round_trip_passed:
                    print(f" (rt_err={round_trip_error:.2e})", end="")
                print()

        except Exception as e:
            print(f"[Rank {rank}] ✗ Exception: {e}")
            if rank == 0:
                import traceback

                traceback.print_exc()

    # Main test execution
    print("Starting distributed all_to_all_balanced_4D test...")

    try:
        rank, world_size, local_rank, device = setup_distributed()

        # Initialize process group for multi-GPU testing
        if world_size > 1:
            group = dist.group.WORLD
            dist.barrier(device_ids=[local_rank])
        else:
            group = None

        if rank == 0:
            print(f"Testing with {world_size} processes")

        # Run comprehensive tests
        test_all_to_all_balanced_4D(rank, world_size, device, group)

        # Synchronize before cleanup
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
            if rank == 0:
                print("✅ All ranks completed tests successfully!")
        else:
            print("✅ Single process test completed!")

        # Clean up distributed environment
        if world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        if rank == 0:
            print(f"❌ Test execution failed: {e}")
            import traceback

            traceback.print_exc()
        if world_size > 1:
            try:
                dist.destroy_process_group()
            except:
                pass
        exit(1)
