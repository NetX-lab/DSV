import math
import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank_0(msg):
    if dist.get_rank() == 0:
        print(f"rank 0: {msg}")


class AlltoallTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, split_dim, gather_dim, context_parallel_group):
        ctx.split_dim = split_dim
        ctx.gather_dim = gather_dim
        ctx.context_parallel_group = context_parallel_group

        world_size = dist.get_world_size(context_parallel_group)
        rank = dist.get_rank(context_parallel_group)

        b, num_heads, seq_len, head_dim = x.shape

        split_dim_size = x.size(split_dim)
        gather_dim_size = x.size(gather_dim)

        print_rank_0(
            f"x.shape: {x.shape}, split_dim_size: {split_dim_size}, gather_dim_size: {gather_dim_size}"
        )
        assert split_dim_size % world_size == 0

        split_dim_size_after_all2all = split_dim_size // world_size
        gather_dim_size_after_all2all = gather_dim_size * world_size

        x_shape_list = [b, num_heads, seq_len, head_dim]

        world_size_dim = split_dim
        split_dim = world_size_dim + 1
        # the view_shape_list is the shape of the tensor after all-to-all operation
        view_shape_list = x_shape_list.copy()
        view_shape_list.insert(
            world_size_dim, world_size
        )  # insert world_size at split_dim
        view_shape_list[split_dim] = split_dim_size_after_all2all

        if gather_dim > world_size_dim:
            gather_dim = gather_dim + 1

        x = x.contiguous().view(view_shape_list)
        # permute the dim 0 and dim split_dim
        x = x.transpose(0, world_size_dim).contiguous()
        # world_size,b,num_heads//world_size,seq_len,head_dim
        output = torch.empty_like(x)
        # world_size,b,num_heads//world_size,seq_len,head_dim
        dist.all_to_all_single(output, x, group=context_parallel_group)

        # torch.cuda.synchronize()

        # output: world_size,b,num_heads//world_size,seq_len,head_dim -> b,num_heads//world_size,seq_len*world_size,head_dim
        dim_index_list = list(range(len(output.shape)))
        # world_size_dim to dim 0 and dim 0 to gather_dim
        dim_index_list.pop(world_size_dim)
        dim_index_list.pop(0)
        dim_index_list.insert(0, world_size_dim)
        if gather_dim > world_size_dim:
            gather_dim = gather_dim - 1
        dim_index_list.insert(gather_dim, 0)
        gather_dim = gather_dim + 1

        print_rank_0(
            f"output shape before permute: {output.shape}, dim_index_list: {dim_index_list},"
        )

        output = output.permute(dim_index_list).contiguous()

        view_shape_list = list(output.shape)

        world_size_dim = gather_dim - 1
        view_shape_list[gather_dim] = gather_dim_size_after_all2all
        view_shape_list.pop(world_size_dim)

        output = output.view(view_shape_list).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = AlltoallTranspose.apply(
            grad_output, ctx.gather_dim, ctx.split_dim, ctx.context_parallel_group
        )

        return grad_output, None, None, None


class AlltoallTranspose_load_balance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        split_dim,
        gather_dim,
        head_dim_permuted_list,
        head_dim_num_list,
        context_parallel_group,
    ):
        ctx.split_dim = split_dim
        ctx.gather_dim = gather_dim
        ctx.context_parallel_group = context_parallel_group
        ctx.head_dim_num_list = head_dim_num_list

        world_size = dist.get_world_size(context_parallel_group)
        rank = dist.get_rank(context_parallel_group)

        b, num_heads, seq_len, head_dim = x.shape

        # x = x[:,ctx.head_dim_permuted_list,:,:]

        split_dim_size = x.size(split_dim)
        gather_dim_size = x.size(gather_dim)

        bs_dim = 0

        print_rank_0(
            f"x is contiguous: {x.is_contiguous()}, x.shape: {x.shape}, split_dim_size: {split_dim_size}, gather_dim_size: {gather_dim_size}"
        )

        x = x.transpose(bs_dim, split_dim).contiguous()

        num_heads_this_rank = ctx.head_dim_num_list[rank]

        output_shape = [num_heads_this_rank * world_size, b, seq_len, head_dim]

        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        # world_size,b,num_heads//world_size,seq_len,head_dim

        input_list_this_rank = ctx.head_dim_num_list
        output_list_this_rank = [num_heads_this_rank for i in range(world_size)]

        ctx.output_list_this_rank = output_list_this_rank
        ctx.input_list_this_rank = input_list_this_rank

        dist.all_to_all_single(
            output,
            x,
            group=context_parallel_group,
            output_list=output_list_this_rank,
            input_list=input_list_this_rank,
        )

        # torch.cuda.synchronize()

        # output: num_heads_this_rank,b,seq_len,head_dim -> b,num_heads//world_size,seq_len*world_size,head_dim

        output = output.transpose(bs_dim, split_dim).contiguous()

        print(f"rank {rank}; output.shape: {output.shape}")

        b, num_heads, seq_len, head_dim = output.shape

        output = (
            output.view(b, num_heads // world_size, world_size, seq_len, head_dim)
            .view(b, num_heads, seq_len * world_size, head_dim)
            .contiguous()
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_list_this_rank_bw = ctx.output_list_this_rank
        output_list_this_rank_bw = ctx.input_list_this_rank

        bs, num_heads, seq_len, head_dim = grad_output.shape

        bs_dim = 0
        num_heads_dim = 1
        seq_len_dim = 2

        seq_len_per_chunk = seq_len // world_size

        grad_output = grad_output.view(
            bs, num_heads, world_size, seq_len // world_size, head_dim
        ).view(bs, num_heads * world_size, seq_len // world_size, head_dim)

        grad_output = grad_output.transpose(bs_dim, num_heads_dim)

        output_shape = [sum(output_list_this_rank_bw), b, seq_len_per_chunk, head_dim]

        output = torch.empty(
            output_shape, dtype=grad_output.dtype, device=grad_output.device
        )

        dist.all_to_all_single(
            output,
            grad_output,
            group=ctx.context_parallel_group,
            output_list=output_list_this_rank_bw,
            input_list=input_list_this_rank_bw,
        )

        output = output.transpose(bs_dim, num_heads_dim).contiguous()

        # recover the head_dim according to the head_dim_permuted_list

        return grad_, None, None, None, None


class UlysessAttention(nn.Module):
    def forward(
        self,
        q,
        k,
        v,
        scale,
        dropout_p=0.0,
        is_causal=False,
        attn_mask=None,
        context_parallel_group=dist.group.WORLD,
    ):
        world_size = dist.get_world_size(context_parallel_group)

        bs, num_heads, seq_len, head_dim = q.shape

        q_alltoall = AlltoallTranspose.apply(q, 1, 2, context_parallel_group)
        k_alltoall = AlltoallTranspose.apply(k, 1, 2, context_parallel_group)
        v_alltoall = AlltoallTranspose.apply(v, 1, 2, context_parallel_group)

        output = torch.nn.functional.scaled_dot_product_attention(
            q_alltoall,
            k_alltoall,
            v_alltoall,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        output = AlltoallTranspose.apply(output, 2, 1, context_parallel_group)
        # output=output.reshape(bs,num_heads//world_size,world_size,seq_len,head_dim)
        # output=output.permute(2,0,1,3,4).contiguous() # world_size,b,num_heads//world_size,seq_len,head_dim

        # output_alltoall=torch.empty_like(output)
        # dist.all_to_all_single(output_alltoall,output,group=context_parallel_group)

        # output=output_alltoall.permute(1,0,2,3,4).contiguous() # b,world_size,num_heads//world_size,seq_len,head_dim
        # output=output.reshape(bs,num_heads,seq_len,head_dim)

        return output


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dtype = torch.float32

    num_heads = 16
    dim = num_heads * 256
    seq_len = 256
    bs = 10
    # init the model

    print(
        f"rank: {rank}, world_size: {world_size} num_heads: {num_heads} head_dim: {dim//num_heads} seq_len: {seq_len}"
    )

    model = DDP(
        MultiHeadAttention(num_heads=num_heads, dim=dim).to(f"cuda:{rank}", dtype=dtype)
    )

    # init the data
    q = torch.randn(bs, seq_len, dim, dtype=dtype, device=f"cuda:{rank}")
    k = torch.randn(bs, seq_len, dim, dtype=dtype, device=f"cuda:{rank}")
    v = torch.randn(bs, seq_len, dim, dtype=dtype, device=f"cuda:{rank}")

    # broadcast the data to all ranks
    dist.broadcast(q, 0)
    dist.broadcast(k, 0)
    dist.broadcast(v, 0)
    torch.cuda.synchronize()

    # split the data into world_size parts in the dim0
    q_split = q.chunk(world_size, dim=1)[rank]
    k_split = k.chunk(world_size, dim=1)[rank]
    v_split = v.chunk(world_size, dim=1)[rank]

    # forward
    output_cp = model(q_split, k_split, v_split, cp_enable=True)

    output_no_cp = model(q, k, v, cp_enable=False).chunk(world_size, dim=1)[rank]

    # print(output_cp)
    # print(output_no_cp)
    print(f"rank {rank}; cp: {torch.allclose(output_cp,output_no_cp)}")

    output_cp = output_cp.reshape(-1)
    output_no_cp = output_no_cp.reshape(-1)
    # return the max diff and the diff ratio of that position
    max_diff = torch.max(torch.abs(output_cp - output_no_cp))
    max_diff_index = torch.argmax(torch.abs(output_cp - output_no_cp))
    max_diff_ratio = (
        torch.abs(output_cp - output_no_cp)[max_diff_index]
        / torch.abs(output_cp)[max_diff_index]
    )
    print(
        f"rank {rank}; max_diff: {max_diff}, max_diff_index: {max_diff_index}, max_diff_ratio: {max_diff_ratio}"
    )
    # the max diff postions's data in the output_cp and output_no_cp
    max_diff_data_cp = output_cp[max_diff_index]
    max_diff_data_no_cp = output_no_cp[max_diff_index]
    print(
        f"rank {rank}; max_diff_data_cp: {max_diff_data_cp}, max_diff_data_no_cp: {max_diff_data_no_cp}"
    )

    # plot the error ratio distribution for rank 0 and save it as png
    if rank == 0:
        error_ratio = torch.abs(output_cp - output_no_cp) / torch.abs(output_cp)
        error_ratio = error_ratio.detach().cpu().numpy()
        plt.hist(error_ratio, bins=10)
        plt.savefig(f"error_ratio_distribution_rank_{rank}.png")
        plt.close()

    dist.destroy_process_group()
