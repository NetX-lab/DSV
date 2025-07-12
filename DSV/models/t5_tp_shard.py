import gc

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)

from .t5 import T5Embedder

device = "cuda"

dtype = torch.bfloat16

# # Initialize 8-way tensor parallel device mesh
# tp_mesh = init_device_mesh(device, (8,))

# torch.set_default_device(device+":"+str(dist.get_rank()))

# embedder = T5Embedder(device=device+":"+str(dist.get_rank()),torch_dtype=dtype)
# model = embedder.model


def convert_t5_to_tp(model, tp_mesh):
    """Transformers T5 model to TP shard model

    Args:
        model: Transformers T5 model
        tp_mesh: Tensor parallel device mesh

    Returns:
        sharded_model: TP shard model
    """
    # number heads 64
    # define parallelization strategy
    parallel_config = {}

    # configure parallelization strategy for all 24 blocks
    for i in range(24):
        block_prefix = f"encoder.block.{i}.layer."

        # parallelize attention layer
        parallel_config.update(
            {
                f"{block_prefix}0.SelfAttention.q": ColwiseParallel(),
                f"{block_prefix}0.SelfAttention.k": ColwiseParallel(),
                f"{block_prefix}0.SelfAttention.v": ColwiseParallel(),
                f"{block_prefix}0.SelfAttention.o": RowwiseParallel(),
                # parallelize feed forward layer
                f"{block_prefix}1.DenseReluDense.wi_0": ColwiseParallel(),
                f"{block_prefix}1.DenseReluDense.wi_1": ColwiseParallel(),
                f"{block_prefix}1.DenseReluDense.wo": RowwiseParallel(),
            }
        )

    # apply parallelization
    sharded_model = parallelize_module(model, tp_mesh, parallel_config)

    # adjust number of attention heads
    original_heads = model.encoder.block[0].layer[0].SelfAttention.n_heads
    size_in_group = dist.get_world_size(tp_mesh.get_group())
    heads_per_rank = original_heads // size_in_group

    rank_in_group = dist.get_rank(tp_mesh.get_group())
    print(
        f"global rank:{dist.get_rank()}; tp_mesh group rank:{rank_in_group}; tp_mesh ranks:{dist.get_process_group_ranks(tp_mesh.get_group())};T5 heads_per_rank:{heads_per_rank}"
    )
    for i in range(24):
        attn_layer = model.encoder.block[i].layer[0].SelfAttention
        attn_layer.n_heads = heads_per_rank
        attn_layer.n_kv_heads = attn_layer.n_heads  # in T5, n_heads equals n_kv_heads
        attn_layer.inner_dim = attn_layer.inner_dim // size_in_group

        if i == 0:
            # shard relative position encoding weights
            original_weight = attn_layer.relative_attention_bias.weight.data

            new_weight = original_weight[
                :, rank_in_group * heads_per_rank : (rank_in_group + 1) * heads_per_rank
            ]
            print(
                f"global rank:{dist.get_rank()}; tp_mesh group rank:{rank_in_group}; original_weight shape:{original_weight.shape}; new_weight shape:{new_weight.shape}"
            )
            attn_layer.relative_attention_bias.weight.data = new_weight

    return sharded_model
