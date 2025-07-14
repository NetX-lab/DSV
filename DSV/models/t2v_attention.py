import math
import sys
from functools import lru_cache, partial
from importlib import import_module
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import einsum, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import DSV.models.global_variable as global_variable
from DSV.models.all_to_all import SeqAllToAll4D
from DSV.triton_plugin.fused_attention_no_causal import \
    attention as triton_no_causal_attention


def get_profile_step(current_step):
    if current_step <= 2000:
        return 25
    else:
        return 100


class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        block_id: str,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        window_based_dict: Optional[dict] = None,
        low_rank_dict: Optional[dict] = None,
    ):
        super().__init__()
        self.block_id = block_id
        self.inner_dim = dim_head * heads

        self.is_cross_attn = cross_attention_dim is not None

        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
            )
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim
            )
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=cross_attention_norm_num_groups,
                eps=1e-5,
                affine=True,
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        # if USE_PEFT_BACKEND:
        #     linear_cls = nn.Linear
        # else:
        #     linear_cls = LoRACompatibleLinear

        linear_cls = nn.Linear

        self.to_q = linear_cls(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = linear_cls(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = linear_cls(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = linear_cls(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = linear_cls(added_kv_proj_dim, self.inner_dim)

        if global_variable.TP_ENABLE == False:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, query_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        self.low_rank_dict = low_rank_dict

        if self.low_rank_dict is not None and self.is_cross_attn == False:
            self.low_rank_head_dim = 16
            self.low_rank_attn_dim = self.low_rank_head_dim * heads

            self.low_rank_qk_proj = nn.Linear(
                dim_head * heads, heads * self.low_rank_head_dim * 2
            )

            self.mse_loss = nn.MSELoss()
            self.cosine_loss = nn.CosineEmbeddingLoss()

        print(f"window_based_dict: {window_based_dict}; low_rank_dict: {low_rank_dict}")

        if processor is None:
            processor = AttnProcessor2_0(window_based_dict, low_rank_dict)

        self.set_processor(processor)

        self.register_buffer("sparsity", torch.tensor(0.0), persistent=True)
        self.register_buffer(
            "sparsity_per_head", torch.zeros(self.heads), persistent=True
        )

        self.disaggregated = (
            self.low_rank_dict["disaggregated"]
            if self.low_rank_dict is not None
            else False
        )

    def calculate_low_rank_loss(self, hidden_states, query, key):
        # Calculate low rank attention loss
        hidden_states = hidden_states
        query = query
        key = key

        B, N, C = hidden_states.shape

        mse_loss_weight = self.low_rank_dict["mse_loss_weight"]
        cosine_loss_weight = self.low_rank_dict["cosine_loss_weight"]

        sampled_interval_across_bs = 40
        sampled_indices = torch.arange(0, N, sampled_interval_across_bs)
        sampled_length = len(sampled_indices)

        qk_low_rank = (
            self.low_rank_qk_proj(hidden_states)
            .contiguous()
            .view(B, N, self.heads, self.low_rank_head_dim * 2)
            .transpose(1, 2)
        )

        q_low_rank, k_low_rank = qk_low_rank.chunk(2, dim=-1)

        q_low_rank = q_low_rank[:, :, sampled_indices, :]

        q_ref = query[:, :, sampled_indices, :]
        k_ref = key

        qk_low_rank = torch.matmul(
            q_low_rank, k_low_rank.transpose(-1, -2)
        ) / math.sqrt(q_low_rank.size(-1))
        qk_ref = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(
            q_ref.size(-1)
        )

        # Calculate MSE loss between low rank and reference attention maps
        qk_low_rank_norm = F.normalize(qk_low_rank, p=2, dim=-1)
        qk_ref_norm = F.normalize(qk_ref, p=2, dim=-1)

        if (
            global_variable.CURRENT_STEP % 200 == 0
            and global_variable.RANK == 0
            and global_variable.CURRENT_STEP < global_variable.LOW_RANK_STAGE0_STEPS
        ):
            k = 64
            _, topk_index_ref = qk_ref.detach().topk(k=k, dim=-1)
            _, topk_index = qk_low_rank.detach().topk(k=k, dim=-1)
            topk_index = topk_index.contiguous().view(-1, k)
            topk_index_ref = topk_index_ref.contiguous().view(-1, k)

            overlap_ratios = []

            for i in range(0, topk_index.size(0), 64):
                row_overlap = (
                    torch.sum(torch.isin(topk_index[i], topk_index_ref[i])).float() / k
                )
                overlap_ratios.append(row_overlap)

            overlap_ratio = torch.stack(overlap_ratios).mean()

            if global_variable.RANK == 0:
                print(
                    f"non-low-rank stage block_id:{self.block_id}, topk_index:{topk_index[:1]}, topk_index_ref:{topk_index_ref[:1]}, low_rank topk and ref topk overlap ratio:{overlap_ratio:.4f}"
                )

            del topk_index, topk_index_ref, overlap_ratios

        mse_loss = self.mse_loss(qk_low_rank_norm, qk_ref_norm)

        target = torch.ones(
            qk_low_rank.size(0) * qk_low_rank.size(1) * qk_low_rank.size(2),
            dtype=torch.long,
            device=qk_low_rank.device,
        )

        cosine_loss = self.cosine_loss(
            qk_low_rank.contiguous().view(-1, qk_low_rank.size(-1)),
            qk_ref.contiguous().view(-1, qk_ref.size(-1)),
            target,
        )

        loss = mse_loss * mse_loss_weight + cosine_loss * cosine_loss_weight

        if (
            global_variable.CURRENT_STEP % global_variable.ATTENTION_LOG_STEP == 0
            and global_variable.RANK == 0
        ):
            print(
                f"block_id: {self.block_id}, low_rank_loss: {float(loss.item()):.4f}, mse_loss: {float(mse_loss.item()):.4f}, cosine_loss: {float(cosine_loss.item()):.4f}"
            )

        return loss

    def profile_sparsity(self, query, key, sum_score_threshold=0.99):
        # Profile attention sparsity
        key_T = key.transpose(-1, -2)

        with torch.no_grad():
            # Compute attention in chunks to avoid OOM
            chunk_size = 1024
            B, H, S, D = query.shape
            sparsity_chunks = []

            for i in range(0, S, chunk_size):
                chunk_end = min(i + chunk_size, S)
                query_chunk = query[:, :, i:chunk_end, :]

                atten_score = (query_chunk @ key_T).float().softmax(-1)
                atten_score, _ = torch.sort(atten_score, dim=-1, descending=True)

                atten_score.cumsum_(dim=-1)

                mask = atten_score >= sum_score_threshold
                chunk_sparsity = mask.sum(-1, keepdim=True).float()
                chunk_sparsity.div_(mask.size(-1))
                sparsity_chunks.append(chunk_sparsity)

                del atten_score, mask

            sparsity_per_query = torch.cat(sparsity_chunks, dim=2)
            sparsity_per_query = sparsity_per_query.squeeze(-1)

            sparsity_flat = sparsity_per_query.view(-1)

            n = len(sparsity_flat)

            sparsity_list = [sparsity_flat.min()]

            percentiles = [0.001, 0.005, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.95]
            k_values = [max(1, int(p * n)) for p in percentiles]

            # Use kthvalue to compute percentiles
            for k in k_values:
                val = torch.kthvalue(sparsity_flat, k).values
                sparsity_list.append(val)

            sparsity_per_head = (
                sparsity_per_query.permute(1, 0, 2).contiguous().view(self.heads, -1)
            )

            sparsity_per_head_list = [
                sparsity_per_head.min(dim=1)[0],
            ]

            percentiles = [0.001, 0.005, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.95]
            k_values = [max(1, int(p * sparsity_per_head.size(1))) for p in percentiles]

            for k in k_values:
                val = torch.kthvalue(sparsity_per_head, k).values
                sparsity_per_head_list.append(val)

            old_sparsity = None
            old_sparsity_per_head = None

            if self.sparsity == 0:
                self.sparsity = sparsity_list[0]
                self.sparsity_per_head = sparsity_per_head_list[0]
            else:
                momentum = 0.9
                current_min = sparsity_list[0]
                old_sparsity = self.sparsity
                self.sparsity = momentum * old_sparsity + (1 - momentum) * current_min

                old_sparsity_per_head = self.sparsity_per_head
                self.sparsity_per_head = (
                    momentum * old_sparsity_per_head
                    + (1 - momentum) * sparsity_per_head_list[0]
                )

            if global_variable.RANK == 0:
                print(
                    f"block_id:{self.block_id}, min:{sparsity_list[0]}, P0.1:{sparsity_list[1]}, P0.5:{sparsity_list[2]}, P1:{sparsity_list[3]}, P2:{sparsity_list[4]}, P5:{sparsity_list[5]}, P25:{sparsity_list[6]}, P50:{sparsity_list[7]}, P75:{sparsity_list[8]}, P95:{sparsity_list[9]}"
                )
                print(
                    f"block_id:{self.block_id}, sparsity_per_head min:{sparsity_per_head_list[0]}, P0.1:{sparsity_per_head_list[1]}, P0.5:{sparsity_per_head_list[2]}, P1:{sparsity_per_head_list[3]}, P2:{sparsity_per_head_list[4]}, P5:{sparsity_per_head_list[5]}, P25:{sparsity_per_head_list[6]}, P50:{sparsity_per_head_list[7]}, P75:{sparsity_per_head_list[8]}, P95:{sparsity_per_head_list[9]}"
                )
                print(
                    f"block_id:{self.block_id}, sparsity before:{old_sparsity}, sparsity new:{self.sparsity}"
                )
                print(
                    f"block_id:{self.block_id}, sparsity_per_head before:{old_sparsity_per_head}, "
                    f"sparsity_per_head new:{self.sparsity_per_head}"
                )

            del sparsity_chunks, sparsity_per_query, sparsity_flat, sparsity_per_head

            return self.sparsity

    def low_rank_predict_and_computation(self, hidden_states, query, key, value):
        B, N, C = hidden_states.shape
        B, H, N, D = query.shape

        with torch.no_grad():
            qk_low_rank = (
                self.low_rank_qk_proj(hidden_states)
                .contiguous()
                .view(B, N, self.heads, self.low_rank_head_dim * 2)
                .transpose(1, 2)
            )

            q_low_rank, k_low_rank = qk_low_rank.chunk(2, dim=-1)

            qk_low_rank = torch.matmul(q_low_rank, k_low_rank.transpose(-1, -2))

            del q_low_rank, k_low_rank

            k = max(int(N * (1 - self.sparsity)), int(N * 0.025))

            # Make k the nearest multiple of 8
            k = math.ceil(k / 8) * 8

            topk, topk_index = qk_low_rank.topk(k=k, dim=-1)

            del qk_low_rank

            topk_index_shape = topk_index.shape

            if global_variable.CURRENT_STEP % 200 == 0 and global_variable.RANK == 0:
                qk_ref = torch.matmul(query.detach(), key.detach().transpose(-1, -2))
                top_value_ref, topk_index_ref = qk_ref.topk(k=k, dim=-1)

                del top_value_ref

                topk_index = topk_index.contiguous().view(-1, k)
                topk_index_ref = topk_index_ref.contiguous().view(-1, k)

                overlap_ratios = []

                for i in range(0, topk_index.size(0), 64):
                    row_overlap = (
                        torch.sum(torch.isin(topk_index[i], topk_index_ref[i])).float()
                        / k
                    )
                    overlap_ratios.append(row_overlap)

                overlap_ratio = torch.stack(overlap_ratios).mean()

                if global_variable.RANK == 0:
                    print(
                        f"block_id:{self.block_id}, topk shape:{topk_index.shape}, low_rank topk and ref topk overlap ratio:{overlap_ratio:.4f}; low_rank topk:{topk_index[:1]}; ref topk:{topk_index_ref[:1]}"
                    )

                del topk_index_ref, overlap_ratios, overlap_ratio, qk_ref

            topk_index = topk_index.view(topk_index_shape)

            mask = torch.zeros(
                (B, self.heads, N, N), dtype=torch.bool, device=hidden_states.device
            )

            mask.scatter_(-1, topk_index, True)

        del topk, topk_index

        # SDPA with mask
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
            )

        del mask

        return output

    def set_processor(
        self, processor: "AttnProcessor", _remove_lora: bool = False
    ) -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
            _remove_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to remove LoRA layers from the model.
        """
        # if not USE_PEFT_BACKEND and hasattr(self, "processor") and _remove_lora and self.to_q.lora_layer is not None:
        #     deprecate(
        #         "set_processor to offload LoRA",
        #         "0.26.0",
        #         "In detail, removing LoRA layers via calling `set_default_attn_processor` is deprecated. Please make sure to call `pipe.unload_lora_weights()` instead.",
        #     )
        #     # TODO(Patrick, Sayak) - this can be deprecated once PEFT LoRA integration is complete
        #     # We need to remove all LoRA layers
        #     # Don't forget to remove ALL `_remove_lora` from the codebase
        #     for module in self.modules():
        #         if hasattr(module, "set_lora_layer"):
        #             module.set_lora_layer(None)

        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(
                f"You are removing possibly trained weights of {self.processor} with {processor}"
            )
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(
        self, return_deprecated_lora: bool = False
    ) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def _save_attention_map(self, hidden_states):
        with torch.no_grad():
            B, N, C = hidden_states.shape
            query = (
                self.to_q(hidden_states)
                .view(B, N, self.heads, self.inner_dim // self.heads)
                .transpose(1, 2)
                .to(torch.bfloat16)
            )
            key = (
                self.to_k(hidden_states)
                .view(B, N, self.heads, self.inner_dim // self.heads)
                .transpose(1, 2)
                .to(torch.bfloat16)
            )
            value = (
                self.to_v(hidden_states)
                .view(B, N, self.heads, self.inner_dim // self.heads)
                .transpose(1, 2)
                .to(torch.bfloat16)
            )

            sample_s = max(1, query.shape[-2] // 16)
            total_s = query.shape[-2]
            indices = torch.randperm(total_s)[:sample_s]

            query_idx = indices

            query = query[:, :, indices, :]

            attn_map = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
                self.inner_dim // self.heads
            )

            accum_threshold = 0.90

            attn_map_original = attn_map.to(torch.float16).softmax(-1).to(torch.float32)

            block_id = int(self.block_id.split("_")[-1])

            attn_map = attn_map_original.sort(dim=-1, descending=True)[0]

            # Compute accumulated sum
            attn_map_accum = attn_map.cumsum(dim=-1)

            # Find elements larger than threshold
            attn_map_accum_threshold = attn_map_accum > accum_threshold

            # Compute sparsity ratio
            attn_map_sparsity_ratio = (
                attn_map_accum_threshold.sum(dim=-1)
                / attn_map_accum_threshold.shape[-1]
            )

            # Compute mean sparsity ratio
            attn_map_sparsity_ratio_mean = attn_map_sparsity_ratio.mean(dim=-1).mean(
                dim=0
            )

            print(
                f"block_id:{self.block_id}, attn_map_sparsity_ratio_mean:{attn_map_sparsity_ratio_mean.cpu().numpy().tolist()}"
            )

            ckpt_path = global_variable.ATTENTION_MAP_SAVE_METADATA["ckpt"]
            save_attn_map_path = global_variable.ATTENTION_MAP_SAVE_METADATA[
                "save_attn_map_path"
            ]
            iteration = ckpt_path.split("/")[-1].split(".")[0]
            iteration = iteration.lstrip("0")
            iteration = int(iteration)

            saved_sample = sample_s // 8

            if (
                "saved_attention_map_tensors"
                not in global_variable.ATTENTION_MAP_SAVE_METADATA
            ):
                global_variable.ATTENTION_MAP_SAVE_METADATA[
                    "saved_attention_map_tensors"
                ] = torch.zeros(
                    (32, 1, 16, saved_sample, total_s),
                    device=hidden_states.device,
                    dtype=torch.float16,
                )
                global_variable.ATTENTION_MAP_SAVE_METADATA[
                    "saved_attention_map_query_ids"
                ] = torch.zeros(
                    (32, saved_sample), device=hidden_states.device, dtype=torch.int32
                )
                global_variable.ATTENTION_MAP_SAVE_METADATA[
                    "saved_attention_sparsity_ratio_tensors"
                ] = torch.zeros(
                    (32, 16), device=hidden_states.device, dtype=torch.float16
                )

            # Save attention score
            global_variable.ATTENTION_MAP_SAVE_METADATA["saved_attention_map_tensors"][
                block_id, 0, :, :saved_sample, :
            ] = attn_map_original[0, :, :saved_sample, :]
            global_variable.ATTENTION_MAP_SAVE_METADATA[
                "saved_attention_map_query_ids"
            ][block_id, :] = query_idx[:saved_sample]
            global_variable.ATTENTION_MAP_SAVE_METADATA[
                "saved_attention_sparsity_ratio_tensors"
            ][block_id, :] = attn_map_sparsity_ratio_mean

            if block_id == 31:
                import os

                print(
                    f"\033[91mSave the attention score and sparsity ratio of iteration {iteration}\033[0m"
                )
                save_path = os.path.join(
                    save_attn_map_path, f"atten_sparsity_pattern_iter_{iteration}.pt"
                )

                save_dict = {
                    "attn_score_for_sampled_queries": global_variable.ATTENTION_MAP_SAVE_METADATA[
                        "saved_attention_map_tensors"
                    ]
                    .cpu()
                    .numpy(),
                    "query_ids_for_sampled_queries": global_variable.ATTENTION_MAP_SAVE_METADATA[
                        "saved_attention_map_query_ids"
                    ]
                    .cpu()
                    .numpy(),
                    "sparsity_ratio": global_variable.ATTENTION_MAP_SAVE_METADATA[
                        "saved_attention_sparsity_ratio_tensors"
                    ]
                    .cpu()
                    .numpy(),
                }
                torch.save(save_dict, save_path)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        if (
            self.is_cross_attn == False
            and global_variable.ATTENTION_MAP_SAVE_PATH != None
        ):
            self._save_attention_map(hidden_states)

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor

    def get_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        # print(f"attention_mask: {attention_mask.shape}")

        return attention_mask

    def norm_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, window_based_dict=None, low_rank_dict=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.window_attn_mask = None
        self.grid_size = None
        self.window_size = None

        self.block_mask = None

        self.device = None

        self.window_based = False

        self.window_based_dict = window_based_dict
        self.low_rank_dict = low_rank_dict

        if self.window_based_dict is not None:
            # self.__init_block_mask(window_based_dict)
            self.window_based = True
            self.get_attn_mask_for_local_window(
                math.prod(self.window_based_dict["grid_size"]),
                self.window_based_dict["grid_size"],
                self.window_based_dict["window_size"],
                torch.bfloat16,
                torch.cuda.current_device(),
            )

        self.low_rank_based = False

        if self.low_rank_dict is not None:
            self.low_rank_based = True

    def get_attn_mask_for_local_window(
        self, seq_len, grid_size, window_size, dtype, device
    ):
        # if self.window_attn_mask != None:
        if global_variable.WINDOW_ATTN_MASK is not None:
            return global_variable.WINDOW_ATTN_MASK

        frames, height, width = grid_size
        window_f, window_h, window_w = window_size
        mask = torch.full(
            (seq_len, seq_len), 0, dtype=torch.bool, device=device
        ).requires_grad_(False)
        mask = mask.contiguous().view(seq_len, frames, height, width)

        for q in range(seq_len):
            query_f, query_h, query_w = (
                q // (width * height),
                (q % (width * height)) // width,
                q % width,
            )
            # get the index of the local window for each query  and complete this without loop
            window_f_start = max(0, query_f - window_f // 2)
            window_f_end = min(frames, query_f + window_f // 2 + 1)
            window_h_start = max(0, query_h - window_h // 2)
            window_h_end = min(height, query_h + window_h // 2 + 1)
            window_w_start = max(0, query_w - window_w // 2)
            window_w_end = min(width, query_w + window_w // 2 + 1)

            mask[
                q,
                window_f_start:window_f_end,
                window_h_start:window_h_end,
                window_w_start:window_w_end,
            ].fill_(True)

        # self.window_attn_mask=mask.contiguous().view(seq_len,seq_len)
        global_variable.WINDOW_ATTN_MASK = mask.contiguous().view(seq_len, seq_len)
        return global_variable.WINDOW_ATTN_MASK

    @lru_cache
    def create_block_mask_cached(self, score_mod, B, H, M, N, device):
        block_mask = create_block_mask(score_mod, B, H, M, N, device)
        return block_mask

    def window_mask_3d(self, b, h, q_idx, kv_idx):
        grid_size, window_size = self.grid_size, self.window_size
        frames, height, width = grid_size
        window_f, window_h, window_w = window_size

        # Convert 1D index to 3D coordinates
        # q_idx: [..., seq_len]
        q_f = q_idx // (height * width)
        q_hw = q_idx % (height * width)
        q_h = q_hw // width
        q_w = q_hw % width

        # kv_idx: [..., seq_len]
        kv_f = kv_idx // (height * width)
        kv_hw = kv_idx % (height * width)
        kv_h = kv_hw // width
        kv_w = kv_hw % width

        # Compute distance in each dimension
        f_dist = torch.abs(q_f - kv_f)
        h_dist = torch.abs(q_h - kv_h)
        w_dist = torch.abs(q_w - kv_w)

        # Check if within window range
        within_f = f_dist <= window_f // 2
        within_h = h_dist <= window_h // 2
        within_w = w_dist <= window_w // 2

        # Only return True when all three dimensions are within the window
        return within_f & within_h & within_w

    def __init_block_mask(self, window_based_dict, device):
        self.grid_size = window_based_dict["grid_size"]
        self.window_size = window_based_dict["window_size"]
        if self.block_mask is None:
            sequence_length = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
            # print(f"recompute the block mask")
            self.block_mask = self.create_block_mask_cached(
                self.window_mask_3d, 1, 1, sequence_length, sequence_length, device
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        if self.device is None:
            self.device = hidden_states.device

        # residual = hidden_states

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # args = () if USE_PEFT_BACKEND else (scale,)
        args = ()

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Context parallel all-to-all
        if global_variable.CP_ENABLE and attn.is_cross_attn == False:
            if global_variable.TP_ENABLE:
                tp_sp_sequence_length = sequence_length * dist.get_world_size(
                    global_variable.TENSOR_PARALLEL_GROUP
                )
            else:
                tp_sp_sequence_length = sequence_length

            query = query.view(batch_size, tp_sp_sequence_length, attn.heads, head_dim)
            key = key.view(batch_size, tp_sp_sequence_length, attn.heads, head_dim)
            value = value.view(batch_size, tp_sp_sequence_length, attn.heads, head_dim)

            query = SeqAllToAll4D.apply(
                global_variable.CONTEXT_PARALLEL_GROUP, query, 2, 1
            )
            key = SeqAllToAll4D.apply(global_variable.CONTEXT_PARALLEL_GROUP, key, 2, 1)
            value = SeqAllToAll4D.apply(
                global_variable.CONTEXT_PARALLEL_GROUP, value, 2, 1
            )

        # Window-based  attention in algorithm 
        if self.window_based:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            grid_size = self.window_based_dict["grid_size"]
            window_size = self.window_based_dict["window_size"]

            mask = self.get_attn_mask_for_local_window(
                sequence_length, grid_size, window_size, query.dtype, query.device
            )

            with torch.nn.attention.sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                hidden_states = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
                )  

        # Low-rank attention
        elif self.low_rank_based:
            head_reduction_ratio = dist.get_world_size(global_variable.CONTEXT_PARALLEL_GROUP) 

            query = query.view(batch_size, -1, attn.heads//head_reduction_ratio, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads//head_reduction_ratio, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads//head_reduction_ratio, head_dim).transpose(1, 2)

            if attn.training and global_variable.TEST_LARGE_SCALE == False:
                if attn.disaggregated == True:
                    low_rank_loss = (
                        global_variable.LOW_RANK_MODULE.compute_low_rank_loss(
                            hidden_states.detach(),
                            query.detach(),
                            key.detach(),
                            int(attn.block_id.split("_")[-1]),
                        )
                    )
                elif attn.disaggregated == False:
                    low_rank_loss = attn.calculate_low_rank_loss(
                        hidden_states.detach(), query.detach(), key.detach()
                    )

                if (
                    global_variable.FORWARD_DONE == False
                    and attn.disaggregated == False
                ):
                    low_rank_loss.backward()
               
            # Profile sparsity
            if (
                global_variable.CURRENT_STEP
                % get_profile_step(global_variable.CURRENT_STEP)
                == 0
                and global_variable.TEST_LARGE_SCALE == False
            ):
                if attn.disaggregated:
                    if global_variable.FORWARD_DONE == False:
                        global_variable.LOW_RANK_MODULE.profile_sparsity(
                            query.detach(),
                            key.detach(),
                            int(attn.block_id.split("_")[-1]),
                        )
                else:
                    attn.profile_sparsity(query.detach(), key.detach())

            # Compute attention
            if global_variable.CURRENT_STEP >= global_variable.LOW_RANK_STAGE0_STEPS:
                if attn.disaggregated == True:
                    if global_variable.LOW_RANK_INFERENCE == True:
                        hidden_states = global_variable.LOW_RANK_MODULE.compute_attention_in_inference(
                            hidden_states,
                            query,
                            key,
                            value,
                            int(attn.block_id.split("_")[-1]),
                        )

                    elif global_variable.TEST_LARGE_SCALE == True:
                        hidden_states = global_variable.LOW_RANK_MODULE.compute_attention_in_training_dummy(
                            hidden_states,
                            query,
                            key,
                            value,
                            int(attn.block_id.split("_")[-1]),
                        )

                    else: # algorithic test
                        mask = global_variable.LOW_RANK_MODULE.get_kv_index_mask(
                            hidden_states.detach(),
                            query.detach(),
                            key.detach(),
                            int(attn.block_id.split("_")[-1]),
                        )

                        with torch.nn.attention.sdpa_kernel(
                            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                        ):
                            hidden_states = F.scaled_dot_product_attention(
                                query,
                                key,
                                value,
                                dropout_p=0.0,
                                attn_mask=mask,
                                is_causal=False,
                            )
                        del mask

                else:
                    hidden_states = attn.low_rank_predict_and_computation(
                        hidden_states.detach(), query, key, value
                    )

            else:
                with torch.nn.attention.sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                ):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )

        else:
            if global_variable.TRITON_ATTENTION == True and attn.is_cross_attn == False:
                head_reduction_ratio = dist.get_world_size(global_variable.CONTEXT_PARALLEL_GROUP) 
                query = query.view(batch_size, -1, attn.heads//head_reduction_ratio, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads//head_reduction_ratio, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads//head_reduction_ratio, head_dim).transpose(1, 2)

                hidden_states = triton_no_causal_attention(
                    query, key, value, False, attn.scale
                )
            else:
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

        if global_variable.CP_ENABLE and attn.is_cross_attn == False:
            # All_to_All to the context parallel group
            hidden_states = SeqAllToAll4D.apply(
                global_variable.CONTEXT_PARALLEL_GROUP,
                hidden_states.transpose(1, 2),
                1,
                2,
            )
            hidden_states = hidden_states.contiguous().view(
                batch_size, -1, attn.heads * head_dim
            )

        else:
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )

        # Linear projection
        if isinstance(attn.to_out, nn.ModuleList):
            hidden_states = attn.to_out[0](hidden_states, *args)
            hidden_states = attn.to_out[1](hidden_states)
        else:
            hidden_states = attn.to_out(hidden_states, *args)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # # would not enter this branch
        # if attn.residual_connection:
        #     hidden_states = hidden_states + residual

        # hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
