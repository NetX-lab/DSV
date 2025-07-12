# Modified from Latte project (https://github.com/Vchitect/Latte)
# Licensed under Apache License 2.0

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import Transformer2DModel
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.embeddings import (CombinedTimestepLabelEmbeddings,
                                         ImagePositionalEmbeddings, PatchEmbed,
                                         SinusoidalPositionalEmbedding,
                                         get_1d_sincos_pos_embed_from_grid)
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange, repeat

try:
    from diffusers.models.embeddings import (CaptionProjection,
                                             CombinedTimestepSizeEmbeddings)
except:
    from diffusers.models.embeddings import (
        PixArtAlphaCombinedTimestepSizeEmbeddings as CombinedTimestepSizeEmbeddings,
    )

import torch
import torch.distributed as dist
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm
from torch import nn
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)

import DSV.models.global_variable as global_variable
from DSV.models.t2v_attention import Attention


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(
            self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype))
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = CombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(
            timestep,
            **added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = (
            x
            + self.alpha_attn.tanh()
            * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        )
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = (
            LoRACompatibleLinear
            if (not USE_PEFT_BACKEND and global_variable.TP_ENABLE == False)
            else nn.Linear
        )

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        if global_variable.TP_ENABLE == False:
            self.net = nn.ModuleList([])
            # project in
            self.net.append(act_fn)
            # project dropout
            self.net.append(nn.Dropout(dropout))
            # project out
            self.net.append(linear_cls(inner_dim, dim_out))
            # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
            if final_dropout:
                print(f"Final dropout is enabled in FeedForward!!!")
                self.net.append(nn.Dropout(dropout))
        else:
            self.net = nn.ModuleList([])
            self.net.append(act_fn)
            self.net.append(nn.Dropout(dropout))

            self.out = nn.Linear(inner_dim, dim_out)

            if final_dropout:
                print(f"Final dropout is enabled in FeedForward!!!")
                self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU,) if USE_PEFT_BACKEND else (GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)

        if global_variable.TP_ENABLE == True:
            hidden_states = self.out(hidden_states)

        return hidden_states


def _chunked_feed_forward(
    ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int
):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        bid: int,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        fused_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.id = bid

        self.only_cross_attention = only_cross_attention

        # self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        # self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_zero = False
        self.use_ada_layer_norm = False

        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"

        self.use_layer_norm = False  # norm_type == "layer_norm"

        assert (
            self.use_ada_layer_norm_single == True
        ), "Only ada_norm_single is supported for now"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        # if self.use_ada_layer_norm:
        #     self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        # elif self.use_ada_layer_norm_zero:
        #     self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        # else:
        # to replace with apex fused layernorm

        if fused_layer_norm:
            self.norm1 = ApexFusedLayerNorm(
                dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
            )
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.window_based_dict = kwargs.get("window_based_dict", None)
        self.low_rank_dict = kwargs.get("low_rank_dict", None)

        self.attn1 = Attention(
            block_id=f"self_attn_{self.id}",
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
            window_based_dict=self.window_based_dict,
            low_rank_dict=self.low_rank_dict,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if fused_layer_norm:
                self.norm2 = ApexFusedLayerNorm(
                    dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
                )
            else:
                self.norm2 = nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )

            self.attn2 = Attention(
                block_id=f"cross_attn_{self.id}",
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                window_based_dict=None,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if not self.use_ada_layer_norm_single:
            if fused_layer_norm:
                self.norm3 = ApexFusedLayerNorm(
                    dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
                )
            else:
                self.norm3 = nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = kwargs.get("feed_forward_chunk_size", None)
        self._chunk_dim = kwargs.get("feed_forward_chunk_dim", 0)

        print(
            f"feed_forward_chunk_size: {self._chunk_size}, feed_forward_chunk_dim: {self._chunk_dim}"
        )

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class CaptionProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py

    origin: PixArtAlphaTextProjection
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(
            in_features=hidden_size, out_features=out_features, bias=True
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class T2V_Model(nn.Module):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        video_length: int = 16,
        data_type: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length

        self.num_layers = num_layers

        print(
            f"model config: number_attenion_heads: {num_attention_heads}, attention_head_dim: {attention_head_dim}, norm_type: {norm_type}, patch_size: {patch_size}, sample_size: {sample_size}, in_channels: {in_channels}, out_channels: {out_channels}, num_layers: {num_layers}"
        )
        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate(
                "norm_type!=num_embeds_ada_norm",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif (
            not self.is_input_continuous
            and not self.is_input_vectorized
            and not self.is_input_patches
        ):
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                eps=1e-6,
                affine=True,
            )
            if use_linear_projection:
                self.proj_in = linear_cls(in_channels, inner_dim)
            else:
                self.proj_in = conv_cls(
                    in_channels, inner_dim, kernel_size=1, stride=1, padding=0
                )
        elif self.is_input_vectorized:
            assert (
                sample_size is not None
            ), "Transformer2DModel over discrete input must provide sample_size"
            assert (
                num_vector_embeds is not None
            ), "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds,
                embed_dim=inner_dim,
                height=self.height,
                width=self.width,
            )
        elif self.is_input_patches:
            assert (
                sample_size is not None
            ), "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size
            self.width = sample_size

            self.patch_size = patch_size

            self.sample_size = sample_size
            interpolation_scale = (
                sample_size // 64
            )  # => 64 (= 512 pixart) has interpolation scale 1
            interpolation_scale = max(interpolation_scale, 1)
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    d,
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    **kwargs,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(
                    inner_dim, in_channels, kernel_size=1, stride=1, padding=0
                )
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(
                inner_dim, patch_size * patch_size * self.out_channels
            )
        elif self.is_input_patches and norm_type == "ada_norm_single":
            # self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.norm_out = ApexFusedLayerNorm(
                inner_dim, elementwise_affine=False, eps=1e-6
            )
            self.scale_shift_table = nn.Parameter(
                torch.randn(2, inner_dim) / inner_dim**0.5
            )
            self.proj_out = nn.Linear(
                inner_dim, patch_size * patch_size * self.out_channels
            )

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            self.use_additional_conditions = (
                False  # self.sample_size == 128 # False, 128 -> 1024
            )
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                inner_dim, use_additional_conditions=self.use_additional_conditions
            )

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(
                in_features=caption_channels, hidden_size=inner_dim
            )

        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)

        # define temporal positional embedding

        temp_pos_embed = self.get_1d_sincos_temp_embed(
            inner_dim, video_length
        )  # 1152 hidden size

        print(f"original temp_pos_embed: {temp_pos_embed.shape}")

        if global_variable.CP_ENABLE or global_variable.TP_ENABLE:
            cp_tp_group_size = dist.get_world_size(
                global_variable.CP_TP_MESH.get_group()
            )
            rank_in_cp_tp_group = dist.get_rank(global_variable.CP_TP_MESH.get_group())
            frames_per_rank = video_length // cp_tp_group_size
            temp_pos_embed = temp_pos_embed[
                rank_in_cp_tp_group
                * frames_per_rank : (rank_in_cp_tp_group + 1)
                * frames_per_rank
            ]

            # (1,video_length, hidden_size)
            print(f"after cp temp_pos_embed: {temp_pos_embed.shape}")

        self.register_buffer(
            "temp_pos_embed",
            torch.from_numpy(temp_pos_embed).float().unsqueeze(0),
            persistent=False,
        )

        print(f"register temp_pos_embed: {self.temp_pos_embed.shape}")

        self.norm_type = norm_type

        self.data_type = data_type

        self.low_rank_dict = kwargs.get("low_rank_dict", None)

        self.low_rank_mode = True if self.low_rank_dict != None else False

        self.register_buffer(
            "sparsity_buffer", torch.zeros(num_layers), persistent=True
        )
        self.register_buffer(
            "sparsity_buffer_per_head",
            torch.zeros(num_layers, num_attention_heads),
            persistent=True,
        )

        self.recover_sparsity_done = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _update_sparsity_buffer(self):
        # iterate over the transformer blocks and assign the sparsity buffer
        for i, block in enumerate(self.transformer_blocks):
            self.sparsity_buffer[i] = block.attn1.sparsity
            self.sparsity_buffer_per_head[i] = block.attn1.sparsity_per_head

    def _restore_sparsity_buffer_per_block(self):
        for i, block in enumerate(self.transformer_blocks):
            block.attn1.sparsity = self.sparsity_buffer[i]
            block.attn1.sparsity_per_head.copy_(self.sparsity_buffer_per_head[i])
        self.recover_sparsity_done = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, frame, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # hidden_states=hidden_states.permute(0, 2,1,3,4) # b,f,c,h,w -> b,c,f,h,w
        # print(f"hidden_states dtype:{hidden_states.dtype}")

        if self.recover_sparsity_done == False:
            self._restore_sparsity_buffer_per_block()
            self.recover_sparsity_done = True

        hidden_states = hidden_states.to(self.data_type)

        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num
        hidden_states = rearrange(
            hidden_states, "b c f h w -> (b f) c h w"
        ).contiguous()

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if (
            encoder_attention_mask is not None and encoder_attention_mask.ndim == 2
        ):  # ndim == 2 means no image joint
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(
                1
            )  # full attention
            # encoder_attention_mask = repeat(encoder_attention_mask, 'b 1 l -> (b f) 1 l', f=frame).contiguous()

        elif (
            encoder_attention_mask is not None and encoder_attention_mask.ndim == 3
        ):  # ndim == 3 means image joint
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(
                encoder_attention_mask_video, "b 1 l -> b (1 f) l", f=frame
            ).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat(
                [encoder_attention_mask_video, encoder_attention_mask_image], dim=1
            )
            encoder_attention_mask = (
                rearrange(encoder_attention_mask, "b n l -> (b n) l")
                .contiguous()
                .unsqueeze(1)
            )

        # Retrieve lora scale.
        # lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0


        if self.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": torch.tensor(
                    [(h, w)] * input_batch_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                "nframe": torch.tensor(
                    [frame] * input_batch_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                "fps": 10,
            }
        else:
            added_cond_kwargs = {
                "resolution": None,
                "aspect_ratio": None,
            }
        # timestep, embedded_timestep = self.adaln_single(
        #     timestep, added_cond_kwargs, batch_size=bsz, hidden_dtype=hidden_states.dtype
        # )

        if self.is_input_patches:  # here
            height, width = (
                hidden_states.shape[-2] // self.patch_size,
                hidden_states.shape[-1] // self.patch_size,
            )
            num_patches = height * width

            hidden_states = self.pos_embed(
                hidden_states
            )  # alrady add positional embeddings

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # batch_size = hidden_states.shape[0]
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep,
                    added_cond_kwargs,
                    batch_size=batch_size,
                    hidden_dtype=hidden_states.dtype,
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(
                encoder_hidden_states
            )  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(
                    encoder_hidden_states_video, "b 1 t d -> b (1 f) t d", f=frame
                ).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states_video, encoder_hidden_states_image], dim=1
                )
                encoder_hidden_states_spatial = rearrange(
                    encoder_hidden_states, "b f t d -> (b f) t d"
                ).contiguous()
            else:
                encoder_hidden_states_full = encoder_hidden_states

        # full attention
        timestep_full = timestep

        # hidden_states_video = hidden_states_video + self.temp_pos_embed
        hidden_states = rearrange(
            hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size, f=frame
        )


        hidden_states = hidden_states + self.temp_pos_embed

        hidden_states = rearrange(
            hidden_states, "(b t) f d -> b (f t) d", b=input_batch_size, f=frame
        )

        # print(f"hidden_states: {hidden_states.shape}")

        for i, full_block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    full_block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_full,
                    encoder_attention_mask,
                    timestep_full,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )

            else:
                hidden_states = full_block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states_full,
                    encoder_attention_mask,
                    timestep_full,
                    cross_attention_kwargs,
                    class_labels,
                )

        if self.is_input_patches:
            if self.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = (
                    self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                )
                hidden_states = self.proj_out_2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame + use_image_num).contiguous()
                embedded_timestep = embedded_timestep

                shift, scale = (
                    self.scale_shift_table[None] + embedded_timestep[:, None]
                ).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)

            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    height,
                    width,
                    self.patch_size,
                    self.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    height * self.patch_size,
                    width * self.patch_size,
                )
            )
            output = rearrange(
                output, "(b f) c h w -> b c f h w", b=input_batch_size
            ).contiguous()

        if (
            self.low_rank_mode
            and global_variable.CURRENT_STEP % global_variable.SPARSITY_UPDATE_STEP == 0
        ):
            self._update_sparsity_buffer()

        if not return_dict:
            # output=output.permute(0,2,1,3,4) # b,c,f,h,w -> b,f,c,h,w
            return output

        return Transformer3DModelOutput(sample=output)

    def get_1d_sincos_temp_embed(self, embed_dim, length):
        pos = torch.arange(0, length).unsqueeze(1)
        return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

    def forward_with_cfg(
        self,
        x,
        t,
        encoder_hidden_states,
        encoder_attention_mask,
        cfg_scale,
        use_image_num,
        return_dict=False,
    ):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        print(
            f"x shape: {x.shape}; combined shape: {combined.shape}; cfg_scale: {cfg_scale}"
        )

        # encoder_hidden_states=text_encoder_hidden_states,encoder_attention_mask=encoder_attention_mask, use_image_num=args.use_image_num,return_dict=False
        model_out = self.forward(
            combined,
            t,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_image_num=use_image_num,
            return_dict=False,
        )
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...]  # 2 16 4 32 32
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)


def convert_T2V_Model_to_tp(model: T2V_Model, tp_mesh):
    """Transformers T2V_Model model to TP shard model

    Args:
        model: T2V_Model model
        tp_mesh: Tensor parallel device mesh

    Returns:
        sharded_model: TP shard model
    """

    from torch.distributed._tensor import Replicate, Shard

    # define parallelization strategy
    parallel_config = {}

    original_heads = model.num_attention_heads

    if global_variable.RANK == 0:
        print(model)

    #   (transformer_blocks): ModuleList(
    #     (0-1): 2 x BasicTransformerBlock(
    #       (norm1): FusedLayerNorm(torch.Size([1024]), eps=1e-05, elementwise_affine=True)
    #       (attn1): Attention(
    #         (to_q): Linear(in_features=1024, out_features=1024, bias=False)
    #         (to_k): Linear(in_features=1024, out_features=1024, bias=False)
    #         (to_v): Linear(in_features=1024, out_features=1024, bias=False)
    #         (to_out): ModuleList(
    #           (0): Linear(in_features=1024, out_features=1024, bias=True)
    #           (1): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #       (norm2): FusedLayerNorm(torch.Size([1024]), eps=1e-05, elementwise_affine=True)
    #       (attn2): Attention(
    #         (to_q): Linear(in_features=1024, out_features=1024, bias=False)
    #         (to_k): Linear(in_features=1024, out_features=1024, bias=False)
    #         (to_v): Linear(in_features=1024, out_features=1024, bias=False)
    #         (to_out): ModuleList(
    #           (0): Linear(in_features=1024, out_features=1024, bias=True)
    #           (1): Dropout(p=0.0, inplace=False)
    #         )
    #       )
    #       (ff): FeedForward(
    #         (net): ModuleList(
    #           (0): GELU(
    #             (proj): Linear(in_features=1024, out_features=4096, bias=True)
    #           )
    #           (1): Dropout(p=0.0, inplace=False)
    #           (2): LoRACompatibleLinear(in_features=4096, out_features=1024, bias=True)
    #         )
    #       )
    #     )
    #   )

    for i in range(model.num_layers):
        block_prefix = f"transformer_blocks.{i}"

        # apply sequential parallelization to norm1 and norm2
        # apply the colwise parallelization to q,k,v
        # apply the rowwise parallelization to o

        # parallelize attention layer
        parallel_config.update(
            {
                # f"{block_prefix}0.norm1": SequentialParallel(),
                # f"{block_prefix}0.attn1": PrepareModuleInput(
                #     input_layouts=(Shard(1)),
                #     desired_input_layouts=(Replicate(), None),
                # ),
                f"{block_prefix}.attn1.to_q": ColwiseParallel(input_layouts=Shard(1)),
                f"{block_prefix}.attn1.to_k": ColwiseParallel(input_layouts=Shard(1)),
                f"{block_prefix}.attn1.to_v": ColwiseParallel(input_layouts=Shard(1)),
                f"{block_prefix}.attn1.to_out": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
                # f"{block_prefix}0.norm2": SequentialParallel(),
                f"{block_prefix}.attn2.to_q": ColwiseParallel(input_layouts=Shard(1)),
                f"{block_prefix}.attn2.to_k": ColwiseParallel(
                    input_layouts=Replicate()
                ),
                f"{block_prefix}.attn2.to_v": ColwiseParallel(
                    input_layouts=Replicate()
                ),
                f"{block_prefix}.attn2.to_out": RowwiseParallel(
                    output_layouts=Shard(1)
                ),
                f"{block_prefix}.ff.net.0.proj": ColwiseParallel(
                    input_layouts=Shard(1)
                ),
                # f"{block_prefix}1.ff.net.1": SequentialParallel(),
                f"{block_prefix}.ff.out": RowwiseParallel(output_layouts=Shard(1)),
            }
        )

        # print(type(model.transformer_blocks[i].attn1.to_out[0]))
        # print(type(model.transformer_blocks[i].attn2.to_out[0]))
        # print(type(model.transformer_blocks[i].ff.net[2]))

    # apply parallelization
    sharded_model = parallelize_module(model, tp_mesh, parallel_config)

    # adjust number of attention heads
    original_heads = model.num_attention_heads
    size_in_group = dist.get_world_size(tp_mesh.get_group())
    heads_per_rank = original_heads // size_in_group

    rank_in_group = dist.get_rank(tp_mesh.get_group())

    print(
        f"global rank:{dist.get_rank()}; tp_mesh group rank:{rank_in_group}; LatteT2V heads_per_rank:{heads_per_rank}"
    )
    for i in range(model.num_layers):
        attn_layer = model.transformer_blocks[i].attn1
        attn_layer.heads = heads_per_rank
        # attn_layer.n_kv_heads = attn_layer.n_heads  # in T5, n_heads equals n_kv_heads
        attn_layer.inner_dim = attn_layer.inner_dim // size_in_group
        attn_layer.cross_attention_dim = attn_layer.cross_attention_dim // size_in_group
        attn_layer.sliceable_head_dim = attn_layer.sliceable_head_dim // size_in_group

        cross_attn_layer = model.transformer_blocks[i].attn2
        cross_attn_layer.heads = heads_per_rank
        # attn_layer.n_kv_heads = attn_layer.n_heads  # in T5, n_heads equals n_kv_heads
        cross_attn_layer.inner_dim = cross_attn_layer.inner_dim // size_in_group
        cross_attn_layer.cross_attention_dim = (
            cross_attn_layer.cross_attention_dim // size_in_group
        )
        cross_attn_layer.sliceable_head_dim = (
            cross_attn_layer.sliceable_head_dim // size_in_group
        )

        # if i == 0:
        #     # shard relative position encoding weights
        #     original_weight = attn_layer.relative_attention_bias.weight.data
        #     new_weight = original_weight[:, rank_in_group*heads_per_rank:(rank_in_group+1)*heads_per_rank]
        #     print(f"global rank:{dist.get_rank()}; tp_mesh group rank:{rank_in_group}; original_weight shape:{original_weight.shape}; new_weight shape:{new_weight.shape}")
        #     attn_layer.relative_attention_bias.weight.data = new_weight

    return sharded_model


if __name__ == "__main__":
    model = T2V_Model(
        in_channels=4,
        patch_size=2,
        sample_size=32,
        num_layers=4,
        norm_type="ada_norm_single",
        caption_channels=1024,
        cross_attention_dim=1024,
        attention_head_dim=64,
        num_attention_heads=16,
        num_cross_attention_heads=16,
        return_dict=False,
        use_additional_conditions=False,
        video_length=16,
    )
    model = model.cuda().to(torch.bfloat16)
    print(model)

    # input batch size: 4, frame: 4, channel: 16, height: 32, width: 32
    input_ = torch.randn((4, 4, 16, 32, 32), device="cuda", dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (4,)).cuda()
    caption = torch.randn((4, 60, 1024), device="cuda", dtype=torch.bfloat16)

    output = model(
        hidden_states=input_, timestep=timestep, encoder_hidden_states=caption
    )

    print(output.shape)
