import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

from .t2v_model import T2V_Model


def customized_lr_scheduler(optimizer, warmup_steps=5000):  # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == "warmup":
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def get_models(args):
    if "T2V" in args.model:
        return T2V_Model(
            num_attention_heads=args.num_heads,
            attention_head_dim=args.head_dim,
            norm_type="ada_norm_single",
            cross_attention_dim=args.num_heads * args.head_dim,
            caption_channels=4096 if args.text_encoder == "t5" else 1024,
            patch_size=2,
            sample_size=args.image_size // 8,
            in_channels=4,
            out_channels=4 * 2 if args.learn_sigma else 4,
            num_layers=args.num_layers,
            video_length=args.num_frames,
            activation_fn=args.get("activation_fn", "geglu"),
            window_based_dict=args.window_based_dict,
            low_rank_dict=args.low_rank_dict,
            feed_forward_chunk_size=args.get("feed_forward_chunk_size", None),
            feed_forward_chunk_dim=args.get("feed_forward_chunk_dim", 0),
        )

    else:
        raise "{} Model Not Supported!".format(args.model)
