# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import gc
import math
import os
import queue
import shutil
import threading
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from glob import glob
from time import time
import json
import numpy as np
import torch.distributed as dist
from DSV.datasets import get_dataset
from DSV.datasets.videogen_datasets import GroupedDistributedSampler
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from DSV.diffusion import create_diffusion
from einops import rearrange
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                    FullStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardedStateDictConfig,
                                    ShardingStrategy, StateDictType)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from webdataset import WebLoader

import DSV.models.global_variable as global_variable
from DSV.models import get_models
from DSV.models.clip import TextEmbedder
from DSV.models.t2v_model import convert_T2V_Model_to_tp
from DSV.models.t5 import T5Embedder
from DSV.models.t5_tp_shard import convert_t5_to_tp
from DSV.rectified_flow.scheduler import rflow as rflow_diffusion
from utils import (cleanup, clip_grad_norm_, cp_tp_region_split_input,
                   create_logger, create_tensorboard, get_experiment_dir,
                   requires_grad, setup_distributed, update_ema,
                   write_tensorboard)

ATTENTION_SAVE_DIR_NAME = "attention_score"
EXP_NAME = None
EXP_INFO = None
formatted_current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
global_logger = None

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Reserved Memory: {reserved:.2f} MB")

def attention_score_save_threading(EXP_DIR):
    def save_tensor_dict():
        nonlocal tmp_step_dict, save_dir, cur_step
        spatial = tmp_step_dict["spatial"]
        temporal = tmp_step_dict["temporal"]
        spatial_save_name = f"step_{cur_step}_spatial.npz"
        temporal_save_name = f"step_{cur_step}_temporal.npz"
        spatial_save_name = os.path.join(save_dir, spatial_save_name)
        temporal_save_name = os.path.join(save_dir, temporal_save_name)

        try:
            np.savez(spatial_save_name, np.array([spatial]))
            np.savez(temporal_save_name, np.array([temporal]))
            global_variable.LOGGER.info(f"save spatial attention score at {spatial_save_name}")
            global_variable.LOGGER.info(f"save temporal attention score at {temporal_save_name}")
        except Exception as e:
            global_variable.LOGGER.error(f"save tensor error dur to {e}")

    save_dir = os.path.join(EXP_DIR, ATTENTION_SAVE_DIR_NAME)
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    tmp_step_dict = {}
    cur_step = -1
    global_variable.LOGGER.info(f"Save theading start -------- Target save dir {save_dir}; Aggregate attention maps across heads: {global_variable.AGGREGATE_ATTENTION_SCORE}")

    while True:
        (st_type, step, block_id, s_or_t_attention_score_cpu_tensor,) = global_variable.ATTENTION_SCORE_QUEUE.get()
        global_variable.LOGGER.info(f"Get {st_type} attention score from block {block_id} in step {step}")

        if st_type == "exit":
            global_variable.EXIT_SINGAL = True

        if global_variable.EXIT_SINGAL == True:
            if len(tmp_step_dict) != 0:
                save_tensor_dict()
            global_variable.LOGGER.info("save theading exit!")
            break

        if cur_step == -1:
            cur_step = step

        if cur_step != step:
            assert len(tmp_step_dict["spatial"]) == len(tmp_step_dict["temporal"])
            save_tensor_dict()
            tmp_step_dict.clear()
            cur_step = step

        if st_type not in tmp_step_dict:
            tmp_step_dict[st_type] = {}

        original = s_or_t_attention_score_cpu_tensor.numpy()
        B, Head, L, L = original.shape

        if global_variable.AGGREGATE_ATTENTION_SCORE:
            averge_head_score = np.mean(original, axis=1)
            assert averge_head_score.shape == (B, L, L)
            tmp_step_dict[st_type][block_id] = averge_head_score
        else:
            tmp_step_dict[st_type][block_id] = original

    return

def get_model_dtype(model):
    dtypes = set()
    if isinstance(model, T5Embedder):
        return model.torch_dtype
    for param in model.parameters():
        dtypes.add(param.dtype)
    return dtypes

def setup_parallel_groups(args):
    world_size = dist.get_world_size()
    rank = int(os.environ["RANK"])

    tp_group_size = args.get("tp_group_size", 1)
    cp_group_size = args.get("cp_group_size", 1)
    dp_group_size = args.get("dp_group_size", world_size)

    print(f"rank:{rank}; tp_group_size:{tp_group_size}; cp_group_size:{cp_group_size}; dp_group_size:{dp_group_size}")

    assert world_size % tp_group_size == 0, "world size must be divisible by tp group size"
    assert world_size % cp_group_size == 0, "world size must be divisible by cp group size"
    assert world_size % dp_group_size == 0, "world size must be divisible by dp group size"

    if tp_group_size > 1 and cp_group_size > 1:
        cp_device_mesh = init_device_mesh(
            "cuda", (world_size // tp_group_size // cp_group_size, cp_group_size, tp_group_size,),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        dp_cp_mesh = cp_device_mesh["dp", "cp"]
        cp_tp_mesh = init_device_mesh(
            "cuda", (world_size // tp_group_size // cp_group_size, cp_group_size * tp_group_size,),
            mesh_dim_names=("dp", "cp_tp"),
        )
        tp_mesh = cp_device_mesh["tp"]
        cp_mesh = cp_device_mesh["cp"]

        global_variable.DATA_PARALLEL_GROUP = dp_cp_mesh._flatten().get_group()
        global_variable.CONTEXT_PARALLEL_GROUP = cp_mesh.get_group()
        global_variable.TENSOR_PARALLEL_GROUP = tp_mesh.get_group()

        global_variable.CP_ENABLE = True
        global_variable.TP_ENABLE = True

        global_variable.TP_MESH = tp_mesh
        global_variable.CP_MESH = cp_mesh
        global_variable.DP_CP_MESH = dp_cp_mesh._flatten()
        global_variable.CP_TP_MESH = cp_tp_mesh["cp_tp"]
        global_variable.FSDP_MESH = dp_cp_mesh._flatten()

        print(f"Rank {rank}; Ranks in its data parallel group: {dist.get_process_group_ranks(global_variable.DATA_PARALLEL_GROUP)}")
        print(f"Rank {rank}; Ranks in its context parallel group: {dist.get_process_group_ranks(global_variable.CONTEXT_PARALLEL_GROUP)}")
        print(f"Rank {rank}; Ranks in its tensor parallel group: {dist.get_process_group_ranks(global_variable.TENSOR_PARALLEL_GROUP)}")
        print(f"Rank {rank}; Ranks in its CP_TP_MESH group: {dist.get_process_group_ranks(global_variable.CP_TP_MESH.get_group())}")

    elif tp_group_size > 1:
        tp_device_mesh = init_device_mesh("cuda", (world_size // tp_group_size, tp_group_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = tp_device_mesh["tp"]
        fsdp_mesh = tp_device_mesh["dp"]

        global_variable.DATA_PARALLEL_GROUP = fsdp_mesh.get_group()
        global_variable.CONTEXT_PARALLEL_GROUP = None
        global_variable.TENSOR_PARALLEL_GROUP = tp_mesh.get_group()

        global_variable.CP_ENABLE = False
        global_variable.TP_ENABLE = True

        global_variable.TP_MESH = tp_mesh
        global_variable.CP_MESH = None
        global_variable.DP_CP_MESH = fsdp_mesh
        global_variable.CP_TP_MESH = tp_mesh
        global_variable.FSDP_MESH = fsdp_mesh

        print(f"Rank {rank}; Ranks in its data parallel group: {dist.get_process_group_ranks(global_variable.DATA_PARALLEL_GROUP)}")
        print(f"Rank {rank}; Ranks in its tensor parallel group: {dist.get_process_group_ranks(global_variable.TENSOR_PARALLEL_GROUP)}")

    elif cp_group_size > 1:
        fsdp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        cp_device_mesh = init_device_mesh("cuda", (world_size // cp_group_size, cp_group_size), mesh_dim_names=("dp", "cp"))
        cp_mesh = cp_device_mesh["cp"]

        global_variable.DATA_PARALLEL_GROUP = fsdp_mesh.get_group()
        global_variable.CONTEXT_PARALLEL_GROUP = cp_mesh.get_group()
        global_variable.TENSOR_PARALLEL_GROUP = None

        global_variable.CP_ENABLE = True
        global_variable.TP_ENABLE = False

        global_variable.TP_MESH = None
        global_variable.CP_MESH = cp_mesh
        global_variable.DP_CP_MESH = fsdp_mesh
        global_variable.CP_TP_MESH = cp_mesh
        global_variable.FSDP_MESH = fsdp_mesh

        if global_variable.DATA_PARALLEL_GROUP != None:
            print(f"Rank {rank}; Ranks in its data parallel group: {dist.get_process_group_ranks(global_variable.DATA_PARALLEL_GROUP)}")
        if global_variable.CONTEXT_PARALLEL_GROUP != None:
            print(f"Rank {rank}; Ranks in its context parallel group: {dist.get_process_group_ranks(global_variable.CONTEXT_PARALLEL_GROUP)}")
        if global_variable.TENSOR_PARALLEL_GROUP != None:
            print(f"Rank {rank}; Ranks in its tensor parallel group: {dist.get_process_group_ranks(global_variable.TENSOR_PARALLEL_GROUP)}")

    else:
        device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        fsdp_mesh = device_mesh["fsdp"]

        global_variable.DATA_PARALLEL_GROUP = fsdp_mesh.get_group()
        global_variable.CONTEXT_PARALLEL_GROUP = None
        global_variable.TENSOR_PARALLEL_GROUP = None

        global_variable.CP_ENABLE = False
        global_variable.TP_ENABLE = False

        global_variable.TP_MESH = None
        global_variable.CP_MESH = None
        global_variable.DP_CP_MESH = fsdp_mesh
        global_variable.CP_TP_MESH = None
        global_variable.FSDP_MESH = fsdp_mesh

        if global_variable.DATA_PARALLEL_GROUP != None:
            print(f"Rank {rank}; Ranks in its data parallel group: {dist.get_process_group_ranks(global_variable.DATA_PARALLEL_GROUP)}")
        if global_variable.CONTEXT_PARALLEL_GROUP != None:
            print(f"Rank {rank}; Ranks in its context parallel group: {dist.get_process_group_ranks(global_variable.CONTEXT_PARALLEL_GROUP)}")
        if global_variable.TENSOR_PARALLEL_GROUP != None:
            print(f"Rank {rank}; Ranks in its tensor parallel group: {dist.get_process_group_ranks(global_variable.TENSOR_PARALLEL_GROUP)}")

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print("-" * 50)
        print(f"args:{args}")
        print("-" * 50)

    global_variable.RANK = rank
    global_variable.SAVE_ATTENTION_SCORE = args.save_attention_score
    global_variable.AGGREGATE_ATTENTION_SCORE = args.aggregate_attention_score

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    setup_parallel_groups(args)

    print(f"TP ENABLE: {global_variable.TP_ENABLE}; CP ENABLE: {global_variable.CP_ENABLE}")

    dtype = torch.bfloat16 if args.dtype == "torch.bfloat16" else torch.float16
    torch.set_default_dtype(dtype)

    print(f"Starting rank={rank}, global_variable.RANK ={global_variable.RANK} ,local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    resume_checkpoint_dir = None

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        num_frame_string = "F" + str(args.num_frames) + "S" + str(args.frame_interval)
        current_time = datetime.now().strftime("%m-%d-%H-%M")

        atten_sparse_mode = args.atten_sparse_mode if args.atten_sparse_mode != None else "no_sparse"
        low_rank_loss = args.low_rank_loss if args.low_rank_loss != None else "no_lr_loss"
        exp_note = "_" + str(args.note) if args.note != None else ""

        experiment_dir = f"{args.results_dir}/{model_string_name}-{num_frame_string}-{args.dataset}-{atten_sparse_mode}-{low_rank_loss}-{current_time}{exp_note}"

        if args.resume_from_checkpoint and (args.create_new_dir_when_resume != True):
            experiment_dir = args.resume_exp_dir
        else:
            experiment_dir = get_experiment_dir(experiment_dir, args)

        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, "config.yaml"))

        if args.resume_from_checkpoint and (args.create_new_dir_when_resume != True):
            logger.info(f"Continue to use previous Exp dir at {experiment_dir}")
        else:
            logger.info(f"Experiment directory created at {experiment_dir}")

        global_variable.LOGGER = logger
    else:
        if args.resume_from_checkpoint:
            experiment_dir = args.resume_exp_dir
        logger = create_logger(None)
        tb_writer = None

    if args.resume_from_checkpoint == True:
        resume_checkpoint_dir = os.path.join(args.resume_exp_dir, "checkpoints")

    logger.info(f"Training Data Info; Batch size: {args.local_batch_size}; Use_image_num: {args.use_image_num}; Image Size: {args.image_size}; Num Frames: {args.num_frames} ")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size

    model = get_models(args).to(torch.float32)

    test_large_scale_flag = args.test_large_scale if "test_large_scale" in args else False
    global_variable.TEST_LARGE_SCALE = test_large_scale_flag

    use_triton_attention_flag = args.get("triton_attention", False)
    global_variable.TRITON_ATTENTION = use_triton_attention_flag

    if test_large_scale_flag == True:
        ema = None
    else:
        ema = deepcopy(model).to(torch.float32)
        requires_grad(ema, False)
        update_ema(ema, model, decay=0)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path).to(device).to(torch.bfloat16)

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    vae.requires_grad_(False)

    if args.extras == 78:
        if args.text_encoder == "clip":
            text_encoder = TextEmbedder(path="stabilityai/stable-diffusion-2-1-base", dropout_prob=0.00001).to(device).to(dtype)
            text_encoder.eval()
            text_encoder.requires_grad_(False)

        elif args.text_encoder == "t5" and args.text_encoder_dummy == False:
            text_encoder = T5Embedder(dir_or_name=args.text_encoder_path, device=device, torch_dtype=dtype,)

            if global_variable.TEST_LARGE_SCALE == True:
                pass
            elif global_variable.TP_ENABLE == True and global_variable.CP_ENABLE == False:
                original_text_encoder_model = text_encoder.model
                sharded_model = convert_t5_to_tp(original_text_encoder_model, global_variable.TP_MESH)
                text_encoder.model = sharded_model
                del original_text_encoder_model

            elif global_variable.CP_ENABLE == True and global_variable.TP_ENABLE == False:
                print(f"rank:{rank}; try to apply tp to t5 in group:{dist.get_world_size(global_variable.CP_MESH.get_group())}")
                original_text_encoder_model = text_encoder.model
                sharded_model = convert_t5_to_tp(original_text_encoder_model, global_variable.CP_MESH)
                text_encoder.model = sharded_model
                del original_text_encoder_model

            elif global_variable.CP_ENABLE == True and global_variable.TP_ENABLE == True:
                print(f"rank:{rank}; try to apply tp to t5 in group:{dist.get_world_size(global_variable.CP_TP_MESH.get_group())}")
                original_text_encoder_model = text_encoder.model
                sharded_model = convert_t5_to_tp(original_text_encoder_model, global_variable.CP_TP_MESH)
                text_encoder.model = sharded_model
                del original_text_encoder_model

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        elif args.text_encoder == "t5" and args.text_encoder_dummy == True:
            text_encoder = None
        else:
            raise NotImplementedError(f"Text encoder {args.text_encoder} not implemented")
    else:
        text_encoder = None

    if args.dataset not in ["webvid"]:
        dataset = get_dataset(args)

        if global_variable.CP_ENABLE == True or global_variable.TP_ENABLE == True:
            sampler = GroupedDistributedSampler(
                dataset,
                world_size=dist.get_world_size(),
                ranks_per_group=dist.get_world_size(global_variable.CP_TP_MESH.get_group()),
                shuffle=True,
                seed=args.global_seed,
            )
            print(f"rank:{rank}; sampler world size:{dist.get_world_size()}; ranks per group:{dist.get_world_size(global_variable.CP_TP_MESH.get_group())}")
        else:
            sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed,)

        loader = DataLoader(dataset, batch_size=int(args.local_batch_size), shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True,)
    else:
        sampler = None
        dataset = get_dataset(args)
        dataset.with_length(100000)

        loader = (WebLoader(dataset, batch_size=None, num_workers=args.num_workers)
            .unbatched()
            .shuffle(1000)
            .batched(args.local_batch_size)
        )
        loader = loader.with_epoch(100000).with_length(100000)

    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path}), num_workers:{args.num_workers}")

    print(f"rank {rank}: Allocated Memory Before FSDP Init: {torch.cuda.memory_allocated(local_rank) / (1024 ** 3):.2f} GB")
    print(f"rank {rank}: Reserved Memory Before FSDP Init: {torch.cuda.memory_reserved(local_rank) / (1024 ** 3):.2f} GB")

    if args.ddp_mode == "ddp":
        model = DDP(model, device_ids=[local_rank], process_group=global_variable.DATA_PARALLEL_GROUP, find_unused_parameters=True,)

        if args.gradient_allreduce_fp32 and args.ddp_mode == "ddp":
            def allreduce_fp32(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
                buffer = bucket.buffer()
                fp32_buffer = buffer.to(torch.float32)
                fut = torch.distributed.all_reduce(fp32_buffer, async_op=True).get_future()

                def cast_to_dtype(fut):
                    all_reduced_tensor = buffer
                    value = fut if isinstance(fut, torch.Tensor) else fut.value()[0]
                    value.div_(dist.get_world_size(group=global_variable.DATA_PARALLEL_GROUP))
                    all_reduced_tensor.copy_(value)
                    return all_reduced_tensor

                return fut.then(cast_to_dtype)

            print("Using gradient allreduce fp32 for DDP")
            model.register_comm_hook(state=None, hook=allreduce_fp32)

        model.train()

    elif args.ddp_mode == "fsdp":
        print("Using FSDP.")

        if global_variable.TP_ENABLE == True:
            print(f"Using TP. Global Rank:{rank}; LatteT2V model to TP format in TP Mesh with ranks:{dist.get_process_group_ranks(global_variable.TP_MESH.get_group())}")
            tp_model = convert_latteT2V_to_tp(model, global_variable.TP_MESH)
            del model
            model = tp_model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        model = FSDP(
            model,
            sharding_strategy=(ShardingStrategy.FULL_SHARD if args.get("zero_stage", 2) == 3 else ShardingStrategy.SHARD_GRAD_OP),
            device_id=local_rank,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32 if test_large_scale_flag == False else None,
                buffer_dtype=torch.bfloat16,
            ),
            use_orig_params=True if global_variable.TP_ENABLE == True else False,
            device_mesh=global_variable.FSDP_MESH,
        )

        if test_large_scale_flag == True:
            ema_fsdp = None
        else:
            ema_fsdp = FSDP(
                ema,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                device_id=local_rank,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32,),
                use_orig_params=True if global_variable.TP_ENABLE == True else False,
                device_mesh=global_variable.FSDP_MESH,
            )
    else:
        raise NotImplementedError(f"DDP mode {args.ddp_mode} not implemented")

    if args.resume_from_checkpoint != True:
        if args.ddp_mode == "ddp":
            low_rank_params = [p for n, p in model.named_parameters() if "low_rank" in n]
            other_params = [p for n, p in model.named_parameters() if "low_rank" not in n]

            if len(low_rank_params) > 0:
                opt = torch.optim.AdamW([
                    {"params": low_rank_params, "lr": args.low_rank_learning_rate},
                    {"params": other_params, "lr": args.learning_rate},
                ], weight_decay=0,)
            else:
                opt = torch.optim.AdamW([{"params": other_params, "lr": args.learning_rate}], weight_decay=0)

        elif args.ddp_mode == "fsdp":
            if test_large_scale_flag == False:
                opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            else:
                opt = None

        if test_large_scale_flag == False:
            lr_scheduler = get_scheduler(
                name="constant",
                optimizer=opt,
                num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            )

    if torch.__version__ >= "2.3":
        scaler = torch.GradScaler("cuda")
    else:
        scaler = torch.cuda.amp.GradScaler()

    torch.cuda.synchronize()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    num_update_steps_per_epoch = math.ceil(len(loader))
    print(f"Steps per epoch {num_update_steps_per_epoch}")
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.resume_from_checkpoint:
        candidate_ckpts = os.listdir(resume_checkpoint_dir)
        print(f"Rank {rank}; Checkpoint files : {candidate_ckpts}")
        candidate_ckpts = [ckpt for ckpt in candidate_ckpts if ckpt.endswith("pt")]
        candidate_ckpts = sorted(candidate_ckpts, key=lambda x: int(x.split(".")[0]))

        if args.ckpt_name != None:
            target_ckpt = args.ckpt_name
            assert target_ckpt in candidate_ckpts
        else:
            target_ckpt = candidate_ckpts[-1]

        logger.info(f"Resuming from checkpoint {target_ckpt}")
        resume_ckpt_path = os.path.join(resume_checkpoint_dir, target_ckpt)
        logger.info(f"Loading checkpoint from {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location="cpu")

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        ), FSDP.state_dict_type(
            ema_fsdp,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
        ):
            model.load_state_dict(checkpoint["model"])
            ema_fsdp.load_state_dict(checkpoint["ema"])

            opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            opt_state_dict = FSDP.optim_state_dict_to_load(model, opt, checkpoint["opt"])
            opt.load_state_dict(opt_state_dict)

            lr_scheduler = get_scheduler(
                name="constant",
                optimizer=opt,
                num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            )
            lr_scheduler.load_state_dict(checkpoint["sch"])

        train_steps = int(target_ckpt.split(".")[0])
        global_variable.CURRENT_STEP = train_steps
        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch
        first_epoch = 0
        resume_step = 0

        del checkpoint["model"]
        del checkpoint["ema"]
        del checkpoint["opt"]
        del checkpoint["sch"]
        del checkpoint

        gc.collect()
        print(f"Rank {rank}; Finish loading checkpoint")

    if rank == 0 and global_variable.SAVE_ATTENTION_SCORE == True:
        save_thead = threading.Thread(target=attention_score_save_threading, args=(experiment_dir,))
        save_thead.start()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    end_to_end_time_per_step = {}

    allocated = torch.cuda.memory_allocated(local_rank) / (1024**3)
    reserved = torch.cuda.memory_reserved(local_rank) / (1024**3)
    print(f"rank {rank}: Allocated Memory: {allocated:.2f} GB")
    print(f"rank {rank}: Reserved Memory: {reserved:.2f} GB")

    if args.low_rank_loss == None:
        print("No low rank loss")
    elif args.low_rank_loss == "KL":
        RL_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        print(f"Low rank loss KL")
    elif args.low_rank_loss == "COS":
        reduction = "sum"
        RL_loss = nn.CosineEmbeddingLoss(reduction=reduction)
        print(f"Low rank loss Cosine; Reduction:{reduction}")
    else:
        raise NotImplementedError

    if "norm_loss" in args.low_rank_config and args.low_rank_config["norm_loss"] == True:
        norm_mse_loss = torch.nn.MSELoss(reduction="sum")
        print(f"MSE Loss for Norm is used and reduction is SUM")
    else:
        norm_mse_loss = None

    if args.atten_sparse_mode == "low_rank":
        global_variable.LOW_RANK_STAGE0_STEPS = args.low_rank_config["low_rank_stage_0_steps"]
        logger.info(f"{args.low_rank_config['low_rank_stage_0_steps']} steps for low rank stage 0")

    if rank == 0:
        if args.model_file_path != None:
            shutil.copy(args.model_file_path, experiment_dir)

    data_type = eval(args.data_type)
    torch.set_default_dtype(torch.float32)

    if args.flow_matching:
        diffusion_scheduler = rflow_diffusion(use_timestep_transform=True, sample_method="logit-normal",)
    else:
        diffusion_scheduler = create_diffusion(timestep_respacing="")

    if test_large_scale_flag == False:
        print(f"ema dtype:{get_model_dtype(ema)}, model dtype:{get_model_dtype(model)}, vae dtype:{get_model_dtype(vae)},text_encoder dtype:{get_model_dtype(text_encoder)}")

    if args.use_profile:
        def trace_handler(prof):
            if rank == 0:
                prof.export_chrome_trace(f"trace_{prof.step_num}.json")

        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=trace_handler,
        )
        print(f"Profile context created")
    else:
        prof_ctx = nullcontext()

    profile_steps = args.profile_steps + train_steps if args.use_profile else None

    with prof_ctx as prof:
        for epoch in range(first_epoch, num_train_epochs):
            if sampler != None:
                sampler.set_epoch(epoch)

            if args.use_profile and train_steps >= profile_steps:
                break

            for step, video_data in enumerate(loader):
                if global_variable.TEST_LARGE_SCALE == True and step >= args.stop_step:
                    dist.barrier()
                    logger.info("Done!")
                    cleanup()
                    exit(0)

                if args.use_profile and train_steps >= profile_steps:
                    break

                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if rank == 0 and step % 1000 == 0:
                        logger.info(f"Skip training step {step} at epoch {epoch}; resume_step is {train_steps}")
                    continue

                if (global_variable.LOW_RANK_STAGE == 0 and args.atten_sparse_mode == "low_rank" and train_steps >= args.low_rank_config["low_rank_stage_0_steps"]):
                    print(f"Switch to LOW RANK STAGE 1")
                    global_variable.LOW_RANK_STAGE = 1

                total_loss = 0
                global_variable.FORWARD_DONE = False

                with nullcontext():
                    if args.dataset not in ["webvid"] and args.extras != 78:
                        x = video_data["video"].to(device, non_blocking=True, dtype=dtype)
                        video_name = video_data["video_name"]

                        if args.dataset == "ucf101_img":
                            image_name = video_data["image_name"]
                            image_names = []
                            for caption in image_name:
                                single_caption = [int(item) for item in caption.split("=====")]
                                image_names.append(torch.as_tensor(single_caption))

                    elif args.extras == 78:
                        if args.dataset in ["ucf101_img", "videogen", "openvid", "dummy"]:
                            x = video_data["video"].to(device, non_blocking=True, dtype=dtype)
                            video_prompt = video_data["video_text_prompt"]

                            if global_variable.CP_ENABLE or global_variable.TP_ENABLE:
                                x = cp_tp_region_split_input(x)

                            if args.text_encoder == "clip":
                                (text_encoder_hidden_states, _, encoder_attention_mask,) = text_encoder(text_prompts=video_prompt, train=False)
                            elif args.text_encoder == "t5":
                                if args.text_encoder_dummy:
                                    text_encoder_hidden_states = torch.randn(len(video_prompt), 120, 4096, dtype=dtype, device=device)
                                    encoder_attention_mask = torch.ones(1, 120, dtype=torch.long, device=device)
                                else:
                                    (text_encoder_hidden_states, _, encoder_attention_mask,) = text_encoder.get_text_embeddings(video_prompt)
                            else:
                                raise NotImplementedError(f"Text encoder {args.text_encoder} not implemented")

                            if global_variable.TP_ENABLE == True or global_variable.CP_ENABLE == True:
                                print(f"rank:{rank}; step:{step}; video_prompt:{video_prompt}")
                                print(f"rank:{rank}; step:{step}; input video shape:{x.shape}; text_encoder_hidden_states shape:{text_encoder_hidden_states.shape}; encoder_attention_mask shape:{encoder_attention_mask.shape}")

                        elif args.dataset == "webvid":
                            x = video_data[0].to(device, non_blocking=True, dtype=dtype)
                            caption = video_data[1]

                            if args.text_encoder == "clip":
                                (text_encoder_hidden_states, _, encoder_attention_mask,) = text_encoder(text_prompts=caption, train=False)
                            elif args.text_encoder == "t5":
                                (text_encoder_hidden_states, _, encoder_attention_mask,) = text_encoder.get_text_embeddings(caption)
                            else:
                                raise NotImplementedError(f"Text encoder {args.text_encoder} not implemented")
                    else:
                        raise NotImplementedError

                    with record_function("vae_encode") if args.use_profile else nullcontext():
                        with torch.no_grad():
                            b, f, c, h, w = x.shape
                            x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
                            if b * f <= 16 * 4:
                                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                            else:
                                chunk_size = 16 * 2
                                chunks = []
                                for i in range(0, b * f, chunk_size):
                                    end_idx = min(i + chunk_size, b * f)
                                    chunk = x[i:end_idx]
                                    chunk_encoded = vae.encode(chunk).latent_dist.sample().mul_(0.18215)
                                    chunks.append(chunk_encoded)
                                x = torch.cat(chunks, dim=0)

                            if args.extras == 78:
                                x = rearrange(x, "(b f) c h w -> b c f h w", b=b).contiguous()
                            else:
                                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()

                    begin_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    if args.extras == 78:
                        print(f"rank:{rank}; step:{step}; text_encoder_hidden_states shape:{text_encoder_hidden_states.shape}; encoder_attention_mask shape:{encoder_attention_mask.shape}")

                        model_kwargs = dict(
                            encoder_hidden_states=text_encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            use_image_num=args.use_image_num,
                            return_dict=False,
                        )

                        model_kwargs["height"] = torch.tensor(args.image_size, device=device).repeat(b)
                        model_kwargs["width"] = torch.tensor(args.image_size, device=device).repeat(b)
                        model_kwargs["num_frames"] = torch.tensor(args.num_frames, device=device).repeat(b)

                    elif args.extras == 2:
                        if args.dataset == "ucf101_img":
                            model_kwargs = dict(y=video_name, y_image=image_names, use_image_num=args.use_image_num,)
                        else:
                            model_kwargs = dict(y=video_name)
                    else:
                        model_kwargs = dict(y=None, use_image_num=args.use_image_num)

                    if args.flow_matching:
                        t = None
                    else:
                        t = torch.randint(0, diffusion_scheduler.num_timesteps, (x.shape[0],), device=device,)

                    begin_event.record()
                    with record_function("model forward") if args.use_profile else nullcontext():
                        loss_dict = diffusion_scheduler.training_losses(model, x, t=t, model_kwargs=model_kwargs)

                    loss = loss_dict["loss"].mean()
                    total_loss += loss

                    assert total_loss.dtype == dtype, f"total_loss dtype:{total_loss.dtype}, desired dtype:{dtype}"

                global_variable.FORWARD_DONE = True

                if rank in [0] and train_steps % 100 == 0 and args.low_rank_dict != None and test_large_scale_flag == False:
                    print(f"rank:{rank}; transformer_blocks.1.attn1.low_rank_qk_proj.weight.grad before global bw", torch.mean(model.module.transformer_blocks[0].attn1.low_rank_qk_proj.weight.grad),)
                    if model.module.transformer_blocks[0].attn1.to_q.weight.grad != None:
                        print(f"rank:{rank}; transformer_blocks.1.attn1.to_q.weight.grad before global bw", torch.mean(model.module.transformer_blocks[0].attn1.to_q.weight.grad),)
                    else:
                        print(f"rank:{rank}; transformer_blocks.1.attn1.to_q.weight.grad is None")

                total_loss.backward()

                end_event.record()

                if global_variable.TEST_LARGE_SCALE == True:
                    torch.cuda.synchronize()
                    print(f"end-to-end time in step {step}:{begin_event.elapsed_time(end_event)} ms")
                    end_to_end_time_per_step[step] = round(begin_event.elapsed_time(end_event), 4)

                    if step == args.stop_step - 1 and rank == 0:
                        if args.save_end_to_end_time_json_path != None:
                            with open(args.save_end_to_end_time_json_path, "w") as f:
                                json.dump(end_to_end_time_per_step, f, indent=4)

                if rank in [0, 1] and train_steps % 100 == 0 and args.low_rank_dict != None and test_large_scale_flag == False:
                    print(f"rank:{rank}; transformer_blocks.1.attn1.low_rank_qk_proj.weight.grad after global bw", torch.mean(model.module.transformer_blocks[0].attn1.low_rank_qk_proj.weight.grad),)
                    print(f"rank:{rank}; transformer_blocks.1.attn1.to_q.weight.grad after global bw", torch.mean(model.module.transformer_blocks[0].attn1.to_q.weight.grad),)

                if train_steps < args.start_clip_iter:
                    if args.ddp_mode == "fsdp":
                        gradient_norm = model.clip_grad_norm_(10000)
                    else:
                        gradient_norm = clip_grad_norm_(model.module.named_parameters(), args.clip_max_norm, clip_grad=False,)
                else:
                    if args.ddp_mode == "fsdp":
                        gradient_norm = model.clip_grad_norm_(args.clip_max_norm)
                    else:
                        gradient_norm = clip_grad_norm_(model.module.named_parameters(), args.clip_max_norm, clip_grad=True,)

                log_steps += 1
                train_steps += 1

                global_variable.CURRENT_STEP += 1

                low_rank_para_grad_norm = 0.0

                if train_steps % args.log_every == 0:
                    def get_grad_norm_per_transformer_block(prefix, named_params, num_blocks=28, norm_type=2.0):
                        grad_dict = defaultdict(list)

                        for name, param in named_params:
                            if "low_rank" in name:
                                continue
                            if param.grad is None:
                                print(f"Param:{name} has no grad")
                                continue

                            if name.startswith(prefix):
                                name_split = name.split(".")
                                block_id = int(name_split[1])
                                grad_dict[block_id].append(param.grad)

                        grad_norm_dict = {}
                        for block_id, grad_list in grad_dict.items():
                            if not grad_list:
                                print(f"block_id:{block_id} has no grad!")
                                continue
                            grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grad_list]), norm_type,)
                            grad_norm_dict[block_id] = grad_norm

                        if grad_dict:
                            print(f"grad_dict value length: {[len(v) for v in grad_dict.values()]}")

                        return grad_norm_dict

                    grad_norm_dict = get_grad_norm_per_transformer_block("transformer_blocks", model.named_parameters())

                if ((args.atten_sparse_mode == "low_rank" or args.low_rank_dict != None) and train_steps % args.log_every == 0 and rank == 0 and test_large_scale_flag == False):
                    def get_lr_grad_list(model):
                        grad_ = []
                        for name, param in model.named_parameters():
                            if "low_rank" in name:
                                grad_.append(param.grad)
                        return grad_

                    torch.cuda.synchronize()
                    grads = get_lr_grad_list(model.module)
                    print(f"low rank related grad list len:{len(grads)}")
                    norm_type = float(2.0)
                    device = grads[0].device
                    low_rank_para_grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type,)

                if test_large_scale_flag == False:
                    opt.step()

                if rank in [0, 1] and train_steps % 100 == 0 and args.low_rank_dict != None and test_large_scale_flag == False:
                    print(f"rank:{rank}; transformer_blocks.10.attn1.low_rank_qk_proj.weight.grad after global opt", torch.mean(model.module.transformer_blocks[0].attn1.low_rank_qk_proj.weight.grad),)
                    print(f"rank:{rank}; transformer_blocks.10.attn1.to_q.weight.grad after global opt", torch.mean(model.module.transformer_blocks[0].attn1.to_q.weight.grad),)

                if test_large_scale_flag == False:
                    lr_scheduler.step()
                    opt.zero_grad()

                if global_variable.TP_ENABLE == False and test_large_scale_flag == False:
                    with torch.no_grad():
                        for (ema_name, ema_param), (model_name, model_param) in zip(ema_fsdp.named_parameters(), model.named_parameters()):
                            assert ema_param.shape == model_param.shape, f"ema_param.shape:{ema_param.shape}, model_param.shape:{model_param.shape}"
                            assert ema_param.dtype == torch.float32, f"ema_param.dtype:{ema_param.dtype}, model_param.dtype:{model_param.dtype}"
                            ema_param.mul_(0.9999).add_(model_param.data, alpha=0.0001)

                        for (ema_name, ema_buffer), (model_name, model_buffer) in zip(ema_fsdp.named_buffers(), model.named_buffers()):
                            assert ema_buffer.shape == model_buffer.shape, f"ema_buffer.shape:{ema_buffer.shape}, model_buffer.shape:{model_buffer.shape}"
                            assert ema_buffer.dtype == torch.float32, f"ema_buffer.dtype:{ema_buffer.dtype}, model_buffer.dtype:{model_buffer.dtype}"
                            ema_buffer.copy_(model_buffer.to(torch.float32))

                running_loss += loss.item()

                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)

                    if test_large_scale_flag == False:
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / dist.get_world_size()
                    else:
                        avg_loss = avg_loss.item()

                    write_tensorboard(tb_writer, "Train Loss", avg_loss, train_steps)
                    write_tensorboard(tb_writer, "Gradient Norm", gradient_norm, train_steps)

                    if args.atten_sparse_mode == "low_rank":
                        if isinstance(loss_low_rank, torch.Tensor):
                            low_rank_loss_value = loss_low_rank.item()
                        else:
                            low_rank_loss_value = loss_low_rank

                        if isinstance(loss_norm, torch.Tensor):
                            loss_norm_value = loss_norm.item()
                        else:
                            loss_norm_value = loss_norm

                        loss_norm_value = loss_norm_value if norm_mse_loss != None else -1

                        logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Low rank Loss: {low_rank_loss_value:.4f}, Norm Loss: {loss_norm_value:.4f}, Total Gradient Norm: {gradient_norm:.4f}, Low rank Param Grad Norm :{low_rank_para_grad_norm:.4f} learning rate: {lr_scheduler.get_last_lr()}, Train Steps/Sec: {steps_per_sec:.2f}")

                        write_tensorboard(tb_writer, "Low rank cos Loss", low_rank_loss_value, train_steps,)
                        write_tensorboard(tb_writer, "Low rank norm Loss", loss_norm_value, train_steps,)
                        write_tensorboard(tb_writer, "Low rank param Gradient Norm", low_rank_para_grad_norm, train_steps,)

                    elif args.low_rank_dict != None:
                        logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, low_rank_grad_norm: {low_rank_para_grad_norm:.8f}, grad_norm_dict: {grad_norm_dict}, learning rate: {lr_scheduler.get_last_lr()}, Train Steps/Sec: {steps_per_sec:.2f}")

                        del low_rank_para_grad_norm
                        del grad_norm_dict
                    else:
                        if global_variable.TEST_LARGE_SCALE == False:
                            logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, grad_norm_dict: {grad_norm_dict}, learning rate: {lr_scheduler.get_last_lr()}, Train Steps/Sec: {steps_per_sec:.2f}")
                        else:
                            logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                            print(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                        del grad_norm_dict

                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    print("begin to save checkpoint")
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    ):
                        scaler_dict = None
                        opt_state_dict = FSDP.optim_state_dict(model, opt, optim_state_dict=opt.state_dict())
                        checkpoint = {
                            "model": model.state_dict(),
                            "opt": opt_state_dict,
                            "sch": lr_scheduler.state_dict(),
                        }

                    print(f"rank:{rank} Save model's state dict")

                    if global_variable.TP_ENABLE == False:
                        with FSDP.state_dict_type(
                            ema_fsdp,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        ):
                            checkpoint["ema"] = ema_fsdp.state_dict()

                        print(f"rank:{rank} Save ema's state dict")

                    if rank == 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    dist.barrier()

                    del checkpoint["model"]
                    del checkpoint["opt"]
                    del checkpoint["sch"]

                    if global_variable.TP_ENABLE == False:
                        del checkpoint["ema"]

                    del checkpoint

                torch.cuda.empty_cache()
                gc.collect()

                if args.use_profile:
                    prof.step()

    model.eval()
    if rank == 0 and global_variable.SAVE_ATTENTION_SCORE == True:
        global_variable.ATTENTION_SCORE_QUEUE.put(["exit", -1, -1, -1])
        save_thead.join()
    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./xxx.yaml")
    parser.add_argument("--DP", type=int, default=1)
    parser.add_argument("--TP", type=int, default=1)
    parser.add_argument("--CP", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--stop_step", type=int, default=30)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.stop_step = args.stop_step
    main(config)
