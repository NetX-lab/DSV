#!/usr/bin/bash
conda activate latte_flex_attn
set -x
torchrun --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port ${MASTER_PORT:-5678} --nnodes $NODE_COUNT --node_rank $NODE_RANK -- ${@:1}