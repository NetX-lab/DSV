from queue import Queue

import torch
import torch.distributed as dist

from DSV.models.low_rank_modules import LowRankModule

SAVE_ATTENTION_SCORE: bool = False
AGGREGATE_ATTENTION_SCORE: bool = False

RANK: int = -1

SAVE_STEP_INTERVAL: int = 500
CURRENT_STEP: int = 0
ATTENTION_LOG_STEP: int = 200
SPARSITY_UPDATE_STEP: int = 500
PROFILE_SPARSITY_STEP = None

ATTENTION_SCORE_QUEUE: Queue = Queue()

EXIT_SINGAL: bool = False

LOGGER = None

LOW_RANK_STAGE = 0  # 0 for align, 1 for decide
LOW_RANK_STAGE0_STEPS = 2500

CONTEXT_PARALLEL_GROUP: dist.ProcessGroup = None
DATA_PARALLEL_GROUP: dist.ProcessGroup = None
TENSOR_PARALLEL_GROUP: dist.ProcessGroup = None

TP_MESH = None
CP_MESH = None
CP_TP_MESH = None
DP_CP_MESH = None
FSDP_MESH = None


FORWARD_DONE = False

GRAD_SCALER = None

LOW_RANK_MODULE: LowRankModule = None

ATTENTION_MAP_SAVE_PATH: str = None
ATTENTION_MAP_SAVE_METADATA: dict = {}

CP_ENABLE: bool = False
TP_ENABLE: bool = False

WINDOW_ATTN_MASK: torch.Tensor = None


LOW_RANK_INFERENCE: bool = False


TEST_LARGE_SCALE: bool = False

TRITON_ATTENTION: bool = False


LOW_RANK_DICT: dict = None
