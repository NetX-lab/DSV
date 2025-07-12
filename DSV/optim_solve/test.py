import os
import sys

import hybrid_optimizer
import numpy as np

server_config = hybrid_optimizer.HardwareConfigManager.create_config_with_user_defined_specs(
    # GPU specifications
    "H100_SXM",  # GPU name
    1979.0,  # FP16 TFLOPS
    1979.0,  # BF16 TFLOPS
    3026.0,  # FP8 TFLOPS
    80.0,  # Memory GB
    2000.0,  # Memory bandwidth GB/s
    # Interconnect specifications
    "SXM_H100_IB",  # Interconnect name
    400.0,  # Intra-node unidirectional bandwidth GB/s
    200.0 / 8,  # Inter-node unidirectional bandwidth GB/s
    8,  # Max GPUs per node
    # Configuration
    8,  # GPUs per node
    1,  # Node count
)

print(server_config)

TOTAL_GPU_NUM = 8
HEAD_NUM = 16  # Attention heads
SEQ_LEN = 128000  # Sequence length
HEAD_DIM = 128  # Head dimension
MEM_BUFFER_PER_GPU = 64e9  # Memory per GPU (GB)
DATA_TYPE = "bf16"  # Data type


sparsity = [
    0.6,
    0.73,
    0.97,
    0.93,
    0.95,
    0.95,
    0.98,
    0.96,
    0.94,
    0.87,
    0.99,
    0.92,
    0.91,
    0.93,
    0.95,
    0.93,
]

sparsity = [
    0.8267,
    0.9984,
    0.9973,
    0.9980,
    0.9817,
    0.9930,
    0.9981,
    0.9918,
    0.9886,
    0.9632,
    0.9962,
    0.9898,
    0.8122,
    0.9097,
    0.9542,
    0.9985,
]

# sparsity = [0.88, 0.82, 0.81, 0.83, 0.85, 0.85, 0.88, 0.86, 0.84, 0.87, 0.86, 0.82, 0.81, 0.83, 0.85, 0.85]
# sparsity = [0.18, 0.15, 0.1, 0.13, 0.15, 0.15, 0.18, 0.16, 0.14, 0.17, 0.19, 0.12, 0.11, 0.13, 0.15, 0.15]

print("Sparsity:", sparsity)

# Create optimizer
optimizer = hybrid_optimizer.HybridSparsityOptimizer(
    server_config=server_config,
    HEAD_NUM=HEAD_NUM,
    SEQ_LEN=SEQ_LEN,
    HEAD_DIM=HEAD_DIM,
    MEM_BUFFER_PER_GPU=MEM_BUFFER_PER_GPU,
    DATA_TYPE=DATA_TYPE,
    compute_efficiency=0.3,
    comm_efficiency=0.7,
    scp_overhead=0.0005,
)

# Run optimization
best_config = optimizer.optimize(sparsity=sparsity, num_threads=1, verbose=True)

# Print optimal configuration
print("Optimal Configuration:")
print(best_config)

# Access fields
print("HCP group size:", best_config.HCP_GROUP_SIZE)
print("SCP group size:", best_config.SCP_GROUP_SIZE)
print("Total cost:", best_config.total_cost, "seconds")
print("Compute time:", best_config.compute_time, "seconds")
print("Communication time:", best_config.communication_time, "seconds")
print("Deployment strategy:", best_config.deployment_strategy)
print("Sparsity group index:", best_config.Reallocated_Sparsity_Idx_Per_Group)

# Get the sparsity value list per group with Reallocated_Sparsity_Idx_Per_Group
sparsity_value_list_per_group = []
comp_value_list_per_group = []
comp_burden_value_per_group = []
for i in range(best_config.HCP_GROUP_SIZE):
    sparsity_value_list_per_group.append([])
    comp_value_list_per_group.append([])
    for j in range(len(best_config.Reallocated_Sparsity_Idx_Per_Group[i])):
        sparsity_value_list_per_group[i].append(
            sparsity[best_config.Reallocated_Sparsity_Idx_Per_Group[i][j]]
        )
        comp_value_list_per_group[i].append(
            1.0 - sparsity[best_config.Reallocated_Sparsity_Idx_Per_Group[i][j]]
        )
    comp_burden_value_per_group.append(
        np.sum(comp_value_list_per_group[i]) / best_config.SCP_GROUP_SIZE
    )

print("Sparsity value list per group:", sparsity_value_list_per_group)
print("Comp burden value per group:", comp_burden_value_per_group)

print(
    f"naive avg comp burden with pure scp: {(len(sparsity) - sum(sparsity)) / TOTAL_GPU_NUM}"
)
