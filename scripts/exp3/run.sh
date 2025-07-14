set -x

time_str=$(date +%Y%m%d_%H%M) 
torchrun --nnodes=1 --nproc-per-node=4 test_sparse_gather_kv_correctness.py | tee output_correctness_${time_str}.log