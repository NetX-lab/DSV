set -x

time_str=$(date +%Y%m%d_%H%M) 
python test_low_rank_kernel.py --test_type breakdown | tee output_correctness_breakdown_${time_str}.log

python benchmark_fw_bw_correctness.py | tee output_correctness_fw_bw_correctness_${time_str}.log 