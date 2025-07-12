set -x

python test_low_rank_kernel.py --test_type breakdown | tee output_correctness_breakdown.log

python benchmark_fw_bw_correctness.py | tee output_correctness_fw_bw_correctness.log 