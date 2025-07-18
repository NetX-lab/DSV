set -x

time_str=$(date +%Y%m%d_%H%M) 
python test_low_rank_kernel.py --test_type speedup | tee output_speedup_${time_str}.log