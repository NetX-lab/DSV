time_str=$(date +%Y%m%d_%H%M) 
echo "Run Case 0"
torchrun --nnodes=1 --nproc-per-node=4 test_comp_comm_hcp_scp.py --case 0 --save_fig | tee output_case0_${time_str}.log

echo "Run Case 1"
torchrun --nnodes=1 --nproc-per-node=4 test_comp_comm_hcp_scp.py --case 1 --save_fig | tee output_case1_${time_str}.log
