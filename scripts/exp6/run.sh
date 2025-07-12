echo "Run Case 0"
torchrun --nnodes=1 --nproc-per-node=4 test_comp_comm_hcp_scp.py --case 0 --save_fig | tee output_case0.log

echo "Run Case 1"
torchrun --nnodes=1 --nproc-per-node=4 test_comp_comm_hcp_scp.py --case 1 --save_fig | tee output_case1.log
