time_str=$(date +%Y%m%d_%H%M)


torchrun --nnodes=1 --nproc_per_node=4 ../../train_fsdp.py --config ../../DSV/configs/t2v/train_full_attn_flow_matching_cp_4_2d7_B.yaml  --stop_step 40 | tee output_full_${time_str}.log  

torchrun --nnodes=1 --nproc_per_node=4 ../../train_fsdp_lr.py --config ../../DSV/configs/t2v/train_full_attn_flow_matching_cp_4_2d7_B_window.yaml  --stop_step 40 | tee output_window_${time_str}.log 

torchrun --nnodes=1 --nproc_per_node=4 ../../train_fsdp_lr.py --config ../../DSV/configs/t2v/train_full_attn_flow_matching_cp_4_2d7_B_low_rank.yaml  --stop_step 40 | tee output_lr_${time_str}.log

python plot_figure.py

