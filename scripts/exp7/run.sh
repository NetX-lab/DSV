torchrun --nnodes=1 --nproc_per_node=4 ../../train_fsdp.py --config /data/DSV/DSV/configs/t2v/train_full_attn_flow_matching_cp_4_2d7_B.yaml  | tee output_full.txt  

torchrun --nnodes=1 --nproc_per_node=4 ../../train_fsdp_lr.py --config /data/DSV/DSV/configs/t2v/train_full_attn_flow_matching_cp_4_2d7_B_low_rank.yaml  | tee output_lr.txt

python plot_figure.py

