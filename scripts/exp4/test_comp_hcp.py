import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from DSV.models.all_to_all import all_to_all_4D
from DSV.models.parallel.sparse_qkv_all_to_all import (
    all_to_all_balanced_4D, solve_optimal_head_allocation)
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group import \
    sparse_group_attention

# to compare the computation difference of the naive hcp and our sparsity-aware hcp 

def init_distributed(rank,world_size):
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    return world_size, rank

def clean_up():
    dist.destroy_process_group() 

def get_sparse_kv_num_per_head(sparsity_per_head, kv_len, multiple_of=256):
    res = [int(((1-s) * kv_len + multiple_of-1) // multiple_of * multiple_of) for s in sparsity_per_head]
    return res



def generate_input_data(total_seq_len,head_num,head_dim,sparsity_per_head,dtype,group_size=32):

    world_size = dist.get_world_size()

    seq_len = total_seq_len//world_size

    q = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    grad = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)

    sparse_kv_num_per_head = get_sparse_kv_num_per_head(sparsity_per_head, seq_len)

    maximum_kv_num = max(sparse_kv_num_per_head)

    sparse_kv_index = torch.full((1, head_num, total_seq_len//group_size, maximum_kv_num), -1, device="cuda", dtype=torch.int32)

    for i in range(head_num):
        for j in range(total_seq_len//group_size):
            this_head_kv_num = sparse_kv_num_per_head[i]
            sparse_kv_index[0,i,j,:this_head_kv_num] = torch.randperm(total_seq_len, device="cuda", dtype=torch.int32)[:this_head_kv_num]

    sparse_kv_num_per_head = torch.tensor(sparse_kv_num_per_head, device="cuda", dtype=torch.int32).unsqueeze(0).repeat(q.shape[0],1)

    return q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head


def naive_all_to_all_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size):

    B,H,S,D = q.shape
    head_dim = D  

    rank = dist.get_rank() 
    world_size = dist.get_world_size()

    even_head_num = H//world_size

    this_rank_sparse_kv_index = sparse_kv_index[:,rank*even_head_num:(rank+1)*even_head_num,:,:]
    this_rank_sparse_kv_num_per_head = sparse_kv_num_per_head[:,rank*even_head_num:(rank+1)*even_head_num]

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    # End-to-end timing
    total_begin_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)    
    # Forward communication
    q_forward = all_to_all_4D(q,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)
    k_forward = all_to_all_4D(k,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)
    v_forward = all_to_all_4D(v,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)

    total_begin_event.record()

    output = sparse_group_attention(q_forward,k_forward,v_forward,False,1.0/(head_dim**0.5),this_rank_sparse_kv_index,this_rank_sparse_kv_num_per_head,group_size)

    # Backward computation
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_begin_event.elapsed_time(total_end_event)

    if rank == 0:
        print(f"[Rank {rank}] naive all_to_all end-to-end time: {total_time:.2f} ms")

    return total_time




def balanced_all_to_all_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head, sparsity_per_head,group_size):
    
    B,H,S,D = q.shape
    head_dim = D 

    rank = dist.get_rank() 
    world_size = dist.get_world_size()
    
    reallocated_head_idx_list, reallocated_head_num_list = solve_optimal_head_allocation(sparsity_per_head,world_size)

    if rank == 0:
        print(f"reallocated_head_idx_list: {reallocated_head_idx_list}, reallocated_head_num_list: {reallocated_head_num_list}") 

    head_start_idx = sum(reallocated_head_num_list[:rank])
    head_end_idx = head_start_idx + reallocated_head_num_list[rank]

    this_rank_head_ids = reallocated_head_idx_list[head_start_idx:head_end_idx]

    this_rank_sparse_kv_index = sparse_kv_index[:,this_rank_head_ids,:,:]
    this_rank_sparse_kv_num_per_head = sparse_kv_num_per_head[:,this_rank_head_ids]    

    q = q.transpose(1,2).contiguous()
    k = k.transpose(1,2).contiguous()
    v = v.transpose(1,2).contiguous()

    # End-to-end timing
    total_begin_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    q_forward = all_to_all_balanced_4D(q,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)
    k_forward = all_to_all_balanced_4D(k,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)
    v_forward = all_to_all_balanced_4D(v,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)

    total_begin_event.record()
    output = sparse_group_attention(q_forward,k_forward,v_forward,False,1.0/(head_dim**0.5),this_rank_sparse_kv_index,this_rank_sparse_kv_num_per_head,group_size) 

    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_begin_event.elapsed_time(total_end_event)

    if rank == 0:
        print(f"[Rank {rank}] balanced all_to_all end-to-end time: {total_time:.2f} ms")

    return total_time

if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc-per-node=4 test_comp_hcp.py
    world_size, rank = init_distributed(int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"]))

    total_seq_len = 256000
    head_num = 16
    head_dim = 128
    group_size = 32
    sparsity_per_head = [0.98,0.93,0.90,0.80,0.92, 0.80, 0.73, 0.99, 0.94,0.96,0.95,0.83, 0.81,0.97,0.95,0.93] # world 4 
    dtype = torch.bfloat16

    q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head = generate_input_data(total_seq_len,head_num,head_dim,sparsity_per_head,dtype)

    total_time_naive_comp = naive_all_to_all_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size)

    total_time_our_comp = balanced_all_to_all_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,sparsity_per_head,group_size)

    if rank == 0:
        print(f"{'='*60}")
        print(f"Total time of naive HCP comp: {total_time_naive_comp:.2f} ms")
        print(f"Total time of our Sparsity-aware HCP comp: {total_time_our_comp:.2f} ms")

        #plot the bar figure 
        methods = ['Naive HCP', 'Our Sparsity-aware HCP']
        total_times = [total_time_naive_comp, total_time_our_comp]
        x = np.arange(len(methods))
        plt.bar(x, total_times,width=0.4)
        plt.xticks(x, methods)
        plt.ylabel('Time (ms)',fontsize=14,fontweight='bold')
        plt.title('Comparison of Computation Time of Naive HCP and Our Sparsity-aware HCP',fontsize=14,fontweight='bold')
        plt.savefig('hcp_comp_comparison.pdf',dpi=300,bbox_inches='tight')
        print(f"Saved figure to hcp_comp_comparison.pdf")

    clean_up()



