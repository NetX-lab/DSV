import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from DSV.models.all_to_all import all_to_all_4D
from DSV.models.parallel.sparse_kv_gather import (CommGroupManager,
                                                  SparseKVGather)
from DSV.models.parallel.sparse_qkv_all_to_all import (
    all_to_all_balanced_4D, solve_optimal_head_allocation)
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group import \
    sparse_group_attention
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group_compressed_kv import \
    compressed_sparse_group_attention


def init_dist(rank,world_size):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()





def get_sparse_kv_num_per_head(sparsity_per_head, kv_len, multiple_of=256):
    res = [int(((1-s) * kv_len + multiple_of-1) // multiple_of * multiple_of) for s in sparsity_per_head]
    return res


def generate_input_data(total_seq_len,head_num,head_dim,sparsity_per_head,dtype,group_size=32):

    torch.manual_seed(42) 

    world_size = dist.get_world_size()

    seq_len = total_seq_len//world_size

    q = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)
    grad = torch.randn(1, head_num, seq_len, head_dim, device="cuda", dtype=dtype)

    sparse_kv_num_per_head = get_sparse_kv_num_per_head(sparsity_per_head, seq_len)

    maximum_kv_num = max(sparse_kv_num_per_head)

    sparse_kv_index = torch.full((1, head_num, seq_len//group_size, maximum_kv_num), -1, device="cuda", dtype=torch.int32)

    for i in range(head_num):
        group_num = seq_len//group_size 
        for j in range(0,group_num,200): # 100 is hardcoded for current total length 256000
            this_head_kv_num = sparse_kv_num_per_head[i]
            sparse_kv_index[0,i,j:j+200,:this_head_kv_num] = torch.randperm(total_seq_len, device="cuda", dtype=torch.int32)[:this_head_kv_num]

    sparse_kv_index_for_all_to_all = torch.full((1, head_num, total_seq_len//group_size, maximum_kv_num), -1, device="cuda", dtype=torch.int32)
    for i in range(head_num):
        group_num = total_seq_len//group_size 
        for j in range(0,group_num,200): # 100 is hardcoded for current total length 256000
            this_head_kv_num = sparse_kv_num_per_head[i]
            sparse_kv_index_for_all_to_all[0,i,j:j+200,:this_head_kv_num] = torch.randperm(total_seq_len, device="cuda", dtype=torch.int32)[:this_head_kv_num]

    sparse_kv_num_per_head = torch.tensor(sparse_kv_num_per_head, device="cuda", dtype=torch.int32).unsqueeze(0).repeat(q.shape[0],1)

    return q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,sparse_kv_index_for_all_to_all


def naive_all_to_all_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size):

    B,H,S,D = q.shape
    head_dim = D  

    rank = dist.get_rank() 
    world_size = dist.get_world_size()

    even_head_num = H//world_size

    this_rank_sparse_kv_index = sparse_kv_index[:,rank*even_head_num:(rank+1)*even_head_num,:,:]
    this_rank_sparse_kv_num_per_head = sparse_kv_num_per_head[:,rank*even_head_num:(rank+1)*even_head_num]

    B,H,num_group, _ = this_rank_sparse_kv_index.shape 

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    # End-to-end timing
    total_begin_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)

    total_begin_event.record()
    
    # Forward communication
    q_forward = all_to_all_4D(q,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)
    k_forward = all_to_all_4D(k,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)
    v_forward = all_to_all_4D(v,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)

    # Forward computation
    output = sparse_group_attention(q_forward,k_forward,v_forward,False,1.0/(head_dim**0.5),this_rank_sparse_kv_index,this_rank_sparse_kv_num_per_head,group_size)

    # Backward computation
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    # Output communication
    output_fwd = all_to_all_4D(output.transpose(1,2),scatter_idx=1,gather_idx=2,group=dist.group.WORLD).transpose(1,2).contiguous()

    # Backward communication
    output_bwd = all_to_all_4D(output.transpose(1,2),scatter_idx=1,gather_idx=2,group=dist.group.WORLD).transpose(1,2).contiguous()  
    q_forward = all_to_all_4D(q,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)
    k_forward = all_to_all_4D(k,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)
    v_forward = all_to_all_4D(v,scatter_idx=2,gather_idx=1,group=dist.group.WORLD).transpose(1,2).contiguous().requires_grad_(True)



    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_begin_event.elapsed_time(total_end_event)

    if rank == 0:
        print(f"[Rank {rank}] naive all_to_all end-to-end time: {total_time:.2f} ms")

    return total_time

def selective_gather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size): 

    B,H,local_seq_len,D = q.shape
    head_dim = D 

    rank = dist.get_rank() 
    world_size = dist.get_world_size()
    total_seq_len = local_seq_len * world_size

    comm_manager = CommGroupManager(cp_group=dist.group.WORLD)
    comm_manager.init_cp_group()
    sparse_kv_gather = SparseKVGather(comm_manager)

    # End-to-end timing
    total_begin_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)


    # Get sparse indices and gather compressed KV
    activated_mask, topk_index_to_packed_index = sparse_kv_gather.get_activated_indices_from_topk_indices(
        sparse_kv_index, sparse_kv_num_per_head[0], total_seq_len)


    total_begin_event.record()
    gathered_k, gathered_v = sparse_kv_gather(k.requires_grad_(True), v.requires_grad_(True), activated_mask, auto_grad=True)

    # Forward and backward computation
    #print(f"q.shape: {q.shape}, gathered_k.shape: {gathered_k.shape}, gathered_v.shape: {gathered_v.shape}")
    output = compressed_sparse_group_attention(q.requires_grad_(True), gathered_k.requires_grad_(True), gathered_v.requires_grad_(True), False, 1.0/(head_dim**0.5), sparse_kv_index, sparse_kv_num_per_head, topk_index_to_packed_index, group_size).requires_grad_(True)
    
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    dk, dv = torch.rand_like(gathered_k), torch.rand_like(gathered_v)

    sparse_kv_gather.backward_kv_gradients(dk, dv, False)
    total_end_event.record()

    torch.cuda.synchronize()

    total_time = total_begin_event.elapsed_time(total_end_event)

    if rank == 0:
        print(f"[Rank {rank}] selective gather end-to-end time: {total_time:.2f} ms")

    return total_time


def naive_allgather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size): 

    B,H,S,D = q.shape
    head_dim = D 

    rank = dist.get_rank() 
    world_size = dist.get_world_size()
    total_seq_len = S * world_size

    # End-to-end timing
    total_begin_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)

    total_begin_event.record()

    # Forward all-gather
    global_k = [torch.empty_like(k) for _ in range(world_size)]
    global_v = [torch.empty_like(v) for _ in range(world_size)]

    dist.all_gather(global_k, k)
    dist.all_gather(global_v, v)

    global_k = torch.cat(global_k, dim=2).requires_grad_(True)
    global_v = torch.cat(global_v, dim=2).requires_grad_(True)

    # Forward computation
    #print(f"q.shape: {q.shape}, global_k.shape: {global_k.shape}, global_v.shape: {global_v.shape}")
    output = sparse_group_attention(q.requires_grad_(True), global_k, global_v, False, 1.0/(head_dim**0.5), sparse_kv_index, sparse_kv_num_per_head, group_size)
    
    # Backward computation
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    # Backward all-gather and reduce-scatter (simulate gradient communication)
    global_k_grad = [torch.empty_like(k) for _ in range(world_size)]
    global_v_grad = [torch.empty_like(v) for _ in range(world_size)]
    
    dist.all_gather(global_k_grad, k)
    dist.all_gather(global_v_grad, v)
    
    reduce_k = torch.empty_like(k)
    reduce_v = torch.empty_like(v)
    dist.reduce_scatter(reduce_k, global_k_grad, op=dist.ReduceOp.SUM) 
    dist.reduce_scatter(reduce_v, global_v_grad, op=dist.ReduceOp.SUM)

    total_end_event.record()

    torch.cuda.synchronize()

    total_time = total_begin_event.elapsed_time(total_end_event)

    if rank == 0:
        print(f"[Rank {rank}] naive allgather end-to-end time: {total_time:.2f} ms")

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

    total_begin_event.record()

    # Forward communication
    q_forward = all_to_all_balanced_4D(q,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)
    k_forward = all_to_all_balanced_4D(k,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)
    v_forward = all_to_all_balanced_4D(v,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)

    # Forward computation
    output = sparse_group_attention(q_forward,k_forward,v_forward,False,1.0/(head_dim**0.5),this_rank_sparse_kv_index,this_rank_sparse_kv_num_per_head,group_size) 

    # Backward communication
    output_fwd = all_to_all_balanced_4D(output.transpose(1,2),scatter_idx=1,gather_idx=2,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous()

    # Backward computation
    grad_output = torch.randn_like(output)
    output.backward(grad_output)


    # backward communication
    output_bwd = all_to_all_balanced_4D(output.transpose(1,2),scatter_idx=1,gather_idx=2,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous()
    q_forward = all_to_all_balanced_4D(q,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)
    k_forward = all_to_all_balanced_4D(k,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)
    v_forward = all_to_all_balanced_4D(v,scatter_idx=2,gather_idx=1,group=dist.group.WORLD,reallocated_head_idx_list=reallocated_head_idx_list,reallocated_head_num_list=reallocated_head_num_list).transpose(1,2).contiguous().requires_grad_(True)

    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_begin_event.elapsed_time(total_end_event)

    if rank == 0:
        print(f"[Rank {rank}] balanced all_to_all end-to-end time: {total_time:.2f} ms")

    return total_time

def run_comparison_sparse_hcp_choice(q,k,v,grad,sparse_kv_index,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,sparsity_per_head,group_size,save_fig=False):
    
    rank = dist.get_rank()

    # Warmup
    for i in range(5):
        _ = naive_allgather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size) 
    # Benchmark naive allgather
    times_naive_allgather = []
    for i in range(10):
        total_time = naive_allgather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size)
        times_naive_allgather.append(total_time)

    for i in range(5):
        _ = naive_all_to_all_comp(q,k,v,grad,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,group_size)

    # Benchmark naive all-to-all
    times_naive_alltoall = []
    for i in range(10):
        total_time = naive_all_to_all_comp(q,k,v,grad,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,group_size)
        times_naive_alltoall.append(total_time)

    for i in range(5):
        _ = balanced_all_to_all_comp(q,k,v,grad,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,sparsity_per_head,group_size)
        torch.cuda.synchronize()
    # Benchmark balanced all-to-all
    times_balanced_alltoall = []
    for i in range(10):
        total_time = balanced_all_to_all_comp(q,k,v,grad,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,sparsity_per_head,group_size)
        torch.cuda.synchronize()
        times_balanced_alltoall.append(total_time)

    # Calculate averages
    avg_time_naive_allgather = np.mean(times_naive_allgather)
    avg_time_naive_alltoall = np.mean(times_naive_alltoall)
    avg_time_balanced_alltoall = np.mean(times_balanced_alltoall)

    if rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        methods = ['Naive HCP + Sparse Attention', 'Naive SCP + Sparse Attention', 'Ours']
        total_times = [avg_time_naive_alltoall, avg_time_naive_allgather, avg_time_balanced_alltoall]
        
        x = np.arange(len(methods))
        width = 0.6
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        bars = ax.bar(x, total_times, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, total_times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(total_times)*0.01,
                    f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Calculate speedup
        baseline_time = avg_time_naive_alltoall  # naive all-to-all as baseline
        balanced_time = avg_time_balanced_alltoall  # balanced all-to-all
        speedup = baseline_time / balanced_time
        
        ax.text(1, max(total_times) * 0.8, f'Our Speedup: {speedup:.2f}x\n(vs Naive HCP)', 
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8),
                fontsize=13, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('End-to-End Time (ms)', fontsize=14, fontweight='bold')
        ax.set_title('Performance Comparison of naive HCP, SCP and Ours', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.set_ylim(0, max(total_times) * 1.15)
        
        plt.tight_layout()
        
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        if save_fig:
            plt.savefig(f'parallelism_benchmark_case1_{current_time}.pdf', dpi=300, bbox_inches='tight')
            print(f"Saved figure to parallelism_benchmark_case1_{current_time}.pdf")
        #plt.show()

        
        print(f"\n{'='*60}")
        print(f"END-TO-END PERFORMANCE COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Naive HCP + Sparse Attention:")
        print(f"  End-to-End Time: {avg_time_naive_alltoall:.2f} ms")
        print(f"Naive SCP + Sparse Attention:")
        print(f"  End-to-End Time: {avg_time_naive_allgather:.2f} ms")
        print(f"Ours:")
        print(f"  End-to-End Time: {avg_time_balanced_alltoall:.2f} ms")
        print(f"\nSpeedup: {speedup:.2f}x (vs Naive HCP + Sparse Attention)")
        print(f"\033[93mNote: When using fewer than 8 GPUs, the difference in communication time between naive SCP's allgather and HCP's alltoall is less pronounced than with larger GPU counts. This is due to the communication complexity of each method becoming more significant as the number of GPUs increases.\033[0m")
        print(f"{'='*60}")

    return avg_time_naive_alltoall, avg_time_balanced_alltoall
    

def run_comparison_sparse_scp_choice(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,sparse_kv_index_for_all_to_all,sparsity_per_head,group_size,save_fig=False):

    rank = dist.get_rank()

    # Warmup
    for i in range(5):
        _ = naive_allgather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size) 
    # Benchmark naive allgather
    times_naive_allgather = []
    for i in range(10):
        total_time = naive_allgather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size)
        torch.cuda.synchronize()
        times_naive_allgather.append(total_time)

    for i in range(5):
        _ = naive_all_to_all_comp(q,k,v,grad,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,group_size)
        torch.cuda.synchronize()

    # Benchmark naive all-to-all
    times_naive_alltoall = []
    for i in range(10):
        total_time = naive_all_to_all_comp(q,k,v,grad,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,group_size)
        torch.cuda.synchronize()
        times_naive_alltoall.append(total_time)

    for i in range(5):
        _ = selective_gather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size)
        torch.cuda.synchronize()

    # Benchmark selective gather
    times_selective_gather = []
    for i in range(10):
        total_time = selective_gather_comp(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,group_size)
        torch.cuda.synchronize()
        times_selective_gather.append(total_time)

    # Calculate averages
    avg_time_naive_allgather = np.mean(times_naive_allgather)
    avg_time_naive_alltoall = np.mean(times_naive_alltoall)
    avg_time_selective_gather = np.mean(times_selective_gather)

    if rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        methods = ['Naive HCP + Sparse Attention', 'Naive SCP + Sparse Attention', 'Ours']
        total_times = [avg_time_naive_alltoall, avg_time_naive_allgather, avg_time_selective_gather]
        
        x = np.arange(len(methods))
        width = 0.6

        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        bars = ax.bar(x, total_times, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, total_times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(total_times)*0.01,
                    f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('End-to-End Time (ms)', fontsize=14, fontweight='bold')
        ax.set_title('Performance Comparison of naive HCP, SCP and Ours', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        ax.set_ylim(0, max(total_times) * 1.15)
        
        plt.tight_layout()
        
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        if save_fig:
            plt.savefig(f'parallelism_benchmark_case0_{current_time}.pdf', dpi=300, bbox_inches='tight')
            print(f"Saved figure to parallelism_benchmark_case0_{current_time}.pdf")
        #plt.show()

        print(f"\n{'='*60}")
        print(f"END-TO-END PERFORMANCE COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Naive HCP + Sparse Attention:")
        print(f"  End-to-End Time: {avg_time_naive_alltoall:.2f} ms")
        print(f"Naive SCP + Sparse Attention:")
        print(f"  End-to-End Time: {avg_time_naive_allgather:.2f} ms")
        print(f"Ours:")
        print(f"  End-to-End Time: {avg_time_selective_gather:.2f} ms")
        print(f"\nSpeedup: {avg_time_naive_alltoall/avg_time_selective_gather:.2f}x (vs Naive HCP + Sparse Attention)")
        print(f"Speedup: {avg_time_naive_allgather/avg_time_selective_gather:.2f}x (vs Naive SCP + Sparse Attention)")
        print(f"{'='*60}")
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=0,choices=[0,1])
    parser.add_argument("--save_fig", action='store_true')
    args = parser.parse_args()

    # torchrun --nnodes=1 --nproc-per-node=4 test_comp_comm_hcp_scp.py --case [1 or 2]

    save_fig = args.save_fig if args.save_fig else False

    init_dist(int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"]))

    if args.case == 1:
        total_seq_len = 128000
        head_num = 16
        head_dim = 128
        group_size = 32
        sparsity_per_head = [0.93,0.93,0.94,0.92, 0.95, 0.94, 0.92, 0.95, 0.95,0.94,0.94,0.92, 0.92,0.91,0.91,0.91]
        dtype = torch.bfloat16

        q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,sparse_kv_index_for_all_to_all = generate_input_data(total_seq_len,head_num,head_dim,sparsity_per_head,dtype)

        run_comparison_sparse_hcp_choice(q,k,v,grad,sparse_kv_index,sparse_kv_index_for_all_to_all,sparse_kv_num_per_head,sparsity_per_head,group_size,save_fig=save_fig)

        cleanup()

    elif args.case == 0:
        total_seq_len = 64000
        head_num = 16
        head_dim = 128
        group_size = 32

        sparsity_per_head = [0.88,0.96,0.98,0.88,0.90, 0.85, 0.82, 0.86, 0.94,0.96,0.95,0.97, 0.96,0.96,0.95,0.93] # world size 4 
        dtype = torch.bfloat16

        q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,sparse_kv_index_for_all_to_all = generate_input_data(total_seq_len,head_num,head_dim,sparsity_per_head,dtype)

        run_comparison_sparse_scp_choice(q,k,v,grad,sparse_kv_index,sparse_kv_num_per_head,sparse_kv_index_for_all_to_all,sparsity_per_head,group_size,save_fig=save_fig)

        cleanup()
    else:
        print("Invalid case")
        exit()







