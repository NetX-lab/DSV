import os

import torch
import torch.distributed as dist

from DSV.models.parallel.sparse_kv_gather import \
    test_sparse_kv_gather_backward_latency


def init_distributed(rank,world_size):
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    return world_size, rank

def clean_up():
    dist.destroy_process_group()


def naive_scp_allgather(q,k,v): 
    rank = dist.get_rank()

    total_begin_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)

    total_begin_event.record()

    global_k = [torch.empty_like(k) for _ in range(world_size)]
    global_v = [torch.empty_like(v) for _ in range(world_size)]

    #forward 
    dist.all_gather(global_k,k)
    dist.all_gather(global_v,v)


    #backward 
    dist.reduce_scatter(k,global_k,op=dist.ReduceOp.SUM)
    dist.reduce_scatter(v,global_v,op=dist.ReduceOp.SUM)

    total_end_event.record()
    torch.cuda.synchronize()
    total_time = total_begin_event.elapsed_time(total_end_event)

    #print(f"Rank {rank} naive scp allgather time: {total_time:.2f} ms")


    return total_time

def benchmark_naive_scp_allgather(B,H,S,D,dtype):
    world_size = dist.get_world_size()
    local_seq_len = S // world_size
    q = torch.randn(B,H,local_seq_len,D,device="cuda",dtype=dtype)
    k = torch.randn(B,H,local_seq_len,D,device="cuda",dtype=dtype)
    v = torch.randn(B,H,local_seq_len,D,device="cuda",dtype=dtype)

    for i in range(5):
        naive_scp_allgather(q,k,v)

    begin_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)


    test_num = 10

    begin_event.record()
    for i in range(test_num):
        naive_scp_allgather(q,k,v)
    end_event.record()

    torch.cuda.synchronize()

    return begin_event.elapsed_time(end_event)/test_num


if __name__ == "__main__":
    # torchrun --nnodes=1 --nproc-per-node=4 test_comm_scp.py 
    B = 1
    H = 16 
    S = 256000
    D = 128
    dtype = torch.bfloat16

    world_size, rank = init_distributed(int(os.environ.get("RANK")),int(os.environ.get("WORLD_SIZE")))

    our_forward_comm_latency, our_backward_comm_latency = test_sparse_kv_gather_backward_latency(B,H,S,D,[0.9]*H)

    naive_scp_allgather_time = benchmark_naive_scp_allgather(B,H,S,D,dtype)

    if rank == 0:
        print(f"{'='*60}")
        print(f"Naive SCP Allgather Time in forward + backward: {naive_scp_allgather_time:.2f} ms")
        print(f"Our Sparsity-aware SCP communication time in forward + backward: {our_forward_comm_latency+our_backward_comm_latency:.2f} ms")
        print(f"Speedup: {naive_scp_allgather_time/(our_forward_comm_latency+our_backward_comm_latency):.2f}x") 

        #plot the figure 
        import matplotlib.pyplot as plt
        import numpy as np

        methods = ['Naive SCP Allgather', 'Our Sparsity-aware SCP']
        total_times = [naive_scp_allgather_time, our_forward_comm_latency+our_backward_comm_latency]

        x = np.arange(len(methods))

        plt.bar(x, total_times,width=0.4)
        plt.xticks(x, methods)
        plt.ylabel('Time (ms)',fontsize=14,fontweight='bold')
        plt.title('Comparison of Communication Time of Naive SCP Allgather and Our Sparsity-aware SCP',fontsize=14,fontweight='bold')
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        plt.savefig(f'scp_comm_time_comparison_{current_time}.pdf',dpi=300,bbox_inches='tight')
        # annotate the speedup for Our Sparsity-aware SCP over Naive SCP Allgather

        print(f"Saved figure to scp_comm_time_comparison_{current_time}.pdf")

    clean_up()

