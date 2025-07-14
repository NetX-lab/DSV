import math
from typing import List, Union, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.attention import SDPBackend, sdpa_kernel

import DSV.models.global_variable as global_variable
from DSV.models.window_utils import generate_window_attention_kvindex_mask_tensors
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group import sparse_group_attention
from DSV.triton_plugin.fused_attention_no_causal_sparse_query_group_w import sparse_window_attention


class CPCommunicationManager:
    def __init__(self, dp_size, cp_size, layer_num):
        self.dp_size = dp_size
        self.cp_size = cp_size

        self.layer_num = layer_num

        self.hcp_size = cp_size
        self.scp_size = 1

        self.hcp_group_per_layer = [None] * self.layer_num
        self.scp_group_per_layer = [None] * self.layer_num

        self.hcp_group_size_per_layer = [None] * self.layer_num
        self.scp_group_size_per_layer = [None] * self.layer_num

        self.__init_hcp_scp_for_each_layer(
            [
                [self.hcp_size, self.scp_size, "hcp_first_intra_node"]
                for _ in range(self.layer_num)
            ]
        )

    def __init_hcp_scp_for_each_layer(
        self, configs_per_layer: List[List[Union[int, str]]]
    ):
        for layer_idx in range(self.layer_num):
            if configs_per_layer[layer_idx] == []:
                new_sub_mesh = init_device_mesh(
                    "cuda",
                    [self.dp_size, self.scp_size, self.hcp_size],
                    mesh_dim_names=["dp", "scp", "hcp"],
                )

                self.hcp_group_per_layer[layer_idx] = new_sub_mesh.get_group("hcp")
                self.scp_group_per_layer[layer_idx] = new_sub_mesh.get_group("scp")

                self.hcp_group_size_per_layer[layer_idx] = dist.get_world_size(
                    self.hcp_group_per_layer[layer_idx]
                )
                self.scp_group_size_per_layer[layer_idx] = dist.get_world_size(
                    self.scp_group_per_layer[layer_idx]
                )

            else:
                hcp_size, scp_size, priority = configs_per_layer[layer_idx]

                assert priority in [
                    "hcp_first_intra_node",
                    "scp_first_intra_node",
                ], f"priority must be 'hcp_first_intra_node' or 'scp_first_intra_node'"

                if priority == "hcp_first_intra_node":
                    new_sub_mesh = init_device_mesh(
                        "cuda",
                        [self.dp_size, self.scp_size, self.hcp_size],
                        mesh_dim_names=["dp", "scp", "hcp"],
                    )
                else:
                    new_sub_mesh = init_device_mesh(
                        "cuda",
                        [self.dp_size, self.hcp_size, self.scp_size],
                        mesh_dim_names=["dp", "hcp", "scp"],
                    )

                self.hcp_group_per_layer[layer_idx] = new_sub_mesh.get_group("hcp")
                self.scp_group_per_layer[layer_idx] = new_sub_mesh.get_group("scp")

                self.hcp_group_size_per_layer[layer_idx] = dist.get_world_size(
                    self.hcp_group_per_layer[layer_idx]
                )
                self.scp_group_size_per_layer[layer_idx] = dist.get_world_size(
                    self.scp_group_per_layer[layer_idx]
                )

    def adjust_cp_config_this_layer(
        self, layer_idx: int, configs: List[Union[int, int, str]]
    ):
        """
        Adjust the cp config for a specific layer.

        args:
            layer_idx: int, the layer index
            configs: List[Union[int,str]], the cp config for the layer (3 elements), [hcp_size, scp_size, priority]
                hcp_size: int, the size of the hcp group
                scp_size: int, the size of the scp group
                priority: str, the priority of the cp config, 'hcp_first_intra_node' or 'scp_first_intra_node'
        """
        hcp_size, scp_size, priority = configs

        assert priority in [
            "hcp_first_intra_node",
            "scp_first_intra_node",
        ], f"priority must be 'hcp_first_intra_node' or 'scp_first_intra_node'"
        assert (
            hcp_size * scp_size == self.cp_size
        ), f"hcp_size * scp_size must be equal to cp_size"

        if priority == "hcp_first_intra_node":
            new_sub_mesh = init_device_mesh(
                "cuda",
                [self.dp_size, self.scp_size, self.hcp_size],
                mesh_dim_names=["dp", "scp", "hcp"],
            )
        else:
            new_sub_mesh = init_device_mesh(
                "cuda",
                [self.dp_size, self.hcp_size, self.scp_size],
                mesh_dim_names=["dp", "hcp", "scp"],
            )

        self.hcp_group_per_layer[layer_idx] = new_sub_mesh.get_group("hcp")
        self.scp_group_per_layer[layer_idx] = new_sub_mesh.get_group("scp")

        self.hcp_group_size_per_layer[layer_idx] = dist.get_world_size(
            self.hcp_group_per_layer[layer_idx]
        )
        self.scp_group_size_per_layer[layer_idx] = dist.get_world_size(
            self.scp_group_per_layer[layer_idx]
        )

    def get_hcp_group_this_layer(self, layer_idx: int):
        return self.hcp_group_per_layer[layer_idx]

    def get_scp_group_this_layer(self, layer_idx: int):
        return self.scp_group_per_layer[layer_idx]

    def get_hcp_group_size_this_layer(self, layer_idx: int):
        return self.hcp_group_size_per_layer[layer_idx]

    def get_scp_group_size_this_layer(self, layer_idx: int):
        return self.scp_group_size_per_layer[layer_idx]


def extract_middle_tokens(tensor, num_groups, seq_len_per_group):
    group_indices = torch.arange(num_groups, device=tensor.device, dtype=torch.long)
    start_positions = group_indices * seq_len_per_group
    middle_positions = start_positions + seq_len_per_group // 2
    return tensor[:, :, middle_positions, :]  # [B, H, num_groups, D]

class LowRankModule(torch.nn.Module):
    def __init__(
        self, num_layers, inner_dim, low_rank_dim, n_heads, device, low_rank_dict
    ):
        super().__init__()
        self.num_layers = num_layers
        self.inner_dim = inner_dim
        self.low_rank_dim = low_rank_dim
        self.low_rank_head_dim = low_rank_dim // n_heads
        self.n_heads = n_heads
        self.device = device

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.mse_loss_weight = low_rank_dict["mse_loss_weight"]
        self.cosine_loss_weight = low_rank_dict["cosine_loss_weight"]

        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

        self._init_low_rank_linears()
        self.__init_window_kv_index_mask()

        self.register_buffer(
            "sparsity_per_layers",
            torch.zeros(self.num_layers, device=self.device, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "sparsity_per_head_per_layers",
            torch.zeros(
                self.num_layers, self.n_heads, device=self.device, dtype=torch.float32
            ),
            persistent=True,
        )


    def __init_window_kv_index_mask(self):
        
        w_cube_size = global_variable.LOW_RANK_DICT.get("window_size", None)

        if w_cube_size is not None:
           
            w_f, w_h, w_w = w_cube_size[0], w_cube_size[1], w_cube_size[2]

            self.window_kv_index, self.window_group_mask = generate_window_attention_kvindex_mask_tensors(
                video_shape=global_variable.LOW_RANK_DICT.get("video_size", None),
                cube_shape=global_variable.LOW_RANK_DICT.get("cube_size", None),
                unified_window_size=global_variable.LOW_RANK_DICT.get("window_size", None),
                device=self.device
            )

            self.cube_size = global_variable.LOW_RANK_DICT.get("cube_size", None)
            self.group_size = math.prod(self.cube_size)
            self.window_size = math.prod(w_cube_size)


        
    def _init_low_rank_linears(self):
        self.low_rank_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.inner_dim,
                    self.low_rank_dim * 2,
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]
        )

        # init the low rank linears
        for i in range(self.num_layers):
            assert (
                self.low_rank_linears[i].weight.requires_grad == True
                and self.low_rank_linears[i].bias.requires_grad == True
            ), f"low rank linear {i} weight and bias requires grad should be True"
            torch.nn.init.kaiming_normal_(self.low_rank_linears[i].weight)
            torch.nn.init.zeros_(self.low_rank_linears[i].bias)

    def sync_gradient_across_ranks(self, group=None):
        params = [p for p in self.parameters() if p.grad is not None]

        # use PyTorch API to flatten all gradients
        flat_grad = _flatten_dense_tensors([p.grad for p in params])

        assert flat_grad.dtype == torch.float32, "Gradient dtype must be float32"

        # sync gradients
        dist.all_reduce(flat_grad)
        flat_grad.div_(dist.get_world_size())

        # restore gradient shape
        grads = _unflatten_dense_tensors(flat_grad, [p.grad for p in params])

        # update gradients
        for param, grad in zip(params, grads):
            param.grad.copy_(grad)

    def profile_sparsity(self, query, key, layer_idx, sum_score_threshold=0.99):
        # query key: b,h,s,d

        key_T = key.transpose(-1, -2)

        with torch.no_grad():
            chunk_size = 1024  # could be modified to any reasonable value, or even be removed; Dependent on the memory
            B, H, S, D = query.shape
            sparsity_chunks = []

            for i in range(0, S, chunk_size):
                chunk_end = min(i + chunk_size, S)
                query_chunk = query[:, :, i:chunk_end, :]

                atten_score = ((query_chunk @ key_T) / math.sqrt(D)).float().softmax(-1)

                del query_chunk

                atten_score, atten_score_idx = torch.sort(
                    atten_score, dim=-1, descending=True
                )

                del atten_score_idx

                atten_score.cumsum_(dim=-1)

                mask = atten_score >= sum_score_threshold

                chunk_sparsity = mask.sum(-1, keepdim=True).float()
                chunk_sparsity.div_(mask.size(-1))
                sparsity_chunks.append(chunk_sparsity)

                del atten_score, mask

            sparsity_per_query = torch.cat(sparsity_chunks, dim=2)  # B,H,S,1
            sparsity_per_query = sparsity_per_query.squeeze(-1)  # B,H,S

            sparsity_flat = sparsity_per_query.view(-1)

            n = len(sparsity_flat)

            sparsity_list = [sparsity_flat.min()]

            percentiles = [0.001, 0.005, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.95]
            k_values = [max(1, int(p * n)) for p in percentiles]

            for k in k_values:
                val = torch.kthvalue(sparsity_flat, k).values
                sparsity_list.append(val)

            # compute sparsity per head
            sparsity_per_head = (
                sparsity_per_query.permute(1, 0, 2).contiguous().view(self.n_heads, -1)
            )

            sparsity_per_head_list = [
                sparsity_per_head.min(dim=1)[0],
            ]

            percentiles = [0.001, 0.005, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.95]
            k_values = [max(1, int(p * sparsity_per_head.size(1))) for p in percentiles]

            for k in k_values:
                val = torch.kthvalue(sparsity_per_head, k).values
                sparsity_per_head_list.append(val)

            old_sparsity = None
            old_sparsity_per_head = None

            if self.sparsity_per_layers[layer_idx] == 0:
                self.sparsity_per_layers[layer_idx] = sparsity_list[1]
                self.sparsity_per_head_per_layers[layer_idx] = sparsity_per_head_list[1]
                # original 0
            else:
                momentum = 0.9
                current_min = sparsity_list[1]
                old_sparsity = self.sparsity_per_layers[layer_idx].clone()
                self.sparsity_per_layers[layer_idx] = (
                    momentum * old_sparsity + (1 - momentum) * current_min
                )

                old_sparsity_per_head = self.sparsity_per_head_per_layers[
                    layer_idx
                ].clone()
                self.sparsity_per_head_per_layers[layer_idx] = (
                    momentum * old_sparsity_per_head
                    + (1 - momentum) * sparsity_per_head_list[1]
                )

            if global_variable.RANK == 0:
                print(
                    f"block_id:{layer_idx}, min:{sparsity_list[0]}, P0.1:{sparsity_list[1]}, P0.5:{sparsity_list[2]}, P1:{sparsity_list[3]}, P2:{sparsity_list[4]}, P5:{sparsity_list[5]}, P25:{sparsity_list[6]}, P50:{sparsity_list[7]}, P75:{sparsity_list[8]}, P95:{sparsity_list[9]}"
                )
                print(
                    f"block_id:{layer_idx}, sparsity_per_head min:{sparsity_per_head_list[0]}, P0.1:{sparsity_per_head_list[1]}, P0.5:{sparsity_per_head_list[2]}, P1:{sparsity_per_head_list[3]}, P2:{sparsity_per_head_list[4]}, P5:{sparsity_per_head_list[5]}, P25:{sparsity_per_head_list[6]}, P50:{sparsity_per_head_list[7]}, P75:{sparsity_per_head_list[8]}, P95:{sparsity_per_head_list[9]}"
                )
                print(
                    f"block_id:{layer_idx}, sparsity before:{old_sparsity}, sparsity new:{self.sparsity_per_layers[layer_idx]}"
                )
                print(
                    f"block_id:{layer_idx}, sparsity_per_head before:{old_sparsity_per_head}, "
                    f"sparsity_per_head new:{self.sparsity_per_head_per_layers[layer_idx]}"
                )

            del sparsity_chunks, sparsity_per_query, sparsity_flat, sparsity_per_head

            return self.sparsity_per_layers[layer_idx]

    def compute_low_rank_loss(self, hidden_states, query, key, layer_idx):
        low_rank_linear = self.low_rank_linears[layer_idx]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            B, N, C = hidden_states.shape

            mse_loss_weight = self.mse_loss_weight
            cosine_loss_weight = self.cosine_loss_weight

            sampled_interval_across_bs = 64  # fixed value for dummy running in ae now
            sampled_indices = torch.arange(0, N, sampled_interval_across_bs)
            sampled_length = len(sampled_indices)

            qk_low_rank = (
                low_rank_linear(hidden_states)
                .contiguous()
                .view(B, N, self.n_heads, self.low_rank_head_dim * 2)
                .transpose(1, 2)
            )

            q_low_rank, k_low_rank = qk_low_rank.chunk(2, dim=-1)

            q_low_rank = q_low_rank[:, :, sampled_indices, :]

            q_ref = query[:, :, sampled_indices, :]
            k_ref = key

            qk_low_rank = torch.matmul(
                q_low_rank, k_low_rank.transpose(-1, -2)
            ) / math.sqrt(q_low_rank.size(-1))
            qk_ref = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(
                q_ref.size(-1)
            )

            with torch.no_grad():
                # log to show the accuracy of the low rank prediction
                if (
                    global_variable.CURRENT_STEP % 200 == 0
                    and global_variable.RANK == 0
                    and global_variable.CURRENT_STEP
                    < global_variable.LOW_RANK_STAGE0_STEPS
                ):
                    # qk_ref=torch.matmul(query.detach(),key.detach().transpose(-1,-2))
                    k = 64
                    _, topk_index_ref = qk_ref.detach().topk(k=k, dim=-1)
                    _, topk_index = qk_low_rank.detach().topk(k=k, dim=-1)
                    topk_index = topk_index.contiguous().view(-1, k)
                    topk_index_ref = topk_index_ref.contiguous().view(-1, k)

                    overlap_ratios = []

                    for i in range(0, topk_index.size(0), 64):
                        row_overlap = (
                            torch.sum(
                                torch.isin(topk_index[i], topk_index_ref[i])
                            ).float()
                            / k
                        )
                        overlap_ratios.append(row_overlap)
                    overlap_ratio = torch.stack(overlap_ratios).mean()

                    if global_variable.RANK == 0:
                        print(
                            f"non-low-rank stage block_id:{layer_idx}, topk_index:{topk_index[:1]}, topk_index_ref:{topk_index_ref[:1]}, low_rank topk and ref topk overlap ratio:{overlap_ratio:.4f}"
                        )

                    del topk_index, topk_index_ref, overlap_ratios


            qk_low_rank_norm = F.normalize(qk_low_rank, p=2, dim=-1)
            qk_ref_norm = F.normalize(qk_ref, p=2, dim=-1)

            mse_loss = self.mse_loss(qk_low_rank_norm, qk_ref_norm)

            target = torch.ones(
                qk_low_rank.size(0) * qk_low_rank.size(1) * qk_low_rank.size(2),
                dtype=torch.long,
                device=qk_low_rank.device,
            )

            cosine_loss = self.cosine_loss(
                qk_low_rank.contiguous().view(-1, qk_low_rank.size(-1)),
                qk_ref.contiguous().view(-1, qk_ref.size(-1)),
                target,
            )

            loss = mse_loss * mse_loss_weight + cosine_loss * cosine_loss_weight

            if (
                global_variable.CURRENT_STEP % global_variable.ATTENTION_LOG_STEP == 0
                and global_variable.RANK == 0
            ):
                print(
                    f"block_id: {layer_idx}, low_rank_loss: {float(loss.item()):.4f}, mse_loss: {float(mse_loss.item()):.4f}, cosine_loss: {float(cosine_loss.item()):.4f}"
                )

        if global_variable.FORWARD_DONE == False:
            loss.backward()

        del qk_low_rank, qk_ref, q_low_rank, k_low_rank, q_ref, k_ref

    def _get_low_rank_linear_of_a_layer(self, layer_idx):
        return self.low_rank_linears[layer_idx]

    def get_kv_index_mask(
        self, hidden_states, query, key, layer_idx, sparse_attn_threshold=0.8
    ):
        B, N, C = hidden_states.shape
        B, H, N, D = query.shape

        if self.sparsity_per_layers[layer_idx] < sparse_attn_threshold:
            return None

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                low_rank_linear = self._get_low_rank_linear_of_a_layer(layer_idx)

                qk_low_rank = (
                    low_rank_linear(hidden_states)
                    .contiguous()
                    .view(B, N, self.n_heads, self.low_rank_head_dim * 2)
                    .transpose(1, 2)
                )

                q_low_rank, k_low_rank = qk_low_rank.chunk(2, dim=-1)
            

                qk_low_rank = torch.matmul(q_low_rank, k_low_rank.transpose(-1, -2))

                k = max(
                    int(N * (1 - self.sparsity_per_layers[layer_idx])), int(N * 0.02)
                )

                k = math.ceil(k / 8) * 8

                topk, topk_index = qk_low_rank.topk(k=k, dim=-1)

                topk_index_shape = topk_index.shape

                mask = torch.zeros(
                    (B, self.n_heads, N, N),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )

                mask.scatter_(-1, topk_index, True)

                if (
                    global_variable.CURRENT_STEP % 200 == 0
                    and global_variable.RANK == 0
                ):
                    qk_ref = torch.matmul(
                        query.detach(), key.detach().transpose(-1, -2)
                    )
                    top_value_ref, topk_index_ref = qk_ref.topk(k=k, dim=-1)

                    del top_value_ref

                    topk_index = topk_index.contiguous().view(-1, k)
                    topk_index_ref = topk_index_ref.contiguous().view(-1, k)

                    overlap_ratios = []
                    # print(f"topk_index shape:{topk_index.shape}, topk_index_ref shape:{topk_index_ref.shape}")

                    for i in range(0, topk_index.size(0), 64):
                        row_overlap = (
                            torch.sum(
                                torch.isin(topk_index[i], topk_index_ref[i])
                            ).float()
                            / k
                        )
                        overlap_ratios.append(row_overlap)
                    # print(f"overlap_ratios:{overlap_ratios}")
                    overlap_ratio = torch.stack(overlap_ratios).mean()

                    if global_variable.RANK == 0:
                        print(
                            f"block_id:{layer_idx}, topk shape:{topk_index.shape}, low_rank topk and ref topk overlap ratio:{overlap_ratio:.4f}; low_rank topk:{topk_index[:1]}; ref topk:{topk_index_ref[:1]}"
                        )

                    del topk_index_ref, overlap_ratios, overlap_ratio, qk_ref

        del qk_low_rank, k_low_rank, q_low_rank, topk, topk_index

        return mask

    def get_kv_index_mask_(
        self,
        hidden_states,
        query,
        key,
        layer_idx,
        sparse_attn_threshold=0.8,
        grid_size=[16, 16, 16],
        cube_size=[2, 2, 2],
    ):
        """ """
        B, N, C = hidden_states.shape
        B, H, N, D = query.shape

        if self.sparsity_per_layers[layer_idx] < sparse_attn_threshold:
            return None

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                low_rank_linear = self._get_low_rank_linear_of_a_layer(layer_idx)

                # Compute the size of the grid and cube
                frames, height, width = grid_size[0], grid_size[1], grid_size[2]
                cube_f, cube_h, cube_w = cube_size[0], cube_size[1], cube_size[2]

                # Ensure the grid can be divided by the cube
                assert (
                    frames % cube_f == 0
                    and height % cube_h == 0
                    and width % cube_w == 0
                )

                qk_low_rank = (
                    low_rank_linear(hidden_states)
                    .contiguous()
                    .view(B, N, self.n_heads, self.low_rank_head_dim * 2)
                    .transpose(1, 2)
                )
                q_low_rank, k_low_rank = qk_low_rank.chunk(2, dim=-1)

                q_low_rank = q_low_rank.view(B, H, frames, height, width, -1)
                q_low_rank = (
                    q_low_rank.unfold(2, cube_f, cube_f)
                    .unfold(3, cube_h, cube_h)
                    .unfold(4, cube_w, cube_w)
                )

                q_grouped = q_low_rank[..., 0, 0, 0, :]

                q_grouped = q_grouped.reshape(
                    B, H, -1, q_grouped.shape[-1]
                )  # [B, H, num_groups, D]
                grouped_attn = torch.matmul(q_grouped, k_low_rank.transpose(-1, -2))

                k = max(
                    int(N * (1 - self.sparsity_per_layers[layer_idx])), int(N * 0.02)
                )
                k = math.ceil(k / 8) * 8
                _, topk_indices = grouped_attn.topk(
                    k=k, dim=-1
                )  

                token_indices = torch.arange(N, device=hidden_states.device)
                f_idx = (token_indices // (height * width)) // cube_f
                h_idx = ((token_indices % (height * width)) // width) // cube_h
                w_idx = (token_indices % width) // cube_w
                group_indices = (
                    f_idx * (height // cube_h) * (width // cube_w)
                    + h_idx * (width // cube_w)
                    + w_idx
                )

                group_lut = group_indices.view(1, 1, N)  # [1, 1, N]

                expanded_topk = topk_indices.gather(
                    dim=2,
                    index=group_lut.expand(B, H, N).unsqueeze(-1).expand(-1, -1, -1, k),
                )  

                mask = torch.zeros(
                    (B, H, N, N), dtype=torch.bool, device=hidden_states.device
                )
                mask.scatter_(-1, expanded_topk, True)

                return mask

    def compute_attention_in_training_dummy( # dummy function here use a uniform sparsity value for throughput test  
        self, hidden_states, query, key, value, layer_idx, sparsity_threshold=0.8
    ):
        B, N, C = hidden_states.shape
        B, H, N, D = query.shape

        sparsity = self.sparsity_per_layers[layer_idx] # uniform value for throughput test  

        if (
            sparsity < sparsity_threshold
            and global_variable.LOW_RANK_DICT.get("window_size", None) is None
        ):
            with torch.nn.attention.sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )

            return hidden_states

        elif global_variable.LOW_RANK_DICT.get("window_size", None) is not None:

            
            if self.window_kv_index is not None and self.window_group_mask.shape[:2]==(1,1):
                self.window_kv_index = self.window_kv_index.repeat(B, H, 1, 1)
                self.window_group_mask = self.window_group_mask.repeat(B, H, 1, 1)
            

            hidden_states = sparse_window_attention(
                query.contiguous(), key.contiguous(), value.contiguous(), False, 1 / math.sqrt(D), 
                self.window_kv_index.contiguous(), self.window_group_mask.contiguous(), 
                self.window_size,
                self.group_size
            )

            return hidden_states

        else:
            with torch.no_grad():
                critical_kv_length = int(N * (1 - sparsity))
                critical_kv_length = max(256, round(critical_kv_length / 256) * 256)
                critical_kv_length = min(critical_kv_length, N)

                query_group_size = 32
                
                kv_index = torch.zeros(
                    (B, H, N // query_group_size, critical_kv_length),
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                
                low_rank_linear = self._get_low_rank_linear_of_a_layer(layer_idx)

                qk_low_rank = (
                    low_rank_linear(hidden_states)
                    .contiguous()
                    .view(B, N, H, self.low_rank_head_dim * 2)
                    .transpose(1, 2)
                )

                q_low_rank, k_low_rank = qk_low_rank.chunk(2, dim=-1)

                q_low_rank = extract_middle_tokens(q_low_rank, N // query_group_size, query_group_size) 

                low_rank_qk = torch.matmul(
                    q_low_rank, k_low_rank.transpose(-2, -1)
                )

                for h in range(H):
                    _, topk_indices = torch.topk(
                        low_rank_qk[:, h, :, :], critical_kv_length, dim=-1, sorted=False
                    )
                    kv_index[:, h, :, :] = topk_indices.to(torch.int32)

                topk_per_head = torch.full(
                    (B, H), critical_kv_length, dtype=torch.int32, device=hidden_states.device
                )

            hidden_states = sparse_group_attention(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                False,
                1 / math.sqrt(D),
                kv_index.contiguous(),
                topk_per_head.contiguous(),
                query_group_size,
            )

            return hidden_states

