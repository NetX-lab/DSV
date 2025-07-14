
import torch
import math
from typing import List, Tuple, Optional


def create_cube_groups(video_shape: Tuple[int, int, int], cube_shape: Tuple[int, int, int]) -> List[List[int]]:
    F, H, W = video_shape
    cube_f, cube_h, cube_w = cube_shape
    
    cube_groups = []

    for f_start in range(0, F, cube_f):
        for h_start in range(0, H, cube_h):
            for w_start in range(0, W, cube_w):
                
                cube_queries = []
                
                for df in range(min(cube_f, F - f_start)):
                    for dh in range(min(cube_h, H - h_start)):
                        for dw in range(min(cube_w, W - w_start)):
                            
                            f_pos = f_start + df
                            h_pos = h_start + dh
                            w_pos = w_start + dw
                            
                            seq_idx = f_pos * H * W + h_pos * W + w_pos
                            cube_queries.append(seq_idx)
                
                if cube_queries:  
                    cube_groups.append(cube_queries)
    
    return cube_groups


def seq_idx_to_3d(seq_idx: int, H: int, W: int) -> Tuple[int, int, int]:
    f = seq_idx // (H * W)
    h = (seq_idx % (H * W)) // W
    w = seq_idx % W
    return f, h, w


def pos_3d_to_seq_idx(f: int, h: int, w: int, H: int, W: int) -> int:
    return f * H * W + h * W + w


def find_cube_center_query(cube_queries: List[int], video_shape: Tuple[int, int, int]) -> Tuple[int, Tuple[int, int, int]]:
    F, H, W = video_shape
    
    cube_3d_coords = []
    for seq_idx in cube_queries:
        f, h, w = seq_idx_to_3d(seq_idx, H, W)
        cube_3d_coords.append((f, h, w))
    
    center_f = sum(coord[0] for coord in cube_3d_coords) // len(cube_3d_coords)
    center_h = sum(coord[1] for coord in cube_3d_coords) // len(cube_3d_coords)
    center_w = sum(coord[2] for coord in cube_3d_coords) // len(cube_3d_coords)
    
    center_3d_pos = (center_f, center_h, center_w)
    
    min_dist = float('inf')
    center_query_idx = cube_queries[0]
    
    for seq_idx in cube_queries:
        f, h, w = seq_idx_to_3d(seq_idx, H, W)
        dist = abs(f - center_f) + abs(h - center_h) + abs(w - center_w)
        if dist < min_dist:
            min_dist = dist
            center_query_idx = seq_idx
    
    return center_query_idx, center_3d_pos


def compute_unified_kv_indices_with_padding_mask(
    center_3d_pos: Tuple[int, int, int], 
    video_shape: Tuple[int, int, int], 
    unified_window_size: Tuple[int, int, int]
) -> Tuple[List[int], List[bool]]:
    F, H, W = video_shape
    W_f, W_h, W_w = unified_window_size
    center_f, center_h, center_w = center_3d_pos
    
    window_size = W_f * W_h * W_w
    
    window_positions = []  # [(kv_idx, is_valid, distance_to_center)]
    
    for df in range(W_f):
        for dh in range(W_h):
            for dw in range(W_w):
                offset_f = df - W_f // 2
                offset_h = dh - W_h // 2
                offset_w = dw - W_w // 2
                
                target_f = center_f + offset_f
                target_h = center_h + offset_h
                target_w = center_w + offset_w
                
                # Check if within valid range
                if 0 <= target_f < F and 0 <= target_h < H and 0 <= target_w < W:
                    kv_idx = target_f * H * W + target_h * W + target_w
                    distance = abs(offset_f) + abs(offset_h) + abs(offset_w)  # Manhattan distance
                    window_positions.append((kv_idx, True, distance))
                else:
                    window_positions.append((None, False, 0))  # Placeholder for invalid positions
    
    valid_positions = [pos[0] for pos in window_positions if pos[1]]
    needed_positions = window_size - len(valid_positions)
    
    if needed_positions > 0:

        max_search_distance = max(W_f, W_h, W_w)  
        
        all_valid_positions = []
        used_positions = set(valid_positions)
        
        search_f_min = max(0, center_f - max_search_distance)
        search_f_max = min(F, center_f + max_search_distance + 1)
        search_h_min = max(0, center_h - max_search_distance)
        search_h_max = min(H, center_h + max_search_distance + 1)
        search_w_min = max(0, center_w - max_search_distance)
        search_w_max = min(W, center_w + max_search_distance + 1)
        
        for f in range(search_f_min, search_f_max):
            for h in range(search_h_min, search_h_max):
                for w in range(search_w_min, search_w_max):
                    kv_idx = f * H * W + h * W + w
                    if kv_idx not in used_positions:
                        distance = abs(f - center_f) + abs(h - center_h) + abs(w - center_w)
                        all_valid_positions.append((kv_idx, distance))
        
        all_valid_positions.sort(key=lambda x: x[1])
        additional_positions = [pos[0] for pos in all_valid_positions[:needed_positions]]
        
        if len(additional_positions) < needed_positions:
            remaining_needed = needed_positions - len(additional_positions)
            fallback_positions = []
            for i in range(remaining_needed):
                if valid_positions:
                    fallback_positions.append(valid_positions[i % len(valid_positions)])
                else:
                    fallback_positions.append(0)  # Ultimate fallback
            additional_positions.extend(fallback_positions)
        
        valid_positions.extend(additional_positions)
    
    kv_indices = []
    padding_mask = []
    valid_idx = 0
    replacement_idx = len([pos for pos in window_positions if pos[1]])  
    
    for _, is_valid, _ in window_positions:
        if is_valid:
            kv_indices.append(valid_positions[valid_idx])
            padding_mask.append(False) 
            valid_idx += 1
        else:
            kv_indices.append(valid_positions[replacement_idx])
            padding_mask.append(True)  
            replacement_idx += 1
    
    assert len(set(kv_indices)) == len(kv_indices), f"Found duplicate KV indices! {len(kv_indices)} total, {len(set(kv_indices))} unique"
    
    kv_mask_pairs = list(zip(kv_indices, padding_mask))
    kv_mask_pairs.sort(key=lambda x: x[0])  
    
    sorted_kv_indices, sorted_padding_mask = zip(*kv_mask_pairs)
    kv_indices = list(sorted_kv_indices)
    padding_mask = list(sorted_padding_mask)
    
    return kv_indices, padding_mask


def generate_cube_padding_mask(
    cube_queries: List[int], 
    kv_indices: List[int], 
    padding_mask: List[bool]
) -> torch.Tensor:
    num_kv_positions = len(kv_indices)
    
    padding_mask_tensor = torch.tensor(padding_mask, dtype=torch.bool)
    
    group_mask = ~padding_mask_tensor  
    
    return group_mask


def reorder_queries_by_cubes(
    q: torch.Tensor, 
    cube_groups: List[List[int]]
) -> Tuple[torch.Tensor, List[int]]:
    #Reorder queries according to cube grouping for efficient processing.
    cube_ordered_q = torch.zeros_like(q)
    query_reorder_map = []
    current_pos = 0
    
    for cube_queries in cube_groups:
        for query_idx in cube_queries:
            cube_ordered_q[:, :, current_pos] = q[:, :, query_idx]
            query_reorder_map.append(query_idx)
            current_pos += 1
    
    return cube_ordered_q, query_reorder_map


def restore_query_order(
    output: torch.Tensor, 
    query_reorder_map: List[int]
) -> torch.Tensor:
    final_output = torch.zeros_like(output)
    for i, original_idx in enumerate(query_reorder_map):
        final_output[:, :, original_idx] = output[:, :, i]
    
    return final_output


def validate_cube_configuration(
    video_shape: Tuple[int, int, int],
    cube_shape: Tuple[int, int, int],
    unified_window_size: Tuple[int, int, int]
) -> bool:
    F, H, W = video_shape
    cube_f, cube_h, cube_w = cube_shape
    W_f, W_h, W_w = unified_window_size
    
    if cube_f <= 0 or cube_h <= 0 or cube_w <= 0:
        return False
    
    if W_f <= 0 or W_h <= 0 or W_w <= 0:
        return False
    
    if W_f > F or W_h > H or W_w > W:
        print(f"Warning: Window size {unified_window_size} is larger than video size {video_shape}")
    
    return True




def generate_window_attention_kvindex_mask_tensors(
    video_shape: Tuple[int, int, int],
    cube_shape: Tuple[int, int, int], 
    unified_window_size: Tuple[int, int, int],
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device('cpu')
    
    if not validate_cube_configuration(video_shape, cube_shape, unified_window_size):
        raise ValueError("Invalid cube configuration")
    
    cube_groups = create_cube_groups(video_shape, cube_shape)
    num_groups = len(cube_groups)
    unified_window_len = unified_window_size[0] * unified_window_size[1] * unified_window_size[2]
    
    #print(f"Generating attention tensors: {num_groups} groups, window size {unified_window_len}")
    
    all_kv_indices = []
    all_group_masks = []
    
    for i, cube_queries in enumerate(cube_groups):
        center_query_idx, center_3d_pos = find_cube_center_query(cube_queries, video_shape)
        kv_indices, padding_mask = compute_unified_kv_indices_with_padding_mask(
            center_3d_pos, video_shape, unified_window_size
        )
        
        group_mask = generate_cube_padding_mask(cube_queries, kv_indices, padding_mask)
        
        unique_kv_count = len(set(kv_indices))
        assert unique_kv_count == len(kv_indices), f"Group {i}: Expected {len(kv_indices)} unique KV indices, got {unique_kv_count}"
        
        all_kv_indices.append(kv_indices)
        all_group_masks.append(group_mask)
    
    kv_indices_tensor = torch.tensor(all_kv_indices, dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(0)
    group_masks_tensor = torch.stack(all_group_masks, dim=0).unsqueeze(0).unsqueeze(0).to(device)
    
    expected_shape = (1, 1, num_groups, unified_window_len)
    assert kv_indices_tensor.shape == expected_shape, f"KV indices shape mismatch: {kv_indices_tensor.shape} vs {expected_shape}"
    assert group_masks_tensor.shape == expected_shape, f"Group masks shape mismatch: {group_masks_tensor.shape} vs {expected_shape}"
    
    kv_indices_tensor = kv_indices_tensor.contiguous()
    group_masks_tensor = group_masks_tensor.bool().contiguous()
    
    print(f"Generated tensors - KV indices: {kv_indices_tensor.shape} {kv_indices_tensor.dtype}, "
          f"Group masks: {group_masks_tensor.shape} {group_masks_tensor.dtype}")
    
    return kv_indices_tensor, group_masks_tensor
