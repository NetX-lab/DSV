import time

import torch


def rearrange_query_tokens_3d(
    q,
    video_f_size,
    video_h_size,
    video_w_size,
    group_f_size,
    group_h_size,
    group_w_size,
):
    """
    Rearrange query tokens so that tokens from the same 3D group are contiguous.

    Args:
        q: [B, H, S, D] where S = video_f_size * video_h_size * video_w_size
        video_f_size, video_h_size, video_w_size: 3D video dimensions
        group_f_size, group_h_size, group_w_size: 3D group dimensions

    Returns:
        q_rearranged: [B, H, S, D] with tokens rearranged by 3D groups
        inverse_indices: indices to restore original order
    """
    B, H, S, D = q.shape
    assert S == video_f_size * video_h_size * video_w_size

    # Create mapping from original 1D index to 3D coordinates
    original_indices = []
    for idx in range(S):
        f = idx // (video_h_size * video_w_size)
        h = (idx % (video_h_size * video_w_size)) // video_w_size
        w = idx % video_w_size
        original_indices.append((f, h, w))

    # Group tokens by their 3D group
    groups_per_f = video_f_size // group_f_size
    groups_per_h = video_h_size // group_h_size
    groups_per_w = video_w_size // group_w_size

    # Vectorized computation of new_indices
    # Generate all 3D coordinates
    f_coords = torch.arange(video_f_size, dtype=torch.long)
    h_coords = torch.arange(video_h_size, dtype=torch.long)
    w_coords = torch.arange(video_w_size, dtype=torch.long)

    # Create meshgrid for all combinations
    f_grid, h_grid, w_grid = torch.meshgrid(f_coords, h_coords, w_coords, indexing="ij")

    # Flatten to get all coordinates
    all_f = f_grid.flatten()
    all_h = h_grid.flatten()
    all_w = w_grid.flatten()

    # Calculate group indices for each coordinate
    group_f_indices = all_f // group_f_size
    group_h_indices = all_h // group_h_size
    group_w_indices = all_w // group_w_size

    # Calculate overall group index (0 to total_groups-1)
    group_indices = (
        group_f_indices * groups_per_h * groups_per_w
        + group_h_indices * groups_per_w
        + group_w_indices
    )

    # Calculate position within each group
    within_group_f = all_f % group_f_size
    within_group_h = all_h % group_h_size
    within_group_w = all_w % group_w_size
    within_group_pos = (
        within_group_f * group_h_size * group_w_size
        + within_group_h * group_w_size
        + within_group_w
    )

    # Create sort keys: first by group, then by within-group position
    sort_keys = (
        group_indices * (group_f_size * group_h_size * group_w_size) + within_group_pos
    )

    # Sort to get the new ordering
    sorted_indices = torch.argsort(sort_keys)

    # Convert back to original 1D indices
    original_1d_indices = (
        all_f * video_h_size * video_w_size + all_h * video_w_size + all_w
    )
    new_indices = original_1d_indices[sorted_indices].tolist()

    # Rearrange the query tensor
    q_rearranged = q[:, :, new_indices, :]

    # Vectorized approach using torch operations
    inverse_indices = torch.empty(S, dtype=torch.long)
    new_indices_tensor = torch.tensor(new_indices, dtype=torch.long)
    positions = torch.arange(S, dtype=torch.long)
    inverse_indices[new_indices_tensor] = positions

    # Convert back to list for compatibility
    inverse_indices = inverse_indices.tolist()

    # print(f"new indices: {new_indices}")
    # print(f"inverse indices: {inverse_indices}")

    return q_rearranged, new_indices, inverse_indices


def restore_output_order(output, inverse_indices):
    B, H, S, D = output.shape


    return output[:, :, inverse_indices, :]


def test_3d_token_reorder():
    print("🔧 Testing 3D Token Reorder Functionality")
    print("=" * 50)

    # Easy 3D video parameters
    video_f_size = 6  # 4 frames
    video_h_size = 6  # 4 height
    video_w_size = 6  # 4 width
    S = video_f_size * video_h_size * video_w_size  # total tokens = 64

    # Group parameters
    group_f_size = 2  # 2 frames
    group_h_size = 3  # 2 height
    group_w_size = 3  # 2 width
    group_size = group_f_size * group_h_size * group_w_size  # 8 tokens

    # Calculate number of groups
    groups_per_f = video_f_size // group_f_size  # 2 groups (f direction)
    groups_per_h = video_h_size // group_h_size  # 2 groups (h direction)
    groups_per_w = video_w_size // group_w_size  # 2 groups (w direction)
    total_groups = groups_per_f * groups_per_h * groups_per_w  # 8 groups

    print(f"3D Video Configuration:")
    print(
        f"   Video dimensions: {video_f_size}×{video_h_size}×{video_w_size} = {S} tokens"
    )
    print(
        f"   Group dimensions: {group_f_size}×{group_h_size}×{group_w_size} = {group_size} tokens per group"
    )
    print(
        f"   Number of groups: {groups_per_f}×{groups_per_h}×{groups_per_w} = {total_groups} groups"
    )

    # Create test data
    B, H, D = 1, 2, 8
    q = (
        torch.arange(S)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(B, H, S, D)
        .float()
    )

    print(f"\nOriginal Token Layout (showing first few tokens):")
    print(f"   Shape: {q.shape}")

    # Display original 3D -> 1D mapping
    print(f"\n📋 Original 3D->1D Index Mapping (f,h,w -> 1D):")
    for f in range(min(2, video_f_size)):  # Only display first 2 frames
        for h in range(min(2, video_h_size)):  # Only display first 2 rows
            for w in range(min(4, video_w_size)):  # Display
                idx_1d = f * video_h_size * video_w_size + h * video_w_size + w
                print(f"   ({f},{h},{w}) -> {idx_1d:2d}", end="  ")
            print()
        if f < min(2, video_f_size) - 1:
            print("   " + "-" * 20)

    # Apply 3D token rearrangement
    q_rearranged, _, inverse_indices = rearrange_query_tokens_3d(
        q,
        video_f_size,
        video_h_size,
        video_w_size,
        group_f_size,
        group_h_size,
        group_w_size,
    )

    print(f"\n After 3D Grouping Rearrangement:")
    print(f"   Rearranged shape: {q_rearranged.shape}")

    # Display rearranged token order (grouped)
    print(f"\nGrouped Token Order (each group has {group_size} tokens):")
    original_tokens = q[0, 0, :, 0].int().tolist()  # Original token sequence
    rearranged_tokens = q_rearranged[0, 0, :, 0].int().tolist()  # Rearranged sequence

    for group_idx in range(min(4, total_groups)):  # Only display first 4 groups
        start_pos = group_idx * group_size
        end_pos = start_pos + group_size
        group_tokens = rearranged_tokens[start_pos:end_pos]
        print(f"   Group {group_idx}: {group_tokens}")

        # Display 3D coordinates of tokens in this group
        print(f"            3D coords:", end="")
        for token_idx in group_tokens:
            f = token_idx // (video_h_size * video_w_size)
            h = (token_idx % (video_h_size * video_w_size)) // video_w_size
            w = token_idx % video_w_size
            print(f" ({f},{h},{w})", end="")
        print()

    # Verify inverse transformation
    q_restored = restore_output_order(q_rearranged, inverse_indices)
    is_correct = torch.equal(q, q_restored)

    print(f"\n✅ Verification:")
    print(f"   Original == Restored: {is_correct}")
    if is_correct:
        print(f"3D Token Reorder works!")
    else:
        print(f"Found problem: inverse transformation is incorrect")

    # Check spatial continuity of tokens in each group
    print(f"\n🔍 Group Spatial Continuity Analysis:")
    for group_idx in range(min(2, total_groups)):  # Check first 2 groups
        start_pos = group_idx * group_size
        end_pos = start_pos + group_size
        group_tokens = rearranged_tokens[start_pos:end_pos]

        # Convert to 3D coordinates
        coords_3d = []
        for token_idx in group_tokens:
            f = token_idx // (video_h_size * video_w_size)
            h = (token_idx % (video_h_size * video_w_size)) // video_w_size
            w = token_idx % video_w_size
            coords_3d.append((f, h, w))

        # Check if forming continuous 3D blocks
        f_coords = [c[0] for c in coords_3d]
        h_coords = [c[1] for c in coords_3d]
        w_coords = [c[2] for c in coords_3d]

        f_range = max(f_coords) - min(f_coords) + 1
        h_range = max(h_coords) - min(h_coords) + 1
        w_range = max(w_coords) - min(w_coords) + 1

        expected_size = group_f_size * group_h_size * group_w_size
        actual_size = len(set(coords_3d))

        print(f"   Group {group_idx}:")
        print(
            f"     3D range: {f_range}×{h_range}×{w_range} (expected: {group_f_size}×{group_h_size}×{group_w_size})"
        )
        print(f"     Unique coordinates: {actual_size} (expected: {expected_size})")

        is_continuous = (
            f_range == group_f_size
            and h_range == group_h_size
            and w_range == group_w_size
            and actual_size == expected_size
        )
        print(f"     3D continuity: {'✅' if is_continuous else '❌'}")

    return is_correct


if __name__ == "__main__":

    test_3d_token_reorder()

    query = torch.randn(1, 16, 64 * 64 * 64, 128, device="cuda", dtype=torch.bfloat16)

    q_rearranged, new_indices, inverse_indices = rearrange_query_tokens_3d(
        query, 64, 64, 64, 4, 4, 2
    )

    q = torch.randn_like(query)

    time_start = time.time()
    q_trans = q[:, :, new_indices, :]
    q_restored = q_trans[:, :, inverse_indices, :]
    torch.cuda.synchronize()
    time_end = time.time()
    print(f"time: {time_end - time_start}")
