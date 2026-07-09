import torch


def prepare_patch_pos_emb(
    grid_thw: torch.Tensor, num_grid_per_side: int = 48, device=None, wt_dtype=None
):
    """
    Qwen3-VL visual part applied finer-grind for HW grid position embedding,
    During Training time, it uses 768 * 768 pixel values while patch_size = 16,
    results in 48 * 48 grids = 2304 (num_position_embeddings)
    During inference, bilinearly sample grid features to the inference's.
    """
    grid_thw = grid_thw.squeeze(0)
    grid_h = int(grid_thw[1].item())
    grid_w = int(grid_thw[2].item())

    # 4 means four anchor points for bilinear interpolation
    idx_list, wt_list = [[] for _ in range(4)], [[] for _ in range(4)]

    h_indices: torch.Tensor = torch.linspace(0, num_grid_per_side - 1, grid_h)
    w_indices: torch.Tensor = torch.linspace(0, num_grid_per_side - 1, grid_w)

    h_idx_floor = h_indices.int()
    h_idx_ceil = (h_indices.int() + 1).clip(max=num_grid_per_side - 1)

    w_idx_floor = w_indices.int()
    w_idx_ceil = (w_indices.int() + 1).clip(max=num_grid_per_side - 1)

    dh = h_indices - h_idx_floor
    dw = w_indices - w_idx_floor

    # flatten to the (0, num_position_embeddings)
    base_h = h_idx_floor * num_grid_per_side
    base_h_ceil = h_idx_ceil * num_grid_per_side

    # bilinear sampling
    indices = [
        (base_h[None].T + w_idx_floor[None]).flatten(),
        (base_h[None].T + w_idx_ceil[None]).flatten(),
        (base_h_ceil[None].T + w_idx_floor[None]).flatten(),
        (base_h_ceil[None].T + w_idx_ceil[None]).flatten(),
    ]

    weights = [
        ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
        ((1 - dh)[None].T * dw[None]).flatten(),
        (dh[None].T * (1 - dw)[None]).flatten(),
        (dh[None].T * dw[None]).flatten(),
    ]

    for i in range(4):
        idx_list[i].extend(indices[i].tolist())
        wt_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    wt_tensor = torch.tensor(wt_list, dtype=wt_dtype, device=device)

    return grid_h, grid_w, idx_tensor, wt_tensor


def vision_rotary_pos_emb(
    dim: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    theta: float = 10000.0,
    device=None,
):
    merge_size = spatial_merge_size
    seqlen = int(grid_thw[:, 1:].max().item())
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    )
    seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(seq, inv_freq)

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = (
            block_rows[:, None, None, None] * merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * merge_size
            + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freqs[pos_ids]  # lookup rotary embeddings
    embeddings = embeddings.flatten(1)
    return embeddings
