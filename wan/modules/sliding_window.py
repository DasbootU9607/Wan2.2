"""Temporal window self-attention for already position-encoded video Q/K/V."""

import math

import torch
import torch.nn.functional as F

from . import attention as attention_backend


def sliding_window_attention(q, k, v, seq_lens, grid_sizes, config,
                             query_chunk_size=128):
    """Average overlapping windows without building a full-video attention mask.

    Inputs are [B, L, H, D] after global RoPE and, with Ulysses, after the
    sequence gather. Each window contains every spatial token in its frames.
    Padding is excluded from both queries and keys and receives zero output.
    FlashAttention processes a window at a time; the SDPA fallback additionally
    chunks queries to bound temporary attention storage.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Sliding-window self-attention requires matching [B, L, H, D] QKV shapes.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Sliding-window QKV must share a device.")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    batch, length = q.shape[:2]
    if grid_sizes.shape != (batch, 3) or seq_lens.shape != (batch,):
        raise ValueError("Sliding-window grids and sequence lengths must match the batch.")

    use_flash = q.is_cuda and q.size(-1) <= 256 and (
        attention_backend.FLASH_ATTN_2_AVAILABLE or attention_backend.FLASH_ATTN_3_AVAILABLE)
    accumulation_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    output = torch.zeros_like(q)
    for sample, (grid, valid_length) in enumerate(zip(grid_sizes.tolist(), seq_lens.tolist())):
        if any(size <= 0 for size in grid) or math.prod(grid) != valid_length or valid_length > length:
            raise ValueError("Sliding-window grid must match the unpadded video token count.")
        frames, height, width = grid
        tokens_per_frame = height * width
        total = torch.zeros_like(q[sample:sample + 1, :valid_length], dtype=accumulation_dtype)
        coverage = torch.zeros(valid_length, device=q.device, dtype=accumulation_dtype)
        for frame_start, frame_end in config.windows(frames):
            start, end = frame_start * tokens_per_frame, frame_end * tokens_per_frame
            window_k = k[sample:sample + 1, start:end]
            window_v = v[sample:sample + 1, start:end]
            if use_flash:
                window_output = attention_backend.flash_attention(
                    q[sample:sample + 1, start:end], window_k, window_v)
                total[:, start:end] += window_output.to(accumulation_dtype)
            else:
                # Wan RoPE returns fp32 Q/K even with half-precision V.
                window_k = window_k.transpose(1, 2).to(v.dtype)
                window_v = window_v.transpose(1, 2)
                for query_start in range(start, end, query_chunk_size):
                    query_end = min(query_start + query_chunk_size, end)
                    window_q = q[sample:sample + 1, query_start:query_end].transpose(1, 2).to(v.dtype)
                    window_output = F.scaled_dot_product_attention(
                        window_q, window_k, window_v, dropout_p=0.0, is_causal=False)
                    total[:, query_start:query_end] += window_output.transpose(1, 2).to(accumulation_dtype)
            coverage[start:end] += 1
        total.div_(coverage[None, :, None, None])
        output[sample:sample + 1, :valid_length] = total.to(q.dtype)
    return output
