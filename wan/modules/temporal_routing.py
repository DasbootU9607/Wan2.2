"""Distance-penalized cross-attention used by Wan Prompt Relay."""

import math

import torch


def offset_prompt_relay(q_token_idx, query_token_offset):
    """Keep global video time when each sequence-parallel rank owns a slice."""
    if not q_token_idx:
        return q_token_idx
    return [dict(segment, query_token_offset=query_token_offset) for segment in q_token_idx]


def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype):
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    if not q_token_idx:
        return offset
    tokens_per_frame = int(q_token_idx[0]['tokens_per_frame'])
    query_offset = q_token_idx[0].get('query_token_offset', 0)
    query_frames = (torch.arange(Lq, device=device) + query_offset) // tokens_per_frame
    for segment in q_token_idx:
        sigma = torch.as_tensor(segment['sigma'], dtype=torch.float32, device=device)
        midpoint = torch.as_tensor(segment['midpoint'], dtype=torch.float32, device=device)
        local = segment['local_token_idx'].to(device=device)
        distance = (query_frames.float()[:, None] - midpoint).abs()
        cost = torch.relu(distance - segment['window']).square() / (2 * sigma.square())
        offset[:, local] = cost.to(dtype)
    return offset


def chunked_softmax_attention(q, k, v, q_token_idx, chunk_size=16):
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    batch, heads, length, dim = q.shape
    # Keep the penalty finite for narrow tails, even with fp16 model weights.
    cost = build_temporal_cost(q_token_idx, length, k.shape[2], q.device, torch.float32)
    out = torch.empty(batch, heads, length, dim, device=q.device, dtype=q.dtype)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        logits = torch.matmul(q[:, :, start:end], k.transpose(-2, -1)) / math.sqrt(dim)
        weights = torch.softmax(logits.float() - cost[start:end], dim=-1)
        out[:, :, start:end] = torch.matmul(weights.to(v.dtype), v)
    return out.transpose(1, 2)
