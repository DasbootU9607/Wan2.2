"""Prompt Relay schedules for Wan T2V; no model weights are needed here."""

import math
from bisect import bisect_left

import torch


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def build_prompt_text(config):
    if not isinstance(config, dict):
        raise ValueError("Prompt Relay configuration must be a JSON object.")
    global_prompt = config.get("global_prompt", "")
    local_prompts = config.get("local_prompts", [])
    if not isinstance(global_prompt, str):
        raise ValueError("global_prompt must be a string.")
    if not isinstance(local_prompts, list) or any(
            not isinstance(p, str) or not p.strip() for p in local_prompts):
        raise ValueError("local_prompts must be a list of nonempty strings.")
    return " ".join([global_prompt.strip()] + [p.strip() for p in local_prompts]).strip()


def prepare_schedule(config, count, frame_num, compression, temporal_patch, video_fps):
    """Return (internal intervals, window parameters) in local-prompt order.

    Explicit intervals are independent half-open ranges. Without them, retain
    Wan Prompt Relay's consecutive ceil-based allocation and legacy decay.
    """
    if not isinstance(frame_num, int) or isinstance(frame_num, bool) or frame_num < 1:
        raise ValueError("frame_num must be a positive integer.")
    latent_frames = (frame_num - 1) // compression + 1
    if (frame_num - 1) % compression or latent_frames % temporal_patch:
        raise ValueError("frame_num must align with the VAE and temporal patch grid.")
    total = latent_frames // temporal_patch
    explicit = "segment_intervals" in config
    if config.get("auto_overlap", False) is not False or any(
            key in config for key in ("overlap_duration", "overlap_ratio")):
        raise ValueError("Wan requires explicit segment_intervals for overlap; omit times for consecutive allocation.")
    if "fps" in config:
        raise ValueError("FPS comes from the model's sample_fps (also used for export), not the prompt JSON.")
    if not explicit:
        if any(key in config for key in ("time_unit", "tail_width", "epsilon", "routing_mode", "sigma", "boundary_margin")):
            raise ValueError("Interval routing options require segment_intervals.")
        lengths = config.get("segment_lengths", [])
        if not isinstance(lengths, list) or (lengths and len(lengths) != count):
            raise ValueError("segment_lengths must contain one length per local prompt.")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lengths):
            raise ValueError("segment_lengths must contain positive integer internal-frame counts.")
        if not count:
            return [], []
        if not lengths:
            lengths = [math.ceil(total / count)] * count
        intervals, start = [], 0
        for length in lengths:
            end = min(start + length, total)
            if end <= start:
                raise ValueError("A local prompt has no frames; reduce prompt count or use explicit intervals.")
            intervals.append([start, end])
            start += length
        # Preserve the original Wan window and fp16-rounded sigma for old JSON.
        sigma = torch.tensor(0.1448, dtype=torch.float16).item()
        params = [((a + b) // 2, (b - a) // 2 - 2, sigma) for a, b in intervals]
        return intervals, params

    if "segment_lengths" in config:
        raise ValueError("Use segment_intervals or segment_lengths, not both.")
    if any(key in config for key in ("sigma", "boundary_margin")):
        raise ValueError("Use tail_width and epsilon for explicit interval decay.")
    if config.get("routing_mode", "overlap") != "overlap":
        raise ValueError("Explicit intervals support routing_mode='overlap' only.")
    unit = config.get("time_unit", "internal_frame")
    if unit not in ("seconds", "internal_frame"):
        raise ValueError("time_unit must be 'seconds' or 'internal_frame'.")
    if not _number(video_fps) or video_fps <= 0:
        raise ValueError("sample_fps must be finite and positive.")
    scale = compression * temporal_patch / video_fps if unit == "seconds" else 1.0
    limit = frame_num / video_fps if unit == "seconds" else total
    epsilon = config.get("epsilon", 1e-3)
    tail = config.get("tail_width", 2 * scale)
    if not _number(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and strictly between zero and one.")
    if not _number(tail) or tail <= 0:
        raise ValueError("tail_width must be finite and positive, in time_unit units.")
    sigma = (tail / scale) / math.sqrt(-2 * math.log(epsilon))
    limits = torch.finfo(torch.float32)
    minimum = max(math.sqrt(limits.tiny), total / math.sqrt(limits.max / 4))
    if not math.isfinite(sigma) or not minimum <= sigma <= math.sqrt(limits.max / 4):
        raise ValueError("tail_width produces an unrepresentable float32 decay.")
    supplied = config["segment_intervals"]
    if not isinstance(supplied, list) or len(supplied) != count:
        raise ValueError("segment_intervals must contain one [start, end] pair per local prompt.")
    times = [compression * (i * temporal_patch + (temporal_patch - 1) / 2) / video_fps
             for i in range(total)]
    intervals = []
    for index, pair in enumerate(supplied):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"segment_intervals[{index}] must be [start, end].")
        a, b = pair
        if not all(_number(v) for v in pair) or not 0 <= a < b <= limit:
            raise ValueError(f"segment_intervals[{index}] must satisfy 0 <= start < end <= {limit}.")
        if unit == "seconds":
            a, b = bisect_left(times, a), bisect_left(times, b)
        elif any(not isinstance(v, int) for v in pair):
            raise ValueError("internal_frame endpoints must be integers.")
        if a == b:
            raise ValueError(f"segment_intervals[{index}] contains no internal timestamp; widen the interval.")
        intervals.append([a, b])
    params = [((a + b - 1) / 2, (b - a - 1) / 2, sigma) for a, b in intervals]
    return intervals, params


def _local_spans(tokenizer, text, config):
    def encode(value, special=False, truncate=False):
        return tokenizer(value, add_special_tokens=special, padding=False,
                         truncation=truncate, return_mask=False)[0].tolist()

    full = encode(text, special=True)
    encoded = encode(text, special=True, truncate=True)
    if full != encoded:
        raise ValueError("Prompt exceeds the T5 token budget; shorten it so all local prompts remain encoded.")
    prefix = encode(config.get("global_prompt", ""))
    if full[:len(prefix)] != prefix:
        raise ValueError("Cannot align global prompt tokens with the encoded text.")
    cursor, spans = len(prefix), []
    for index, prompt in enumerate(config.get("local_prompts", [])):
        tokens = encode(prompt)
        if not tokens:
            raise ValueError(f"local_prompts[{index}] has no tokens after text cleaning.")
        for start in range(cursor, len(full) - len(tokens) + 1):
            end = start + len(tokens)
            if full[start:end] == tokens:
                spans.append(torch.arange(start, end, dtype=torch.long))
                cursor = end
                break
        else:
            raise ValueError(f"Cannot align local_prompts[{index}] with the encoded text; revise the prompt boundary.")
    return spans


def prepare_prompt_relay(config, tokenizer, frame_num, size, vae_stride,
                         patch_size, video_fps):
    text = build_prompt_text(config)
    intervals, params = prepare_schedule(
        config, len(config.get("local_prompts", [])), frame_num,
        vae_stride[0], patch_size[0], video_fps)
    if not intervals:
        return None, text
    width, height = size
    tokens_per_frame = ((height // vae_stride[1]) // patch_size[1]
                        * ((width // vae_stride[2]) // patch_size[2]))
    if tokens_per_frame <= 0:
        raise ValueError("Video size must contain at least one spatial patch.")
    spans = _local_spans(tokenizer, text, config)
    payload = [dict(midpoint=mid, window=window, sigma=sigma,
                    tokens_per_frame=tokens_per_frame, local_token_idx=span,
                    frame_interval=interval)
               for interval, (mid, window, sigma), span in zip(intervals, params, spans)]
    return payload, text
