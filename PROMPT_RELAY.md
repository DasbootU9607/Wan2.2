# Overlapping Prompt Relay for Wan2.2

This branch adds independent temporal intervals to the existing **T2V-A14B**
Prompt Relay path, in both high-noise and low-noise experts. No retraining or
new model weights are required. [中文说明](PROMPT_RELAY_ZH.md)

## Usage

From this Wan2.2 directory, with the usual Wan dependencies and weights installed:

```bash
python generate.py --task t2v-A14B --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --size "832*480" --frame_num 81 --offload_model True --convert_model_dtype \
  --prompt_filepath prompt_relay_overlap.json
```

The [example](prompt_relay_overlap.json) assigns the robber entering a car to
`[0, 3)` seconds and an explosion at the bank behind the car to `[2, 4)` seconds.
Both local prompts are available without temporal penalty during `[2, 3)`.
They can describe different objects. Specify the subject and location in each
prompt; this is temporal control, without spatial masks or object identity binding.

## JSON fields

| Field | Meaning |
| --- | --- |
| `global_prompt` | Persistent scene context, with no temporal penalty. |
| `local_prompts` | One description per event; their order matches the interval list. |
| `segment_intervals` | Independent `[start, end)` pairs. Overlap, nesting, identical intervals, gaps and unsorted intervals are allowed. |
| `time_unit` | `seconds` or `internal_frame` (default). Internal endpoints must be integers. |
| `tail_width` | Distance outside the interval at which the temporal prior falls to `epsilon`. Uses `time_unit`; defaults to two internal time steps. |
| `epsilon` | Default `0.001`; must be strictly between 0 and 1. |

Each interval must be nonempty, within the video duration, and contain at least
one internal timestamp. Times are half-open: the end is excluded. Do not combine
`segment_intervals` with `segment_lengths`, even an empty length list.

Seconds use `config.sample_fps`, the same value used by `generate.py` to export
the video (16 FPS for T2V-A14B). Do not put a separate `fps` value in the JSON.
For the standard temporal stride of 4 and patch size of 1, nominal internal
timestamps are `0, 0.25, 0.5, ...` seconds at 16 FPS. Thus the example maps to
`[0, 12)` and `[8, 16)` internal frames. This scheduling grid is not the VAE's
full receptive field. Very short intervals may contain no grid point and produce
a clear error; the resolved internal intervals are logged before text encoding.

## Default behavior and compatibility

Without `segment_intervals`, the original consecutive allocation is retained:
explicit `segment_lengths`, or the original `ceil(internal_frames / prompt_count)`
step when lengths are absent/empty. The last segment is clipped to the video.
Invalid allocations that give a prompt no frames are rejected. Legacy lengths
may leave an uncovered tail; the old temporal decay is preserved there too.
Legacy lengths and default schedules keep the original Wan midpoint, window
and sigma settings.

**Missing times never trigger overlap detection or overlapping allocation.**
This Wan branch requires explicit intervals for overlap. Hunyuan's optional
`auto_overlap` heuristic is not enabled by this Wan implementation; requesting
it raises an error. `auto_overlap: false` is accepted.

Repeated prompt text is mapped to each occurrence in order, including when the
global prompt contains the same words. Prompts exceeding the T5 token budget are
rejected instead of silently dropping a routed event. JSON uses UTF-8. Prompt
extension must be disabled with Prompt Relay JSON, since it would change the
text being routed. Omitting `--prompt_filepath` uses the baseline pipeline.
Other Wan tasks are not connected to this JSON route.

## Attention

The attention formula remains:

```text
softmax(Q K^T / sqrt(d) - C) V
```

For a local prompt assigned internal range `[a, b)`, explicit interval mode uses
the distance to the nearest integer frame in that range:

```text
midpoint = (a + b - 1) / 2
window   = (b - a - 1) / 2
distance = max(abs(frame - midpoint) - window, 0)
C        = distance^2 / (2 sigma^2)
sigma    = tail_width_in_internal_steps / sqrt(2 log(1 / epsilon))
```

Every local prompt has its own text token columns. In an overlap, several sets
of columns have zero penalty simultaneously. They still share the same softmax;
equal weights or successful realization of both events are not guaranteed.
Outside each interval, attention decays smoothly rather than being hard blocked.
Global tokens, special tokens, text padding and the unconditional CFG pass retain
their previous routing behavior. Video self-attention is unchanged.

The routing kernel now stores temporal costs in float32 and matches the value
tensor dtype during attention multiplication, including outside autocast. Small
rounding differences from the earlier low-precision implementation are possible.
Sequence-parallel ranks carry their global query offset, so the same timestamp
gets the same penalty on every rank.

## Validation

```bash
python -m unittest discover -s tests -p test_prompt_relay.py -v
```

Tests exercise timing, token truncation/repetition, independent reference
attention on CPU/CUDA in float32/float16/bfloat16, legacy costs, real small Wan
cross-attention and Transformer layers, and simulated sequence-parallel offsets.
Model-layer tests require Wan's model dependencies; CUDA tests require CUDA.
For small model tests, FlashAttention self-attention is replaced with dense
attention, and distributed communication is simulated. These tests do not replace
full pretrained video generation or a real multi-GPU inference run.
