# Temporal sliding-window self-attention for Wan2.2

This optional inference mode applies to **T2V-A14B**, with or without Prompt
Relay. It needs no new model weights. It is available in the
[`wan2.2-sliding-window` Prompt Relay branch](https://github.com/GordonChen19/Prompt-Relay/tree/wan2.2-sliding-window)
and the standalone
[`feat/prompt-relay-sliding-window` Wan branch](https://github.com/DasbootU9607/Wan2.2/tree/feat/prompt-relay-sliding-window).

## Usage

Run from the Wan2.2 directory with the usual dependencies and checkpoints:

```bash
python generate.py --task t2v-A14B --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --size "832*480" --frame_num 241 --offload_model True --convert_model_dtype \
  --prompt "A continuous wide shot of a hiker walking beside a lake, with a dog exploring nearby." \
  --sliding_window --window_length 31 --window_stride 16
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--sliding_window` | `false` | Enable with the bare flag or `true`/`1`; disable with `false`/`0`. |
| `--window_length` | `31` | Temporal window size in transformer latent frames. |
| `--window_stride` | `16` | Distance between window starts in the same units. |

When enabled, both lengths must be positive integers and stride must not exceed
length. Invalid options and unsupported tasks fail before model loading.
Disabled mode uses the existing attention path and ignores window sizes.
The Python API exposes the same three arguments on `WanT2V.generate()`.

The stock T2V VAE has temporal stride 4 and the transformer temporal patch size
is 1: `internal_frames = (frame_num - 1) // 4 + 1`. Thus 81 output frames are
21 internal frames, and 241 output frames are 61 internal frames. A window of
31 does **not** split an 81-frame video. Use 12/6 on that short video to exercise
multiple windows. Increasing `--frame_num` sets output duration; changing the
window size alone does not add output frames.

## How it works

The implementation follows the same fixed-stride window and averaging method
as the earlier Hunyuan implementation, adapted to Wan's separate self-attention
and text cross-attention layers:

1. Apply RoPE using the full video's absolute positions.
2. Slice video Q/K/V by contiguous temporal windows, retaining all spatial
   tokens in each selected frame. The final window is shortened if needed.
3. Compute bidirectional self-attention inside each window.
4. Accumulate overlapping results in float32 for half-precision inputs and
   divide by the number of covering windows. Exclude sequence padding.

For eight internal frames, length 3 and stride 2 produce `[0,3)`, `[2,5)`,
`[4,7)`, `[6,8)`. Frames 2, 4, and 6 receive the average of two window results.
A single window covering the video is equivalent to global self-attention on
valid tokens, within attention backend precision differences.

FlashAttention 2/3, when available, receives each physically sliced window;
this does not depend on its token-local `window_size` option. Otherwise the
implementation uses [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
with queries additionally chunked at 128 tokens. No full-video square mask is
created. This fallback applies to the new window kernel; other Wan attention
paths retain their existing dependency requirements.

Both high-noise and low-noise experts, and both conditional and unconditional
CFG predictions, receive the same window configuration. The configuration is
per call, so enabling it for one generation does not enable it for later calls.
Ulysses gathers the global sequence before window attention, then scatters the
result; rank boundaries do not restart windows or the timeline.

## Combining with Prompt Relay

Add `--prompt_filepath your_schedule.json` to use explicit overlapping events.
For a short check with the supplied schedule:

```bash
python generate.py --task t2v-A14B --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --size "832*480" --frame_num 81 --offload_model True --convert_model_dtype \
  --prompt_filepath prompt_relay_overlap.json --base_seed 42 \
  --sliding_window --window_length 12 --window_stride 6
```

Prompt Relay controls which text tokens guide each video time. Sliding windows
limit which other video frames each frame can attend to. The cross-attention
schedule, global prompt, overlapping intervals, and sequence-parallel time
offsets are preserved. Do not reset prompt intervals at each window start.
For a longer video, schedule events against its full timeline. Prompt extension
must remain disabled with Prompt Relay JSON.

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests cover independent dense attention references in float32/float16/bfloat16,
overlap averaging, tail coverage, padding exclusion, locality, full-window
equivalence, absolute RoPE positions, unchanged Prompt Relay timing, enable/
disable behavior, CLI forwarding, and a real small-model denoising loop through
both experts and CFG branches. CUDA tests also exercise autocast with real small
Wan layers. Pretrained text/VAE weights are replaced by tiny fixtures.
FlashAttention dispatch and distributed collectives are simulated; the SDPA
numerics are real. Missing model/CLI dependencies or CUDA cause explicit skips.

For an end-to-end comparison, keep the seed, prompt JSON, resolution, frame
count, and sampling settings identical. On an 81-frame clip, compare disabled,
full-window 31/16, and shorter-window 12/6 runs, using different `--save_file`
paths. The first two should be numerically close; check the third for continuity
and correct overlapping event timing. Then increase the frame count and compare
peak GPU memory and runtime. Review several seeds before judging video quality.

Full pretrained A14B generation, real FlashAttention kernels, and real multi-GPU
inference have not been validated by these tests.

## Limits

The windows reduce the temporal span of self-attention. They do not stream the
entire diffusion process or VAE: full-video latents, Q/K/V, feed-forward layers,
text cross-attention, and decoding still consume memory. Actual memory and
runtime changes depend on resolution, window overlap, and attention backend;
overlap repeats some work. A smaller window weakens direct long-range temporal
connections and can change motion or identity consistency. Longer generation
is still constrained by available memory and Wan's existing positional tables
(1024 positions per grid axis); this does not guarantee arbitrary duration or
unchanged video quality. Other Wan task pipelines are not enabled by this CLI.
