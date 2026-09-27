## Sliding-window and Prompt Relay tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The sliding-window suite checks independent attention references, overlap
averaging, padding, temporal locality, full-window equivalence, CLI validation,
real small Wan layers, mixed-precision CUDA, and the T2V denoising loop through
both experts and both CFG branches. Pretrained text/VAE weights are replaced
with tiny fixtures. FlashAttention dispatch and Ulysses collectives are simulated;
the CPU/CUDA SDPA calculations are real. No full model weights are required.
Install the usual Wan T2V/CLI dependencies for all integration tests; unavailable
dependencies or CUDA produce explicit skips. See [validation](../SLIDING_WINDOW.md#validation).

## Prompt Relay timing and attention tests

Run `python -m unittest discover -s tests -p test_prompt_relay.py -v` from the Wan2.2 directory. No model weights are needed. The suite checks explicit overlap, legacy timing, token spans, CPU/CUDA reference attention and small Wan layers. Distributed communication is simulated; see [validation details](../PROMPT_RELAY.md#validation).


Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
