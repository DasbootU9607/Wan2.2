## Prompt Relay timing and attention tests

Run `python -m unittest discover -s tests -p test_prompt_relay.py -v` from the Wan2.2 directory. No model weights are needed. The suite checks explicit overlap, legacy timing, token spans, CPU/CUDA reference attention and small Wan layers. Distributed communication is simulated; see [validation details](../PROMPT_RELAY.md#validation).


Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```
