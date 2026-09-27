"""Weight-free window references and real small Wan model/pipeline checks.

Private package namespaces avoid importing unrelated speech/animation models.
Pipeline tests replace pretrained text/VAE weights, not the denoising loop.
Distributed tests simulate collectives; they are not a multi-GPU validation.
"""

import contextlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "_window_test_wan"
for suffix in ("", ".modules", ".distributed"):
    package = types.ModuleType(PREFIX + suffix)
    package.__path__ = [str(ROOT / "wan" / suffix.lstrip("."))]
    sys.modules[package.__name__] = package
options = importlib.import_module(PREFIX + ".sliding_window")
kernel = importlib.import_module(PREFIX + ".modules.sliding_window")


def dense_attention(q, k, v, k_lens=None, **kwargs):
    mask = None
    if k_lens is not None:
        mask = torch.arange(k.shape[1], device=q.device)[None, :] < k_lens.to(q.device)[:, None]
        mask = mask[:, None, None, :]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2).to(v.dtype), k.transpose(1, 2).to(v.dtype),
        v.transpose(1, 2), attn_mask=mask).transpose(1, 2).to(q.dtype)


def reference(q, k, v, grids, windows):
    """Independent per-token masked softmax in double precision."""
    output = torch.zeros_like(q, dtype=torch.float64)
    for sample, (frames, height, width) in enumerate(grids.tolist()):
        spatial = height * width
        valid = frames * spatial
        scores = torch.einsum("qhd,khd->hqk", q[sample, :valid].double(),
                              k[sample, :valid].double()) / math.sqrt(q.shape[-1])
        for token in range(valid):
            containing = [(a, b) for a, b in windows[sample] if a <= token // spatial < b]
            for a, b in containing:
                logits = scores[:, token].clone()
                logits[:, :a * spatial] = -torch.inf
                logits[:, b * spatial:] = -torch.inf
                output[sample, token] += torch.einsum(
                    "hk,khd->hd", logits.softmax(-1), v[sample, :valid].double())
            output[sample, token] /= len(containing)
    return output.to(q.dtype)


class WindowConfigTests(unittest.TestCase):
    def test_tail_single_frame_and_no_gaps(self):
        self.assertEqual(list(options.SlidingWindowConfig(3, 2).windows(8)),
                         [(0, 3), (2, 5), (4, 7), (6, 8)])
        self.assertEqual(list(options.SlidingWindowConfig(31, 16).windows(21)), [(0, 21)])
        self.assertEqual(list(options.SlidingWindowConfig(1, 1).windows(1)), [(0, 1)])
        for length in range(1, 10):
            for stride in range(1, length + 1):
                windows = list(options.SlidingWindowConfig(length, stride).windows(23))
                self.assertTrue(all(any(a <= t < b for a, b in windows) for t in range(23)))

    def test_invalid_configuration_and_explicit_disable(self):
        for length, stride in [(0, 1), (-1, 1), (3, 0), (3, -1), (3, 4), (True, 1), (3, 1.5)]:
            with self.subTest(length=length, stride=stride), self.assertRaises(ValueError):
                options.make_sliding_window_config(True, length, stride)
        with self.assertRaises(ValueError):
            options.make_sliding_window_config("false")
        with self.assertRaises(ValueError):
            list(options.SlidingWindowConfig().windows(0))
        self.assertIsNone(options.make_sliding_window_config(False, 0, 0))


class WindowKernelTests(unittest.TestCase):
    def inputs(self, device="cpu", dtype=torch.float32):
        torch.manual_seed(23)
        q = torch.randn(2, 18, 2, 8, device=device, dtype=dtype)
        return q, torch.randn_like(q), torch.randn_like(q), torch.tensor([16, 10]), torch.tensor([[8, 1, 2], [5, 2, 1]])

    def compare(self, device, dtype):
        q, k, v, lengths, grids = self.inputs(device, dtype)
        expected = reference(q, k, v, grids, [[(0, 3), (2, 5), (4, 7), (6, 8)], [(0, 3), (2, 5)]])
        for chunk in (1, 5, 128):
            actual = kernel.sliding_window_attention(q, k, v, lengths, grids, options.SlidingWindowConfig(3, 2), chunk)
            tolerance = 2e-2 if dtype == torch.bfloat16 else (2e-3 if dtype == torch.float16 else 1e-5)
            if device == "cuda" and dtype == torch.float32 and (
                    kernel.attention_backend.FLASH_ATTN_2_AVAILABLE or kernel.attention_backend.FLASH_ATTN_3_AVAILABLE):
                tolerance = 2e-2  # Wan's FlashAttention wrapper computes fp32 inputs in bf16.
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            self.assertEqual(actual.dtype, q.dtype)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.equal(actual[1, 10:], torch.zeros_like(actual[1, 10:])))

    def test_cpu_overlap_and_padding_match_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.compare("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_overlap_matches_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.compare("cuda", dtype)

    def test_full_window_matches_global_attention_on_valid_tokens(self):
        q, k, v, lengths, grids = self.inputs()
        expected = dense_attention(q, k, v, k_lens=lengths)
        actual = kernel.sliding_window_attention(q, k, v, lengths, grids, options.SlidingWindowConfig(31, 16))
        for b, length in enumerate(lengths):
            torch.testing.assert_close(actual[b, :length], expected[b, :length])

    def test_nonoverlapping_windows_and_single_frames(self):
        q, k, v, lengths, grids = self.inputs()
        for length, windows in [(3, [[(0, 3), (3, 6), (6, 8)], [(0, 3), (3, 5)]]),
                                (1, [[(i, i + 1) for i in range(8)], [(i, i + 1) for i in range(5)]])]:
            expected = reference(q, k, v, grids, windows)
            actual = kernel.sliding_window_attention(q, k, v, lengths, grids, options.SlidingWindowConfig(length, length))
            torch.testing.assert_close(actual, expected)

    def test_padding_and_distant_frames_cannot_leak_into_window(self):
        q, k, v, lengths, grids = self.inputs()
        config = options.SlidingWindowConfig(3, 2)
        baseline = kernel.sliding_window_attention(q, k, v, lengths, grids, config)
        k[:, 14:] = 10000
        v[:, 14:] = -10000
        k[1, 10:] = 10000
        v[1, 10:] = -10000
        changed = kernel.sliding_window_attention(q, k, v, lengths, grids, config)
        torch.testing.assert_close(baseline[:, :4], changed[:, :4], atol=0, rtol=0)
        torch.testing.assert_close(baseline[1], changed[1], atol=0, rtol=0)

    def test_sdpa_fallback_bounds_query_and_key_lengths(self):
        q, k, v, lengths, grids = self.inputs()
        with patch.object(kernel.F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as calls:
            kernel.sliding_window_attention(q, k, v, lengths, grids, options.SlidingWindowConfig(3, 2), 3)
        self.assertTrue(calls.call_args_list)
        for call in calls.call_args_list:
            self.assertLessEqual(call.args[0].shape[2], 3)
            self.assertLessEqual(call.args[1].shape[2], 6)

    def test_mixed_rope_and_value_dtypes(self):
        q, k, v, lengths, grids = self.inputs(dtype=torch.bfloat16)
        out = kernel.sliding_window_attention(q.float(), k.float(), v, lengths, grids, options.SlidingWindowConfig(3, 2))
        expected = reference(q, k, v, grids, [[(0, 3), (2, 5), (4, 7), (6, 8)], [(0, 3), (2, 5)]])
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, expected.float(), atol=.02, rtol=.02)

    def test_invalid_grid_or_shape_fails(self):
        q, k, v, lengths, grids = self.inputs()
        for bad_lengths, bad_grids in [(lengths + 1, grids), (lengths, grids[:1]),
                                      (lengths, torch.tensor([[0, 1, 2], [5, 2, 1]]))]:
            with self.assertRaises(ValueError):
                kernel.sliding_window_attention(q, k, v, bad_lengths, bad_grids, options.SlidingWindowConfig())
        with self.assertRaises(ValueError):
            kernel.sliding_window_attention(q, k[:, :-1], v, lengths, grids, options.SlidingWindowConfig())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_flash_dispatch_receives_sliced_windows_without_padding(self):
        q, k, v, lengths, grids = self.inputs("cuda")
        with patch.object(kernel.attention_backend, "FLASH_ATTN_2_AVAILABLE", True), \
                patch.object(kernel.attention_backend, "flash_attention", side_effect=dense_attention) as flash:
            actual = kernel.sliding_window_attention(q, k, v, lengths, grids, options.SlidingWindowConfig(3, 2))
        self.assertEqual([call.args[0].shape[1] for call in flash.call_args_list], [6, 6, 6, 4, 6, 6])
        expected = reference(q, k, v, grids, [[(0, 3), (2, 5), (4, 7), (6, 8)], [(0, 3), (2, 5)]])
        torch.testing.assert_close(actual, expected)


class WanIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model_module = importlib.import_module(PREFIX + ".modules.model")
            cls.sp = importlib.import_module(PREFIX + ".distributed.sequence_parallel")
            cls.ulysses = importlib.import_module(PREFIX + ".distributed.ulysses")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"Install Wan model dependencies: {error}")

    def small_model(self):
        model = self.model_module.WanModel(dim=24, ffn_dim=48, freq_dim=8, text_dim=8,
                                          in_dim=2, out_dim=2, num_heads=2, num_layers=1,
                                          text_len=8, patch_size=(1, 2, 2))
        torch.nn.init.normal_(model.head.head.weight, std=.02)
        return model.eval()

    def test_small_model_preserves_relay_timing_and_disable_does_not_leak(self):
        from test_prompt_relay import WordTokenizer
        relay = importlib.import_module(PREFIX + ".prompt_relay")
        routing = importlib.import_module(PREFIX + ".modules.temporal_routing")
        payload, _ = relay.prepare_prompt_relay(
            dict(global_prompt="scene", local_prompts=["enter car", "bank explodes"],
                 segment_intervals=[[0, 5], [3, 8]]), WordTokenizer(), 29,
            (32, 16), (4, 8, 8), (1, 2, 2), 16)
        model = self.small_model()
        inputs = dict(x=[torch.randn(2, 8, 2, 4)], t=torch.tensor([500.]),
                      context=[torch.randn(8, 8)], seq_len=18, cross_attn_q_token_idx=payload)
        with torch.no_grad(), patch.object(self.model_module, "flash_attention", side_effect=dense_attention), \
                patch.object(self.model_module, "chunked_softmax_attention", wraps=routing.chunked_softmax_attention) as routed:
            baseline = model(**inputs)[0]
            full = model(**inputs, sliding_window_config=options.SlidingWindowConfig(31, 16))[0]
            windowed = model(**inputs, sliding_window_config=options.SlidingWindowConfig(3, 2))[0]
            disabled = model(**inputs, sliding_window_config=None)[0]
        torch.testing.assert_close(full, baseline)
        torch.testing.assert_close(disabled, baseline, atol=0, rtol=0)
        self.assertFalse(torch.allclose(windowed, baseline))
        self.assertTrue(torch.isfinite(windowed).all())
        self.assertEqual(windowed.shape, inputs["x"][0].shape)
        for call in routed.call_args_list:
            self.assertIs(call.kwargs["q_token_idx"], payload)
            self.assertEqual(call.args[0].shape[1], 18)
        self.assertNotIn("query_token_offset", payload[0])

    def test_self_attention_windows_use_global_rope(self):
        layer = self.model_module.WanSelfAttention(24, 2, window_size=[-1, -1])
        x, grids = torch.randn(1, 16, 24), torch.tensor([[8, 1, 2]])
        freqs = self.small_model().freqs
        expected_q = self.model_module.rope_apply(layer.norm_q(layer.q(x)).view(1, 16, 2, 12), grids, freqs)
        with patch.object(self.model_module, "sliding_window_attention", wraps=kernel.sliding_window_attention) as windows:
            layer(x, torch.tensor([16]), grids, freqs, sliding_window_config=options.SlidingWindowConfig(3, 2))
        torch.testing.assert_close(windows.call_args.args[0], expected_q)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_model_and_sequence_parallel_autocast(self):
        model = self.small_model().cuda()
        inputs = dict(x=[torch.randn(2, 8, 2, 4, device="cuda")], t=torch.tensor([500.], device="cuda"),
                      context=[torch.randn(8, 8, device="cuda")], seq_len=18,
                      sliding_window_config=options.SlidingWindowConfig(3, 2))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), \
                patch.object(self.model_module, "flash_attention", side_effect=dense_attention):
            expected = model(**inputs)[0]
            model.blocks[0].self_attn.forward = types.MethodType(self.sp.sp_attn_forward, model.blocks[0].self_attn)
            with patch.object(self.sp, "get_world_size", return_value=1), patch.object(self.sp, "get_rank", return_value=0), \
                    patch.object(self.sp, "gather_forward", side_effect=lambda x, dim: x), \
                    patch.object(self.ulysses.dist, "is_initialized", return_value=True), \
                    patch.object(self.ulysses, "all_to_all", side_effect=lambda x, **kw: x):
                actual = self.sp.sp_dit_forward(model, **inputs)[0]
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)

    def test_ulysses_windows_are_applied_after_global_sequence_gather(self):
        torch.manual_seed(41)
        q = torch.randn(1, 18, 4, 8)
        k, v = torch.randn_like(q), torch.randn_like(q)
        lengths, grids = torch.tensor([16]), torch.tensor([[8, 1, 2]])
        config = options.SlidingWindowConfig(3, 2)
        expected = kernel.sliding_window_attention(q, k, v, lengths, grids, config)
        for rank in range(2):
            local = [tensor[:, rank * 9:(rank + 1) * 9] for tensor in (q, k, v)]
            gathered = [tensor[:, :, rank * 2:(rank + 1) * 2] for tensor in (q, k, v)]
            calls = []
            def exchange(tensor, scatter_dim, gather_dim):
                i = len(calls)
                calls.append((scatter_dim, gather_dim))
                if i < 3:
                    torch.testing.assert_close(tensor, local[i])
                    self.assertEqual((scatter_dim, gather_dim), (2, 1))
                    return gathered[i]
                self.assertEqual((scatter_dim, gather_dim), (1, 2))
                torch.testing.assert_close(tensor, expected[:, :, rank * 2:(rank + 1) * 2])
                return expected[:, rank * 9:(rank + 1) * 9]
            with patch.object(self.ulysses.dist, "is_initialized", return_value=True), \
                    patch.object(self.ulysses, "all_to_all", side_effect=exchange):
                actual = self.ulysses.distributed_attention(*local, lengths, grid_sizes=grids, sliding_window_config=config)
            self.assertEqual(len(calls), 4)
            torch.testing.assert_close(actual, expected[:, rank * 9:(rank + 1) * 9])

    def test_sequence_parallel_model_passes_window_config_through_real_blocks(self):
        model = self.small_model()
        inputs = dict(x=[torch.randn(2, 8, 2, 4)], t=torch.tensor([500.]),
                      context=[torch.randn(8, 8)], seq_len=18,
                      sliding_window_config=options.SlidingWindowConfig(3, 2))
        with torch.no_grad(), patch.object(self.model_module, "flash_attention", side_effect=dense_attention):
            expected = model(**inputs)[0]
            sp = self.sp
            def forward(layer, *args, **kwargs):
                return sp.sp_attn_forward(layer, *args, dtype=torch.float32, **kwargs)
            model.blocks[0].self_attn.forward = types.MethodType(forward, model.blocks[0].self_attn)
            with patch.object(sp, "get_world_size", return_value=1), patch.object(sp, "get_rank", return_value=0), \
                    patch.object(sp, "gather_forward", side_effect=lambda x, dim: x), \
                    patch.object(self.ulysses.dist, "is_initialized", return_value=True), \
                    patch.object(self.ulysses, "all_to_all", side_effect=lambda x, **kw: x):
                actual = sp.sp_dit_forward(model, **inputs)[0]
        torch.testing.assert_close(actual, expected)


class PipelineAndCliTests(unittest.TestCase):
    small_model = WanIntegrationTests.small_model

    @classmethod
    def setUpClass(cls):
        WanIntegrationTests.setUpClass.__func__(cls)
        try:
            cls.pipeline_module = importlib.import_module(PREFIX + ".text2video")
            aliases = {"wan" + name[len(PREFIX):]: module for name, module in list(sys.modules.items())
                       if name == PREFIX or name.startswith(PREFIX + ".")}
            # Restore only Wan aliases: clearing unrelated imports can re-register
            # torchvision CUDA operators when later tests move a model to GPU.
            previous = {name: module for name, module in sys.modules.items()
                        if name == "wan" or name.startswith("wan.")}
            sys.modules.update(aliases)
            try:
                spec = importlib.util.spec_from_file_location("_window_test_generate", ROOT / "generate.py")
                cls.cli = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cls.cli)
            finally:
                for name in list(sys.modules):
                    if name == "wan" or name.startswith("wan."):
                        del sys.modules[name]
                sys.modules.update(previous)
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"Install Wan T2V/CLI dependencies: {error}")

    def test_cli_defaults_boolean_flags_and_invalid_options(self):
        base = ["--ckpt_dir", "unused"]
        args = self.cli._parse_args(base)
        self.assertEqual((args.sliding_window, args.window_length, args.window_stride), (False, 31, 16))
        for value, enabled in [(None, True), ("true", True), ("1", True), ("false", False), ("0", False)]:
            args = self.cli._parse_args(base + ["--sliding_window"] + ([] if value is None else [value]))
            self.assertEqual(args.sliding_window, enabled)
        for invalid in [["--window_length", "0"], ["--window_stride", "0"], ["--window_stride", "32"],
                        ["--task", "i2v-A14B"], ["--task", "ti2v-5B"], ["maybe"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.cli._parse_args(base + ["--sliding_window"] + invalid)
            self.assertEqual(error.exception.code, 2)

    def test_cli_forwards_options_to_t2v_pipeline(self):
        args = self.cli._parse_args(["--ckpt_dir", "unused", "--sliding_window", "--window_length", "12", "--window_stride", "6"])
        pipeline = Mock()
        pipeline.generate.return_value = torch.zeros(3, 1, 2, 2)
        with patch.object(self.cli.wan, "WanT2V", return_value=pipeline, create=True), \
                patch.object(self.cli, "save_video"), patch.object(self.cli, "_init_logging"), \
                patch.object(self.cli.torch.cuda, "synchronize"), \
                patch.dict(self.cli.os.environ, {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}):
            self.cli.generate(args)
        received = pipeline.generate.call_args.kwargs
        self.assertEqual((received["sliding_window"], received["window_length"], received["window_stride"]), (True, 12, 6))

    def test_denoising_loop_windows_both_cfg_branches_and_both_experts(self):
        from test_prompt_relay import WordTokenizer
        pipe = self.pipeline_module.WanT2V.__new__(self.pipeline_module.WanT2V)
        pipe.device, pipe.rank, pipe.sp_size = torch.device("cpu"), 0, 1
        pipe.t5_cpu, pipe.init_on_cpu = True, False
        pipe.vae_stride, pipe.patch_size = (4, 8, 8), (1, 2, 2)
        pipe.param_dtype, pipe.num_train_timesteps, pipe.boundary = torch.bfloat16, 1000, .75
        pipe.config = types.SimpleNamespace(sample_fps=16)
        pipe.sample_neg_prompt = "blur"
        class Encoder:
            tokenizer = WordTokenizer()
            def __call__(self, prompts, device):
                return [torch.full((8, 8), float(len(text)) / 100, device=device) for text in prompts]
        pipe.text_encoder = Encoder()
        pipe.vae = types.SimpleNamespace(model=types.SimpleNamespace(z_dim=2), decode=lambda x: x)
        pipe.low_noise_model, pipe.high_noise_model = self.small_model(), self.small_model()
        schedule = dict(global_prompt="scene", local_prompts=["enter car", "bank explodes"], segment_intervals=[[0, 5], [3, 8]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(schedule), encoding="utf-8")
            with patch.object(self.model_module, "flash_attention", side_effect=dense_attention), \
                    patch.object(pipe.low_noise_model, "forward", wraps=pipe.low_noise_model.forward) as low, \
                    patch.object(pipe.high_noise_model, "forward", wraps=pipe.high_noise_model.forward) as high:
                out = pipe.generate("unused", size=(32, 16), frame_num=29, shift=1.0, sampling_steps=3,
                                    seed=7, offload_model=False, prompt_filepath=str(path),
                                    sliding_window=True, window_length=3, window_stride=2)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (2, 8, 2, 4))
        self.assertGreater(high.call_count, 0)
        self.assertGreater(low.call_count, 0)
        for expert in (high, low):
            self.assertEqual(expert.call_count % 2, 0)
            for i, call in enumerate(expert.call_args_list):
                self.assertEqual(call.kwargs["sliding_window_config"], options.SlidingWindowConfig(3, 2))
                self.assertEqual(bool(call.kwargs.get("cross_attn_q_token_idx")), i % 2 == 0)


if __name__ == "__main__":
    unittest.main()
