"""Weight-free timing/attention tests, plus small real Wan layers when available.

Load a private package namespace to avoid importing unrelated speech/animation
pipelines from wan.__init__. No production modules are replaced in that namespace.
"""

import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
for suffix in ("", ".modules", ".distributed"):
    name = "_relay_test_wan" + suffix
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "wan" / suffix.lstrip("."))]
    sys.modules[name] = package
relay = importlib.import_module("_relay_test_wan.prompt_relay")
routing = importlib.import_module("_relay_test_wan.modules.temporal_routing")


class WordTokenizer:
    """T5-style no-BOS, EOS and truncation behavior, using whole-word tokens."""

    def __init__(self, limit=512):
        self.limit = limit
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True, truncation=True, **kwargs):
        ids = [self.vocab.setdefault(word, len(self.vocab) + 2) for word in text.split()]
        if truncation:
            ids = ids[:self.limit - int(add_special_tokens)]
        return torch.tensor([ids + ([1] if add_special_tokens else [])], dtype=torch.long)


def prepare(intervals=None, **changes):
    config = dict(global_prompt="scene", local_prompts=["robber enters car", "bank explodes"])
    if intervals is not None:
        config["segment_intervals"] = intervals
    config.update(changes)
    return relay.prepare_prompt_relay(config, WordTokenizer(), 81, (32, 16),
                                      (4, 8, 8), (1, 2, 2), 16)


class ScheduleTests(unittest.TestCase):
    def test_missing_times_preserve_legacy_nonoverlapping_allocation(self):
        payload, _ = prepare()
        self.assertEqual([p["frame_interval"] for p in payload], [[0, 11], [11, 21]])
        self.assertEqual([(p["midpoint"], p["window"]) for p in payload], [(5, 3), (16, 3)])
        self.assertEqual(payload[0]["sigma"], torch.tensor(.1448, dtype=torch.float16).item())

    def test_legacy_lengths_remain_consecutive(self):
        payload, _ = prepare(segment_lengths=[8, 13])
        self.assertEqual([p["frame_interval"] for p in payload], [[0, 8], [8, 21]])

    def test_seconds_use_export_fps_and_half_open_intervals(self):
        payload, _ = prepare([[0, 3], [2, 4]], time_unit="seconds")
        self.assertEqual([p["frame_interval"] for p in payload], [[0, 12], [8, 16]])
        config = dict(segment_intervals=[[0, 3], [2, 4]], time_unit="seconds")
        intervals, _ = relay.prepare_schedule(config, 2, 81, 4, 1, 20)
        self.assertEqual(intervals, [[0, 15], [10, 20]])

    def test_temporal_patch_center_timestamps(self):
        config = dict(segment_intervals=[[0, .5], [.5, 1]], time_unit="seconds")
        intervals, _ = relay.prepare_schedule(config, 2, 29, 4, 2, 16)
        self.assertEqual(intervals, [[0, 1], [1, 2]])

    def test_unsorted_nested_adjacent_and_gapped_intervals(self):
        for intervals in ([[8, 15], [0, 10]], [[0, 21], [3, 6]],
                          [[0, 8], [8, 21]], [[0, 4], [10, 21]], [[0, 21], [0, 21]]):
            with self.subTest(intervals=intervals):
                payload, _ = prepare(intervals)
                self.assertEqual([p["frame_interval"] for p in payload], intervals)

    def test_triple_overlap_and_more_prompts_than_frames(self):
        payload, _ = prepare([[0, 1]] * 25, local_prompts=["same event"] * 25)
        self.assertEqual(len(payload), 25)
        self.assertEqual(len(set(p["local_token_idx"][0].item() for p in payload)), 25)

    def test_repeated_prompts_do_not_match_global_or_each_other(self):
        payload, text = prepare([[0, 12], [8, 21]], global_prompt="same event",
                                local_prompts=["same event", "same event"])
        self.assertEqual(text, "same event same event same event")
        self.assertEqual([p["local_token_idx"].tolist() for p in payload], [[2, 3], [4, 5]])

    def test_truncated_local_prompt_is_rejected(self):
        config = dict(global_prompt="scene", local_prompts=["one two three", "four five six"])
        with self.assertRaisesRegex(ValueError, "token budget"):
            relay.prepare_prompt_relay(config, WordTokenizer(limit=6), 81, (32, 16),
                                        (4, 8, 8), (1, 2, 2), 16)

    def test_empty_local_list_has_no_routing(self):
        payload, text = prepare(local_prompts=[])
        self.assertIsNone(payload)
        self.assertEqual(text, "scene")

    def test_invalid_explicit_inputs(self):
        invalid = [dict(segment_intervals=[[0, 2]]),
                   dict(segment_intervals=[[0, 0], [2, 4]]),
                   dict(segment_intervals=[[-1, 2], [2, 4]]),
                   dict(segment_intervals=[[0, 22], [2, 4]]),
                   dict(segment_intervals=[[0, float("nan")], [2, 4]]),
                   dict(segment_intervals=[[False, 2], [2, 4]]),
                   dict(segment_intervals=[[0, 2.5], [2, 4]]),
                   dict(segment_intervals=[[0], [2, 4]]),
                   dict(segment_lengths=[]), dict(time_unit="frames"),
                   dict(epsilon=0), dict(epsilon=1), dict(epsilon=float("inf")),
                   dict(tail_width=0), dict(tail_width=1e-100), dict(tail_width=1e100),
                   dict(routing_mode="other"), dict(sigma=1), dict(fps=24),
                   dict(auto_overlap=True), dict(overlap_ratio=.2)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare([[0, 12], [8, 21]], **changes)

    def test_options_without_intervals_cannot_silently_enable_overlap(self):
        for changes in (dict(time_unit="seconds"), dict(tail_width=2), dict(epsilon=.01),
                        dict(auto_overlap=True), dict(segment_lengths=[1]),
                        dict(segment_lengths=[0, 21]), dict(local_prompts=["", "action"])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare(**changes)
        self.assertEqual([p["frame_interval"] for p in prepare(auto_overlap=False)[0]],
                         [[0, 11], [11, 21]])

    def test_unresolvable_times_and_incompatible_grid(self):
        with self.assertRaisesRegex(ValueError, "no internal timestamp"):
            prepare([[.01, .02], [2, 4]], time_unit="seconds")
        for frame_num in (80, 0, True):
            with self.assertRaises(ValueError):
                relay.prepare_schedule(dict(segment_intervals=[[0, 1]]), 1, frame_num, 4, 1, 16)


class AttentionTests(unittest.TestCase):
    def test_both_prompts_have_zero_cost_throughout_overlap(self):
        payload, _ = prepare([[0, 12], [8, 16]])
        cost = routing.build_temporal_cost(payload, 42, 8, "cpu", torch.float32)
        for frame in range(21):
            for p in payload:
                a, b = p["frame_interval"]
                local_cost = cost[frame * 2, p["local_token_idx"]]
                self.assertTrue(torch.all(local_cost == 0) if a <= frame < b else torch.all(local_cost > 0))
        self.assertTrue(torch.all(cost[:, [0, 6, 7]] == 0))  # Global, EOS and padding retain baseline behavior.

    def test_tail_width_has_requested_decay(self):
        payload, _ = prepare([[4, 8], [10, 14]], tail_width=2, epsilon=.001)
        cost = routing.build_temporal_cost(payload, 42, 6, "cpu", torch.float32)
        torch.testing.assert_close(torch.exp(-cost[2 * 2, 1]), torch.tensor(.001))

    def test_simulated_shards_match_full_cost_even_across_frame_boundaries(self):
        payload, _ = prepare([[0, 12], [8, 21]])
        full = routing.build_temporal_cost(payload, 42, 6, "cpu", torch.float32)
        parts, start = [], 0
        for length in (13, 13, 16):
            routed = routing.offset_prompt_relay(payload, start)
            parts.append(routing.build_temporal_cost(routed, length, 6, "cpu", torch.float32))
            start += length
        torch.testing.assert_close(torch.cat(parts), full)
        self.assertNotIn("query_token_offset", payload[0])
        self.assertIsNone(routing.offset_prompt_relay(None, 13))

    def _compare_reference(self, device, dtype):
        torch.manual_seed(7)
        payload, _ = prepare([[0, 12], [8, 21]])
        q = torch.randn(2, 42, 2, 8, device=device, dtype=dtype)
        k = torch.randn(2, 8, 2, 8, device=device, dtype=dtype)
        v = torch.randn_like(k)
        # Independent reference: distance to the nearest integer inside [a,b).
        frames = torch.arange(42, device=device) // 2
        cost = torch.zeros(42, 8, device=device)
        for segment in payload:
            a, b = segment["frame_interval"]
            distance = torch.maximum((a - frames).clamp(min=0), (frames - (b - 1)).clamp(min=0)).float()
            cost[:, segment["local_token_idx"].to(device)] = (distance.square() / (2 * segment["sigma"] ** 2))[:, None]
        logits = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2) / math.sqrt(8)
        expected = (torch.softmax(logits - cost, -1) @ v.transpose(1, 2).float()).transpose(1, 2)
        tolerance = 1e-5 if dtype == torch.float32 else .025
        for chunk in (1, 16, 64):
            actual = routing.chunked_softmax_attention(q, k, v, payload, chunk_size=chunk)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)

    def test_cpu_attention_matches_independent_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._compare_reference("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_attention_matches_independent_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._compare_reference("cuda", dtype)

    def test_overlap_retrieves_values_from_both_events(self):
        payload, _ = prepare([[0, 12], [8, 21]])
        q = torch.zeros(1, 42, 1, 2)
        k, v = torch.zeros(1, 6, 1, 2), torch.zeros(1, 6, 1, 2)
        v[:, 1:4, :, 0], v[:, 4:6, :, 1] = 1, 1
        out = routing.chunked_softmax_attention(q, k, v, payload)[0, ::2, 0]
        self.assertLess(out[0, 1].item(), 1e-8)
        self.assertGreater(out[9, 0].item(), .1)
        self.assertGreater(out[9, 1].item(), .1)
        self.assertLess(out[20, 0].item(), 1e-8)

    def test_legacy_attention_matches_original_penalty(self):
        payload, _ = prepare()
        frames = (torch.arange(42) // 2).float()
        expected = torch.zeros(42, 6)
        for a, b, indices in ((0, 11, [1, 2, 3]), (11, 21, [4, 5])):
            sigma = torch.tensor(.1448, dtype=torch.float16).float()
            distance = (frames - ((a + b) // 2)).abs()
            expected[:, indices] = (torch.relu(distance - ((b - a) // 2 - 2)).square() / (2 * sigma.square()))[:, None]
        actual = routing.build_temporal_cost(payload, 42, 6, "cpu", torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class WanLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = importlib.import_module("_relay_test_wan.modules.model")
            cls.sp = importlib.import_module("_relay_test_wan.distributed.sequence_parallel")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"Install Wan model dependencies for layer tests: {error}")

    @staticmethod
    def dense_flash(q, k, v, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    def test_real_cross_attention_routes_only_when_payload_present(self):
        layer = self.model.WanCrossAttention(dim=16, num_heads=2)
        payload, _ = prepare([[0, 12], [8, 21]])
        x, context = torch.randn(1, 42, 16), torch.randn(1, 8, 16)
        with patch.object(self.model, "flash_attention", side_effect=self.dense_flash) as flash:
            routed = layer(x, context, None, payload)
            flash.assert_not_called()
            baseline = layer(x, context, None, None)
            flash.assert_called_once()
        self.assertEqual(routed.shape, x.shape)
        self.assertFalse(torch.allclose(routed, baseline))

    def _small_model(self):
        model = self.model.WanModel(dim=24, ffn_dim=48, freq_dim=8, text_dim=8,
                                   in_dim=2, out_dim=2, num_heads=2, num_layers=1,
                                   text_len=8, patch_size=(1, 2, 2))
        torch.nn.init.normal_(model.head.head.weight, std=.02)
        return model.eval()

    def test_small_transformer_uses_prepared_overlapping_schedule(self):
        torch.manual_seed(11)
        model = self._small_model()
        payload, _ = prepare([[0, 12], [8, 21]])
        inputs = dict(x=[torch.randn(2, 21, 2, 4)], t=torch.tensor([500.]),
                      context=[torch.randn(8, 8)], seq_len=42)
        with torch.no_grad(), patch.object(self.model, "flash_attention", side_effect=self.dense_flash):
            routed = model(**inputs, cross_attn_q_token_idx=payload)[0]
            baseline = model(**inputs)[0]
        self.assertEqual(routed.shape, inputs["x"][0].shape)
        self.assertTrue(torch.isfinite(routed).all())
        self.assertFalse(torch.allclose(routed, baseline))

    def test_sequence_parallel_forward_passes_global_query_offset(self):
        model = self._small_model()
        payload, _ = prepare([[0, 12], [8, 21]])
        inputs = dict(x=[torch.randn(2, 21, 2, 4)], t=torch.tensor([500.]),
                      context=[torch.randn(8, 8)], seq_len=42,
                      cross_attn_q_token_idx=payload)
        for rank in (0, 1):
            with torch.no_grad(), patch.object(self.sp, "get_world_size", return_value=2), \
                    patch.object(self.sp, "get_rank", return_value=rank), \
                    patch.object(self.sp, "gather_forward", side_effect=lambda x, dim: x), \
                    patch.object(model, "unpatchify", side_effect=lambda x, grid: [x]), \
                    patch.object(model.blocks[0].self_attn, "forward", side_effect=lambda x, *args, **kw: torch.zeros_like(x)), \
                    patch.object(self.model, "chunked_softmax_attention", wraps=routing.chunked_softmax_attention) as routed:
                self.sp.sp_dit_forward(model, **inputs)
                received = routed.call_args.kwargs["q_token_idx"]
                self.assertEqual(received[0]["query_token_offset"], rank * 21)
        self.assertNotIn("query_token_offset", payload[0])


if __name__ == "__main__":
    unittest.main()
