import asyncio
import threading
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

import activation_codec as codec
from benchmarking import BenchmarkRun
from cache_migration import (
    assert_cache_lengths,
    extract_kv_delta,
    install_kv_delta,
)
from kv_streaming import stream_from_worker
from edge_quantization import quantize_overlap_layers


class FakeDynamicCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, key, value, layer_idx):
        if layer_idx > len(self.key_cache):
            raise ValueError("gap")
        if layer_idx == len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value], dim=-2
            )


class ProtocolTests(unittest.TestCase):
    def test_session_round_trip(self):
        frame = codec.pack_session_start(
            4, 12, 10, True, 64, True, "fp16", True,
            model_name="example/model", torch_dtype="torch.float16",
        )
        fields = codec.unpack_session_start(frame)
        self.assertEqual(fields["prefill_edge_layers"], 4)
        self.assertEqual(fields["decode_edge_layers"], 12)
        self.assertEqual(fields["prefill_chunk_size"], 64)
        self.assertTrue(fields["benchmark_enabled"])
        self.assertFalse(fields["overlap_kv_transfer"])
        self.assertEqual(fields["kv_transfer_queue_depth"], 1)
        self.assertFalse(fields["allow_model_load"])
        self.assertEqual(fields["model_name"], "example/model")
        self.assertEqual(fields["torch_dtype"], "torch.float16")

    def test_session_overlap_round_trip(self):
        fields = codec.unpack_session_start(codec.pack_session_start(
            4,
            12,
            10,
            overlap_kv_transfer=True,
            kv_transfer_queue_depth=2,
        ))
        self.assertTrue(fields["overlap_kv_transfer"])
        self.assertEqual(fields["kv_transfer_queue_depth"], 2)

    def test_warmup_session_allows_model_loading(self):
        fields = codec.unpack_session_start(codec.pack_session_start(
            4,
            12,
            2,
            allow_model_load=True,
        ))
        self.assertTrue(fields["allow_model_load"])

    def test_prefill_chunk_round_trip(self):
        hidden = torch.randn(1, 7, 16)
        positions = torch.arange(64, 71).reshape(1, -1)
        fields = codec.unpack_prefill_chunk(
            codec.pack_prefill_chunk(hidden, positions, "fp16", 64, True)
        )
        self.assertEqual(fields["token_offset"], 64)
        self.assertEqual(fields["seq_len"], 7)
        self.assertTrue(fields["final_chunk"])
        decoded = codec.decode_activation(
            fields["payload"], fields["scales"], fields["dtype"],
            fields["seq_len"], fields["hidden_dim"], "cpu",
        )
        torch.testing.assert_close(decoded, hidden, atol=1e-3, rtol=1e-3)

    def test_kv_round_trip(self):
        key = torch.randn(1, 2, 7, 8, dtype=torch.float16)
        value = torch.randn_like(key)
        fields = codec.unpack_kv_delta(
            codec.pack_kv_delta(6, 64, key, value)
        )
        self.assertEqual(fields[0:2], (6, 64))
        torch.testing.assert_close(fields[2], key)
        torch.testing.assert_close(fields[3], value)

    def test_completion_round_trips(self):
        self.assertEqual(
            codec.unpack_chunk_complete(
                codec.pack_chunk_complete(64, 3, {"kv_bytes": 12})
            ),
            (64, 3, {"kv_bytes": 12}),
        )
        self.assertEqual(
            codec.unpack_prefill_complete(
                codec.pack_prefill_complete(42, {"done": 1})
            ),
            (42, {"done": 1}),
        )


class BenchmarkBreakdownTests(unittest.TestCase):
    def test_breakdown_accounts_for_ttft_and_decode_round(self):
        run = BenchmarkRun("prompt", SimpleNamespace(), 0.0)
        run.setup = {
            "tokenization_ms": 2.0,
            "session_handshake_ms": 3.0,
        }
        run.requests = [
            {
                "type": "prefill_chunk",
                "timings_ms": {
                    "round_total": 100.0,
                    "edge_forward": 30.0,
                    "activation_encode": 5.0,
                    "websocket_send": 2.0,
                },
                "cloud": {
                    "cloud_prefill_ms": 15.0,
                    "chunk_total_ms": 20.0,
                },
            },
            {
                "type": "decode",
                "timings_ms": {
                    "round_total": 50.0,
                    "edge_forward": 10.0,
                    "activation_encode": 1.0,
                    "websocket_send": 1.0,
                },
                "cloud": {
                    "cloud_forward_ms": 12.0,
                    "server_processing_ms": 20.0,
                },
            },
        ]

        ttft, decode = run._latency_breakdowns(110.0)

        self.assertEqual(ttft["network_tunnel_and_queue"], 43.0)
        self.assertEqual(ttft["cloud_other"], 5.0)
        self.assertEqual(ttft["client_other"], 5.0)
        self.assertEqual(decode["network_tunnel_and_queue"], 18.0)
        self.assertEqual(decode["cloud_other"], 8.0)
        self.assertEqual(decode["round_total"], 50.0)


class CacheTests(unittest.TestCase):
    def test_extract_and_install_partial_chunks(self):
        source = FakeDynamicCache()
        destination = FakeDynamicCache()
        first_key = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        first_value = torch.randn_like(first_key)
        source.update(first_key, first_value, 0)
        key, value = extract_kv_delta(source, 0, 0, 4)
        install_kv_delta(destination, 0, 0, key, value)

        second_key = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        second_value = torch.randn_like(second_key)
        source.update(second_key, second_value, 0)
        key, value = extract_kv_delta(source, 0, 4, 3)
        install_kv_delta(destination, 0, 4, key, value)
        assert_cache_lengths(destination, [0], 7)
        torch.testing.assert_close(destination.key_cache[0], source.key_cache[0])

    def test_rejects_duplicate_or_gap(self):
        cache = FakeDynamicCache()
        key = torch.zeros(1, 2, 2, 8, dtype=torch.float16)
        install_kv_delta(cache, 0, 0, key, key)
        with self.assertRaisesRegex(ValueError, "offset mismatch"):
            install_kv_delta(cache, 0, 0, key, key)
        with self.assertRaisesRegex(ValueError, "offset mismatch"):
            install_kv_delta(cache, 0, 3, key, key)

    def test_rejects_bad_dtype_and_payload_size(self):
        cache = FakeDynamicCache()
        key = torch.zeros(1, 2, 2, 8)
        with self.assertRaisesRegex(ValueError, "fp16"):
            install_kv_delta(cache, 0, 0, key, key)
        valid = codec.pack_kv_delta(
            0, 0, key.half(), key.half()
        )
        with self.assertRaisesRegex(ValueError, "payload length"):
            codec.unpack_kv_delta(valid[:-1])


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_in_order_while_producer_continues(self):
        sender_started = asyncio.Event()
        release_sender = asyncio.Event()
        producer_continued = threading.Event()
        sent = []

        def produce(emit):
            emit(b"layer-4")
            producer_continued.set()
            emit(b"layer-5")
            emit(b"layer-6")
            return "logits"

        async def send(frame):
            sent.append(frame)
            if len(sent) == 1:
                sender_started.set()
                await release_sender.wait()

        task = asyncio.create_task(stream_from_worker(
            produce, send, max_queue_size=1
        ))
        await asyncio.wait_for(sender_started.wait(), timeout=1)
        continued = await asyncio.wait_for(
            asyncio.to_thread(producer_continued.wait, 1),
            timeout=2,
        )
        self.assertTrue(continued)
        self.assertFalse(task.done())

        release_sender.set()
        result, metrics = await asyncio.wait_for(task, timeout=2)
        self.assertEqual(result, "logits")
        self.assertEqual(
            sent,
            [b"layer-4", b"layer-5", b"layer-6"],
        )
        self.assertTrue(metrics["kv_compute_send_overlap"])
        self.assertEqual(metrics["kv_frames_sent"], 3)
        self.assertEqual(metrics["kv_max_queue_depth"], 1)

    async def test_producer_failure_propagates_after_queued_frames(self):
        sent = []

        def produce(emit):
            emit(b"layer-4")
            raise ValueError("forward failed")

        async def send(frame):
            sent.append(frame)

        with self.assertRaisesRegex(ValueError, "forward failed"):
            await asyncio.wait_for(
                stream_from_worker(produce, send),
                timeout=2,
            )
        self.assertEqual(sent, [b"layer-4"])

    async def test_sender_failure_cancels_blocked_producer(self):
        def produce(emit):
            for layer in range(100):
                emit(bytes([layer]))

        async def send(_frame):
            raise ConnectionError("transport failed")

        with self.assertRaisesRegex(ConnectionError, "transport failed"):
            await asyncio.wait_for(
                stream_from_worker(produce, send, max_queue_size=1),
                timeout=2,
            )


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Sequential(
            nn.Linear(16, 16, bias=False),
            nn.Linear(16, 16, bias=False),
        )
        self.mlp = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.Linear(32, 16, bias=False),
        )


class _FakeModel(nn.Module):
    def __init__(self, layer_count=3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            _FakeLayer() for _ in range(layer_count)
        ])


def _quant_config(**overrides):
    fields = {
        "prefill_edge_layers": 1,
        "decode_edge_layers": 3,
        "edge_overlap_quantization": "mixed",
        "edge_overlap_attention_bits": 8,
        "edge_overlap_ffn_bits": 4,
        "edge_int4_group_size": 64,
        "edge_quantization_backend": "pytorch",
        "allow_quantization_fallback": True,
        "device": "cpu",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class QuantizationTests(unittest.TestCase):
    def test_none_leaves_every_layer_unmodified(self):
        model = _FakeModel()
        first_types = [
            type(layer.self_attn[0]) for layer in model.model.layers
        ]
        result = quantize_overlap_layers(
            model,
            _quant_config(edge_overlap_quantization="none"),
        )
        self.assertEqual(result["effective_attention_bits"], 16)
        self.assertEqual(result["effective_ffn_bits"], 16)
        self.assertEqual(
            [type(layer.self_attn[0]) for layer in model.model.layers],
            first_types,
        )

    def test_pytorch_quantizes_only_overlap_layers(self):
        model = _FakeModel()
        result = quantize_overlap_layers(model, _quant_config())

        self.assertIsInstance(model.model.layers[0].self_attn[0], nn.Linear)
        for layer in model.model.layers[1:3]:
            self.assertIsInstance(
                layer.self_attn[0],
                torch.ao.nn.quantized.dynamic.Linear,
            )
            self.assertIsInstance(
                layer.mlp[0],
                torch.ao.nn.quantized.dynamic.Linear,
            )
            output = layer.self_attn(torch.randn(1, 1, 16))
            self.assertEqual(output.shape, (1, 1, 16))

        self.assertEqual(result["effective_attention_bits"], 8)
        self.assertEqual(result["effective_ffn_bits"], 8)
        self.assertEqual(result["overlap_start_layer"], 1)
        self.assertEqual(result["overlap_end_layer"], 3)
        self.assertIn("INT4 backend unavailable", result["fallback_reason"])
        self.assertEqual(
            model._edge_quantization_activation_dtype,
            torch.float32,
        )
        self.assertLess(
            result["estimated_weight_bytes_after"],
            result["estimated_weight_bytes_before"],
        )

    def test_strict_int4_rejects_fallback(self):
        model = _FakeModel()
        with self.assertRaisesRegex(RuntimeError, "INT4 backend unavailable"):
            quantize_overlap_layers(
                model,
                _quant_config(allow_quantization_fallback=False),
            )

    def test_quantized_qwen_overlap_forward_and_cache_growth(self):
        from spec_decoding import run_edge_layers

        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )).half().eval()
        quantize_overlap_layers(model, _quant_config())
        cache = DynamicCache()

        hidden, _ = run_edge_layers(
            model,
            torch.tensor([[1, 2, 3]]),
            cache,
            edge_layers=3,
            past_len=0,
        )
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertEqual([item.shape[-2] for item in cache.key_cache], [3] * 3)
        self.assertEqual(
            [item.dtype for item in cache.key_cache],
            [torch.float16, torch.float32, torch.float32],
        )

        hidden, _ = run_edge_layers(
            model,
            torch.tensor([[4]]),
            cache,
            edge_layers=3,
            past_len=3,
        )
        self.assertEqual(hidden.shape, (1, 1, 16))
        self.assertEqual([item.shape[-2] for item in cache.key_cache], [4] * 3)

    def test_quantized_decode_accepts_migrated_fp16_prompt_cache(self):
        from spec_decoding import run_edge_layers

        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )).half().eval()
        quantize_overlap_layers(model, _quant_config())
        cache = DynamicCache()

        run_edge_layers(
            model,
            torch.tensor([[1, 2, 3]]),
            cache,
            edge_layers=1,
            past_len=0,
        )
        migrated = torch.randn(1, 2, 3, 4, dtype=torch.float16)
        install_kv_delta(cache, 1, 0, migrated, migrated.clone())
        install_kv_delta(cache, 2, 0, migrated, migrated.clone())

        hidden, _ = run_edge_layers(
            model,
            torch.tensor([[4]]),
            cache,
            edge_layers=3,
            past_len=3,
        )
        self.assertEqual(hidden.shape, (1, 1, 16))
        self.assertEqual([item.shape[-2] for item in cache.key_cache], [4] * 3)
        self.assertEqual(cache.key_cache[0].dtype, torch.float16)
        self.assertEqual(cache.key_cache[1].dtype, torch.float32)
        self.assertEqual(cache.key_cache[2].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
