import asyncio
import threading
import unittest

import torch

import activation_codec as codec
from cache_migration import (
    assert_cache_lengths,
    extract_kv_delta,
    install_kv_delta,
)
from kv_streaming import stream_from_worker


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


if __name__ == "__main__":
    unittest.main()
