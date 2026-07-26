"""Bounded producer/async-sender bridge for overlapped KV migration."""

import asyncio
import queue
import threading
import time
from dataclasses import dataclass


class StreamCancelled(RuntimeError):
    """Raised inside a producer when its consumer can no longer send."""


@dataclass
class _Completed:
    result: object
    finished_ns: int


@dataclass
class _Failed:
    error: BaseException


def _elapsed_ms(start_ns, end_ns):
    return (end_ns - start_ns) / 1_000_000


async def stream_from_worker(produce, send, max_queue_size=1):
    """Run ``produce(emit)`` in a worker while asynchronously sending frames.

    ``emit`` is a blocking, thread-safe callback. Frames are sent by exactly
    one async consumer in FIFO order. The bounded queue applies backpressure
    when transport is slower than production.
    """
    if max_queue_size < 1:
        raise ValueError("max_queue_size must be at least 1")

    outbox = queue.Queue(maxsize=max_queue_size)
    cancelled = threading.Event()
    stats_lock = threading.Lock()
    producer_stats = {
        "queue_wait_ms": 0.0,
        "max_queue_depth": 0,
    }

    def put_until_available(item):
        wait_started = time.perf_counter_ns()
        while not cancelled.is_set():
            try:
                outbox.put(item, timeout=0.05)
                waited_until = time.perf_counter_ns()
                with stats_lock:
                    producer_stats["queue_wait_ms"] += _elapsed_ms(
                        wait_started, waited_until
                    )
                    producer_stats["max_queue_depth"] = max(
                        producer_stats["max_queue_depth"],
                        outbox.qsize(),
                    )
                return
            except queue.Full:
                continue
        raise StreamCancelled("KV sender stopped before production completed")

    def run_producer():
        try:
            result = produce(put_until_available)
            put_until_available(_Completed(result, time.perf_counter_ns()))
        except BaseException as error:
            if cancelled.is_set():
                return
            try:
                put_until_available(_Failed(error))
            except StreamCancelled:
                pass

    producer = threading.Thread(
        target=run_producer,
        name="kv-prefill-producer",
        daemon=True,
    )
    producer.start()

    send_ms = 0.0
    frames_sent = 0
    bytes_sent = 0
    first_send_started_ns = None
    final_send_finished_ns = None
    completed = None

    try:
        while True:
            item = await asyncio.to_thread(outbox.get)
            if isinstance(item, _Completed):
                completed = item
                break
            if isinstance(item, _Failed):
                raise item.error

            send_started = time.perf_counter_ns()
            if first_send_started_ns is None:
                first_send_started_ns = send_started
            await send(item)
            send_finished = time.perf_counter_ns()
            final_send_finished_ns = send_finished
            send_ms += _elapsed_ms(send_started, send_finished)
            frames_sent += 1
            bytes_sent += len(item)
    except BaseException:
        cancelled.set()
        while True:
            try:
                outbox.get_nowait()
            except queue.Empty:
                break
        raise
    finally:
        if completed is None:
            cancelled.set()
        await asyncio.to_thread(producer.join)

    with stats_lock:
        queue_wait_ms = producer_stats["queue_wait_ms"]
        max_queue_depth = producer_stats["max_queue_depth"]
    return completed.result, {
        "kv_send_ms": send_ms,
        "kv_queue_wait_ms": queue_wait_ms,
        "kv_max_queue_depth": max_queue_depth,
        "kv_frames_sent": frames_sent,
        "kv_bytes_sent": bytes_sent,
        "kv_first_send_started_ns": first_send_started_ns,
        "kv_final_send_finished_ns": final_send_finished_ns,
        "kv_producer_finished_ns": completed.finished_ns,
        "kv_compute_send_overlap": bool(
            first_send_started_ns
            and first_send_started_ns < completed.finished_ns
        ),
    }
