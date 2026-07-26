import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

import activation_codec as codec
from benchmarking import (
    BenchmarkRun,
    PowermetricsSampler,
    TelemetrySampler,
    append_jsonl,
    elapsed_ms,
    encoded_activation_bytes,
    now_ns,
)
from config import CONFIG
from model_loading import load_partial_model
from spec_decoding import draft_and_prepare_verification, run_edge_layers


class EdgeClient:
    """Resident edge inference client that serves multiple prompts."""

    def __init__(self, config=CONFIG):
        self.config = config
        self.ws = None
        self._benchmark = None
        self._telemetry = None
        self._power_sampler = None

        load_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if config.split_inference:
            # Speculative drafting needs the LM head; standard split inference
            # only needs embeddings and the edge layer range.
            self.model = load_partial_model(
                config.model_name,
                config.torch_dtype,
                config.device,
                0,
                config.edge_layers,
                need_embed=True,
                need_lm_head=config.speculative_decoding,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=config.torch_dtype,
            ).to(config.device)
            self.model.eval()
        self.model_load_seconds = time.perf_counter() - load_started

    def connect(self):
        """Open the persistent edge-to-cloud WebSocket if required."""
        if not self.config.split_inference:
            return
        self.close_connection()
        self.ws = connect(
            self.config.cloud_ws_url,
            open_timeout=self.config.request_timeout_seconds,
            close_timeout=5,
        )

    def close_connection(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def close(self):
        self.close_connection()

    def _read_reply(self, unpack):
        data = self.ws.recv(timeout=self.config.request_timeout_seconds)
        if codec.message_type(data) == codec.MSG_ERROR:
            raise RuntimeError(data[1:].decode())
        return unpack(data), len(data)

    def _start_session(self):
        """Reset cloud KV state, reconnecting once if the socket went stale."""
        session_started = now_ns()
        reused_connection = self.ws is not None
        for attempt in range(2):
            try:
                if self.ws is None:
                    self.connect()
                self.ws.send(codec.pack_session_start(
                    self.config.edge_layers,
                    benchmark_enabled=self._benchmark is not None,
                ))
                session_metrics, _ = self._read_reply(codec.unpack_ok)
                if self._benchmark is not None:
                    self._benchmark.session_metadata = session_metrics
                    self._benchmark.setup["session_handshake_ms"] = elapsed_ms(
                        session_started
                    )
                    self._benchmark.setup["websocket_reused"] = reused_connection
                    self._benchmark.setup["session_attempts"] = attempt + 1
                return
            except (ConnectionClosed, OSError, TimeoutError):
                self.close_connection()
                if attempt == 1:
                    raise

    def _record_request(
        self,
        request_type,
        hidden,
        frame,
        reply_bytes,
        output_tokens,
        timings,
        cloud_metrics,
        request_started,
        request_finished,
    ):
        if self._benchmark is None:
            return
        raw_bytes = hidden.numel() * hidden.element_size()
        encoded_bytes = encoded_activation_bytes(
            hidden,
            self.config.activation_dtype,
        )
        scales_bytes = hidden.shape[1] * 4 if self.config.activation_dtype == "int4" else 0
        self._benchmark.add_request({
            "index": len(self._benchmark.requests),
            "type": request_type,
            "input_sequence_length": hidden.shape[1],
            "output_tokens": output_tokens,
            "timings_ms": timings,
            "bytes": {
                "raw_activation": raw_bytes,
                "encoded_activation": encoded_bytes,
                "quantization_scales": scales_bytes,
                "metadata": len(frame) - encoded_bytes - scales_bytes,
                "frame_out": len(frame),
                "frame_in": reply_bytes,
            },
            "edge_telemetry": (
                self._telemetry.summarize(request_started, request_finished)
                if self._telemetry else {}
            ),
            "cloud": cloud_metrics,
        })

    def _synchronize_for_timing(self):
        if self._benchmark is None:
            return
        if self.config.device == "mps":
            torch.mps.synchronize()
        elif self.config.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _generate_standard(self, input_ids, max_new_tokens):
        edge_cache = DynamicCache()
        cur_ids = input_ids
        past_len = 0
        generated = []
        round_times = []

        for step in range(max_new_tokens):
            started = time.perf_counter()
            request_started = now_ns()
            if self._telemetry:
                self._telemetry.capture()
            step_input = cur_ids if step == 0 else cur_ids[:, -1:]
            forward_started = now_ns()
            hidden, position_ids = run_edge_layers(
                self.model,
                step_input,
                edge_cache,
                self.config.edge_layers,
                past_len,
            )
            self._synchronize_for_timing()
            forward_finished = now_ns()
            encode_started = now_ns()
            frame = codec.pack_decode(
                hidden,
                position_ids,
                self.config.activation_dtype,
            )
            encode_finished = now_ns()
            send_started = now_ns()
            self.ws.send(frame)
            send_finished = now_ns()
            receive_started = now_ns()
            (next_token, cloud_metrics), reply_bytes = self._read_reply(
                codec.unpack_decode_reply
            )
            receive_finished = now_ns()
            generated.append(next_token)
            cur_ids = torch.cat([
                cur_ids,
                torch.tensor([[next_token]], device=cur_ids.device),
            ], dim=1)
            past_len += step_input.shape[1]
            round_times.append(time.perf_counter() - started)
            if self._benchmark:
                self._benchmark.mark_tokens(1)
            if self._telemetry:
                self._telemetry.capture()
            self._record_request(
                "prefill" if step == 0 else "decode",
                hidden,
                frame,
                reply_bytes,
                1,
                {
                    "edge_forward": elapsed_ms(forward_started, forward_finished),
                    "activation_encode": elapsed_ms(encode_started, encode_finished),
                    "websocket_send": elapsed_ms(send_started, send_finished),
                    "response_wait": elapsed_ms(receive_started, receive_finished),
                    "round_total": elapsed_ms(request_started, receive_finished),
                },
                cloud_metrics,
                request_started,
                receive_finished,
            )
            if next_token == self.tokenizer.eos_token_id:
                break

        return generated, round_times

    def _generate_speculative(self, input_ids, max_new_tokens):
        cur_ids = input_ids
        generated = []
        round_times = []
        proposed_total = 0
        accepted_total = 0
        draft_seconds = 0.0
        preparation_seconds = 0.0
        ws_seconds = 0.0
        acceptance_by_round = []

        while len(generated) < max_new_tokens:
            started = time.perf_counter()
            request_started_ns = now_ns()
            if self._telemetry:
                self._telemetry.capture()
            remaining = max_new_tokens - len(generated)
            draft_count = min(self.config.num_draft_tokens, remaining)
            forward_started = now_ns()
            draft_ids, hidden, position_ids, edge_timings = (
                draft_and_prepare_verification(
                    self.model,
                    cur_ids,
                    draft_count,
                    self.config.edge_layers,
                )
            )
            self._synchronize_for_timing()
            forward_finished = now_ns()
            draft_seconds += edge_timings["draft_seconds"]
            preparation_seconds += edge_timings["preparation_seconds"]

            encode_started = now_ns()
            frame = codec.pack_verify(
                hidden,
                position_ids,
                self.config.activation_dtype,
                cur_ids.shape[1],
                draft_ids[0].tolist(),
                self.config.edge_layers,
            )
            encode_finished = now_ns()
            request_started = time.perf_counter()
            send_started = now_ns()
            self.ws.send(frame)
            send_finished = now_ns()
            receive_started = now_ns()
            (
                accepted_count,
                bonus_token,
                cloud_metrics,
            ), reply_bytes = self._read_reply(
                codec.unpack_verify_reply
            )
            receive_finished = now_ns()
            ws_seconds += time.perf_counter() - request_started

            proposed_total += draft_count
            accepted_total += accepted_count
            acceptance_by_round.append(f"{accepted_count}/{draft_count}")
            accepted = draft_ids[0, :accepted_count].tolist()
            new_tokens = (accepted + [bonus_token])[:remaining]
            generated.extend(new_tokens)
            cur_ids = torch.cat([
                cur_ids,
                torch.tensor([new_tokens], device=cur_ids.device),
            ], dim=1)
            round_times.append(time.perf_counter() - started)
            if self._benchmark:
                self._benchmark.mark_tokens(len(new_tokens))
            if self._telemetry:
                self._telemetry.capture()
            self._record_request(
                "verify",
                hidden,
                frame,
                reply_bytes,
                len(new_tokens),
                {
                    "edge_draft_and_prepare": elapsed_ms(
                        forward_started,
                        forward_finished,
                    ),
                    "activation_encode": elapsed_ms(encode_started, encode_finished),
                    "websocket_send": elapsed_ms(send_started, send_finished),
                    "response_wait": elapsed_ms(receive_started, receive_finished),
                    "round_total": elapsed_ms(
                        request_started_ns,
                        receive_finished,
                    ),
                },
                cloud_metrics,
                request_started_ns,
                receive_finished,
            )

            if self.tokenizer.eos_token_id in new_tokens:
                eos_index = generated.index(self.tokenizer.eos_token_id)
                generated = generated[:eos_index + 1]
                break

        metrics = {
            "proposed": proposed_total,
            "accepted": accepted_total,
            "draft_seconds": draft_seconds,
            "preparation_seconds": preparation_seconds,
            "ws_seconds": ws_seconds,
            "acceptance_by_round": acceptance_by_round,
        }
        return generated, round_times, metrics

    def _generate_local(self, input_ids, max_new_tokens):
        cache = DynamicCache()
        cur_ids = input_ids
        generated = []
        round_times = []

        for step in range(max_new_tokens):
            started = time.perf_counter()
            request_started = now_ns()
            if self._telemetry:
                self._telemetry.capture()
            step_input = cur_ids if step == 0 else cur_ids[:, -1:]
            outputs = self.model(
                step_input,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(-1).item()
            request_finished = now_ns()
            generated.append(next_token)
            cur_ids = torch.cat([
                cur_ids,
                torch.tensor([[next_token]], device=cur_ids.device),
            ], dim=1)
            round_times.append(time.perf_counter() - started)
            if self._benchmark:
                self._benchmark.mark_tokens(1)
                if self._telemetry:
                    self._telemetry.capture()
                self._benchmark.add_request({
                    "index": len(self._benchmark.requests),
                    "type": "local_prefill" if step == 0 else "local_decode",
                    "input_sequence_length": step_input.shape[1],
                    "output_tokens": 1,
                    "timings_ms": {
                        "local_forward": elapsed_ms(request_started, request_finished),
                        "round_total": elapsed_ms(request_started, request_finished),
                    },
                    "bytes": {
                        "raw_activation": 0,
                        "encoded_activation": 0,
                        "quantization_scales": 0,
                        "metadata": 0,
                        "frame_out": 0,
                        "frame_in": 0,
                    },
                    "edge_telemetry": (
                        self._telemetry.summarize(request_started, request_finished)
                        if self._telemetry else {}
                    ),
                    "cloud": {},
                })
            if next_token == self.tokenizer.eos_token_id:
                break

        return generated, round_times

    def _finish_benchmark(self, generated_count, status="completed", error=None):
        if self._benchmark is None:
            return None
        generation_end_ns = now_ns()
        if self._telemetry:
            self._telemetry.stop()
            edge_summary = self._telemetry.summarize()
        else:
            edge_summary = {}
        if self._power_sampler:
            self._power_sampler.stop()
            edge_summary.update(self._power_sampler.summarize())
        else:
            edge_summary.update({
                "edge_energy_j": None,
                "edge_energy_source": "unsupported",
            })
        record = self._benchmark.finish(
            generated_count,
            edge_summary,
            status=status,
            error=error,
            end_ns=generation_end_ns,
        )
        append_jsonl(self.config.benchmark_output, record)
        self._benchmark = None
        self._telemetry = None
        self._power_sampler = None
        return record

    def generate(
        self,
        prompt,
        max_new_tokens=None,
        show_metrics=True,
        benchmark=None,
    ):
        """Generate one response while keeping model weights resident."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        benchmark_enabled = (
            self.config.benchmark_enabled if benchmark is None else benchmark
        )
        if benchmark_enabled:
            self._benchmark = BenchmarkRun(
                prompt,
                self.config,
                self.model_load_seconds,
            )
            self._telemetry = TelemetrySampler(
                interval_ms=self.config.telemetry_interval_ms,
            )
            self._telemetry.start()
            if self.config.edge_power_sampler == "powermetrics":
                self._power_sampler = PowermetricsSampler(
                    interval_ms=self.config.telemetry_interval_ms,
                )
                self._power_sampler.start()

        tokenization_started = now_ns()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.config.device
        )
        tokenization_finished = now_ns()
        if self._benchmark:
            self._benchmark.prompt_tokens = input_ids.shape[1]
            self._benchmark.setup["tokenization_ms"] = elapsed_ms(
                tokenization_started,
                tokenization_finished,
            )
        started = time.perf_counter()
        generated = []

        try:
            with torch.no_grad():
                if not self.config.split_inference:
                    generated, round_times = self._generate_local(
                        input_ids,
                        max_new_tokens,
                    )
                    mode = "local"
                    metrics = None
                else:
                    # Every prompt gets fresh edge/cloud KV caches while model
                    # weights and the WebSocket remain resident.
                    self._start_session()
                    if self.config.speculative_decoding:
                        generated, round_times, metrics = (
                            self._generate_speculative(input_ids, max_new_tokens)
                        )
                        mode = "speculative"
                    else:
                        generated, round_times = self._generate_standard(
                            input_ids,
                            max_new_tokens,
                        )
                        mode = "standard"
                        metrics = None
        except (ConnectionClosed, OSError, TimeoutError):
            # A mid-generation reconnect would leave the two KV caches out of
            # sync. Abort this prompt and reconnect safely on the next one.
            self.close_connection()
            message = (
                "Cloud connection was lost during generation; retry the prompt"
            )
            self._finish_benchmark(
                len(generated),
                status="failed",
                error=message,
            )
            raise RuntimeError(message) from None
        except Exception as error:
            self._finish_benchmark(
                len(generated),
                status="failed",
                error=str(error),
            )
            raise

        total = time.perf_counter() - started
        benchmark_record = self._finish_benchmark(len(generated))
        if show_metrics:
            formatted_times = [f"{value:.3f}" for value in round_times]
            dtype_note = (
                f", activation_dtype: {self.config.activation_dtype}"
                if self.config.split_inference else ""
            )
            print(
                f"mode: {mode}{dtype_note}, total: {total:.3f}s, "
                f"rounds: {formatted_times}"
            )
            if metrics is not None:
                proposed = metrics["proposed"]
                accepted = metrics["accepted"]
                acceptance_rate = accepted / proposed if proposed else 0.0
                tokens_per_request = (
                    len(generated) / len(round_times) if round_times else 0.0
                )
                print(
                    "speculative metrics: "
                    f"requests={len(round_times)}, proposed={proposed}, "
                    f"accepted={accepted}, acceptance={acceptance_rate:.1%}, "
                    f"tokens/request={tokens_per_request:.2f}"
                )
                print(f"accepted by round: {metrics['acceptance_by_round']}")
                print(
                    "timing breakdown: "
                    f"draft={metrics['draft_seconds']:.3f}s, "
                    f"verification_prep={metrics['preparation_seconds']:.3f}s, "
                    f"ws_and_serialization={metrics['ws_seconds']:.3f}s"
                )
            if benchmark_record is not None:
                latency = benchmark_record["latency"]
                transport = benchmark_record["transport"]
                itl_mean = (
                    latency["inter_token_summary_ms"] or {}
                ).get("mean", 0)
                print(
                    "benchmark: "
                    f"TTFT={latency['ttft_ms']:.2f}ms, "
                    f"ITL mean={itl_mean:.2f}ms, "
                    f"throughput={latency['output_tokens_per_second']:.2f} tok/s, "
                    f"bytes={transport['total_application_bytes']}, "
                    f"saved={self.config.benchmark_output}"
                )

        return self.tokenizer.decode(input_ids[0].tolist() + generated)

    def warmup(self):
        """Warm edge/cloud kernels once, discarding the generated token."""
        input_ids = self.tokenizer(
            "Hello",
            return_tensors="pt",
        ).input_ids.to(self.config.device)
        with torch.no_grad():
            if self.config.split_inference:
                self._start_session()
                self._generate_standard(input_ids, max_new_tokens=1)
            else:
                self._generate_local(input_ids, max_new_tokens=1)


_default_client = None


def generate(prompt, max_new_tokens=None):
    """Backwards-compatible API with a process-resident default client."""
    global _default_client
    if _default_client is None:
        _default_client = EdgeClient()
        if CONFIG.warmup_on_start:
            _default_client.warmup()
    return _default_client.generate(prompt, max_new_tokens)


def main():
    print("Loading edge model...")
    client = EdgeClient()
    print(f"Model ready in {client.model_load_seconds:.2f}s on {CONFIG.device}.")

    try:
        if CONFIG.split_inference:
            print("Connecting to cloud...")
            client.connect()
        if CONFIG.warmup_on_start:
            print("Warming up edge and cloud inference...")
            client.warmup()
        print("Ready. Type /quit or /exit to stop.\n")

        while True:
            try:
                prompt = input("Prompt> ").strip()
            except EOFError:
                print()
                break

            if not prompt:
                continue
            if prompt.lower() in {"/quit", "/exit"}:
                break

            try:
                result = client.generate(prompt)
                print(f"Result: {result}\n")
            except Exception as error:
                print(f"Generation failed: {error}\n")
    except KeyboardInterrupt:
        print()
    finally:
        print("Shutting down...")
        client.close()


if __name__ == "__main__":
    main()
