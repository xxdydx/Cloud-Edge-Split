import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

import activation_codec as codec
from config import CONFIG
from model_loading import load_partial_model
from spec_decoding import draft_and_prepare_verification, run_edge_layers


class EdgeClient:
    """Resident edge inference client that serves multiple prompts."""

    def __init__(self, config=CONFIG):
        self.config = config
        self.ws = None

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
        return unpack(data)

    def _start_session(self):
        """Reset cloud KV state, reconnecting once if the socket went stale."""
        for attempt in range(2):
            try:
                if self.ws is None:
                    self.connect()
                self.ws.send(codec.pack_session_start(self.config.edge_layers))
                self._read_reply(codec.unpack_ok)
                return
            except (ConnectionClosed, OSError, TimeoutError):
                self.close_connection()
                if attempt == 1:
                    raise

    def _generate_standard(self, input_ids, max_new_tokens):
        edge_cache = DynamicCache()
        cur_ids = input_ids
        past_len = 0
        generated = []
        round_times = []

        for step in range(max_new_tokens):
            started = time.perf_counter()
            step_input = cur_ids if step == 0 else cur_ids[:, -1:]
            hidden, position_ids = run_edge_layers(
                self.model,
                step_input,
                edge_cache,
                self.config.edge_layers,
                past_len,
            )
            self.ws.send(codec.pack_decode(
                hidden,
                position_ids,
                self.config.activation_dtype,
            ))
            next_token = self._read_reply(codec.unpack_decode_reply)
            generated.append(next_token)
            cur_ids = torch.cat([
                cur_ids,
                torch.tensor([[next_token]], device=cur_ids.device),
            ], dim=1)
            past_len += step_input.shape[1]
            round_times.append(time.perf_counter() - started)
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
            remaining = max_new_tokens - len(generated)
            draft_count = min(self.config.num_draft_tokens, remaining)
            draft_ids, hidden, position_ids, edge_timings = (
                draft_and_prepare_verification(
                    self.model,
                    cur_ids,
                    draft_count,
                    self.config.edge_layers,
                )
            )
            draft_seconds += edge_timings["draft_seconds"]
            preparation_seconds += edge_timings["preparation_seconds"]

            request_started = time.perf_counter()
            self.ws.send(codec.pack_verify(
                hidden,
                position_ids,
                self.config.activation_dtype,
                cur_ids.shape[1],
                draft_ids[0].tolist(),
                self.config.edge_layers,
            ))
            accepted_count, bonus_token = self._read_reply(
                codec.unpack_verify_reply
            )
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
            step_input = cur_ids if step == 0 else cur_ids[:, -1:]
            outputs = self.model(
                step_input,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(-1).item()
            generated.append(next_token)
            cur_ids = torch.cat([
                cur_ids,
                torch.tensor([[next_token]], device=cur_ids.device),
            ], dim=1)
            round_times.append(time.perf_counter() - started)
            if next_token == self.tokenizer.eos_token_id:
                break

        return generated, round_times

    def generate(self, prompt, max_new_tokens=None, show_metrics=True):
        """Generate one response while keeping model weights resident."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.config.device
        )
        started = time.perf_counter()

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
            raise RuntimeError(
                "Cloud connection was lost during generation; retry the prompt"
            ) from None

        total = time.perf_counter() - started
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
