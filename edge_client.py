import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from websockets.sync.client import connect

import activation_codec as codec
from config import CONFIG
from model_loading import load_partial_model
from spec_decoding import draft_and_prepare_verification, run_edge_layers


tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
if CONFIG.split_inference:
    # Only the edge's own layer range + embed_tokens; no lm_head is needed
    model = load_partial_model(
        CONFIG.model_name, CONFIG.torch_dtype, CONFIG.device,
        0, CONFIG.edge_layers, need_embed=True, need_lm_head=False,
    )
else:
    # split_inference=False needs the complete model for a regular forward pass.
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_name, torch_dtype=CONFIG.torch_dtype,
    ).to(CONFIG.device)
    model.eval()


def _read_reply(ws, unpack):
    data = ws.recv()
    if codec.message_type(data) == codec.MSG_ERROR:
        raise RuntimeError(data[1:].decode())
    return unpack(data)


def _generate_standard(ws, input_ids, max_new_tokens):
    ws.send(codec.pack_session_start(CONFIG.edge_layers))
    _read_reply(ws, codec.unpack_ok)
    edge_cache = DynamicCache()
    cur_ids = input_ids
    past_len = 0
    generated = []
    round_times = []

    for step in range(max_new_tokens):
        started = time.perf_counter()
        step_input = cur_ids if step == 0 else cur_ids[:, -1:]
        hidden, position_ids = run_edge_layers(
            model, step_input, edge_cache, CONFIG.edge_layers, past_len
        )
        ws.send(codec.pack_decode(hidden, position_ids, CONFIG.activation_dtype))
        next_token = _read_reply(ws, codec.unpack_decode_reply)
        generated.append(next_token)
        cur_ids = torch.cat(
            [cur_ids, torch.tensor([[next_token]], device=cur_ids.device)], dim=1
        )
        past_len += step_input.shape[1]
        round_times.append(time.perf_counter() - started)
        if next_token == tokenizer.eos_token_id:
            break

    return generated, round_times


def _generate_speculative(ws, input_ids, max_new_tokens):
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
        draft_count = min(CONFIG.num_draft_tokens, remaining)
        draft_ids, hidden, position_ids, edge_timings = draft_and_prepare_verification(
            model, cur_ids, draft_count, CONFIG.edge_layers
        )
        draft_seconds += edge_timings["draft_seconds"]
        preparation_seconds += edge_timings["preparation_seconds"]

        request_started = time.perf_counter()
        ws.send(codec.pack_verify(
            hidden,
            position_ids,
            CONFIG.activation_dtype,
            cur_ids.shape[1],
            draft_ids[0].tolist(),
            CONFIG.edge_layers,
        ))
        accepted_count, bonus_token = _read_reply(ws, codec.unpack_verify_reply)
        ws_seconds += time.perf_counter() - request_started

        proposed_total += draft_count
        accepted_total += accepted_count
        acceptance_by_round.append(f"{accepted_count}/{draft_count}")
        accepted = draft_ids[0, :accepted_count].tolist()
        new_tokens = accepted + [bonus_token]
        new_tokens = new_tokens[:remaining]
        generated.extend(new_tokens)
        new_tensor = torch.tensor([new_tokens], device=cur_ids.device)
        cur_ids = torch.cat([cur_ids, new_tensor], dim=1)
        round_times.append(time.perf_counter() - started)

        if tokenizer.eos_token_id in new_tokens:
            eos_index = generated.index(tokenizer.eos_token_id)
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


def _generate_local(input_ids, max_new_tokens):
    """Baseline: the regular, unsplit pytorch forward pass, no network calls."""
    cache = DynamicCache()
    cur_ids = input_ids
    generated = []
    round_times = []

    for step in range(max_new_tokens):
        started = time.perf_counter()
        step_input = cur_ids if step == 0 else cur_ids[:, -1:]
        outputs = model(step_input, past_key_values=cache, use_cache=True)
        next_token = outputs.logits[:, -1, :].argmax(-1).item()
        generated.append(next_token)
        cur_ids = torch.cat(
            [cur_ids, torch.tensor([[next_token]], device=cur_ids.device)], dim=1
        )
        round_times.append(time.perf_counter() - started)
        if next_token == tokenizer.eos_token_id:
            break

    return generated, round_times


def generate(prompt, max_new_tokens=None):
    max_new_tokens = max_new_tokens or CONFIG.max_new_tokens
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(CONFIG.device)
    started = time.perf_counter()

    if not CONFIG.split_inference:
        with torch.no_grad():
            generated, round_times = _generate_local(input_ids, max_new_tokens)
        mode = "local"
    else:
        with connect(CONFIG.cloud_ws_url) as ws:
            with torch.no_grad():
                if CONFIG.speculative_decoding:
                    generated, round_times, metrics = _generate_speculative(
                        ws, input_ids, max_new_tokens
                    )
                    mode = "speculative"
                else:
                    generated, round_times = _generate_standard(ws, input_ids, max_new_tokens)
                    mode = "standard"

    total = time.perf_counter() - started
    formatted_times = [f"{value:.3f}" for value in round_times]
    dtype_note = f", activation_dtype: {CONFIG.activation_dtype}" if CONFIG.split_inference else ""
    print(f"mode: {mode}{dtype_note}, total: {total:.3f}s, rounds: {formatted_times}")
    if CONFIG.split_inference and CONFIG.speculative_decoding:
        proposed = metrics["proposed"]
        accepted = metrics["accepted"]
        acceptance_rate = accepted / proposed if proposed else 0.0
        tokens_per_request = len(generated) / len(round_times) if round_times else 0.0
        print(
            "speculative metrics: "
            f"requests={len(round_times)}, proposed={proposed}, accepted={accepted}, "
            f"acceptance={acceptance_rate:.1%}, tokens/request={tokens_per_request:.2f}"
        )
        print(f"accepted by round: {metrics['acceptance_by_round']}")
        print(
            "timing breakdown: "
            f"draft={metrics['draft_seconds']:.3f}s, "
            f"verification_prep={metrics['preparation_seconds']:.3f}s, "
            f"ws_and_serialization={metrics['ws_seconds']:.3f}s"
        )
    return tokenizer.decode(input_ids[0].tolist() + generated)


if __name__ == "__main__":
    print("Result:", generate("The capital of France is"))
