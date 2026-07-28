"""Reusable edge-side helpers for greedy speculative decoding."""

import inspect
import time

import torch
from transformers.cache_utils import DynamicCache

def _layer_hidden(layer_output):
    return layer_output[0] if isinstance(layer_output, tuple) else layer_output


def _cache_argument(layer, cache):
    """Support both old and new Transformers decoder-layer cache APIs."""
    parameters = inspect.signature(layer.forward).parameters
    if "past_key_values" in parameters:
        return {"past_key_values": cache}
    if "past_key_value" in parameters:
        return {"past_key_value": cache}
    raise RuntimeError("Decoder layer does not expose a supported KV-cache argument")


def causal_attention_mask(hidden, position_ids, past_len):
    """Return a 4D additive causal mask for direct decoder-layer calls."""
    sequence_length = hidden.shape[1]
    key_length = past_len + sequence_length
    key_positions = torch.arange(key_length, device=hidden.device).view(1, 1, -1)
    blocked = key_positions > position_ids.unsqueeze(-1)
    mask = torch.zeros(
        (hidden.shape[0], 1, sequence_length, key_length),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return mask.masked_fill(blocked.unsqueeze(1), torch.finfo(hidden.dtype).min)


def run_edge_layers(model, input_ids, cache, edge_layers, past_len=0):
    """Run embeddings and the configured edge portion of the model."""
    hidden = model.model.embed_tokens(input_ids)
    seq_len = hidden.shape[1]
    position_ids = torch.arange(
        past_len,
        past_len + seq_len,
        device=hidden.device,
    ).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    attention_mask = causal_attention_mask(hidden, position_ids, past_len)

    quantization_start = getattr(
        model,
        "_edge_quantization_start_layer",
        None,
    )
    quantization_dtype = getattr(
        model,
        "_edge_quantization_activation_dtype",
        None,
    )
    for layer_idx, layer in enumerate(model.model.layers[:edge_layers]):
        if (
            layer_idx == quantization_start
            and quantization_dtype is not None
            and hidden.dtype != quantization_dtype
        ):
            hidden = hidden.to(quantization_dtype)
            attention_mask = attention_mask.to(quantization_dtype)
        hidden = _layer_hidden(layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            position_embeddings=position_embeddings,
            **_cache_argument(layer, cache),
        ))

    return hidden, position_ids


def draft_early_exit(model, cur_ids, num_draft, edge_layers):
    """Greedily draft tokens using only the edge layers plus the LM head."""
    past_len = 0
    cache = DynamicCache()
    draft_ids = None

    for _ in range(num_draft):
        step_input = cur_ids if draft_ids is None else draft_ids[:, -1:]
        hidden, _ = run_edge_layers(
            model, step_input, cache, edge_layers, past_len
        )
        logits = model.lm_head(model.model.norm(hidden))
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        draft_ids = (
            next_token
            if draft_ids is None
            else torch.cat([draft_ids, next_token], dim=1)
        )
        past_len += step_input.shape[1]

    return draft_ids


def draft_and_prepare_verification(model, cur_ids, num_draft, edge_layers):
    """Draft tokens and retain their edge states for cloud verification."""
    cache = DynamicCache()
    draft_ids = None
    hidden_parts = []
    position_parts = []
    past_len = 0
    draft_started = time.perf_counter()

    for _ in range(num_draft):
        step_input = cur_ids if draft_ids is None else draft_ids[:, -1:]
        hidden, position_ids = run_edge_layers(
            model, step_input, cache, edge_layers, past_len
        )
        hidden_parts.append(hidden)
        position_parts.append(position_ids)
        logits = model.lm_head(model.model.norm(hidden))
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        draft_ids = (
            next_token
            if draft_ids is None
            else torch.cat([draft_ids, next_token], dim=1)
        )
        past_len += step_input.shape[1]

    draft_seconds = time.perf_counter() - draft_started

    # The last draft was predicted from the preceding hidden state but has not
    # itself passed through the edge layers. Process that one token so the
    # cloud receives states for context + every draft token.
    preparation_started = time.perf_counter()
    final_hidden, final_positions = run_edge_layers(
        model,
        draft_ids[:, -1:],
        cache,
        edge_layers,
        past_len,
    )
    hidden_parts.append(final_hidden)
    position_parts.append(final_positions)
    verification_hidden = torch.cat(hidden_parts, dim=1)
    verification_positions = torch.cat(position_parts, dim=1)
    preparation_seconds = time.perf_counter() - preparation_started

    return (
        draft_ids,
        verification_hidden,
        verification_positions,
        {
            "draft_seconds": draft_seconds,
            "preparation_seconds": preparation_seconds,
        },
    )


def verify_full_model(model, cur_ids, draft_ids):
    """Local reference verifier used by tests and debugging."""
    total_ids = torch.cat([cur_ids, draft_ids], dim=1)
    cache = DynamicCache()
    layers = model.model.layers
    hidden, _ = run_edge_layers(model, total_ids, cache, len(layers), 0)
    logits = model.lm_head(model.model.norm(hidden))

    N = cur_ids.shape[1]
    num_draft = draft_ids.shape[1]
    predicted = logits[0].argmax(-1)     # shape (N + num_draft,)
    draft_list = draft_ids[0].tolist()

    accepted = 0
    for i in range(num_draft):
        if predicted[N - 1 + i].item() != draft_list[i]:
            break
        accepted += 1

    accepted_ids = draft_ids[:, :accepted]
    bonus_token = predicted[N - 1 + accepted].view(1, 1)

    return accepted_ids, bonus_token
