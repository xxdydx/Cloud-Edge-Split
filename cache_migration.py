"""Exact per-layer KV delta extraction and installation helpers."""

import torch


def _layer_tensors(cache, layer_idx):
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        # Old DynamicCache stores entries compactly in call order even when
        # decoder layers retain non-zero original layer_idx values.
        physical_idx = layer_idx - getattr(cache, "_split_cache_base", 0)
        if physical_idx < 0 or physical_idx >= len(cache.key_cache):
            raise ValueError(f"cache has no layer {layer_idx}")
        return cache.key_cache[physical_idx], cache.value_cache[physical_idx]

    if hasattr(cache, "layers"):
        if layer_idx >= len(cache.layers):
            raise ValueError(f"cache has no layer {layer_idx}")
        layer = cache.layers[layer_idx]
        key = getattr(layer, "keys", getattr(layer, "key_cache", None))
        value = getattr(layer, "values", getattr(layer, "value_cache", None))
        if key is None or value is None:
            raise ValueError(f"cache layer {layer_idx} is uninitialized")
        return key, value
    raise TypeError("unsupported Transformers cache representation")


def layer_length(cache, layer_idx):
    try:
        key, value = _layer_tensors(cache, layer_idx)
    except ValueError:
        return 0
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch at layer {layer_idx}")
    return key.shape[-2]


def extract_kv_delta(cache, layer_idx, token_offset, token_count, dtype="fp16"):
    if dtype != "fp16":
        raise ValueError("only fp16 KV migration is supported")
    if token_offset < 0 or token_count < 1:
        raise ValueError("invalid KV token range")
    key, value = _layer_tensors(cache, layer_idx)
    end = token_offset + token_count
    if key.shape != value.shape or key.ndim != 4 or end > key.shape[-2]:
        raise ValueError(f"invalid KV cache shape/range at layer {layer_idx}")
    return (
        key[..., token_offset:end, :].to(torch.float16).contiguous(),
        value[..., token_offset:end, :].to(torch.float16).contiguous(),
    )


def install_kv_delta(cache, layer_idx, token_offset, key, value):
    if key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("migrated KV tensors must be fp16")
    if (
        key.shape != value.shape
        or key.ndim != 4
        or key.shape[0] != 1
        or key.shape[-2] < 1
    ):
        raise ValueError("invalid migrated KV shapes")
    current = layer_length(cache, layer_idx)
    if token_offset != current:
        raise ValueError(
            f"KV offset mismatch for layer {layer_idx}: expected {current}, "
            f"got {token_offset}"
        )
    if current:
        old_key, old_value = _layer_tensors(cache, layer_idx)
        if (
            old_key.shape[:-2] != key.shape[:-2]
            or old_key.shape[-1] != key.shape[-1]
            or old_value.shape[:-2] != value.shape[:-2]
            or old_value.shape[-1] != value.shape[-1]
        ):
            raise ValueError(f"KV shape mismatch for layer {layer_idx}")
    # DynamicCache.update is the stable public API and appends on sequence dim.
    cache.update(key, value, layer_idx)


def assert_cache_lengths(cache, layer_indexes, expected_length):
    for layer_idx in layer_indexes:
        actual = layer_length(cache, layer_idx)
        if actual != expected_length:
            raise ValueError(
                f"cache layer {layer_idx} has {actual} tokens; "
                f"expected {expected_length}"
            )
