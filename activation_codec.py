"""Binary framing and quantization codec for edge<->cloud activation transfer.

Every message exchanged over the `/session` websocket is one binary frame:
a fixed-layout struct header followed by variable-length sections (position
ids, draft ids, per-token scales, activation payload) in a fixed order.
Quantization only applies to the activation payload — the one tensor that
has to physically cross the network at the edge/cloud split point.
"""

import struct
import json

import numpy as np
import torch

MSG_SESSION_START = 1
MSG_DECODE = 2
MSG_VERIFY = 3
MSG_DECODE_REPLY = 4
MSG_VERIFY_REPLY = 5
MSG_ERROR = 6
MSG_OK = 7
MSG_PREFILL_CHUNK = 8
MSG_KV_DELTA = 9
MSG_CHUNK_COMPLETE = 10
MSG_PREFILL_COMPLETE = 11

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT4 = 2

_DTYPE_CODES = {"fp32": DTYPE_FP32, "fp16": DTYPE_FP16, "int4": DTYPE_INT4}
_DTYPE_NAMES = {code: name for name, code in _DTYPE_CODES.items()}

_HEADER_FMT = "<BBII"        # msg_type, dtype, seq_len, hidden_dim
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_VERIFY_EXTRA_FMT = "<IHH"   # context_length, num_draft, edge_layers
_VERIFY_EXTRA_SIZE = struct.calcsize(_VERIFY_EXTRA_FMT)
_SESSION_START_FMT = "<BHHI?I?B?"  # both splits, limit, chunking, KV settings
_PREFILL_EXTRA_FMT = "<I?"  # token offset, final chunk
_KV_HEADER_FMT = "<BHII4I"  # type, layer, offset, count, B/H/S/D
_CHUNK_COMPLETE_FMT = "<BII"  # type, offset, count
_PREFILL_COMPLETE_FMT = "<Bi"  # type, first generated token
_DECODE_REPLY_FMT = "<Bi"    # msg_type, next_token
_VERIFY_REPLY_FMT = "<BIi"   # msg_type, accepted_count, bonus_token
_METRICS_LENGTH_FMT = "<I"
_METRICS_LENGTH_SIZE = struct.calcsize(_METRICS_LENGTH_FMT)


def dtype_code(dtype):
    try:
        return _DTYPE_CODES[dtype]
    except KeyError:
        raise ValueError(f"Unknown wire dtype: {dtype}") from None


def dtype_name(code):
    try:
        return _DTYPE_NAMES[code]
    except KeyError:
        raise ValueError(f"Unknown wire dtype code: {code}") from None


# --- quantization -----------------------------------------------------

def _pack_nibbles(quantized):
    flat = quantized.reshape(-1).cpu().numpy().astype(np.int8)
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).astype(np.uint8).tobytes()


def _unpack_nibbles(packed, count):
    packed_bytes = np.frombuffer(packed, dtype=np.uint8)
    low = (packed_bytes & 0x0F).astype(np.int8)
    high = ((packed_bytes >> 4) & 0x0F).astype(np.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    interleaved = np.empty(packed_bytes.size * 2, dtype=np.int8)
    interleaved[0::2] = low
    interleaved[1::2] = high
    return interleaved[:count]


def _quantize_int4(hidden):
    """Per-token symmetric quantization. hidden: (1, seq_len, hidden_dim), even hidden_dim."""
    rows = hidden[0]
    scales = rows.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    quantized = (rows / scales.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    payload = _pack_nibbles(quantized)
    scales_bytes = scales.to(torch.float32).cpu().numpy().tobytes()
    return payload, scales_bytes


def _dequantize_int4(payload, scales_bytes, seq_len, hidden_dim):
    values = _unpack_nibbles(payload, seq_len * hidden_dim).astype(np.float32)
    scales = np.frombuffer(scales_bytes, dtype=np.float32)
    values = values.reshape(seq_len, hidden_dim) * scales.reshape(seq_len, 1)
    return torch.from_numpy(values.copy()).reshape(1, seq_len, hidden_dim)


def encode_activation(hidden, dtype):
    """hidden: (1, seq_len, hidden_dim) float tensor -> (payload_bytes, scales_bytes)."""
    if dtype == "fp32":
        return hidden.to(torch.float32).cpu().numpy().tobytes(), b""
    if dtype == "fp16":
        return hidden.to(torch.float16).cpu().numpy().tobytes(), b""
    if dtype == "int4":
        return _quantize_int4(hidden)
    raise ValueError(f"Unknown activation dtype: {dtype}")


def decode_activation(payload, scales, dtype, seq_len, hidden_dim, device, model_dtype=torch.float32):
    """model_dtype is the receiving model's compute dtype, independent of the
    wire dtype (`dtype`) the activation was quantized with for transport."""
    if dtype == "fp32":
        array = np.frombuffer(payload, dtype=np.float32)
        tensor = torch.from_numpy(array.copy()).reshape(1, seq_len, hidden_dim)
    elif dtype == "fp16":
        array = np.frombuffer(payload, dtype=np.float16)
        tensor = torch.from_numpy(array.astype(np.float32)).reshape(1, seq_len, hidden_dim)
    elif dtype == "int4":
        tensor = _dequantize_int4(payload, scales, seq_len, hidden_dim)
    else:
        raise ValueError(f"Unknown activation dtype: {dtype}")
    return tensor.to(device=device, dtype=model_dtype)


# --- framing ------------------------------------------------------------

def _pack_metrics(metrics):
    payload = (
        json.dumps(metrics, separators=(",", ":")).encode()
        if metrics else b""
    )
    return struct.pack(_METRICS_LENGTH_FMT, len(payload)) + payload


def _unpack_metrics(data, offset):
    if len(data) < offset + _METRICS_LENGTH_SIZE:
        return {}
    (length,) = struct.unpack(
        _METRICS_LENGTH_FMT,
        data[offset: offset + _METRICS_LENGTH_SIZE],
    )
    if not length:
        return {}
    payload = data[
        offset + _METRICS_LENGTH_SIZE:
        offset + _METRICS_LENGTH_SIZE + length
    ]
    return json.loads(payload)


def pack_session_start(
    prefill_edge_layers, decode_edge_layers, max_new_tokens,
    chunked_prefill=True, prefill_chunk_size=64, stream_kv_layers=True,
    kv_transfer_dtype="fp16", benchmark_enabled=False,
):
    return struct.pack(
        _SESSION_START_FMT,
        MSG_SESSION_START,
        prefill_edge_layers,
        decode_edge_layers,
        max_new_tokens,
        chunked_prefill,
        prefill_chunk_size,
        stream_kv_layers,
        dtype_code(kv_transfer_dtype),
        int(benchmark_enabled),
    )


def unpack_session_start(data):
    (
        _, prefill_layers, decode_layers, max_new_tokens, chunked,
        chunk_size, stream_layers, kv_dtype, benchmark_enabled,
    ) = struct.unpack(
        _SESSION_START_FMT,
        data,
    )
    return {
        "prefill_edge_layers": prefill_layers,
        "decode_edge_layers": decode_layers,
        "max_new_tokens": max_new_tokens,
        "chunked_prefill": chunked,
        "prefill_chunk_size": chunk_size,
        "stream_kv_layers": stream_layers,
        "kv_transfer_dtype": dtype_name(kv_dtype),
        "benchmark_enabled": bool(benchmark_enabled),
    }


def pack_ok(metrics=None):
    return bytes([MSG_OK]) + _pack_metrics(metrics)


def unpack_ok(data):
    return _unpack_metrics(data, 1)


def pack_decode(hidden, position_ids, dtype):
    seq_len, hidden_dim = hidden.shape[1], hidden.shape[2]
    payload, scales = encode_activation(hidden, dtype)
    header = struct.pack(_HEADER_FMT, MSG_DECODE, dtype_code(dtype), seq_len, hidden_dim)
    position_bytes = position_ids.reshape(-1).to(torch.int32).cpu().numpy().tobytes()
    return header + position_bytes + scales + payload


def unpack_decode(data):
    _, dtype_c, seq_len, hidden_dim = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
    dtype = dtype_name(dtype_c)
    offset = _HEADER_SIZE
    position_ids = np.frombuffer(data, dtype=np.int32, count=seq_len, offset=offset).copy()
    offset += seq_len * 4
    if dtype == "int4":
        scales = data[offset: offset + seq_len * 4]
        offset += seq_len * 4
    else:
        scales = b""
    payload = data[offset:]
    return dtype, seq_len, hidden_dim, position_ids, scales, payload


def pack_prefill_chunk(hidden, position_ids, dtype, token_offset, final_chunk):
    frame = bytearray(pack_decode(hidden, position_ids, dtype))
    frame[0] = MSG_PREFILL_CHUNK
    extra = struct.pack(_PREFILL_EXTRA_FMT, token_offset, final_chunk)
    return bytes(frame[:_HEADER_SIZE] + extra + frame[_HEADER_SIZE:])


def unpack_prefill_chunk(data):
    extra_size = struct.calcsize(_PREFILL_EXTRA_FMT)
    token_offset, final_chunk = struct.unpack(
        _PREFILL_EXTRA_FMT, data[_HEADER_SIZE:_HEADER_SIZE + extra_size]
    )
    decode_frame = bytes([MSG_DECODE]) + data[1:_HEADER_SIZE] + data[_HEADER_SIZE + extra_size:]
    dtype, seq_len, hidden_dim, positions, scales, payload = unpack_decode(decode_frame)
    return {
        "dtype": dtype, "seq_len": seq_len, "hidden_dim": hidden_dim,
        "position_ids": positions, "scales": scales, "payload": payload,
        "token_offset": token_offset, "final_chunk": final_chunk,
    }


def pack_kv_delta(layer_idx, token_offset, key, value):
    if key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("KV wire dtype must be fp16")
    if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
        raise ValueError("invalid KV shapes")
    shape = tuple(key.shape)
    header = struct.pack(
        _KV_HEADER_FMT, MSG_KV_DELTA, layer_idx, token_offset, shape[-2], *shape
    )
    return header + key.cpu().numpy().tobytes() + value.cpu().numpy().tobytes()


def unpack_kv_delta(data, device="cpu"):
    header_size = struct.calcsize(_KV_HEADER_FMT)
    _, layer_idx, offset, count, *shape = struct.unpack(
        _KV_HEADER_FMT, data[:header_size]
    )
    if shape[-2] != count or any(dimension < 1 for dimension in shape):
        raise ValueError("invalid KV frame shape")
    tensor_bytes = int(np.prod(shape)) * 2
    if len(data) != header_size + 2 * tensor_bytes:
        raise ValueError("KV payload length does not match shape")
    key_array = np.frombuffer(data, dtype=np.float16, count=int(np.prod(shape)), offset=header_size)
    value_array = np.frombuffer(data, dtype=np.float16, count=int(np.prod(shape)), offset=header_size + tensor_bytes)
    key = torch.from_numpy(key_array.copy()).reshape(shape).to(device)
    value = torch.from_numpy(value_array.copy()).reshape(shape).to(device)
    return layer_idx, offset, key, value


def pack_chunk_complete(token_offset, token_count, metrics=None):
    return struct.pack(
        _CHUNK_COMPLETE_FMT, MSG_CHUNK_COMPLETE, token_offset, token_count
    ) + _pack_metrics(metrics)


def unpack_chunk_complete(data):
    size = struct.calcsize(_CHUNK_COMPLETE_FMT)
    _, offset, count = struct.unpack(_CHUNK_COMPLETE_FMT, data[:size])
    return offset, count, _unpack_metrics(data, size)


def pack_prefill_complete(next_token, metrics=None):
    return struct.pack(
        _PREFILL_COMPLETE_FMT, MSG_PREFILL_COMPLETE, next_token
    ) + _pack_metrics(metrics)


def unpack_prefill_complete(data):
    size = struct.calcsize(_PREFILL_COMPLETE_FMT)
    _, token = struct.unpack(_PREFILL_COMPLETE_FMT, data[:size])
    return token, _unpack_metrics(data, size)


def pack_verify(hidden, position_ids, dtype, context_length, draft_ids, edge_layers):
    seq_len, hidden_dim = hidden.shape[1], hidden.shape[2]
    payload, scales = encode_activation(hidden, dtype)
    header = struct.pack(_HEADER_FMT, MSG_VERIFY, dtype_code(dtype), seq_len, hidden_dim)
    header += struct.pack(_VERIFY_EXTRA_FMT, context_length, len(draft_ids), edge_layers)
    position_bytes = position_ids.reshape(-1).to(torch.int32).cpu().numpy().tobytes()
    draft_bytes = np.asarray(draft_ids, dtype=np.int32).tobytes()
    return header + position_bytes + draft_bytes + scales + payload


def unpack_verify(data):
    _, dtype_c, seq_len, hidden_dim = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
    dtype = dtype_name(dtype_c)
    offset = _HEADER_SIZE
    context_length, num_draft, edge_layers = struct.unpack(
        _VERIFY_EXTRA_FMT, data[offset: offset + _VERIFY_EXTRA_SIZE]
    )
    offset += _VERIFY_EXTRA_SIZE
    position_ids = np.frombuffer(data, dtype=np.int32, count=seq_len, offset=offset).copy()
    offset += seq_len * 4
    draft_ids = np.frombuffer(data, dtype=np.int32, count=num_draft, offset=offset).copy()
    offset += num_draft * 4
    if dtype == "int4":
        scales = data[offset: offset + seq_len * 4]
        offset += seq_len * 4
    else:
        scales = b""
    payload = data[offset:]
    return {
        "dtype": dtype,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "context_length": context_length,
        "edge_layers": edge_layers,
        "position_ids": position_ids,
        "draft_ids": draft_ids,
        "scales": scales,
        "payload": payload,
    }


def pack_decode_reply(next_token, metrics=None):
    return (
        struct.pack(_DECODE_REPLY_FMT, MSG_DECODE_REPLY, next_token)
        + _pack_metrics(metrics)
    )


def unpack_decode_reply(data):
    reply_size = struct.calcsize(_DECODE_REPLY_FMT)
    _, next_token = struct.unpack(_DECODE_REPLY_FMT, data[:reply_size])
    return next_token, _unpack_metrics(
        data,
        reply_size,
    )


def pack_verify_reply(accepted_count, bonus_token, metrics=None):
    return (
        struct.pack(
            _VERIFY_REPLY_FMT,
            MSG_VERIFY_REPLY,
            accepted_count,
            bonus_token,
        )
        + _pack_metrics(metrics)
    )


def unpack_verify_reply(data):
    reply_size = struct.calcsize(_VERIFY_REPLY_FMT)
    _, accepted_count, bonus_token = struct.unpack(
        _VERIFY_REPLY_FMT,
        data[:reply_size],
    )
    return accepted_count, bonus_token, _unpack_metrics(data, reply_size)


def message_type(data):
    return data[0]
