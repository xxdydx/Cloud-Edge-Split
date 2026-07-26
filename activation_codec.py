"""Binary framing and quantization codec for edge<->cloud activation transfer.

Every message exchanged over the `/session` websocket is one binary frame:
a fixed-layout struct header followed by variable-length sections (position
ids, draft ids, per-token scales, activation payload) in a fixed order.
Quantization only applies to the activation payload — the one tensor that
has to physically cross the network at the edge/cloud split point.
"""

import struct

import numpy as np
import torch

MSG_SESSION_START = 1
MSG_DECODE = 2
MSG_VERIFY = 3
MSG_DECODE_REPLY = 4
MSG_VERIFY_REPLY = 5
MSG_ERROR = 6
MSG_OK = 7

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT4 = 2

_DTYPE_CODES = {"fp32": DTYPE_FP32, "fp16": DTYPE_FP16, "int4": DTYPE_INT4}
_DTYPE_NAMES = {code: name for name, code in _DTYPE_CODES.items()}

_HEADER_FMT = "<BBII"        # msg_type, dtype, seq_len, hidden_dim
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_VERIFY_EXTRA_FMT = "<IHH"   # context_length, num_draft, edge_layers
_VERIFY_EXTRA_SIZE = struct.calcsize(_VERIFY_EXTRA_FMT)
_SESSION_START_FMT = "<BH"   # msg_type, edge_layers
_DECODE_REPLY_FMT = "<Bi"    # msg_type, next_token
_VERIFY_REPLY_FMT = "<BIi"   # msg_type, accepted_count, bonus_token


def dtype_code(dtype):
    return _DTYPE_CODES[dtype]


def dtype_name(code):
    return _DTYPE_NAMES[code]


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

def pack_session_start(edge_layers):
    return struct.pack(_SESSION_START_FMT, MSG_SESSION_START, edge_layers)


def unpack_session_start(data):
    _, edge_layers = struct.unpack(_SESSION_START_FMT, data)
    return edge_layers


def pack_ok():
    return bytes([MSG_OK])


def unpack_ok(data):
    return None


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


def pack_decode_reply(next_token):
    return struct.pack(_DECODE_REPLY_FMT, MSG_DECODE_REPLY, next_token)


def unpack_decode_reply(data):
    _, next_token = struct.unpack(_DECODE_REPLY_FMT, data)
    return next_token


def pack_verify_reply(accepted_count, bonus_token):
    return struct.pack(_VERIFY_REPLY_FMT, MSG_VERIFY_REPLY, accepted_count, bonus_token)


def unpack_verify_reply(data):
    _, accepted_count, bonus_token = struct.unpack(_VERIFY_REPLY_FMT, data)
    return accepted_count, bonus_token


def message_type(data):
    return data[0]
