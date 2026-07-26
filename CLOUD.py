"""Cloud-side server for split and speculative LLM inference."""

import importlib.util
import gc
import inspect
import os
import subprocess
import sys
import threading
import time


def _install_missing_dependencies():
    """Install Kaggle runtime dependencies before importing them."""
    packages = {
        "accelerate": "accelerate",
        "fastapi": "fastapi",
        "pyngrok": "pyngrok",
        "psutil": "psutil",
        "pynvml": "nvidia-ml-py",
        "transformers": "transformers",
        "uvicorn": "uvicorn[standard]",
    }
    missing = [
        package
        for module, package in packages.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            *missing,
        ])


_install_missing_dependencies()

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pyngrok import ngrok
from transformers.cache_utils import DynamicCache

import activation_codec as codec
from benchmarking import TelemetrySampler, elapsed_ms, now_ns
from model_loading import load_partial_model, num_hidden_layers
from cache_migration import assert_cache_lengths, extract_kv_delta


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cuda")
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Model identity, dtype, and layer residency are selected by the edge's
# session-start frame. Deployment-only settings such as DEVICE remain local.
model = None
layers = None
MODEL_NAME = None
TORCH_DTYPE = None
TOTAL_LAYERS = None
PREFILL_EDGE_LAYERS = None
MODEL_LOAD_SECONDS = None
MODEL_SESSION_LOCK = threading.Lock()

app = FastAPI()


def _get_ngrok_auth_token():
    """Read ngrok credentials from the environment or Kaggle Secrets."""
    auth_token = os.getenv("AUTH_TOKEN")
    if auth_token:
        return auth_token

    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("AUTH_TOKEN")
    except Exception as error:
        raise RuntimeError(
            "Add an AUTH_TOKEN Kaggle secret containing your ngrok token, "
            "or set the AUTH_TOKEN environment variable"
        ) from error


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


def _causal_attention_mask(hidden, position_ids, past_len):
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


def _wire_torch_dtype(name):
    normalized = str(name).removeprefix("torch.")
    if normalized not in _DTYPES:
        raise ValueError(f"unsupported cloud compute dtype: {name}")
    return _DTYPES[normalized]


def _ensure_model(model_name, torch_dtype_name, prefill_layers, decode_layers):
    """Load the cloud range requested by the edge, reusing it when possible."""
    global model, layers, MODEL_NAME, TORCH_DTYPE, TOTAL_LAYERS
    global PREFILL_EDGE_LAYERS, MODEL_LOAD_SECONDS

    if not model_name:
        raise ValueError("edge session did not provide model_name")
    requested_dtype = _wire_torch_dtype(torch_dtype_name)
    residency_matches = (
        model is not None
        and MODEL_NAME == model_name
        and TORCH_DTYPE == requested_dtype
        and PREFILL_EDGE_LAYERS == prefill_layers
    )
    if residency_matches:
        if not prefill_layers <= decode_layers < TOTAL_LAYERS:
            raise ValueError(
                "layer splits must satisfy prefill <= decode < total"
            )
        return False

    total_layers = num_hidden_layers(model_name)
    if not 0 < prefill_layers <= decode_layers < total_layers:
        raise ValueError(
            "layer splits must satisfy 0 < prefill <= decode < total"
        )

    model = None
    layers = None
    gc.collect()
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_started = time.perf_counter()
    model = load_partial_model(
        model_name, requested_dtype, DEVICE,
        prefill_layers, total_layers, need_embed=False, need_lm_head=True,
    )
    layers = model.model.layers
    MODEL_NAME = model_name
    TORCH_DTYPE = requested_dtype
    TOTAL_LAYERS = total_layers
    PREFILL_EDGE_LAYERS = prefill_layers
    MODEL_LOAD_SECONDS = time.perf_counter() - load_started
    return True


def cloud_forward(hidden, position_ids, cache, past_len, start_layer=None):
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    attention_mask = _causal_attention_mask(hidden, position_ids, past_len)
    start_layer = (
        PREFILL_EDGE_LAYERS if start_layer is None else start_layer
    )
    local_start = start_layer - PREFILL_EDGE_LAYERS
    for layer in layers[local_start:]:
        hidden = _layer_hidden(layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            position_embeddings=position_embeddings,
            **_cache_argument(layer, cache),
        ))
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


def _decode_tensors(dtype, seq_len, hidden_dim, position_ids_array, scales, payload):
    hidden = codec.decode_activation(
        payload, scales, dtype, seq_len, hidden_dim, DEVICE, model_dtype=TORCH_DTYPE
    )
    positions = torch.tensor(position_ids_array, dtype=torch.long, device=DEVICE).reshape(1, -1)
    return hidden, positions


@app.websocket("/session")
async def session(websocket: WebSocket):
    await websocket.accept()

    cache = None
    prefill_layers = None
    decode_layers = None
    max_new_tokens = None
    past_len = 0
    telemetry = None
    owns_model_lock = False

    try:
        while True:
            data = await websocket.receive_bytes()
            msg_type = codec.message_type(data)

            if msg_type == codec.MSG_SESSION_START:
                fields = codec.unpack_session_start(data)
                prefill_layers = fields["prefill_edge_layers"]
                decode_layers = fields["decode_edge_layers"]
                max_new_tokens = fields["max_new_tokens"]
                benchmark_enabled = fields["benchmark_enabled"]
                if not owns_model_lock:
                    owns_model_lock = MODEL_SESSION_LOCK.acquire(blocking=False)
                    if not owns_model_lock:
                        raise RuntimeError(
                            "cloud model is busy with another active session"
                        )
                if max_new_tokens < 1:
                    raise ValueError("max_new_tokens must be at least 1")
                if fields["prefill_chunk_size"] < 1:
                    raise ValueError("prefill chunk size must be at least 1")
                if fields["kv_transfer_dtype"] != "fp16":
                    raise ValueError("only fp16 KV transfer is supported")
                if prefill_layers != decode_layers and not fields["stream_kv_layers"]:
                    raise ValueError("varied split requires streamed KV layers")
                model_reloaded = _ensure_model(
                    fields.get("model_name"),
                    fields.get("torch_dtype"),
                    prefill_layers,
                    decode_layers,
                )
                cache = DynamicCache()
                # Used by migration helpers with legacy compact-list caches.
                cache._split_cache_base = PREFILL_EDGE_LAYERS
                past_len = 0
                if telemetry is not None:
                    telemetry.stop()
                telemetry = (
                    TelemetrySampler(
                        interval_ms=int(os.getenv("TELEMETRY_INTERVAL_MS", "100")),
                        enable_nvml=True,
                    )
                    if benchmark_enabled else None
                )
                if telemetry:
                    telemetry.start()
                await websocket.send_bytes(codec.pack_ok({
                    "cloud_model_load_ms": MODEL_LOAD_SECONDS * 1000,
                    "cloud_compute_dtype": str(TORCH_DTYPE),
                    "cloud_device": DEVICE,
                    "max_new_tokens": max_new_tokens,
                    "cloud_model_reloaded": model_reloaded,
                    **fields,
                } if benchmark_enabled else None))
                continue

            if msg_type == codec.MSG_PREFILL_CHUNK:
                request_started = now_ns()
                if cache is None:
                    raise ValueError("Session not started")
                fields = codec.unpack_prefill_chunk(data)
                offset = fields["token_offset"]
                count = fields["seq_len"]
                if offset != past_len:
                    raise ValueError(
                        f"prefill offset mismatch: expected {past_len}, got {offset}"
                    )
                if count < 1:
                    raise ValueError("prefill chunk cannot be empty")
                hidden, position_ids_t = _decode_tensors(
                    fields["dtype"], count, fields["hidden_dim"],
                    fields["position_ids"], fields["scales"], fields["payload"],
                )
                position_embeddings = model.model.rotary_emb(
                    hidden, position_ids_t
                )
                attention_mask = _causal_attention_mask(
                    hidden, position_ids_t, past_len
                )
                kv_bytes = 0
                extraction_ms = 0.0
                forward_started = now_ns()
                with torch.no_grad():
                    for local_idx, layer in enumerate(layers):
                        global_idx = PREFILL_EDGE_LAYERS + local_idx
                        hidden = _layer_hidden(layer(
                            hidden,
                            attention_mask=attention_mask,
                            position_ids=position_ids_t,
                            use_cache=True,
                            position_embeddings=position_embeddings,
                            **_cache_argument(layer, cache),
                        ))
                        if global_idx < decode_layers:
                            extraction_started = now_ns()
                            key, value = extract_kv_delta(
                                cache, global_idx, offset, count, "fp16"
                            )
                            frame = codec.pack_kv_delta(
                                global_idx, offset, key, value
                            )
                            extraction_ms += elapsed_ms(extraction_started)
                            kv_bytes += len(frame)
                            await websocket.send_bytes(frame)
                    hidden = model.model.norm(hidden)
                    logits = model.lm_head(hidden)
                forward_finished = now_ns()
                past_len += count
                metrics = {
                    "cloud_prefill_ms": elapsed_ms(
                        forward_started, forward_finished
                    ),
                    "kv_extraction_ms": extraction_ms,
                    "kv_bytes": kv_bytes,
                    "chunk_total_ms": elapsed_ms(request_started),
                    "request_bytes": len(data),
                }
                await websocket.send_bytes(
                    codec.pack_chunk_complete(offset, count, metrics)
                )
                if fields["final_chunk"]:
                    assert_cache_lengths(
                        cache, range(decode_layers, TOTAL_LAYERS), past_len
                    )
                    next_token = logits[:, -1, :].argmax(-1).item()
                    await websocket.send_bytes(
                        codec.pack_prefill_complete(next_token, metrics)
                    )
                continue

            if msg_type == codec.MSG_DECODE:
                request_started = now_ns()
                if telemetry:
                    telemetry.capture()
                if cache is None:
                    raise ValueError("Session not started")
                decode_started = now_ns()
                dtype, seq_len, hidden_dim, position_ids, scales, payload = codec.unpack_decode(data)
                hidden, position_ids_t = _decode_tensors(
                    dtype, seq_len, hidden_dim, position_ids, scales, payload
                )
                decode_finished = now_ns()
                forward_started = now_ns()
                with torch.no_grad():
                    logits = cloud_forward(
                        hidden, position_ids_t, cache, past_len,
                        start_layer=decode_layers,
                    )
                    next_token = logits[:, -1, :].argmax(-1).item()
                forward_finished = now_ns()
                past_len += hidden.shape[1]
                if telemetry:
                    telemetry.capture()
                metrics = {
                    "activation_decode_ms": elapsed_ms(decode_started, decode_finished),
                    "cloud_forward_ms": elapsed_ms(forward_started, forward_finished),
                    "server_processing_ms": elapsed_ms(request_started),
                    "request_bytes": len(data),
                    "telemetry": (
                        telemetry.summarize(request_started, now_ns())
                        if telemetry else {}
                    ),
                } if telemetry else None
                await websocket.send_bytes(
                    codec.pack_decode_reply(next_token, metrics)
                )
                continue

            if msg_type == codec.MSG_VERIFY:
                request_started = now_ns()
                if telemetry:
                    telemetry.capture()
                decode_started = now_ns()
                fields = codec.unpack_verify(data)
                if prefill_layers != decode_layers:
                    raise ValueError(
                        "speculative decoding is unavailable for varied split"
                    )
                if fields["edge_layers"] != decode_layers:
                    raise ValueError(
                        "verification split does not match the active session"
                    )
                context_length = fields["context_length"]
                draft_ids = fields["draft_ids"]
                if context_length < 1 or len(draft_ids) == 0:
                    raise ValueError(
                        "Verification requires a context and at least one draft token"
                    )
                hidden, position_ids_t = _decode_tensors(
                    fields["dtype"],
                    fields["seq_len"],
                    fields["hidden_dim"],
                    fields["position_ids"],
                    fields["scales"],
                    fields["payload"],
                )
                decode_finished = now_ns()
                expected_length = context_length + len(draft_ids)
                if hidden.shape[1] != expected_length:
                    raise ValueError("Hidden-state length does not match context plus drafts")

                forward_started = now_ns()
                with torch.no_grad():
                    logits = cloud_forward(hidden, position_ids_t, DynamicCache(), 0)
                    predicted = logits[0].argmax(-1)
                forward_finished = now_ns()

                accepted_count = 0
                for index, draft_id in enumerate(draft_ids.tolist()):
                    verifier_id = predicted[context_length - 1 + index].item()
                    if verifier_id != draft_id:
                        break
                    accepted_count += 1

                bonus_index = context_length - 1 + accepted_count
                bonus_token = predicted[bonus_index].item()
                if telemetry:
                    telemetry.capture()
                metrics = {
                    "activation_decode_ms": elapsed_ms(decode_started, decode_finished),
                    "cloud_forward_ms": elapsed_ms(forward_started, forward_finished),
                    "server_processing_ms": elapsed_ms(request_started),
                    "request_bytes": len(data),
                    "telemetry": (
                        telemetry.summarize(request_started, now_ns())
                        if telemetry else {}
                    ),
                } if telemetry else None
                await websocket.send_bytes(codec.pack_verify_reply(
                    accepted_count,
                    bonus_token,
                    metrics,
                ))
                continue

    except WebSocketDisconnect:
        pass
    except Exception as error:
        await websocket.send_bytes(bytes([codec.MSG_ERROR]) + str(error).encode())
        await websocket.close()
    finally:
        if telemetry is not None:
            telemetry.stop()
        if owns_model_lock:
            MODEL_SESSION_LOCK.release()


@app.get("/ping")
async def ping():
    return {
        "status": "alive",
        "model": MODEL_NAME,
        "device": DEVICE,
        "total_layers": TOTAL_LAYERS,
        "prefill_edge_layers": PREFILL_EDGE_LAYERS,
        "cloud_layers": (
            f"{PREFILL_EDGE_LAYERS}:{TOTAL_LAYERS}"
            if model is not None else None
        ),
        "configuration_source": "edge session handshake",
    }


def main():
    ngrok.set_auth_token(_get_ngrok_auth_token())
    ngrok.kill()
    public_url = ngrok.connect(PORT).public_url
    print(f"Public URL: {public_url}")
    print(f"Ping URL: {public_url}/ping")

    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="uvicorn-server")
    try:
        # Kaggle/IPython already has an asyncio event loop. Running Uvicorn in
        # its own thread gives it a separate loop and also works as a .py file.
        server_thread.start()
        server_thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
        server_thread.join()
    finally:
        ngrok.kill()


if __name__ == "__main__":
    main()
