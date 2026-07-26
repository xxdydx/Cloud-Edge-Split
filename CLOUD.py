"""Cloud-side server for split and speculative LLM inference."""

import importlib.util
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
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

import activation_codec as codec
from benchmarking import TelemetrySampler, elapsed_ms, now_ns
from model_loading import load_partial_model, num_hidden_layers


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cuda")
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TORCH_DTYPE = _DTYPES[os.getenv("TORCH_DTYPE", "float16")]
# Fixed split point this cloud instance is configured to serve — it only
# ever loads layers[EDGE_LAYERS:], so it cannot serve any other value.
EDGE_LAYERS = int(os.getenv("EDGE_LAYERS", "14"))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model_load_started = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
TOTAL_LAYERS = num_hidden_layers(MODEL_NAME)
# Only this device's own layer range + lm_head; no embed_tokens is needed
# since the cloud never receives raw token ids, only hidden states.
model = load_partial_model(
    MODEL_NAME, TORCH_DTYPE, DEVICE,
    EDGE_LAYERS, TOTAL_LAYERS, need_embed=False, need_lm_head=True,
)
MODEL_LOAD_SECONDS = time.perf_counter() - model_load_started
layers = model.model.layers

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


def _validate_edge_layers(edge_layers):
    if edge_layers != EDGE_LAYERS:
        raise ValueError(
            f"this cloud instance is only configured for edge_layers={EDGE_LAYERS}, "
            f"got {edge_layers}"
        )


def cloud_forward(hidden, position_ids, cache, past_len):
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    attention_mask = _causal_attention_mask(hidden, position_ids, past_len)
    # `layers` already only holds this device's own range ([EDGE_LAYERS:] of
    # the full model) after partial loading, so no further slicing here.
    for layer in layers:
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
    edge_layers = None
    past_len = 0
    telemetry = None

    try:
        while True:
            data = await websocket.receive_bytes()
            msg_type = codec.message_type(data)

            if msg_type == codec.MSG_SESSION_START:
                edge_layers, benchmark_enabled = codec.unpack_session_start(data)
                _validate_edge_layers(edge_layers)
                cache = DynamicCache()
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
                } if benchmark_enabled else None))
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
                    logits = cloud_forward(hidden, position_ids_t, cache, past_len)
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
                _validate_edge_layers(fields["edge_layers"])
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
    except (ValueError, RuntimeError) as error:
        await websocket.send_bytes(bytes([codec.MSG_ERROR]) + str(error).encode())
        await websocket.close()
    finally:
        if telemetry is not None:
            telemetry.stop()


@app.get("/ping")
async def ping():
    return {
        "status": "alive",
        "model": MODEL_NAME,
        "device": DEVICE,
        "total_layers": TOTAL_LAYERS,
        "cloud_layers": f"{EDGE_LAYERS}:{TOTAL_LAYERS}",
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
