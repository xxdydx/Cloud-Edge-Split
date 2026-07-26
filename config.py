"""Configuration shared by the edge inference entry point."""

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    # Override this with the URL printed by CLOUD.py:
    # CLOUD_URL=https://example.ngrok-free.app python3 edge_client.py
    cloud_url: str = os.getenv(
        "CLOUD_URL",
        "https://liquid-cycling-kindle.ngrok-free.dev",
    )
    split_inference: bool = True
    prefill_edge_layers: int = 4
    decode_edge_layers: int = 12
    chunked_prefill: bool = True
    prefill_chunk_size: int = 64
    stream_kv_layers: bool = True
    kv_transfer_dtype: str = "fp16"
    overlap_kv_transfer: bool = True
    kv_transfer_queue_depth: int = 4
    speculative_decoding: bool = False
    num_draft_tokens: int = 3
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "100"))
    request_timeout_seconds: float = 60.0
    model_load_timeout_seconds: float = 900.0
    torch_dtype: torch.dtype = torch.float16
    activation_dtype: str = "fp16"  # "fp32", "fp16", or "int4"
    # device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    device: str =  "cpu"

    warmup_on_start: bool = True
    benchmark_enabled: bool = True
    benchmark_output: str = "benchmarks/results.jsonl"
    telemetry_interval_ms: int = 100
    edge_power_sampler: str = "powermetrics"  

    @property
    def edge_layers(self):
        """Compatibility alias for code paths which have a single split."""
        return self.decode_edge_layers

    def validate(self, total_layers):
        if not 0 < self.prefill_edge_layers <= self.decode_edge_layers < total_layers:
            raise ValueError(
                "layer splits must satisfy 0 < prefill_edge_layers <= "
                "decode_edge_layers < total_layers"
            )
        if self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be at least 1")
        if self.model_load_timeout_seconds <= 0:
            raise ValueError("model_load_timeout_seconds must be positive")
        if self.kv_transfer_dtype != "fp16":
            raise ValueError("only exact fp16 KV transfer is currently supported")
        if self.kv_transfer_queue_depth < 1:
            raise ValueError("kv_transfer_queue_depth must be at least 1")
        if self.overlap_kv_transfer and not self.stream_kv_layers:
            raise ValueError("KV transfer overlap requires streamed KV layers")
        if self.speculative_decoding and (
            self.prefill_edge_layers != self.decode_edge_layers
        ):
            raise ValueError(
                "speculative decoding requires equal prefill/decode split points"
            )

    @property
    def cloud_ws_url(self):
        return self.cloud_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/session"


CONFIG = InferenceConfig()
