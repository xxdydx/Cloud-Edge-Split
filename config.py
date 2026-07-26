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
    edge_layers: int = 14  # of 28 total layers in Qwen2.5-7B
    speculative_decoding: bool = True
    num_draft_tokens: int = 3
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "10"))
    request_timeout_seconds: float = 60.0
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
    def cloud_ws_url(self):
        return self.cloud_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/session"


CONFIG = InferenceConfig()
