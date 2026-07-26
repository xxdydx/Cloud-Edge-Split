"""Lightweight inference benchmarking and device telemetry utilities."""

import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


def now_ns():
    return time.perf_counter_ns()


def elapsed_ms(start_ns, end_ns=None):
    end_ns = now_ns() if end_ns is None else end_ns
    return (end_ns - start_ns) / 1_000_000


def _summary(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }


class TelemetrySampler:
    """Sample process, MPS, and optional NVIDIA device telemetry."""

    def __init__(self, interval_ms=100, enable_nvml=False):
        self.interval_seconds = interval_ms / 1000
        self.enable_nvml = enable_nvml
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.process = psutil.Process(os.getpid()) if psutil else None
        self.nvml_handle = None
        self.nvml_error = None

        if enable_nvml and pynvml:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as error:
                self.nvml_error = str(error)
        elif enable_nvml:
            self.nvml_error = "nvidia-ml-py is unavailable"

    def start(self):
        if self._thread is not None:
            return
        if self.process:
            self.process.cpu_percent(None)
        self.capture()
        self._thread = threading.Thread(
            target=self._run,
            name="telemetry-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self.capture()

    def capture(self):
        sample = {"timestamp_ns": now_ns()}
        if self.process:
            try:
                memory = self.process.memory_info()
                sample.update({
                    "process_cpu_percent": self.process.cpu_percent(None),
                    "process_rss_bytes": memory.rss,
                    "system_cpu_percent": psutil.cpu_percent(None),
                    "system_memory_percent": psutil.virtual_memory().percent,
                })
            except Exception:
                pass

        if torch.backends.mps.is_available():
            try:
                sample.update({
                    "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                    "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                })
            except Exception:
                pass

        if self.nvml_handle is not None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                sample.update({
                    "gpu_util_percent": utilization.gpu,
                    "gpu_memory_util_percent": utilization.memory,
                    "gpu_memory_used_bytes": memory.used,
                    "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(
                        self.nvml_handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    ),
                    "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(
                        self.nvml_handle
                    ) / 1000,
                })
                try:
                    sample["gpu_total_energy_mj"] = (
                        pynvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml_handle)
                    )
                except Exception:
                    pass
            except Exception as error:
                self.nvml_error = str(error)

        self.samples.append(sample)
        return sample

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2)
        self.capture()
        self._thread = None

    def summarize(self, start_ns=None, end_ns=None):
        selected = [
            sample for sample in self.samples
            if (start_ns is None or sample["timestamp_ns"] >= start_ns)
            and (end_ns is None or sample["timestamp_ns"] <= end_ns)
        ]
        if not selected and self.samples:
            selected = [self.samples[-1]]

        keys = {
            key
            for sample in selected
            for key in sample
            if key != "timestamp_ns" and not key.endswith("total_energy_mj")
        }
        result = {
            key: _summary([sample.get(key) for sample in selected])
            for key in sorted(keys)
        }
        result["sample_count"] = len(selected)

        energy_values = [
            sample.get("gpu_total_energy_mj")
            for sample in selected
            if sample.get("gpu_total_energy_mj") is not None
        ]
        if len(energy_values) >= 2:
            result["gpu_energy_j"] = (
                energy_values[-1] - energy_values[0]
            ) / 1000
            result["gpu_energy_source"] = "nvml_total_energy"
        else:
            power_samples = [
                sample for sample in selected
                if sample.get("gpu_power_w") is not None
            ]
            if len(power_samples) >= 2:
                energy_j = 0.0
                for first, second in zip(power_samples, power_samples[1:]):
                    seconds = (
                        second["timestamp_ns"] - first["timestamp_ns"]
                    ) / 1_000_000_000
                    energy_j += (
                        first["gpu_power_w"] + second["gpu_power_w"]
                    ) * 0.5 * seconds
                result["gpu_energy_j"] = energy_j
                result["gpu_energy_source"] = "integrated_nvml_power"
            elif self.enable_nvml:
                result["gpu_energy_j"] = None
                result["gpu_energy_source"] = "unsupported"

        if self.enable_nvml and self.nvml_error:
            result["nvml_error"] = self.nvml_error
        return result


class PowermetricsSampler:
    """Best-effort Apple power sampler; never prompts for sudo credentials."""

    _CPU_PATTERN = re.compile(r"CPU Power:\s+([0-9.]+)\s+mW")
    _GPU_PATTERN = re.compile(r"GPU Power:\s+([0-9.]+)\s+mW")

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.process = None
        self.thread = None
        self.samples = []
        self.error = None
        self._latest_cpu_w = None

    def start(self):
        if platform.system() != "Darwin" or not shutil.which("powermetrics"):
            self.error = "powermetrics is unavailable"
            return
        try:
            self.process = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power",
                    "-i",
                    str(self.interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.thread = threading.Thread(
                target=self._read,
                name="powermetrics-sampler",
                daemon=True,
            )
            self.thread.start()
        except Exception as error:
            self.error = str(error)

    def _read(self):
        for line in self.process.stdout:
            cpu_match = self._CPU_PATTERN.search(line)
            if cpu_match:
                self._latest_cpu_w = float(cpu_match.group(1)) / 1000
            gpu_match = self._GPU_PATTERN.search(line)
            if gpu_match:
                self.samples.append({
                    "timestamp_ns": now_ns(),
                    "cpu_power_w": self._latest_cpu_w,
                    "gpu_power_w": float(gpu_match.group(1)) / 1000,
                })

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if self.thread:
            self.thread.join(timeout=1)
        if self.process.returncode not in (0, -15) and not self.samples:
            stderr = self.process.stderr.read().strip()
            self.error = stderr or "powermetrics authorization failed"

    def summarize(self):
        if len(self.samples) < 2:
            return {
                "edge_energy_j": None,
                "edge_energy_source": "unsupported",
                "powermetrics_error": self.error or "insufficient samples",
            }
        energy_j = 0.0
        for first, second in zip(self.samples, self.samples[1:]):
            seconds = (
                second["timestamp_ns"] - first["timestamp_ns"]
            ) / 1_000_000_000
            first_power = (first.get("cpu_power_w") or 0) + first["gpu_power_w"]
            second_power = (second.get("cpu_power_w") or 0) + second["gpu_power_w"]
            energy_j += (first_power + second_power) * 0.5 * seconds
        return {
            "edge_energy_j": energy_j,
            "edge_energy_source": "powermetrics_cpu_plus_gpu",
            "cpu_power_w": _summary([
                sample.get("cpu_power_w") for sample in self.samples
            ]),
            "gpu_power_w": _summary([
                sample.get("gpu_power_w") for sample in self.samples
            ]),
            "powermetrics_scope": "device_components_not_process",
        }


class BenchmarkRun:
    def __init__(self, prompt, config, model_load_seconds):
        self.start_ns = now_ns()
        self.first_token_ns = None
        self.token_timestamps_ns = []
        self.requests = []
        self.config = config
        self.model_load_seconds = model_load_seconds
        self.prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        self.prompt_tokens = None
        self.session_metadata = {}
        self.setup = {}
        self.status = "running"
        self.error = None

    def mark_tokens(self, count):
        timestamp = now_ns()
        if count and self.first_token_ns is None:
            self.first_token_ns = timestamp
        self.token_timestamps_ns.extend([timestamp] * count)

    def add_request(self, request):
        self.requests.append(request)

    def finish(
        self,
        generated_tokens,
        edge_telemetry,
        status="completed",
        error=None,
        end_ns=None,
    ):
        end_ns = now_ns() if end_ns is None else end_ns
        self.status = status
        self.error = error
        token_times = self.token_timestamps_ns
        itl_ms = [
            (second - first) / 1_000_000
            for first, second in zip(token_times, token_times[1:])
        ]
        outgoing = sum(item["bytes"]["frame_out"] for item in self.requests)
        incoming = sum(item["bytes"]["frame_in"] for item in self.requests)
        raw = sum(item["bytes"]["raw_activation"] for item in self.requests)
        encoded = sum(item["bytes"]["encoded_activation"] for item in self.requests)
        kv_migration = sum(
            item["bytes"].get("kv_migration", 0) for item in self.requests
        )
        total_ms = elapsed_ms(self.start_ns, end_ns)
        return {
            "schema_version": 1,
            "timestamp_unix": time.time(),
            "status": status,
            "error": error,
            "prompt_hash": self.prompt_hash,
            "configuration": {
                "model": self.config.model_name,
                "split_inference": self.config.split_inference,
                "edge_layers": self.config.edge_layers,
                "prefill_edge_layers": self.config.prefill_edge_layers,
                "decode_edge_layers": self.config.decode_edge_layers,
                "chunked_prefill": self.config.chunked_prefill,
                "prefill_chunk_size": self.config.prefill_chunk_size,
                "stream_kv_layers": self.config.stream_kv_layers,
                "kv_transfer_dtype": self.config.kv_transfer_dtype,
                "speculative_decoding": self.config.speculative_decoding,
                "edge_compute_dtype": str(self.config.torch_dtype),
                "activation_dtype": self.config.activation_dtype,
                "edge_device": self.config.device,
            },
            "tokens": {
                "prompt": self.prompt_tokens,
                "generated": generated_tokens,
            },
            "latency": {
                "model_load_ms": self.model_load_seconds * 1000,
                "ttft_ms": (
                    elapsed_ms(self.start_ns, self.first_token_ns)
                    if self.first_token_ns else None
                ),
                "total_generation_ms": total_ms,
                "output_tokens_per_second": (
                    generated_tokens / (total_ms / 1000) if total_ms else None
                ),
                "inter_token_ms": itl_ms,
                "inter_token_summary_ms": _summary(itl_ms),
            },
            "transport": {
                "edge_to_cloud_bytes": outgoing,
                "cloud_to_edge_bytes": incoming,
                "total_application_bytes": outgoing + incoming,
                "raw_activation_bytes": raw,
                "encoded_activation_bytes": encoded,
                "kv_migration_bytes": kv_migration,
                "activation_compression_ratio": raw / encoded if encoded else None,
                "excludes_tcp_tls_websocket_overhead": True,
            },
            "edge_summary": edge_telemetry,
            "cloud_summary": self._cloud_summary(),
            "cloud_session": self.session_metadata,
            "setup": self.setup,
            "requests": self.requests,
        }

    def _cloud_summary(self):
        numeric = {}
        for request in self.requests:
            for key, value in request.get("cloud", {}).items():
                if isinstance(value, (int, float)):
                    numeric.setdefault(key, []).append(value)
        return {key: _summary(values) for key, values in numeric.items()}


def append_jsonl(path, record):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def encoded_activation_bytes(hidden, dtype):
    count = hidden.numel()
    if dtype == "fp32":
        return count * 4
    if dtype == "fp16":
        return count * 2
    if dtype == "int4":
        return (count + 1) // 2
    raise ValueError(f"Unknown activation dtype: {dtype}")
