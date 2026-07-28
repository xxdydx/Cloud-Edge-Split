"""Portable, capability-driven quantization for edge-only overlap layers."""

import platform
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

try:
    import psutil
except ImportError:
    psutil = None


@dataclass(frozen=True)
class QuantizationCapabilities:
    backend: str
    int8_linear: bool
    int4_weight_only_linear: bool
    requires_float32_activations: bool


@dataclass
class QuantizationResult:
    requested_mode: str
    requested_attention_bits: int
    requested_ffn_bits: int
    effective_attention_bits: int
    effective_ffn_bits: int
    backend: str
    overlap_start_layer: int
    overlap_end_layer: int
    attention_parameters: int
    ffn_parameters: int
    estimated_weight_bytes_before: int
    estimated_weight_bytes_after: int
    quantization_ms: float
    process_rss_bytes_before: int | None
    process_rss_bytes_after: int | None
    fallback_reason: str | None = None

    def to_dict(self):
        return asdict(self)


class QuantizationBackend:
    capabilities = QuantizationCapabilities(
        backend="none",
        int8_linear=False,
        int4_weight_only_linear=False,
        requires_float32_activations=False,
    )
    effective_attention_bits = 16
    effective_ffn_bits = 16

    def quantize_attention(self, module):
        return module

    def quantize_ffn(self, module):
        return module


class NoQuantizationBackend(QuantizationBackend):
    def __init__(self, reason=None):
        self.reason = reason


class PyTorchDynamicInt8Backend(QuantizationBackend):
    """Dynamic INT8 using the portable QNNPACK/FBGEMM CPU engines."""

    effective_attention_bits = 8
    effective_ffn_bits = 8

    def __init__(self):
        engines = set(torch.backends.quantized.supported_engines)
        machine = platform.machine().lower()
        preferred = (
            ("qnnpack", "fbgemm")
            if machine in {"arm64", "aarch64"}
            else ("fbgemm", "qnnpack")
        )
        engine = next((item for item in preferred if item in engines), None)
        if engine is None:
            raise RuntimeError("no PyTorch dynamic INT8 CPU engine is available")
        torch.backends.quantized.engine = engine
        self.engine = engine
        self.capabilities = QuantizationCapabilities(
            backend=f"pytorch_dynamic_int8:{engine}",
            int8_linear=True,
            int4_weight_only_linear=False,
            requires_float32_activations=True,
        )

    @staticmethod
    def _quantize(module):
        def replace_linears(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    # Pack one projection at a time to avoid materializing an
                    # entire multi-projection block in FP32 simultaneously.
                    wrapper = nn.Sequential(child.float())
                    torch.ao.quantization.quantize_dynamic(
                        wrapper,
                        {nn.Linear},
                        dtype=torch.qint8,
                        inplace=True,
                    )
                    setattr(parent, name, wrapper[0])
                else:
                    replace_linears(child)

        replace_linears(module)
        return module

    def quantize_attention(self, module):
        return self._quantize(module)

    def quantize_ffn(self, module):
        return self._quantize(module)


class TorchAOMixedBackend(QuantizationBackend):
    """Optional exact INT8-attention/INT4-FFN TorchAO backend."""

    effective_attention_bits = 8
    effective_ffn_bits = 4

    def __init__(self, device, group_size):
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            quantize_,
        )

        self.device = device
        self.group_size = group_size
        self._quantize = quantize_
        self._attention_config = Int8DynamicActivationInt8WeightConfig()
        self._ffn_config = Int4WeightOnlyConfig(group_size=group_size)
        self.capabilities = QuantizationCapabilities(
            backend="torchao_mixed_int8_int4",
            int8_linear=True,
            int4_weight_only_linear=True,
            requires_float32_activations=False,
        )
        self._probe()

    def _probe(self):
        """Reject installations which import but lack usable target kernels."""
        attention = nn.Sequential(nn.Linear(128, 128, bias=False)).to(
            self.device
        )
        ffn = nn.Sequential(nn.Linear(128, 128, bias=False)).to(self.device)
        sample = torch.randn(1, 1, 128, device=self.device)
        self._quantize(attention, self._attention_config)
        self._quantize(ffn, self._ffn_config)
        with torch.no_grad():
            attention(sample)
            ffn(sample)

    def quantize_attention(self, module):
        self._quantize(module, self._attention_config)
        return module

    def quantize_ffn(self, module):
        self._quantize(module, self._ffn_config)
        return module


def _module_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _estimated_bytes(parameter_count, bits):
    return (parameter_count * bits + 7) // 8


def select_quantization_backend(config):
    """Resolve the strongest backend supported by this edge device."""
    mode = config.edge_overlap_quantization
    if mode == "none":
        return NoQuantizationBackend(), None

    exact_mixed_requested = (
        config.edge_overlap_attention_bits == 8
        and config.edge_overlap_ffn_bits == 4
    )
    requested_backend = config.edge_quantization_backend
    torchao_error = None

    if requested_backend in {"auto", "torchao"} and exact_mixed_requested:
        try:
            return TorchAOMixedBackend(
                config.device,
                config.edge_int4_group_size,
            ), None
        except Exception as error:
            torchao_error = str(error)
            if requested_backend == "torchao" and not config.allow_quantization_fallback:
                raise RuntimeError(
                    f"requested TorchAO mixed quantization is unavailable: {error}"
                ) from error

    if requested_backend not in {"auto", "pytorch", "torchao"}:
        raise ValueError(
            f"unknown edge quantization backend: {requested_backend}"
        )

    if config.device != "cpu":
        reason = (
            f"portable PyTorch INT8 fallback only supports CPU, got "
            f"{config.device}"
        )
        if not config.allow_quantization_fallback:
            raise RuntimeError(reason)
        return NoQuantizationBackend(reason), reason

    try:
        backend = PyTorchDynamicInt8Backend()
    except RuntimeError as error:
        if not config.allow_quantization_fallback:
            raise
        return NoQuantizationBackend(str(error)), str(error)

    fallback_reason = None
    if config.edge_overlap_ffn_bits == 4:
        fallback_reason = (
            "efficient INT4 backend unavailable; FFN fell back to INT8"
        )
        if torchao_error:
            fallback_reason += f" ({torchao_error})"
        if not config.allow_quantization_fallback:
            raise RuntimeError(fallback_reason)
    return backend, fallback_reason


def quantize_overlap_layers(model, config):
    """Quantize only layers used on edge decode but not edge prefill."""
    started = time.perf_counter_ns()
    process = psutil.Process() if psutil else None
    rss_before = process.memory_info().rss if process else None
    start = config.prefill_edge_layers
    end = config.decode_edge_layers
    mode = config.edge_overlap_quantization
    backend, fallback_reason = select_quantization_backend(config)

    if not 0 <= start <= end <= len(model.model.layers):
        raise ValueError("overlap quantization layer range is out of bounds")

    attention_parameters = 0
    ffn_parameters = 0
    for layer in model.model.layers[start:end]:
        attention_parameters += _module_parameters(layer.self_attn)
        ffn_parameters += _module_parameters(layer.mlp)

    if mode != "none" and start < end:
        for layer in model.model.layers[start:end]:
            if config.edge_overlap_attention_bits < 16:
                backend.quantize_attention(layer.self_attn)
            if config.edge_overlap_ffn_bits < 16:
                backend.quantize_ffn(layer.mlp)

    capabilities = backend.capabilities
    uses_quantized_modules = (
        mode != "none"
        and start < end
        and (
            config.edge_overlap_attention_bits < 16
            or config.edge_overlap_ffn_bits < 16
        )
    )
    if capabilities.requires_float32_activations and uses_quantized_modules:
        model._edge_quantization_start_layer = start
        model._edge_quantization_activation_dtype = torch.float32

    effective_attention_bits = 16
    effective_ffn_bits = 16
    if mode != "none" and start < end:
        if config.edge_overlap_attention_bits < 16:
            effective_attention_bits = backend.effective_attention_bits
        if config.edge_overlap_ffn_bits < 16:
            effective_ffn_bits = backend.effective_ffn_bits
    before = _estimated_bytes(
        attention_parameters + ffn_parameters,
        16,
    )
    after = (
        _estimated_bytes(attention_parameters, effective_attention_bits)
        + _estimated_bytes(ffn_parameters, effective_ffn_bits)
    )
    result = QuantizationResult(
        requested_mode=mode,
        requested_attention_bits=config.edge_overlap_attention_bits,
        requested_ffn_bits=config.edge_overlap_ffn_bits,
        effective_attention_bits=effective_attention_bits,
        effective_ffn_bits=effective_ffn_bits,
        backend=capabilities.backend,
        overlap_start_layer=start,
        overlap_end_layer=end,
        attention_parameters=attention_parameters,
        ffn_parameters=ffn_parameters,
        estimated_weight_bytes_before=before,
        estimated_weight_bytes_after=after,
        quantization_ms=(time.perf_counter_ns() - started) / 1_000_000,
        process_rss_bytes_before=rss_before,
        process_rss_bytes_after=(
            process.memory_info().rss if process else None
        ),
        fallback_reason=fallback_reason or getattr(backend, "reason", None),
    )
    return result.to_dict()
