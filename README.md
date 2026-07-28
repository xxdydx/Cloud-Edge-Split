# Cloud Edge Split

Research project exploring vertical LLM inference partitioning between edge and cloud devices to reduce edge compute while maintaining/reducing latency.

## Configuration

Edge inference settings live in `config.py`. In particular:

- Set `speculative_decoding` to `True` to use batched draft verification, or `False` to use the original token-by-token path.
- Set `prefill_edge_layers` and `decode_edge_layers` to the edge layer counts for prompt processing and token decoding. They must satisfy `0 < prefill_edge_layers <= decode_edge_layers < total_layers`.
- `chunked_prefill=True` divides prompts into `prefill_chunk_size` token chunks (64 by default).
- `overlap_kv_transfer=True` streams each migrated layer's KV frame while the cloud computes subsequent layers. `kv_transfer_queue_depth` bounds buffered frames; set `overlap_kv_transfer=False` for the synchronous comparison path.
- `edge_overlap_quantization="mixed"` enables selective quantisation of the decode-only edge layers described below. `edge_quantization_backend="auto"` uses TorchAO when available and records any fallback in the benchmark metadata.
- Set `num_draft_tokens` to the maximum speculative block size.
- Set `max_new_tokens`, or the `MAX_NEW_TOKENS` environment variable, to the generation limit selected by the edge and sent to the cloud for each prompt.
- Set `warmup_on_start` to `True` to warm the edge and cloud kernels before accepting the first prompt.
- Set `benchmark_enabled` to record per-generation and per-request metrics in `benchmark_output` (default: `benchmarks/results.jsonl`).

Start the cloud server with `AUTH_TOKEN=<your-ngrok-token> python CLOUD.py`, then copy its printed Public URL into the edge process:

```bash
CLOUD_URL=https://your-current-ngrok-url.ngrok-free.app python3 edge_client.py
```

The edge model is loaded once and the process waits for prompts until `/quit`, `/exit`, EOF, or Ctrl-C. Each prompt receives fresh edge and cloud KV caches while the model weights and WebSocket connection remain resident.

## Benchmarking

When benchmarking is enabled, every prompt prints a short TTFT, inter-token latency, throughput, and byte-count summary. A full JSONL record is appended to `benchmarks/results.jsonl`, including:

- TTFT, inter-token latency distribution, total generation time, and tokens/s
- Edge forward, activation encoding, WebSocket, and cloud-forward timings
- Raw/encoded activation sizes, complete binary-frame sizes, and compression ratio (TCP/TLS/WebSocket framing overhead is excluded)
- Edge process CPU/RAM and MPS allocation statistics
- Cloud CPU/RAM plus NVIDIA GPU utilization, memory, temperature, power, and energy when NVML exposes them
- Speculative acceptance and response-chunk information

On macOS, `powermetrics` energy sampling is best-effort and uses non-interactive `sudo -n`; if it is not already authorized, the JSON record marks edge energy as unsupported rather than prompting for a password or estimating a value.

## Current Implementation

Disaggregated inference—the edge device computes the LLM forward pass through the first K layers and maintains its own KV cache. The hidden states are then sent to the cloud, which computes the forward pass through the remaining N−K layers.

<!-- ### Cons of Current Implementation
- Every generated token costs one full round trip: the edge device computes K layers -> network hop -> the cloud computes N-K layers -> network hop back to the edge device.
- Two network hops are required per generated token. In measurements using a T4 GPU in the cloud and an Apple M3 CPU on the edge, generating 10 tokens with Qwen 0.5B took 4 seconds.
- Using a `Session` object means every call incurs HTTP request overhead, including the usual headers and response. This can be slow for autoregressive generation. -->


## Experiments and Optimisations

### Network Transport Layer

The initial implementation used FP32 encoding and sent hidden states from the edge to the cloud as JSON. Each value in the JSON-encoded float lists occupied approximately 15–20 ASCII bytes, making transfer extremely slow.

The JSON representation was replaced with raw binary encoding of the activation tensor and further optimised through reduced-precision quantisation.
- FP16: Native half-precision floating point, which halves the bit width.
- INT4: A per-row scale is calculated as max(|row|) / 7. Each value is stored as a 4-bit signed integer using round(value / scale), with two values packed per byte. The scale is sent alongside the payload so the cloud can dequantise it using value ≈ int_value × scale.

<u>Benchmark: 5-token boundary tensor, Qwen2.5-0.5B</u>

| Encoding | Bytes | vs. old JSON |
|---|---|---|
| Old: JSON float list | 92,613 | 1× |
| New: binary, FP32 | 17,950 | 5.2× smaller |
| New: binary, FP16 | 8,990 | 10.3× smaller |
| New: binary, INT4 | 2,290 | **40.4× smaller** |

### Device-Specific Model Residency
Each device only holds the portion of the model it is responsible for computing. For example, if K=4, the edge device holds layers 0 through K−1, while the cloud device holds layers K through N−1.

This reduces resident device memory because unneeded layers are discarded on each device.

However, `load_partial_model()` first loads the entire model into CPU memory on each device, discards the unnecessary layers, and then moves the remaining layers to MPS/CUDA. This reduces final device memory usage but not peak CPU memory usage during loading.

## Decode vs. Prefill Layer Split

During the initial experiments, the model was split evenly, with 14 layers on the edge and 14 layers in the cloud. These were the results:

| Workload | Edge CPU (14 layers) | Cloud GPU (14 layers) | Ratio |
|---|---:|---:|---:|
| Single-token decode forward | 60–88 ms (mean ~68 ms) | 35.9–37.0 ms (mean ~36.8 ms) | ~1.9× cloud faster |
| Prefill forward, per token (~88 tokens) | 28.5 ms/token | 0.91 ms/token | ~31× cloud faster |

For decode, the performance difference between the GPU and CPU is relatively small. The difference is substantially larger for prefill.

This motivates using different split points by phase: assign fewer prefill layers to the edge and more to the cloud to exploit GPU parallelism, while choosing the decode split mainly around edge capacity, boundary-transfer cost, and per-token network latency. This also requires reworking the model-residency logic between devices.

### Edge Overlap-Layer Quantisation

Layers between the prefill and decode splits run on the edge only during decoding and are called **overlap layers** (e.g. layers 4–11 for a 4/12 split). Their attention and FFN linear weights use INT8 and INT4 respectively to reduce edge memory.

### First Dual-K Experiment

The first experiment used `prefill_edge_layers=4`, `decode_edge_layers=12`, chunked prefill, synchronous FP16 KV migration, and CPU execution on the edge. Reducing edge prefill work improved time to first token (TTFT).

| Metric | Fixed K=14, 88-token prompt | Dual-K 4/12, 96-token prompt | Observed change |
|---|---:|---:|---:|
| TTFT | 4137.7 ms | **3070.4 ms** | **−1067.3 ms (−25.8%)** |
| Mean decode ITL | ~455 ms | **743.6 ms** | +288.6 ms (+63.4%) |
| Decode ITL range | ~422–581 ms | **355–2425 ms** | Much less stable |

Prefill was where the edge CPU had the largest disadvantage relative to the cloud GPU, so reducing edge prefill from 14 layers to 4 outweighed the added KV-migration cost.

#### KV-Transfer Validation

The first 64-token chunk transferred 1,048,792 bytes of KV frames, or approximately 16,387 bytes per prompt token. The predicted FP16 payload was:

```text
2,048 bytes/layer/token × 8 migrated layers = 16,384 bytes/token
```

The roughly three-byte-per-token difference is framing metadata. This closely matches the GQA-based transfer-cost model and confirms that only the intended new KV positions were migrated.

#### Synchronous Migration Timing

| Prompt chunk | Tokens | Round total | First KV arrival | Final KV arrival | Cloud prefill compute |
|---|---:|---:|---:|---:|---:|
| 0 | 64 | 1986.9 ms | 1555.2 ms | 1986.5 ms | ~106 ms |
| 1 | 32 | 812.4 ms | 757.1 ms | 795.6 ms | ~106 ms |

For the first chunk, the first KV frame arrived after 1555.2 ms, about 78% of the complete chunk round. KV extraction itself took only about 19 ms. The current implementation sends each layer's KV synchronously before computing the next layer, so CPU copying, serialisation, WebSocket/tunnel delay, and network transfer remain exposed in TTFT rather than overlapping subsequent cloud computation.

The benchmark field `kv_migration_bandwidth_bytes_per_second` should not be interpreted as raw link bandwidth. It divides total KV bytes by complete prefill wall time, which also includes compute, queuing, serialization, and tunnel latency.

#### Decode Instability

The same run showed a decode regression: mean ITL increased from roughly 455 ms to 743.6 ms, with individual steps reaching 932 ms and 2425 ms. Telemetry during the spikes showed process RSS approaching 5.75 GB and system memory usage reaching 84.7%, so memory pressure is a plausible contributor.

Potential sources of the observed memory-pressure pattern include:

- **macOS compressed memory and swap:** macOS compresses cold pages under pressure, which can reduce reported RSS while consuming CPU and adding latency. Keeping the process working set smaller is the main application-level mitigation.
- **PyTorch CPU allocator behavior:** freed tensor blocks may be cached and later returned to the OS in batches, producing an allocation-and-release sawtooth that is not aligned with request boundaries.
- **Python garbage collection:** short-lived Python objects can accumulate until a generational GC threshold is reached, causing a periodic collection pause and batch release.
- **Dynamic KV-cache growth:** append operations based on `torch.cat` temporarily retain both the old and enlarged buffers, creating repeated allocation spikes before the old storage is reclaimed.
- **WebSocket buffering:** networking libraries may grow and shrink internal buffers with message traffic, although this is unlikely to explain gigabyte-scale RSS changes by itself.
- **Unrelated system pressure:** other applications on the shared machine can trigger global reclaim; process RSS and whole-system memory usage should therefore be evaluated separately.

### Second Dual-K Experiment: Overlapped KV Transfer

The synchronous migration in the first experiment fully exposed CPU copy, serialisation, and tunnel transfer time inside TTFT — nothing overlapped with subsequent layer compute. The natural next step was to test whether overlapping the KV send with ongoing cloud computation (via pinned buffers and a bounded in-flight queue) would recover a meaningful fraction of that exposed time.

#### Queue depth = 1 (config not yet effective)

The first several overlap runs were executed with `kv_transfer_queue_depth=1` still in effect at the transport layer — confirmed directly, since `kv_max_queue_depth` reported `min=1, max=1` in every run despite `overlap_kv_transfer=true`. This gave a controlled comparison: overlap machinery enabled, but with no spare buffer slots for it to use.

| Run | Chunk 0 `first_kv_arrival` | Chunk 0 compute (`cloud_prefill_ms` + `kv_extraction_ms`) |
|---|---:|---:|
| 1 | 1826 ms | ~163 ms |
| 2 | 1673 ms | ~89 ms |
| 3 | 1548 ms | ~90 ms |
| 4 | 1560 ms | ~91 ms |
| 5 | 1676 ms | ~90 ms |
| 6 | 1586 ms | ~87 ms |

Across six repeated runs, `first_kv_arrival` stayed within a tight 1548–1826 ms band, indistinguishable from the synchronous baseline (1555 ms). TTFT likewise showed no consistent improvement (3142–3623 ms overlap runs vs. 3070 ms synchronous baseline). The overlap flag being enabled had no measurable effect while the buffer pool was limited to one slot.

#### Queue depth = 4

Raising `kv_transfer_queue_depth` to 4 and confirming the change took effect (`kv_max_queue_depth` now varied `min=1, max=4` across chunks) isolated whether the bottleneck was local buffer contention.

| Metric | Depth = 1 | Depth = 4 |
|---|---:|---:|
| Chunk 0 `first_kv_arrival` | 1524–1826 ms | 1466 ms |
| `kv_transfer_drain_ms` | 0.0 ms | 4.86 ms |
| TTFT | 3070–3623 ms | 3259.8 ms |

Increasing queue depth produced a small, real effect — `kv_transfer_drain_ms` moved off zero for the first time, indicating some local queuing wait was genuinely eliminated. But `first_kv_arrival` remained within the same band as every depth = 1 run, and TTFT showed no consistent improvement.

#### Byte-scaling analysis

Comparing the two prefill chunks within a single run isolates whether the migration cost scales with payload size or is dominated by a fixed per-message cost:

| Chunk | Tokens | KV bytes | `kv_frames_sent` | `first_kv_arrival` |
|---|---:|---:|---:|---:|
| 0 | 64 | 1,048,792 | 8 | ~1466–1826 ms |
| 1 | 32 | 524,504 | 8 | ~757–1170 ms |

Frame count is identical (one message per migrated layer, 8 layers) across both chunks, but arrival time scales roughly with payload size. 



### Speculative Decoding
- Use a draft model to compute the first K layers on the edge device. Draft up to N tokens at a time and send them to the cloud for verification.
- In the cloud, verify the full pass consisting of the initial prompt tokens and N newly generated tokens. Use the resulting logits to determine whether the N draft tokens match the verifier model's predictions. When the verifier disagrees with the draft model, stop the chain, retain the accepted tokens plus the bonus token, and have the draft model generate another N tokens.
- This can be faster than the standard implementation because it reduces network trips to the cloud verifier. A potential disadvantage is poor draft quality when the edge model uses too few layers, which requires calibration of K and N.
- Experiments showed limited effectiveness. With 20 or fewer edge layers, the draft-token acceptance rate was at most 30%. With 24 edge layers, the acceptance rate reached 100% and provided a modest improvement (3.1 seconds versus 4.7 seconds), but this primarily resulted from fewer batched network requests rather than effective speculative decoding.

**Improvements**
- Achieving a tangible improvement would require fewer than 12 edge layers, but this is impractical because the final LM head was not trained to decode intermediate representations reliably.
- A meaningful improvement would require training a separate auxiliary early-exit head or using a cheaper draft model and a larger cloud verifier model.


## To-dos

### Join prefill/decode split selection
- Profiling each layer on both devices, for prefill/decode time, weight & KV memory, boundary-trf cost (time & bytes) for diff token lengths
- T_req = T_prefill(s_p) + T_handoff(s_p, s_d) + N x T_decode(s_d)

### Break-even-aware switching
do only if decode savings recover the handoff cost

N_breakeven = T_handoff / (T_decode(s_p) - T_decode(s_d))

### Incremental KV streaming
Divide prefill into token chunks and migrate concurrently with subsequent layer of prefill compute

### Transfer vs remat
Transfer when networking is faster, replay when edge compute is faster

### Hybrid suffix remat
Keep received KV pages, identify the missing causal suffix, replay only that suffix on the Mac

### Deadline-aware scheduling
Each KV chunk has a deadline,
