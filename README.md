# Cloud Edge Split

Research project exploring vertical LLM inference partitioning between edge and cloud devices to reduce edge compute while maintaining/reducing latency.

## configuration

Edge inference settings live in `config.py`. In particular:

- Set `speculative_decoding` to `True` to use batched draft verification, or
  `False` to use the original token-by-token path.
- Set `prefill_edge_layers` and `decode_edge_layers` to the edge layer counts
  for prompt processing and token decoding. They must satisfy
  `0 < prefill_edge_layers <= decode_edge_layers < total_layers`.
- `chunked_prefill=True` divides prompts into `prefill_chunk_size` token
  chunks (64 by default). `kv_transfer_dtype` is currently restricted to
  exact FP16 and transfers are synchronous.
- Set `num_draft_tokens` to the maximum speculative block size.
- Set `max_new_tokens`, or the `MAX_NEW_TOKENS` environment variable, to the
  generation limit selected by the edge and sent to the cloud for each prompt.
- Set `warmup_on_start` to `True` to warm the edge and cloud kernels before
  accepting the first prompt.
- Set `benchmark_enabled` to record per-generation and per-request metrics in
  `benchmark_output` (default: `benchmarks/results.jsonl`).

Start the cloud server with `AUTH_TOKEN=<your-ngrok-token> python CLOUD.py`,
then copy its printed Public URL into the edge process:

```bash
CLOUD_URL=https://your-current-ngrok-url.ngrok-free.app python3 edge_client.py
```

The edge model is loaded once and the process waits for prompts until `/quit`,
`/exit`, EOF, or Ctrl-C. Each prompt receives fresh edge and cloud KV caches
while the model weights and WebSocket connection remain resident.

## benchmarking

When benchmarking is enabled, every prompt prints a short TTFT, inter-token
latency, throughput, and byte-count summary. A full JSONL record is appended to
`benchmarks/results.jsonl`, including:

- TTFT, inter-token latency distribution, total generation time, and tokens/s
- edge forward, activation encoding, WebSocket, and cloud-forward timings
- raw/encoded activation sizes, complete binary-frame sizes, and compression
  ratio (TCP/TLS/WebSocket framing overhead is excluded)
- edge process CPU/RAM and MPS allocation statistics
- cloud CPU/RAM plus NVIDIA GPU utilization, memory, temperature, power, and
  energy when NVML exposes them
- speculative acceptance and response-chunk information

On macOS, `powermetrics` energy sampling is best-effort and uses non-interactive
`sudo -n`; if it is not already authorized, the JSON record marks edge energy
as unsupported rather than prompting for a password or estimating a value.

## current implementation

disaggregated inference — edge device will compute the forward pass of the LLM
up to first K layers. edge device will have its own KV cache. then, the hidden
states will be sent up to the cloud, and forward pass for remaining N-K layers
will be computed.

<!-- ### cons of current implementation
- every generated token costs one full round trip, edge device computes K layers -> network hop -> cloud computes N-K layers -> network hop back to edge device.
- 2 network hops per token generated. from my measurements using T4 GPU on cloud and Apple M3 CPU on edge, took 4s to generate 10 tokens with Qwen 0.5B, which is incredibly slow.
- using `Session` object; every call has the HTTP request overhead, with the usual sending headers and response. can be slow for every call, especially with autoregressive generation. -->


## experiments/optimisations done

### network transport layer

initial implementation was using FP32 encoding + sending hidden states from edge to cloud via JSON (JSON-encoded float lists, each value sent as ASCII text, ~15-20 bytes/float). yeah i know, that's incredibly slow. 

replaced it with raw binary encoding of the activation tensor. further optimised it with quantisation; reduced per-value precision.
- fp16: native half-precision float, halves bit-width.
- int4: per-row scale computed as max(|row|) / 7, each value stored as a 4-bit signed integer (round(value / scale)), two values packed per byte. scale is sent alongside so the cloud side can dequantize (value ≈ int_value * scale).

<u>benchmark: 5-token boundary tensor, Qwen2.5-0.5B</u>

| Encoding | Bytes | vs. old JSON |
|---|---|---|
| Old: JSON float list | 92,613 | 1× |
| New: binary, fp32 | 17,950 | 5.2× smaller |
| New: binary, fp16 | 8,990 | 10.3× smaller |
| New: binary, int4 | 2,290 | **40.4× smaller** |

### device-specific model residency
each device only holds the portion of model that it is supposed to compute. for example, if K=4, edge device only holds 0 to K-1 layers, while the cloud device will hold K to N-1 layers.

this reduces resident device memory, as the unneeded layers are discarded from memory on each device.

however, `load_partial_model()` loads the full model in entirety on each device into CPU, discards the unnecessary layers before moving it to MPS/CUDA. this reduces final device memory but not peak CPU loading memory.

## decode vs prefill layer split

during initial experiments, there was an even split of 14 layers between cloud vs edge. these were the results.

| Workload | Edge CPU (14 layers) | Cloud GPU (14 layers) | Ratio |
|---|---:|---:|---:|
| Single-token decode forward | 60–88 ms (mean ~68 ms) | 35.9–37.0 ms (mean ~36.8 ms) | ~1.9× cloud faster |
| Prefill forward, per token (~88 tokens) | 28.5 ms/token | 0.91 ms/token | ~31× cloud faster |

for decode, the margin between the GPU and CPU is pretty close, but it's very much different for prefill.

this motivates using different split points by phase: assign fewer prefill layers to the edge and more to the cloud to exploit GPU parallelism, while choosing the decode split mainly around edge capacity, boundary-transfer cost, and per-token network latency. this also requires a re-working of the model residency logic between devices.


### speculative decoding 
- use draft model, compute first K layers on edge device. draft up to N tokens each time, and send over to cloud device to verify.
- on cloud device, verify the full pass (Initial Prompt Tokens + N newly generated tokens). from logits generated, verify if the N tokens match with the verifier model's prediction probability distributions. at the point where the verifier model disagrees with draft model, stop the chain, take the accepted tokens + bonus token and get draft model to generate N tokens again. 
- faster than current implementation as this reduces the network hops done to the verifier model in cloud. potential con is if the draft model in edge device is bad and doesn't predict the tokens well as compared to verifier model. that will require calibration of parameters K and N.
- from experiments, seems pretty useless. when trying with edge layers <= 20, acceptance rate for draft tokens is always <= 30%. if trying with edge layers = 24, acceptance rate is 100%, provides a modest improvement (3.1s vs 4.7s), but that's due to the reduced number of network trips (due to batched network requests), rather than the efficacy of speculative decoding.

**improvements**
- for it to see tangible improvement, edge layers shd be reduced to less than 12, but not practical as the final LM head was not trained to decode intermediate representations reliably.
- a meaningful improvement wld require training a separate auxillary early-exit head, or using a cheaper draft model and verify using larger model on cloud.
