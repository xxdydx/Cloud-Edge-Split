# Cloud Edge Split

Research project exploring vertical LLM inference partitioning between edge and cloud devices to reduce edge compute while maintaining/reducing latency.

## configuration

Edge inference settings live in `config.py`. In particular:

- Set `speculative_decoding` to `True` to use batched draft verification, or
  `False` to use the original token-by-token path.
- Set `edge_layers` to the number of transformer layers executed on the edge.
- Set `num_draft_tokens` to the maximum speculative block size.

Restart the edge client after changing the configuration. Start the cloud
server with `AUTH_TOKEN=<your-ngrok-token> python CLOUD.py`; its `/verify`
endpoint is used automatically when speculative decoding is enabled.

## current implementation

disaggregated inference — edge device will compute the forward pass of the LLM
up to first K layers. edge device will have its own KV cache. then, the hidden
states will be sent up to the cloud, and forward pass for remaining N-K layers
will be computed.

## to-do items
- replace JSON float lists with quantised (fp16/fp4/int4) binary activation transfer
- boundary activation compression
- replace HTTP requests w/ persistent websockets
- cloud shd only load layers K...N-1. same for edge, 0...N
- diff K split for prefill & decode

- benchmarks: TTFT, Inter-token latency, boundary activation size, total gen time, bytes transferred, edge mem/energy util, cloud util, gpu/cpu utils

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

### speculative decoding 
- use draft model, compute first K layers on edge device. draft up to N tokens each time, and send over to cloud device to verify.
- on cloud device, verify the full pass (Initial Prompt Tokens + N newly generated tokens). from logits generated, verify if the N tokens match with the verifier model's prediction probability distributions. at the point where the verifier model disagrees with draft model, stop the chain, take the accepted tokens + bonus token and get draft model to generate N tokens again. 
- faster than current implementation as this reduces the network hops done to the verifier model in cloud. potential con is if the draft model in edge device is bad and doesn't predict the tokens well as compared to verifier model. that will require calibration of parameters K and N.
- from experiments, seems pretty useless. when trying with edge layers <= 20, acceptance rate for draft tokens is always <= 30%. if trying with edge layers = 24, acceptance rate is 100%, provides a modest improvement (3.1s vs 4.7s), but that's due to the reduced number of network trips (due to batched network requests), rather than the efficacy of speculative decoding.

**improvements**
- for it to see tangible improvement, edge layers shd be reduced to less than 12, but not practical as the final LM head was not trained to decode intermediate representations reliably.
- a meaningful improvement wld require training a separate auxillary early-exit head, or using a cheaper draft model and verify using larger model on cloud.





