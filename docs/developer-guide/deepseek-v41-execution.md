# DeepSeek V4.1 execution implementation

The V4.1 integration now includes arithmetic, streamed weight access, packed
cache storage, a built-in A5 backend, TP/EP rank execution, DSpark and image
requests. This is a correctness-oriented implementation prepared without an
A5 machine. Source completeness and CPU tests do not establish real-checkpoint
accuracy, A5 runtime compatibility, model-serving acceptance or performance.

The pinned reference is
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference),
revision `dba1be0a40aa45a94ad051997016db3960a90277`. The earlier
[framework report](deepseek-v41-framework.md) describes revision `9080f6f`;
its missing-arithmetic statements are historical.

## Implemented computation

| Component | Source | Behavior |
|---|---|---|
| Quantized linear | `numerics.py`, `tensor_store.py` | Decode normalized E4M3/E2M1 values; FP32 dot products over K32 blocks; activation scale then weight scale; FP32 accumulation and reference output cast |
| PyPTO Cube | `kernels.py`, `pypto_ops.py` | Actual JIT BF16 matrix kernel, FP32 accumulation, bounded padding, shape LRU and instance-owned cleanup |
| Hyper-Connections | `numerics.py` | Pre/post coefficients, residual orientation, Sinkhorn normalization and delayed pre-mix use |
| CSA2 | `attention.py` | Full/Reindex/Reuse, source ownership, hierarchical candidates, rotary/YaRN, sink softmax, grouped output projections and arbitrary prefill boundaries |
| MoE | `numerics.py` | Selection bias, original-score routing weights, top-k normalization, asymmetric SwiGLU clamps, FP4 experts and shared expert |
| Engram | `engram.py`, `numerics.py`, `tensor_store.py` | Token history/hash, sparse row reads, TP row ownership, projection, signed-square-root gate and image exclusion |
| Cache | `cache.py`, `packed_cache.py` | Scheduler page mapping, actual FP8/FP4 payload/scale bytes, complete compressed rows, SWA ring and reversible writes |
| Serving | `backend.py`, `npu_runner.py`, `npu_executor.py` | All backbone layers, terminal prefill, greedy/stochastic autoregressive decode, request generations and atomic failure recovery |
| Multi-rank | `distributed.py` | Spawned rank workers, HCCL, reference TP+EP, mirrored page metadata, finite RPC deadlines and owned-process cleanup |
| DSpark | `draft.py`, `dspark.py` | Three stages, target attention-input states, temporary bidirectional draft KV, Markov correction, confidence, bounded greedy verification and recovery |
| Vision | `vision.py`, `vision_chat.py` | Resize/patchify, 2D rotary ViT, aligner, span embeddings and OpenAI image content through existing HTTP/Engine/Scheduler/Worker |

The official forward executes all 40 backbone layers for both prefill and
decode. Compression ratios do not define a 20/20 encoder/decoder split. Engram
is mandatory in text inference. Neither behavior is removed to fit the older
adaptation plan's phase boundaries.

## Precision and implementation limits

Dense weights remain FP8 with 32x32 UE8M0 scales; routed experts remain packed
FP4 E2M1 with one UE8M0 scale per row/K32 block. FP4 uses the low nibble first.
The normalized values are representable in BF16, so the initial PyPTO kernel
multiplies explicitly decoded BF16 values and accumulates FP32. Scale application
occurs after each K32 dot product. This does not reinterpret FP4 as FP8 or claim
a native mixed FP8-by-FP4 instruction.

The A5 backend uses Torch-NPU for FP32-sensitive and vector arithmetic, and
PyPTO for BF16 matrix products. The bridge copies inputs/results through host
memory. Small K32 launches and repeated checkpoint reads are expensive. Fusion,
device-resident staging, overlap and performance tuning remain future hardware
work; this implementation makes no throughput improvement claim.

`DeepSeekV41TensorStore` uses actual safetensors slices, checks source metadata
and finite values, then owns only the returned bytes. It streams matrix tiles
and performs selected Engram/vocabulary row reads. The default per-operation
tensor-buffer budget is 256 MiB; the decoded embedding LRU is separately bounded
at 64 MiB. `placement="uncached"` disables row retention. Neither mode loads the
full Engram table or promises HBM-resident weights. Header/index Python objects,
compiler workspaces and caller-retained outputs are separate memory costs.

The source index includes text, vision and draft schemas. The original
`DeepSeekV41WeightLoader` remains available for bounded inspection and selected
layer conversion. Production arithmetic uses the streaming tensor store.

## Cache and transactions

The scheduler is the only page allocator. Every rank stores the same logical
cache groups and receives the scheduler's page tables. `PackedRows` resolves
each physical read/write through the active `CacheTransaction`. Persistent SWA
rows use actual packed bytes; no extra persistent BF16 history masquerades as
the packed cache. Compressed rows publish only when the complete source group
has arrived, including groups spanning prefill chunks.

Ordinary batch abort restores pages, compressor partials, SWA endpoints, draft
state and Engram history. An outer speculative checkpoint preserves the state
before all sequential verification steps. Inner abort/retry cannot mutate that
outer snapshot. Visibility restoration occurs only after tensor restoration.
Failed restoration or rank death makes the executor unusable instead of allowing
subsequent requests to read uncertain state. Prefix caching remains disabled:
immutable page sharing, copy-on-write and auxiliary-state prefix reuse are a
separate optimization, not required for this request lifecycle.

## A5 startup

Use the repository's normal PyPTO/CANN environment with Torch-NPU, safetensors
with native E4M3/UE8M0 dtype support, NumPy, tokenizers and the complete pinned
checkpoint/tokenizer. Pillow is additionally required for image preprocessing.
The repository does not provision the roughly 510 GB checkpoint automatically.

The CLI selects the built-in V4.1 backend. A custom
`--v41-kernel-factory MODULE:CALLABLE` remains an optional trusted-code override.
No CPU fallback runs when an A5 launch fails.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pypto-serving \
  --model /path/to/DeepSeek-V4.1-Flash \
  --platform a5 --devices 0,1,2,3,4,5,6,7 --dp 1 --tp 8 --ep 8 \
  --max-model-len 9216 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --prompt 'Explain sparse attention.' \
  --generate-config '{"max_new_tokens":128,"temperature":0.0}'
```

These are commands for future A5 bring-up, not recorded successful runs. With
compatible checkpoint dimensions, one rank is also supported. TP ranks must
divide output groups, attention/index heads and vocabulary. EP uses global
expert IDs and the same group; no V4-specific TP4/DP4 topology is inherited.

For HTTP, omit `--prompt` and add `--port 8899`. The same worker, scheduler,
streaming usage and release paths execute both text and image requests.
Multi-rank startup/RPC deadlines are configured with
`PYPTO_V41_INIT_TIMEOUT` and `PYPTO_V41_RPC_TIMEOUT` in positive seconds
(defaults 300 and 600). Increase them explicitly for slow initial compilation;
zero does not disable timeouts. Persistent cross-launch compiled caching is not
implemented by this bridge; the shape cache lasts for the backend lifetime.

## DSpark scheduling

Enable greedy speculation with:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

The target first consumes the pending token and emits the next anchor. DSpark
reads the already seeded target-state window and proposes a block. Sequential
target forwards verify the proposed prefix until the first mismatch. The last
emitted correction stays pending; rejected drafts never enter committed target
cache/history. An all-match round emits up to the reserved token capacity.
This initial sequential verifier is correct-by-construction scheduling work,
not an accelerated speculative implementation.

`--v41-draft-confidence-threshold FLOAT` optionally stops verification at the
first raw confidence score below the supplied floor. There is no default
calibration or conversion of scores into probabilities. Without a floor, the
reserved capacity is used. Positive-temperature batches use the existing normal
sampler; the greedy verifier does not approximate stochastic rejection sampling.
Near the maximum sequence length, insufficient temporary draft space causes an
ordinary single-token step.

`executor.diagnostics()` reports committed speculation counters and backend
memory observations. Counter updates are discarded with a failed outer round.
Actual packed-page bytes, configured capacity and Torch-NPU allocator counters
are separate quantities; CPU runs report no NPU memory measurement. The PyPTO
runtime may own additional device memory outside Torch's allocator.

## Image requests

OpenAI chat `content` may contain `text` and `image_url` blocks. Images use bounded
`data:image/...;base64,...` URLs. Arbitrary server files and remote image fetching
are not accepted by this preprocessing path. A prompt is encoded once, image
markers expand to the official span, and versioned metadata travels with the
request registration. Input token counts use the expanded IDs.

All image spans must fit the first prefill chunk, as required by the pinned
reference. Admission validates this before streaming starts. If the current
scheduler step lacks enough room, the request waits for sufficient capacity;
an intrinsically oversized first chunk is rejected. Later text-only chunks may
continue normally. Vision transport validates dimensions, bytes, normalized
pixels, token types and image-marker IDs before model execution.

## Reproducible validation and benchmark

Run the bounded CPU suite with an environment containing Torch, safetensors,
NumPy, Pillow, tokenizers and the normal serving dependencies:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest --confcutdir=tests/unit/model/deepseek_v41 \
  tests/unit/model/deepseek_v41 -q
```

The tests distinguish miniature random-checkpoint arithmetic, captured real
quantization bytes, independent dtype/formula goldens, real CPU collectives and
recording protocol backends. Dependency skips are not numerical passes. The
kernel tests require actual PyPTO compiler bindings; source parsing alone does
not establish IR generation or device execution.

`pypto-benchmark-http` records TTFT, TPOT, request latency, throughput, success
rate and optional SLO goodput from real streaming requests. It requires server
usage counts and a complete stream. Workload JSONL contains one `prompt` or
`messages` object per line; prompt text is not copied into the result artifact.

```bash
pypto-benchmark-http --endpoint http://127.0.0.1:8899/v1/completions \
  --model DeepSeek-V4.1-Flash --prompts workloads/s1.jsonl \
  --max-tokens 128 --requests 1 --concurrency 1 \
  --server-manifest artifacts/server-manifest.json \
  --output artifacts/s1-http.json
```

The server manifest should record checkpoint revision, config digest, serving/
PyPTO/runtime commits, actual devices/toolchain, startup and warmup procedure,
allocator peaks and device-wide memory observations. Missing provenance remains
missing; the client does not invent it. Use actual prompt usage to verify the
8K/128/C=1 workload and use explicit SLOs for goodput. Run OOM/capacity sweeps
only on available authorized A5 hardware, with separately recorded failures.

Real checkpoint layer comparisons, 64-to-128-token golden generation, A5 EP8
collectives, HTTP model inference, long-context capacity, peak device-wide HBM
and the formal M0/P0 acceptance remain unverified until weights and A5 hardware
are available. Performance measurements must use matched workloads on the same
machine; none is inferred from CPU unit tests.
