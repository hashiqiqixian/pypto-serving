# DeepSeek V4.1 execution implementation

The subsequent A3/A5 branch supports both `--platform a2a3` (the default,
including A3) and explicit `--platform a5`. Platform selection is passed through
the executor, single-rank or distributed backend, and PyPTO RunConfig; a backend
reporting another platform is rejected. Both targets use the same checkpoint
and packed-cache contracts. The A5-only descriptions and validation snapshot
below record the earlier `9a73a7a` implementation. For current bring-up commands
and evidence, see [Ascend platform validation](deepseek-v41-ascend.md).

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

The reference leaves equal-score `torch.topk` ordering unspecified. This adapter
breaks cutoff ties by the smallest absolute position or candidate-block ID so
masked future columns cannot change an earlier token's selection across prefill
chunks. Scores are not perturbed. Equal-score selected indices may therefore
differ from a particular reference device's unspecified tie ordering.

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

## Validation snapshot and plan position

On 2026-09-11, implementation revision `9a73a7a` passed the bounded V4.1 CPU
suite: **530 passed, 2 skipped** in 74.55 seconds. The skipped tests required
optional downloaded references. Both were subsequently run separately: the
official ViT tensor comparison passed on the CPU validation host, and the full
draft checkpoint metadata comparison passed locally. Neither test uses complete
real model weights. The ViT oracle uses the pinned source with SHA256
`5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c`.

The CPU environment was Python 3.10.9, Torch 2.10.0+cpu, safetensors 0.8.0 and
PyPTO revision `c9af90508b674f1a568bba3f293051d3c2ef5823`, with one Torch/BLAS
thread. The two PyPTO tests generated actual A5 PTO containing BF16 Cube
operations and FP32 accumulators; assembler and device execution were disabled.
The two-rank Gloo tests exercise transport, collectives and mirrored transactions,
not complete-model TP/EP numerical agreement. The Windows dependency-limited
suite passed 293 tests with 14 skips before the final test-context correction.

Shared CLI, tokenizer, V4 model-component and serving tests at `1d05f99` reported
225 passed and two failures. Both failures reproduced at baseline `290213a`:
`test_deepseek_mtp_prefill_reads_only_selected_owner_outputs` has a test worker
without the `src_offset` argument, and
`test_worker_releases_preempted_state_before_same_command_reregistration` creates
a worker without its sampler. Revision `9a73a7a` only corrects a V4.1 test's
inference-mode mutation context; shared production code is identical. These
baseline failures remain unresolved and are not reported as passing regression.

| Plan step | Code and CPU evidence | Required acceptance still pending |
|---|---|---|
| P0-1 | Independent config, checkpoint contracts, sliced loading and quantized arithmetic tested | Real checkpoint layer load/pack/upload and A5 tensor goldens |
| P0-2 | Backbone, CSA2, MoE, Engram and packed cache arithmetic tested with miniature weights | Real model short-token goldens, A5 TP/EP and 8K prefill/decode |
| Minimum serving / P0-3 | Existing request pipeline, streaming, lifecycle and recovery integrated and tested | Real-weight A5 HTTP requests and deterministic reference agreement |
| M0 | NOT_RUN / BLOCKED by unavailable A5 and complete checkpoint | All five model/device acceptance items below |
| P1 | DSpark, Engram, vision and HTTP measurement code implemented | Real-weight goldens, accelerated speculation, prefix reuse, capacity and performance measurements |

The five M0 items remain NOT_RUN: complete real-checkpoint inference; target
multi-rank 8K prefill and single-token decode; 64-to-128-token golden plus
continuous real-model HTTP requests; a matching device artifact including peak
HBM; and S1 smoke with 8K input, 128 output and concurrency one. CPU codegen and
test artifacts do not replace any of these. P0 and the final project are not
accepted as complete.

The next device step is one real checkpoint layer with captured intermediate
tensors, followed by short-model logits, 8K chunk prefill and the M0 HTTP/S1
workload. Start with ordinary greedy decode and private pages. Prefix sharing,
native low-precision optimization and performance tuning remain separate work.
Local evidence is retained under `artifacts/deepseek-v41-implementation/`:
`progress.json`, `cpu-9a73a7a.log`, `official-vision-9a73a7a.log`,
`shared-1d05f99.log` and `baseline-two-failures-290213a.log`.
