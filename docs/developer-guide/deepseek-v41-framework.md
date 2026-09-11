# DeepSeek V4.1 host execution framework

This is the historical framework report for `9080f6f`. For the subsequent
arithmetic, packed cache, DSpark, vision and built-in backend implementation,
see [V4.1 execution](deepseek-v41-execution.md).

This increment implements host-side serving integration while A5 arithmetic
kernels are unavailable. It does not establish checkpoint-to-text inference,
model accuracy, an FP4 device cache, or performance acceptance. The earlier
[P0 evidence](deepseek-v41-p0.md) remains a historical report for its recorded
revision. This document describes the subsequent framework changes.

The reference remains `deepseek-ai/DeepSeek-V4.1-Flash` at revision
`dba1be0a40aa45a94ad051997016db3960a90277`. Configuration and tensor contracts
come from that checkpoint's config, index, shard headers, and inference sources.

## Implemented paths

| Component | Implemented behavior | Remaining device work |
|---|---|---|
| Model registration | Distinguishes V4.1 from V4 and Qwen; validates the full text index, local shard paths, config and scheduler cache topology; leaves weights lazy | Full real checkpoint provisioning and device upload |
| Chat encoding | Independent pure-text adapter matching the pinned official encoder, including merged user turns and thinking budgets | Tools, vision and other unsupported prompt fields are rejected |
| Weight loading | Reads selected tensors through `LazySafetensorsStore`; validates headers, payload dtype/shape and finite values; applies EP ownership and explicit reference TP slicing | Device packing, communication and actual low-precision matrix kernels |
| Cache state | Scheduler-owned pages, private SWA groups, shared main KV/index keys, causal visibility, complete compression rows, request generations, atomic commit and failure recovery | Tensor allocation/writes and compressor state rollback in the backend |
| Engram | Official token normalization, prime layout, PCG64 multipliers, signed int64 hashing, image boundaries, bounded per-request history and sparse row ownership | Device lookup, projections, gate and integration with actual layer arithmetic |
| Execution | Packed prefill and autoregressive decode through the existing Worker; all backbone layers, mandatory Engram before each configured layer, output validation and cancellation release | An implemented A5 kernel factory |
| DSpark | Explicit draft-prefix selection, aligned greedy target verification, EOS/budget truncation, accepted-prefix rollback across supplied state participants | Draft/target kernels, Markov state adapter, scheduling policy and serving integration |

No CED encoder/decoder split is inferred from compression ratios. The pinned
reference runs all 40 backbone layers on every forward. Vision execution, an
MTP/DSpark serving fast path, and shared prefix restoration are not enabled.

The checkpoint has no Hugging Face chat template. Its independent text adapter
merges adjacent user messages, emits the V4.1 system delimiter, and supports
official thinking budgets (`low=50`, `high=75`, `max=100`, or integers 1..100).
Unsupported tools/vision/schema inputs and unsupported effort names are rejected.
Eighteen fixed official prompt goldens and a separate 1,360-sequence differential
check establish string-encoding parity only, not real tokenizer IDs or HTTP/A5 inference.

## Scheduler and cache ownership

`build_v41_cache_group_specs` builds the existing `KVCacheGroupSpec` objects.
The serving scheduler remains the sole page allocator. The runner receives
`block_ids_by_group`, validates page ownership, and never allocates a second pool.
Reference TP and EP use the same device group and one scheduler cache partition.
The CLI requires `dp=1`, `tp=ep=device_count`, with compatible group/head sizes.
For programmatic serving, `EngineConfig.resolve_runtime_config` resolves the same
groups before scheduler construction and requires `enable_prefix_cache=False`.

The pinned model has 48 permanent cache groups:

- `swa.0` through `swa.39`: each layer owns its sliding attention cache.
- `main_kv.2`, `.8`, `.14`, `.20`: full-history compressed KV producers.
- `index_k.2`, `.8`, `.14`, `.20`: index-key producers. Layers 24, 28, 32,
  and 36 recompute index results using shared keys; they do not own new key pools.

Index results and hierarchical candidate results live only within one execution
transaction. Consumers resolve their producer through `config.layer_plan`.
A transaction exposes completed compression rows only, including groups completed
across chunk boundaries. SWA reserves space for the old attention tail and the
pending chunk. Older rollback positions may become unavailable after ring writes.

`v41_packed_v1` is this integration's explicit proposed device cache ABI:

| Group | Value bytes per row | Scale bytes per row | Rows per source-token page |
|---|---:|---:|---:|
| SWA | `head_dim` (E4M3) | `head_dim / 32` (UE8M0) | `page_size` |
| Main KV | `head_dim / 2` (packed E2M1) | `head_dim / 16` (E4M3) | `page_size / ratio` |
| Index keys | `index_head_dim / 2` (packed E2M1) | `index_head_dim / 32` (UE8M0) | `page_size / ratio` |

These are storage contracts and allocation accounting, not measured HBM usage.
Within each row, contiguous value bytes precede contiguous scale bytes. FP4
pairs use the low nibble for the earlier element; scales follow increasing
16/32-element blocks. Rows follow in logical order within a page. Backend-specific
padding must be budgeted separately and must not change the exposed row mapping.
The reference writes dequantized values back to BF16 caches through fake
quantization. A backend must explicitly implement the packed ABI; it cannot
claim compatibility merely because its values have the same quantization error.

On failure, the runner aborts the backend batch, closes cache transactions,
restores Engram snapshots, releases newly created requests, and restores existing
page bindings together. A backend recovery failure poisons the runner and requires
executor recreation. The backend must complete/cancel device writes before returning
from abort or release. Request generations prevent reused request IDs from naming
old backend state. Prefix caching remains disabled until immutable page sharing,
copy-on-write, and all auxiliary-state restoration are implemented.

## Weight conversions and limits

`DeepSeekV41WeightLoader.load_layer` and `load_globals` accept full checkpoint
tensor names. Selecting a quantized weight or scale includes its companion.
`parallel_mode="ep_only"` partitions routed experts while replicating dense weights.
`parallel_mode="reference_tp"` additionally applies reference tensor parallelism;
these modes have different dense layouts and communication requirements.

- Dense E4M3 and UE8M0 tensors retain their storage formats.
- Expert packed FP4 retains its original signed-byte representation and scales.
- `wo_a` is dequantized from FP8 blocks to BF16 and reshaped by local output group.
- Compressor weights with ratio greater than one, routing scores where required,
  and the output head receive their documented FP32 conversions.
- Engram row sharding uses ceil division and zero padding outside valid rows.

The default per-call tensor-buffer budget is 256 MiB. The estimator includes
full source tensors, outputs and conservative conversion scratch before any payload
is read. The shared store currently materializes an entire selected tensor: TP
slicing does not make a huge source tensor cheap. In particular, full Engram tables
are rejected under the default budget. `SparseEngramLookup` accepts a bounded row
reader, but an actual storage/device reader is still needed for those tables.

## Backend factory contract

The CLI exposes `--v41-kernel-factory MODULE:CALLABLE` with `--platform a5`.
No default arithmetic factory is installed. Omitting it produces an explicit
error before starting an inference worker. The factory is trusted application
code, supplied by the operator, and must clean up its allocations if construction
raises before it can return a backend.

The factory receives `config`, `runtime`, `cache_layouts`, `weight_loader`,
`device_ids`, `pypto_build_dir`, and `use_compile_cache`. It returns an object with:

- `capabilities`: `V41BackendCapabilities` with concrete world size, batch,
  chunk, sequence and layer limits; ABI version 1, A5, `v41_packed_v1`,
  `reference_tp`, Engram, and transactional behavior are required.
- `num_pages`: allocated primary-group pages. It must be a positive multiple
  of that group's full-context block count; other pools scale consistently.
  Short requests may share this capacity, so this is not a request-count limit.
- The methods described by `V41KernelBackend` in `npu_runner.py`.

For each batch, the runner prepares contexts and calls `begin_batch`. For each
request it calls `embed`, then `engram` where configured and `layer` for every
backbone layer, then `head`. The opaque state passed between methods must include
the Hyper-Connections/pre-mix state required by the reference. `head` performs the
final `hc_pre`, normalization and output projection, returning a floating
`[vocab_size]` tensor for the chunk's last token. The executor copies these rows
to owned CPU FP32 storage and rejects nonfinite results before commit.

`commit_batch` is atomic. After a failed commit, `abort_batch` must still restore
the prior tensor state, including compressed attention state and all auxiliary
state. `begin_batch` must clean up internally if it raises before returning a
ticket. The host serializes transactions using one runner lock; backend-internal
rank parallelism remains possible. `release_request` receives both ID and generation.

Terminal prefill is explicitly finalized after Worker sampling. Later decode
consumes exactly one sampled token at the committed cache position. A decode
before terminal prefill, a repeated finalization, a gap/overlap, a wrong group,
an out-of-range page, or another request's page is rejected.

## DSpark host semantics

`select_draft_prefix` validates anchor plus draft tokens and raw finite confidence
scores. The caller must choose `verify_count`; there is no invented confidence
threshold. Scores are not treated as probabilities.

`verify_greedy` takes K drafts and K+1 aligned target predictions. It accepts the
matching prefix and emits the first correction, or the extra target token when
all drafts match. The correction/bonus has not been cached and must be consumed
by the next forward. EOS and the remaining output budget truncate emissions.

`DSparkTransaction` coordinates request-owned participants implementing position,
snapshot, restore and rollback. The base already includes the cached anchor;
after validation, only accepted draft positions remain. Cache tensor snapshots and
Markov state are backend responsibilities. This standalone utility does not make
the autoregressive executor a speculative executor; CLI speculation is rejected.

## Validation boundaries

The unit suite distinguishes actual CPU conversions and hashes from recording
backends that only test lifecycle behavior. Cache integration tests exercise the
existing allocator with scheduler-generated page tables. Runner tests use all
40 pinned layer plans and actual Engram/cache state, but do not run layer arithmetic.
Weight-loader tests use small real safetensors payloads, including captured real
K32 bytes, while serving metadata tests intentionally prohibit payload reads.

Run the bounded host suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  --confcutdir=tests/unit/model/deepseek_v41 tests/unit/model/deepseek_v41 -q
```

Torch, native UE8M0 support and safetensors are required for payload/executor
tests; NumPy and tokenizers cover the official hash multiplier and normalization
paths. Skipped dependency tests are not passing numerical validation. Full-weight
real logits, generated-token comparisons, A5 kernels, EP collectives, HTTP model
acceptance, million-token memory validation and performance measurements remain
separate acceptance gates.

### Recorded validation for `9080f6f` (2026-09-11)

- Windows bounded host suite: **266 passed, 4 skipped** for unavailable Torch-related paths.
- `ci-69`, Python 3.10.9 / Torch 2.10.0 CPU, serial execution: **328 passed, no skips**.
- Existing CLI, tokenizer, Engine and V4 component regression: **132 passed, 1 failed**.
  The failing `test_deepseek_mtp_prefill_reads_only_selected_owner_outputs` reports
  `FakeWorker.copy_from()` rejecting `src_offset`; the exact same failure was
  reproduced at the pre-change `290213a` revision. It was not repaired in this scope.
- Local review, Ruff, copyright headers, documentation navigation and diff checks passed.

The remote checkout was restored to `9080f6f`, and its pre-existing `pypto-lib`
modification was preserved. No NPU allocation, full-weight loading, HTTP inference,
or performance run was performed. Detailed logs and account usage snapshots are
stored locally under `artifacts/deepseek-v41-framework/` and are not source files.
