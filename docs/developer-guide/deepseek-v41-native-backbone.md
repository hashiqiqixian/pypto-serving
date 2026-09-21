# V4.1 A5 TP1 native backbone bring-up

The opt-in `pypto_serving.model.deepseek_v41.native_backbone:create_backend`
factory extends the existing native-attention adapter with library mHC,
RMS normalization, routing, routed FP4 experts and shared FP8 experts.
Engram retains the existing tokenizer normalization, hash history, sparse
checkpoint lookup, projection and gate. It executes before each configured
backbone block, on all HC streams. Embeddings and the vocabulary projection
also retain the bridge. The default factory is unchanged.

This is the first integration stage, not a performance implementation or
checkpoint-to-text acceptance claim. Activations cross the host boundary;
the host schedules experts in tiles of at most 32 rows and combines routed
outputs in FP32. No EP collective is used. The library is configured TP1/EP2
only to import its existing shape-specialized **local** kernels; this does
not allocate a second rank or run the EP2 distributed entry.

## Execution and ownership

- Requires one A5 device, Flash arithmetic dimensions and routing semantics.
  A3, multiple devices, DSpark and multimodal requests are rejected.
- Retains the native attention adapter's scheduler-owned page mapping,
  page-size restrictions, compressor state, cache journals and rollback.
- Preserves delayed HC mixes: attention consumes the preceding block's
  `pre_mix`, FFN consumes the current attention mix, and the next block/head
  consumes the current FFN mix. Engram changes the residual, not `pre_mix`.
- Handles the full prefill length, including the final partial expert tile.
  Top-k weights enter SwiGLU before its BF16 rounding and down projection.
- Keeps expert payloads packed. Routed matrices retain their checkpoint
  FP4 bytes and E8M0 scales; shared matrices use input-major FP8 and packed
  MX_B_NN scales. The tensor store checks the complete returned matrix budget
  before payload reads.
- Shares the native attention ChipWorker. Backbone weights have a separate
  bounded 256 MiB device LRU, released before the worker closes. This budget
  excludes attention weights, cache, activations and compiler workspaces.
- Diagnostics report actual kernel calls, resident backbone bytes and the
  retained reference components. CPU mock dispatch tests do not count as
  native numerical evidence.

## Run

Use the existing V4.1 CLI, complete checkpoint and tokenizer, with:

```bash
pypto-serving --model /path/to/DeepSeek-V4.1-Flash \
  --platform a5 --devices 0 --dp 1 --tp 1 --ep 1 --block-size 256 \
  --v41-kernel-factory pypto_serving.model.deepseek_v41.native_backbone:create_backend \
  --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 256 \
  --prompt 'Explain sparse attention.' \
  --generate-config '{"max_new_tokens":8,"temperature":0.0}'
```

Use a source-token cache page size divisible by 256, as required by the native
attention adapter. Confirm the selected runtime page size before startup.
The factory explicitly rejects an incompatible value. Set `PYPTO_LIB_ROOT`
when using a library checkout other than the serving submodule. The tested
host contract is the V4.1 `local_attention` interface at library `d6aa8ad`.

## Validation

Run host integration and byte-layout tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest --confcutdir=tests/unit/model/deepseek_v41 \
  tests/unit/model/deepseek_v41/test_native_backbone.py \
  tests/unit/model/deepseek_v41/test_native_weights.py -q
```

`test_native_backbone_kernels.py` has seven actual A5 compiler cases. Each
runs in its own process; lack of a compiler is an explicit skip. It also has
opt-in actual-shape device cases for mHC/norm/router and FP4/FP8 experts at
1, 3 and 32 active rows, compared against library Torch goldens.

Inside a single-device, bounded `task-submit` allocation, with `TASK_DEVICE`
set to the allocated card and the verified PyPTO/CANN toolchain active:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYPTO_V41_NATIVE_NPU_TESTS=1 \
python -m pytest --confcutdir=tests/unit/model/deepseek_v41 \
  tests/unit/model/deepseek_v41/test_native_backbone_kernels.py -k real_a5 -q
```

Keep compiler, random-kernel, real-checkpoint layer, short-logit and generated
token evidence separate. Full-checkpoint A5 logits/tokens must still be
compared against the pinned independent reference. Passing host tests or
random kernel tests does not establish that acceptance.

### Local validation snapshot (2026-09-21)

On Windows, Python 3.12.10 / Torch 2.14.0, the final targeted suite covering
backbone composition, native weight packing, native attention lifecycle,
the miniature backend and tensor store reported **69 passed, 21 skipped**.
Ruff, source headers, source language and diff whitespace checks passed.
Seven new compiler cases and nine new A5 device cases were skipped because
the native compiler/device environment was unavailable locally.

The broader V4.1 suite, before four additional host-only tests were added,
reported **733 passed, 47 skipped, 1 failed**. The FP32 parameterization of
`test_indexer_batches_queries_without_changing_causal_candidates` failed
identically at unmodified serving `2830741` in a separate baseline worktree:
307/2080 elements differed, maximum absolute difference 0.0362466. The new
host tests subsequently passed in the targeted run above.

A5 SSH public-key authentication was rejected before environment inspection.
No NPU job was submitted. Actual PyPTO compilation, A5 kernel numerics and
real-checkpoint logits/generated tokens remain **NOT RUN** for this change.
