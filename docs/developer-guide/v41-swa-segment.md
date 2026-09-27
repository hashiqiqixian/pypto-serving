# V4.1 bounded SWA segment

`pypto_serving/model/deepseek_v41/swa_segment.py` adds the concrete half-layer
execution boundary inspected against lib `216456332c2a74d89cca23b7824dab264ce34bff`:

1. `prefill_swa.make_hc_program(..., epochs=1)` includes mHC, input Norm,
   attention, TP communication and mHC post.
2. `moe.l3_moe` includes mHC, Norm, packed-FP4 routed/shared experts and mHC post.
3. The returned FP32 `LayerState` is passed directly to the next SWA layer.

The shared V4 `KernelCompiler` compiles both entries. `SwaSegment` uses one
persistent `DistributedWorker`, matching V4's runtime ownership. It does not
call small operators or copy intermediate state to CPU. The caller owns
uploaded weights, per-layer caches, positions, shared count metadata and scratch.
The example uses independent caches and tied synthetic weights for two layers.

Current bound: 16 physical token rows per rank, as fixed by lib `MOE_TOKENS`.
TP4 permits 64 rows per DP group; TP2 permits 32. Counts use contiguous slabs,
including padded ranks. Exceeding this capacity is rejected. Capacity checks and
count conversion do not establish numerical correctness of empty/ragged cases.
Device failures poison the segment; its worker must be closed before recovery.
Do not reset only the request cache and resume an uncertain collective.

Run the explicit synthetic diagnostic on A5 through the device queue:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3
```

`--compile-only` checks code generation without device execution. The device
smoke checks completion and finite/nonzero outputs, not numerical acceptance.
CPU dispatch tests use a mocked worker and do not establish NPU correctness.

This segment does not enable `load_composite_bindings()` for complete serving:
checkpoint-to-resident bundle assembly, input initialization, all attention
modes, decode, cache lifecycle and the final output boundary still need adapters
and validation. Engram is excluded. Full TP4/DP2/EP8 8K-to-128 M0 is not claimed.
