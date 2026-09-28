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
production resident bundle management, input initialization, all attention
modes, decode, cache lifecycle and the final output boundary still need adapters
and validation. Engram is excluded. Full TP4/DP2/EP8 8K-to-128 M0 is not claimed.

## Real checkpoint bundles and numerical diagnostic

`load_swa_layer_weights(model_dir, layer_id, topology)` returns CPU Attention
and MoE weight maps for a SWA layer. It uses the selective checkpoint loader;
TP projections and EP expert ownership retain their existing rules. Routed
payloads stay packed FP4. Per-expert scales must be unpacked and repacked into
the combined expert/K-group MX layout, not concatenated in their already packed
order. The caller still owns upload, request metadata and cache allocation.

To exercise distinct real weights for layers 0 and 1 and compare the final
state against composed Torch references:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3 \
  --model-dir /path/to/DeepSeek-V4.1-Flash --reference --ring-heap-mib 4096
```

The diagnostic uses controlled activations and request metadata, including in
checkpoint mode. It does not yet validate embedding initialization, real prompt
semantics, Engram or generation. The reference runs only after the device worker
closes, never supplies intermediate device inputs, and shares read-only weights
with the device path. Final actual/expected residual and pre_mix are saved to
`--artifact-dir/comparison.pt` even when the numerical comparison fails. The
residual gate uses lib's local MoE relative-error comparator; pre_mix uses
rtol=0.01 and atol=0.0001. A completed smoke is not a numerical pass.

The real-weight TP2/DP2/EP4 diagnostic exhausted the temporary heap at both
512 MiB and 1024 MiB per ring. With 4096 MiB per ring, both device layers and
their Torch references completed. This runtime has four rings, so a per-ring
setting is not the total allocation; check device headroom before running.
The saved final outputs did not pass the numerical gate (residual relative L2
about 0.00760, pre_mix about 0.00105). These are diagnostic observations, not
an accepted end-to-end tolerance or a production memory recommendation.

`--stage-reference` also captures each half-layer's outputs after the entire
device chain completes, and applies lib's stage comparators to references
computed on that half-layer's actual input. This localizes accumulated errors;
it does not replace the independent end-to-end check or feed CPU values back
to the device. The final check still determines success.

At serving `7ef97ca`, all stage checks passed for both real-weight SWA/MoE
layers, including Attention caches and MoE residuals. The independent two-layer
check still failed with the errors above. Agreement on each stage's actual input
does not establish the accumulated numerical budget across layers; that boundary
still needs validation before full-model acceptance.

To recheck a saved final comparison without compiling or allocating devices:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3 \
  --compare-only /path/to/comparison.pt
```
