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

The default `--input-source stress` uses controlled random HC activations and
request metadata, including in checkpoint mode. `--input-source embeddings`
instead loads checkpoint embedding rows, broadcasts each row to four HC lanes,
and uses an identity pre-mix selecting lane zero. By default it selects sequential
token IDs. Supply `--token-ids /path/to/ids.json` for an integer JSON array shaped
`[DP, capacity]`, such as tokenizer-produced full token slabs. IDs are not padded,
repeated or truncated by the diagnostic. Both controls retain fixture RoPE/pages;
neither establishes complete prompt semantics, Engram or generation. The selected
IDs and exact initial state are saved for CPU replay. This diagnostic input setup
does not enable the production composite adapter.

The reference runs only after the device worker
closes, never supplies intermediate device inputs, and shares read-only weights
with the device path. Final actual/expected residual and pre_mix are saved to
`--artifact-dir/comparison.pt` even when the numerical comparison fails. The
residual gate defaults to `--residual-profile dsv4-layer`: the DSV4 complete-layer
comparator `ratio_reldiff(diff_thd=0.01, pct_thd=0.05)`, applied separately to each
rank. `--residual-profile v41-local` retains the historical V4.1 single-MoE
comparator (0.003/2%, with its single-point cap). Accumulated pre_mix uses the user-approved provisional rtol=0.01 and
atol=0.001, requiring every element to pass; local stage gates are unchanged.
This is limited to this two-layer diagnostic, not a V4/vLLM multi-layer standard. A completed smoke is not a numerical pass.

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

Under the historical `v41-local` profile, the baseline embedding controls fail the accumulated gate, while
their 18 native half-layer checks pass:

| Initial state | Rank-zero outliers | Residual relative L2 | Pre-mix relative L2 |
| --- | ---: | ---: | ---: |
| Independent random HC streams | 32.572% | 0.00760 | 0.00105 |
| Sequential checkpoint embedding rows | 3.896% | 0.01316 | 0.00281 |
| Checkpoint embeddings for text token prefixes | 4.094% | 0.01348 | 0.00284 |

The text control uses 32 tokens per DP group with fixture metadata. The lower
outlier fraction does not mean relative L2 improved: these are distinct
measurements on distinct workloads. The original stress failure remains an
unresolved regression case. On the sequential embedding control, CPU replay
matches the saved full reference bitwise; accumulated residual relative L2 at
Attention 0, MoE 0, Attention 1, and MoE 1 is 0.00101, 0.00353, 0.01029, and
0.01316 respectively. CPU router replay on inherited device inputs changes four
second-layer expert sets in the random control and none in this embedding
control. Routing changes alone therefore do not explain the chain failure.

For precision bisection, follow lib's `docs/debug-and-tune/precision-tuning.md`
and PyPTO's `docs/en/user/precision/00-workflow.md`. Check dtype, rounding and
reference operation order before changing kernels. The historical full-chain gate
reused a single-MoE comparator; it was not an agreed model-wide error budget.
Report relative L2, maximum absolute error and outlier fraction separately, and
retain the historical results when changing acceptance profiles.
FP8 trace comparisons decode payloads with their own scales; different encoding
pairs can represent identical values. Local checks on actual device inputs and
CPU boundary substitutions only localize errors, never replace full-chain
acceptance or supply intermediate values to device execution.

Rechecking those same saved tensors with `dsv4-layer` passes residual on every
rank. Worst-rank outlier fractions are 3.3542% (random streams), 0.02167%
(sequential embeddings), and 0.12879% (text embeddings), below the 5% budget.
This is a requested acceptance-profile change, not reduced numerical error.
With the historical pre_mix atol=0.0001, 14/256, 30/256 and 24/256 entries
respectively failed, so that historical overall diagnostic failed. DSV4's layer comparison is not an independent
43-layer accuracy guarantee, and it does not specify V4.1's delayed pre_mix
contract. These results do not establish complete model or M0 acceptance.

To recheck a saved final comparison without compiling or allocating devices:

```bash
PYTHONPATH=. python tools/validate_v41_swa_segment.py \
  --lib-root /path/to/current/pypto-lib --tp 2 --devices 0,1,2,3 \
  --compare-only /path/to/comparison.pt
```


On 2026-09-28 the user approved provisional accumulated pre_mix tolerances
rtol=0.01, atol=0.001, with no allowed failing elements. CPU rechecks of the
three saved device runs still fail in 2/256 (random), 7/256 (sequential
embeddings), and 7/256 (text embeddings) entries. Residual and all native stage
checks pass, but overall two-layer acceptance remains unresolved. This changes
the acceptance budget only; numerical errors are unchanged. No new NPU run was
performed. Evidence: `.validation-artifacts/approved-premix-budget-recheck.json`
on the A5 validation checkout, using baseline lib `21645633`.


`segment_inputs.prepare_segment_inputs` prepares fresh host HC input buffers
from packed BF16 embeddings and a `ForwardStep`. Like V4 host input preparation,
it maps request rows before device upload. The initial four FP32 lanes and
lane-zero pre-mix follow lib `input_pack.pack_x_hc` and `golden.identity_pre_mix`
at `fbe92bfc`. This is data packing only; it does not run mHC or normalization.

Requests may interleave DP partitions. Each partition keeps its own stable
packed order, split into contiguous TP token slabs. Returned source-row indices
also define the mapping for positions and other token metadata; returned final
row locations preserve the original request order for later output selection.
Inactive rows have zero residual/pre-mix, including fully empty DP partitions.
The helper rejects over-capacity steps before allocation instead of truncating.
It is used by the embedding diagnostic; cache lowering, device upload and the
complete production adapter remain separate work. Never invoke it between layers
to overwrite the residual/pre-mix produced by the previous composite.


`swa_metadata.prepare_swa_window_metadata` lowers scheduler-owned full-history
128-row window pages into the lib's INT64 write slots and INT32 causal read
indices. TP peers receive identical metadata within each DP group; physical
page IDs may be reused across DP groups but never shared by active requests in
one group. Each query reads at most 128 positions ending at itself. Full-history
pages avoid overwriting an early query's history when an entire new chunk is
published before attention. Rolling/modulo page reuse is not implemented.

The metadata helper covers page crossing, chunk continuation into decode,
interleaved requests, padding and empty partitions. It does not allocate or clear
the cache or choose a RoPE profile. Its positions must select the corresponding
checkpoint RoPE rows before dispatch. Integration into a complete model adapter
and device validation of that lifecycle remain pending.
