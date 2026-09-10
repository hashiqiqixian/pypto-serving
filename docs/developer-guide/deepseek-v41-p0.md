# DeepSeek V4.1 P0 progress

P0 is **BLOCKED**, not complete. This change implements independent configuration,
source weight specifications, bounded checkpoint inspection, and small CPU quantization
references. It does not register an executable V4.1 model or implement NPU packing,
CED execution, paged CSA2, or HTTP serving. No V4 dispatch, constants, or loader behavior
is changed.

## Fixed reference and scope

The reference is `deepseek-ai/DeepSeek-V4.1-Flash` at
`dba1be0a40aa45a94ad051997016db3960a90277`, retrieved on 2026-09-10:

- [HF configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json)
- [Reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
- [Reference conversion](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/convert.py)

The original `deepseek-v41-pypto-adaptation-plan.md` and
`deepseek-v4-flash-benchmark-baseline.md` were not found in the selected workspace.
The supplied P0 task description is retained in the task attachment. Its later P1
performance, DSpark, Engram deployment, and vision goals have not been implemented.
The user requested A5 hardware, superseding the attachment's A3 target.

The checked configuration has hidden size 5120, 40 backbone layers, 3 appended draft
layers, 384 routed experts, one shared expert, and top-6 routing. EP8 owns 48 routed
experts per rank. KV producers are 2/8/14/20; index producers additionally include
24/28/32/36. Ownership follows the most recent producer, with private SWA per layer.

Two reference facts prevent silently applying the proposed execution split:

1. `Transformer.forward` invokes all 40 backbone layers for every `start_pos`.
   Neither configuration exposes a CED encoder/decoder split. A transition in
   `compress_ratios` is not evidence that autoregressive decode may skip 20 layers.
2. Text forward computes Engram hashes and applies Engram at layers 1 and 14 before
   their blocks. Deferring all Engram computation conflicts with a true-checkpoint
   text golden. Removing it would change the model. This needs a reference-backed
   resolution before implementing the proposed separate CED entrypoints.

The reference's nonzero-position compressor/SWA paths assume a single token; its
first-prefill branch handles multiple tokens. Arbitrary chunk prefill therefore
requires additional implementation and comparison, not just a sequence-length change.

## Implemented contracts

`pypto_serving/model/deepseek_v41/` contains:

- `config.py`: standalone HF parser/profile, validation, immutable source metadata,
  per-layer ownership plan, and expert ownership. V4 metadata and unsupported formats
  are rejected. `ced_split` remains `None` because the reference supplies no split.
- `weight_spec.py`: 93,418 required text tensor keys, physical shape/dtype, and explicit
  source-to-reference conversion descriptions, including required Engram tensors.
  Conversion descriptions are not an Ascend runtime pack implementation.
- `checkpoint.py`: duplicate-key, missing/unknown key, shard assignment, shape/dtype,
  byte range, overlap, truncated file, and total-size checks. Reads only safetensors
  headers. Missing shards produce `BLOCKED`; successful subset checks are separate.
- `quantization.py`: byte-accurate E2M1 packing, UE8M0 and E4M3FN encoding, expert/dense
  dequantization, activation quantization, exact FP4-to-FP8 conversion when representable,
  and small CPU GEMM references. They use Python floating point, not a BF16/FP32 kernel
  execution emulator. Unrepresentable allegedly lossless conversions fail explicitly.

The quantization contracts are distinct:

| Data | Format | Scale layout |
| --- | --- | --- |
| Dense weight | E4M3FN | UE8M0 per 32x32 block |
| Routed expert weight | E2M1, low nibble first, stored as I8 `[N,K/2]` | UE8M0 `[N,K/32]` |
| Index key/query | FP4 reference quantization | UE8M0 per K32 |
| Compressed main KV | FP4 reference quantization | E4M3 per K16 |

The official in-place cache quantizer writes dequantized values back to its cache
tensor. Its memory use is not evidence of packed FP4 KV storage. Compressor weights
are BF16 in the checkpoint and promoted to F32 for ratio>1. `wo_a` is FP8 plus scale
in the checkpoint and dequantized to BF16 by reference conversion.

The local PyPTO/PTOAS/PTO ISA trees expose FP8/FP4/UE8M0 types. However, the inspected
A5 MX matmul checks accept FP8xFP8 or FP4xFP4, while the reference expert path uses
FP8 activations with FP4 weights. Type availability is not proof of this mixed GEMM
ABI. No compiler, ISA, or kernel change is included here.

## Evidence and reproduction

Local ignored evidence is under `artifacts/deepseek-v41-p0/`. It contains the pinned
reference files, all 48 HTTP-range header snapshots, their hashes, `inspection.json`,
test logs, environment/progress records, and usage snapshots. No full checkpoint was
downloaded. The index declares 510,286,023,000 storage bytes; this is a metadata value,
not static HBM, peak HBM, or downloaded bytes.

Full metadata inspection validated all 93,418 required text tensors. The remaining
266 vision and 2,401 draft tensors are explicitly outside text shape specifications;
their header structure and index assignment are checked, not their model semantics.

`tests/fixtures/deepseek_v41/config.json` is the pinned configuration, SHA256
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
`real_quant_blocks.json` retains exactly two real K32 blocks (FP8 dense and FP4 expert),
with byte ranges, revision and hashes. Native Torch CPU comparisons cover these
blocks only. They do not constitute a real-layer or NPU golden.

Run isolated CPU tests without loading the parent tests' Torch fixtures:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  --confcutdir=tests/unit/model/deepseek_v41 tests/unit/model/deepseek_v41 -q
```

The native Torch block test is explicitly skipped if Torch is unavailable. A skip is
not a pass. Use a Torch build exposing E4M3FN and E8M0 to execute that comparison.

Inspect a complete local checkpoint without allocating its tensor payload:

```bash
python scripts/inspect_deepseek_v41.py \
  --config "$CHECKPOINT/config.json" \
  --index "$CHECKPOINT/model.safetensors.index.json" \
  --checkpoint-dir "$CHECKPOINT" \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --report artifacts/deepseek-v41-p0/local-checkpoint-inspection.json
```

Reproduce the saved metadata-only HTTP-range inspection:

```bash
python scripts/inspect_deepseek_v41.py \
  --config artifacts/deepseek-v41-p0/reference/config.json \
  --index artifacts/deepseek-v41-p0/reference/model.safetensors.index.json \
  --header-dir artifacts/deepseek-v41-p0/reference \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --report artifacts/deepseek-v41-p0/inspection.json
```

Exit codes are 0=complete requested metadata inspection, 1=invalid contract, and
2=missing metadata or invalid CLI arguments. Reports never claim tensor numerics or
NPU upload from metadata success. Header snapshots cannot establish that all tensor
payloads exist locally. Reports must be outside the input checkpoint/header directory.

V4 CPU regression command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/unit/model/deepseek -q --disable-warnings
```

The initially found remote serving revision `5ddabfc` has an existing
`test_deepseek_prefill_context_preserves_prepare_reserved_state` failure due to
a test input without `local_rows` (86 passed, 1 failed before maxfail stopped it).
That revision differs from the selected local base `4afc60e`; use the captured
baseline/after logs for comparisons. No V4 full-checkpoint test was run.

The reviewed implementation was committed and CPU-tested as `eb00bfd`:

| Check | Actual result |
| --- | --- |
| Local isolated tests, Python 3.12 | 116 PASS; one module SKIPPED because Torch is absent |
| Remote isolated tests, Python 3.10.9 / Torch 2.10.0+cpu | 121 PASS, including five real-block fixture/numeric checks |
| V4 baseline at `4afc60e` | 121 PASS, 1 FAIL |
| V4 after at `eb00bfd` | 121 PASS, same 1 FAIL |
| Ruff check/format, headers, diff whitespace | PASS |

The matched baseline/after failure is
`test_deepseek_mtp_prefill_reads_only_selected_owner_outputs`: its `FakeWorker.copy_from`
does not accept `src_offset`. No new failures appeared in this bounded V4 CPU suite.
This is not evidence of full-checkpoint or device non-regression. Logs are
`v41-tests.log`, `v4-baseline.log`, and `v4-after.log` in the artifact directory.

No working V4.1 build, weight upload, serving start, token golden, or S1 benchmark
command exists in this patch. Supplying a V4 command with a V4.1 checkpoint would
misrepresent its support.

## Acceptance and next checkpoint

| Stage | Status | Evidence or remaining requirement |
| --- | --- | --- |
| P0-1 | BLOCKED | Config, all text metadata, CPU format tests implemented; full real layer pack/upload/golden missing |
| P0-2 | BLOCKED | Ownership plan only; CED scope conflict, mixed GEMM ABI, paged CSA2 and chunk/decode execution unresolved |
| M0 | BLOCKED | All five model/service acceptance items below remain unverified |
| P0-3 | BLOCKED | No true-weight V4.1 executor/HTTP integration |

| M0 item | Status | Evidence location / missing evidence |
| --- | --- | --- |
| True V4.1 checkpoint loaded, V4 coexists | BLOCKED | `inspection.json` proves metadata only; complete loading not run |
| 8K prefill and single-token decode on requested topology | BLOCKED | Requested A5 endpoint unavailable; kernels not implemented |
| 64-to-128 greedy golden and continuous HTTP requests | NOT_RUN | No executable V4.1 path or full checkpoint |
| Reproducible artifact including peak HBM | BLOCKED | Source/environment/metadata artifacts saved; peak HBM not measured |
| S1 8K input / 128 output / concurrency 1 | NOT_RUN | Serving smoke not run |

Both authorized endpoints were inspected through bounded task-queue metadata jobs:
69 reports Ascend910_9362, and 686 reports Ascend910_9392. Both are A3, not the
requested Ascend950/A5. No A5 kernel was run on those devices. The selected slot's
remote kernel repository also has pre-existing changes, which were preserved.

The next checkpoint needs an authorized A5 environment and an exact accessible
V4.1 checkpoint path, followed by one full real layer's packing/upload and numerical
comparison. Resolve the Engram/CED reference conflict without changing model semantics.
First establish a supported FP8-activation/FP4-weight execution path and fixed tolerances;
then implement cache/kernel integration. P1 remains out of this change's scope.
