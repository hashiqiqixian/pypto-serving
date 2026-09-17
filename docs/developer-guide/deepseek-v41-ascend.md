# DeepSeek V4.1 A3 and A5 support

Branch `feat/deepseek-v41-ascend-a3-a5` starts from `4ca74d2`, preserving the
previous implementation branch. The model, checkpoint conversions, cache ABI
and serving pipeline are shared between the two platforms.

| Hardware target | CLI / Python platform | Default |
|---|---|---|
| A3 | `a2a3` | Yes |
| A5 | `a5` | Explicit selection |

The name `a2a3` is PyPTO's backend identifier. A3 device validation does not
establish results on A2. Selection reaches both the compiler and every rank;
requested and reported backend capabilities must match. Custom V4.1 kernel
factories now receive the `platform` keyword. Their capabilities must report it.
An unavailable backend fails instead of falling back to CPU.

PyPTO fixes the compiler architecture for the lifetime of a process. Use separate
workers for A3 and A5; changing an existing worker's architecture is unsupported.
The dual-platform codegen checks therefore use separate, bounded subprocesses.

FP8 dense and FP4 expert values are decoded to normalized BF16 values, with
FP32 accumulation and explicit scale application. This works without assuming
a native mixed FP8-by-FP4 instruction on A3. Native checkpoint float8 dtypes are
used on the CPU weight-reading side; device caches store uint8 payloads and
scales. The initial bridge still copies matrix inputs/results through the host.

Quantization reads IEEE sign bits to preserve negative zero and constructs exact
FP32 powers of two. Scale rounding follows the pinned reference's FP32 reciprocal
multiply and exponent/mantissa extraction. These operations avoid the observed
A3 `signbit(-0)` and approximate `pow` differences without moving tensors to CPU.
UE8M0 normalization multiplies by the exact inverse power of two because A5
division flushes subnormal inputs. Non-power-of-two E4M3 scales retain division.

## Optional native attention

The default backend remains the shared A3/A5 bridge. An alternative factory
connects the library's SWA, C2A Full/Reuse and C1A Full/Reindex/Reuse kernels to
the existing serving backend. Select it with:

```text
--platform a5 --devices 0 --block-size 256
--v41-kernel-factory pypto_serving.model.deepseek_v41.native_attention:create_backend
```

This factory requires A5, TP1, Flash attention dimensions and a source-token
page size divisible by 256. Select the existing request chunk limit at 8192 or
below. Set `PYPTO_LIB_ROOT` if the library checkout is not beside the serving
package. Native TP and EP dimensions are configured before importing kernels;
an incompatible configuration already imported in the process is rejected.

The native worker retains separate packed payload and scale pools and bounded
attention weight bundles. It derives all addresses from existing scheduler
leases. Index backing follows the corresponding main-KV physical order while
validating the independent index lease, so the scheduler need not allocate
identical page IDs. Failed writes and speculative checkpoints restore bytes at
the same device addresses, including ratio-2 compressor state.

Each layer transfers its current activation to the L2 worker and returns its
local FP32 result through the existing reduction interface. Engram, MoE, HC,
head and serving continue through the existing backend. This is an attention
integration, not native whole-model or multi-rank acceptance. CPU transport
tests cover dispatch, independent page namespaces, isolation and recovery;
the library's native compilation/numerical harness and real-checkpoint M0
validation are separate requirements.

## Functional validation

The CPU suite includes both platform selections, rank capability checks and
actual compiler/codegen tests for A3 and A5. Hardware tests are opt-in and must
run under the machine's device allocator. Missing dependencies or an invalid
device are errors after opt-in, not skips or implicit CPU execution.

The first hardware scope is one allocated A3 card:

- Four small BF16 GEMM shapes, including padding and K accumulation. Exact
  binary-fraction inputs allow zero-tolerance FP32 CPU comparisons. Repeated
  dispatch checks output ownership and reuse with changed inputs.
- Three packed quantization formats compared byte for byte with CPU, plus
  RMSNorm and Hyper-Connections against independent CPU formulas.
- All UE8M0 scale encodings, FP8/FP4 midpoint neighbors and signed zero,
  including non-power-of-two main-KV scales and subnormal FP32 scale values.
- A miniature random checkpoint with five backbone layers and two Engram
  layers: two-token prefill, one decode step, NPU packed pages and request release.

These are functional tests, not complete real-checkpoint inference or performance
benchmarks. The miniature test disables DSpark. It does not establish multi-rank
HCCL correctness, full-checkpoint goldens, HTTP inference, long-context capacity,
M0 acceptance or A5 hardware compatibility.

Use a matching PyTorch, TorchNPU and CANN installation. The initial A3 validation
environment uses Torch 2.10.0, TorchNPU 2.10.0 and CANN 9.0.0, following the
[official compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).
Torch and safetensors must expose the checkpoint's native UE8M0 dtype.

Set these paths to existing installations before submitting a device job:

```bash
source /path/to/cann/set_env.sh
export PTOAS_ROOT=/path/to/directory-containing-ptoas
export PTO_ISA_ROOT=/path/to/pto-isa
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
```

`PTO_ISA_ROOT` must contain `include/pto/pto-inst.hpp`. `ASCEND_HOME_PATH`, set by
CANN, must contain `bin/ccec` and `bin/ld.lld`. Keep TMPDIR and generated test
artifacts inside the authorized workspace. For the bounded CPU run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python -m pytest --confcutdir=tests/unit/model/deepseek_v41 \
  tests/unit/model/deepseek_v41 -q -rs
```

For a single allocated card, `task-submit` supplies `TASK_DEVICE`. Start with
primitives; run the miniature model after those pass:

```bash
task-submit --device auto --max-time 600 --run \
  "PYPTO_V41_NPU_TESTS=1 PYPTO_V41_NPU_PLATFORM=a2a3 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   python -m pytest --confcutdir=tests/unit/model/deepseek_v41 \
   tests/unit/model/deepseek_v41/test_pypto_kernels.py \
   tests/unit/model/deepseek_v41/test_numerics.py -k real_npu -q"

task-submit --device auto --max-time 600 --run \
  "PYPTO_V41_NPU_TESTS=1 PYPTO_V41_NPU_PLATFORM=a2a3 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   python -m pytest --confcutdir=tests/unit/model/deepseek_v41 \
   tests/unit/model/deepseek_v41/test_backend.py -k real_npu -q"
```

Use `PYPTO_V41_NPU_PLATFORM=a5` only on an allocated A5 device. For subsequent
real-model serving, use the launch parameters in the execution guide and select
`--platform a2a3` for A3 or `--platform a5` for A5. Full checkpoint provisioning,
real-layer goldens and model-scale acceptance remain required.

## Validation on 2026-09-11

Revision `6166bea` was checked on machine 69 using one allocated device, reported
as `Ascend910_9362`, with the `a2a3` backend. The environment used Python 3.10.9,
Torch/TorchNPU 2.10.0, CANN 9.0.0, ptoas 0.59, PyPTO `c9af905` and runtime
`77fa017`. Existing runtime binaries were reused; the tests compiled only their
small kernels and orchestration.

| Check | Result |
|---|---|
| Full V4.1 CPU suite, including both platforms' actual codegen | 580 passed, 16 skipped |
| Shared CLI regression (`06736ab`; CLI unchanged afterward) | 9 passed |
| A3 matrix and numerical primitives | 14 passed |
| A3 five-layer miniature model, including two Engram layers | 1 passed |

The CPU skips comprise 15 explicitly opted-in NPU cases and one optional
checkpoint-index test. Primitive job `task_20260911_114935_333431424306` exited
with status 0. Miniature job `task_20260911_115027_34316829463` also exited with
status 0, covering two-token prefill, one decode, CPU logit tolerance and identical
argmax, NPU packed pages and request release. Logs are under
`artifacts/deepseek-v41-a3/` as `cpu-6166bea.log`, `cli-06736ab.log`,
`npu-primitives-6166bea.log` and `npu-miniature-6166bea.log`.

An earlier run on `06736ab` hit AICPU error 507018 on its first GEMM; the runtime
reported device recovery. The same case passed unchanged in a separate job, and
all four GEMMs passed in the final primitive run. The initial error remains in
`npu-primitives-06736ab.log`; its cause has not been established, and these
results do not establish long-running runtime reliability.
