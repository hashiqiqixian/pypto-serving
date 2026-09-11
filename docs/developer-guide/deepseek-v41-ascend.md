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

FP8 dense and FP4 expert values are decoded to normalized BF16 values, with
FP32 accumulation and explicit scale application. This works without assuming
a native mixed FP8-by-FP4 instruction on A3. Native checkpoint float8 dtypes are
used on the CPU weight-reading side; device caches store uint8 payloads and
scales. The initial bridge still copies matrix inputs/results through the host.

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
