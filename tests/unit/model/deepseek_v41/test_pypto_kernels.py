# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3/A5 compiler checks and explicitly opted-in bounded NPU correctness checks.

The default tests run real IR/codegen without assembler or hardware execution.
Each compiler case uses a fresh, bounded subprocess because the PyPTO backend
is process-global: an initialized A3 backend cannot be replaced by A5 in place.
Set PYPTO_V41_NPU_TESTS=1, TASK_DEVICE to the allocated single device, and
PYPTO_V41_NPU_PLATFORM=a2a3 (default) or a5 to run the hardware cases. Hardware
tests require existing CANN/PTO-ISA/ptoas installations; they do not fetch them.
The A5 8K regression additionally requires PYPTO_V41_NPU_8K_TESTS=1. Run it
under task-submit with a 600-second process-tree limit; its oracle uses no CPU GEMM.
"""

import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


RUN_NPU = os.environ.get("PYPTO_V41_NPU_TESTS") == "1"
RUN_NPU_8K = os.environ.get("PYPTO_V41_NPU_8K_TESTS") == "1"
if RUN_NPU or RUN_NPU_8K:
    import torch
else:
    torch = pytest.importorskip("torch", reason="kernel IR tests require Torch tensor metadata")
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("rows,columns,inner", [(16, 64, 64), (32, 128, 128)])
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_bf16_matmul_lowers_to_fp32_cube_with_out_parameter(tmp_path, rows, columns, inner, platform):
    if not RUN_NPU:
        pytest.importorskip("pypto.pypto_core", reason="real PyPTO compiler extension is required")
    # Never reset the compiler singleton. Each subprocess selects exactly one
    # backend, runs serially, and cannot invoke ptoas/C++ builds or a device.
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    child = (
        "import runpy, sys; "
        "case = runpy.run_path(sys.argv[1]); "
        "case['_compile_and_check_ir'](sys.argv[2], "
        "int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                child,
                str(Path(__file__).resolve()),
                platform,
                str(rows),
                str(columns),
                str(inner),
                str(tmp_path),
            ],
            env=environment,
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{platform} compiler subprocess exceeded 60s:\n{exc.stdout!r}\n{exc.stderr!r}")
    assert completed.returncode == 0, (
        f"{platform} compiler subprocess failed ({completed.returncode}):\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    _assert_generated_cube(tmp_path, platform, inner)


def _compile_and_check_ir(platform, rows, columns, inner, output_directory):
    """Run only in a fresh child, retaining actual compiler/IR assertions."""
    from pypto import ir
    from pypto.pypto_core import DataType
    from pypto.runtime import RunConfig

    decorator = importlib.import_module("pypto.jit.decorator")
    # Exercise the real specializer, parser, passes and selected code generator. Only
    # the external assembler is disabled, so this test cannot launch C++ builds.
    spec = importlib.util.spec_from_file_location(
        "_v41_real_kernel_test", ROOT / "pypto_serving/model/deepseek_v41/kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.make_bf16_matmul_kernel()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)
        compiled = kernel.compile(
            torch.empty((rows, inner), dtype=torch.bfloat16),
            torch.empty((inner, columns), dtype=torch.bfloat16),
            torch.empty((rows, columns), dtype=torch.float32),
            config=RunConfig(
                platform=platform, codegen_only=True, save_kernels=True, save_kernels_dir=output_directory
            ),
        )
    assert compiled.platform == platform
    assert compiled.output_indices == [2]
    program = compiled.program
    orchestration = next(
        function
        for function in program.functions.values()
        if function.func_type == ir.FunctionType.Orchestration
    )
    assert [parameter.type.dtype for parameter in orchestration.params] == [
        DataType.BF16,
        DataType.BF16,
        DataType.FP32,
    ]
    assert [
        [dimension.value for dimension in parameter.type.shape] for parameter in orchestration.params
    ] == [[rows, inner], [inner, columns], [rows, columns]]

    # CompiledProgram.program retains the input tensor IR. The lowered Cube
    # operations must be checked in the actual emitted PTO below.
    class MatmulVisitor(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls = []

        def visit_call(self, call):
            if call.op.name in ("tensor.matmul", "tensor.matmul_acc"):
                self.calls.append(call)
            super().visit_call(call)

    visitor = MatmulVisitor()
    visitor.visit_program(program)
    assert {call.op.name for call in visitor.calls} == {"tensor.matmul", "tensor.matmul_acc"}
    for call in visitor.calls:
        assert call.type.dtype == DataType.FP32
        assert [argument.type.dtype for argument in call.args[-2:]] == [DataType.BF16, DataType.BF16]


def _assert_generated_cube(tmp_path, platform, inner):
    """Inspect the child's actual emitted artifacts in the parent process."""
    artifacts = list(tmp_path.rglob("*.pto"))
    assert artifacts, f"{platform} PTO codegen must emit kernel artifacts"
    emitted = "\n".join(path.read_text() for path in artifacts)
    assert "#pto.kernel_kind<cube>" in emitted
    cube_calls = re.findall(r"^\s*pto\.tmatmul(?:\.acc)?\s+.*$", emitted, re.MULTILINE)
    assert cube_calls, "lowering must emit actual Cube matrix operations"
    assert any("pto.tmatmul ins(" in call for call in cube_calls)
    if inner > 64:
        assert any("pto.tmatmul.acc ins(" in call for call in cube_calls)
    for call in cube_calls:
        inputs, outputs = call.split(" outs(", 1)
        assert inputs.count("dtype=bf16") == 2
        assert "loc=left, dtype=bf16" in inputs and "loc=right, dtype=bf16" in inputs
        assert "loc=acc, dtype=f32" in outputs


@pytest.fixture
def real_npu_ops(tmp_path, monkeypatch, request):
    if not RUN_NPU:
        if RUN_NPU_8K:
            pytest.fail("PYPTO_V41_NPU_8K_TESTS also requires PYPTO_V41_NPU_TESTS=1")
        pytest.skip("set PYPTO_V41_NPU_TESTS=1 under an allocated task-submit device")
    platform = os.environ.get("PYPTO_V41_NPU_PLATFORM", "a2a3")
    if platform not in ("a2a3", "a5"):
        pytest.fail("PYPTO_V41_NPU_PLATFORM must select hardware 'a2a3' or 'a5'")
    default_budget = getattr(request, "param", None) == "default_budget"
    if default_budget and platform != "a5":
        pytest.fail("the opted-in 8K regression requires PYPTO_V41_NPU_PLATFORM=a5")
    device_text = os.environ.get("TASK_DEVICE", "")
    if not device_text.isascii() or not device_text.isdecimal():
        pytest.fail("TASK_DEVICE must be the allocated single nonnegative device index")
    pto_isa = os.environ.get("PTO_ISA_ROOT")
    if not pto_isa or not (Path(pto_isa) / "include").is_dir():
        pytest.fail("PTO_ISA_ROOT must name an existing PTO-ISA checkout containing include/")
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home or not all(
        (Path(ascend_home) / "bin" / name).is_file() for name in ("ccec", "ld.lld")
    ):
        pytest.fail("ASCEND_HOME_PATH must contain the installed bin/ccec and bin/ld.lld")
    ptoas_root = os.environ.get("PTOAS_ROOT")
    ptoas = str(Path(ptoas_root) / "ptoas") if ptoas_root else shutil.which("ptoas")
    if not ptoas or not Path(ptoas).is_file() or not os.access(ptoas, os.X_OK):
        pytest.fail("an executable ptoas is required on PATH or directly inside PTOAS_ROOT")

    # Load the real bridge without importing the unrelated serving model registry.
    # Its PyPTO runtime and JIT kernel imports remain real and cannot select Torch.
    package_name = "_v41_npu_leaf_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "pypto_serving/model/deepseek_v41")]
    monkeypatch.setitem(sys.modules, package_name, package)
    name = package_name + ".pypto_ops"
    spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0]) / "pypto_ops.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    def no_cpu_provider(*args, **kwargs):
        pytest.fail("real NPU test must not dispatch the CPU reference provider")

    monkeypatch.setattr(module.TorchMatmulOps, "matmul", no_cpu_provider)
    budget = {} if default_budget else {"max_buffer_bytes": 8 << 20}
    ops = module.PyptoMatmulOps(
        platform=platform,
        device_id=int(device_text),
        build_dir=tmp_path,
        max_cached_shapes=1,
        **budget,
    )
    try:
        yield ops
    finally:
        ops.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rows,columns,inner", [(5, 7, 3), (17, 65, 33), (16, 64, 64), (32, 128, 128)])
def test_real_npu_bf16_matmul_matches_fp32_cpu_oracle(real_npu_ops, rows, columns, inner):
    # Binary fractions are exact in BF16. All products and partial sums fit FP32
    # exactly, so this checks both padded and multi-K-tile execution with no tolerance.
    a = ((torch.arange(rows * inner).reshape(rows, inner) % 17 - 8).float() / 8).to(torch.bfloat16)
    b = ((torch.arange(inner * columns).reshape(inner, columns) * 7 % 19 - 9).float() / 16).to(torch.bfloat16)
    before_a, before_b = a.clone(), b.clone()
    expected = a.float() @ b.float()
    result = real_npu_ops.matmul(a, b)
    assert result.device.type == "cpu" and result.dtype == torch.float32
    assert result.shape == (rows, columns) and result.is_contiguous()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # Same compiled shape with changed values catches stale execution outputs;
    # keeping the first result also checks caller ownership across runtime reuse.
    repeated = real_npu_ops.matmul(a, -b)
    torch.testing.assert_close(repeated, -expected, rtol=0, atol=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.zero_()
    assert torch.equal(a, before_a) and torch.equal(b, before_b)


@pytest.mark.skipif(not RUN_NPU_8K, reason="set PYPTO_V41_NPU_8K_TESTS=1 for the allocated A5 8K run")
@pytest.mark.parametrize("real_npu_ops", ["default_budget"], indirect=True)
def test_real_a5_8k_group_projection_matches_exact_structured_oracle(real_npu_ops, monkeypatch):
    # A[i,k] = r[i]*s[k], B[k,j] = s[k]*c[j], with s[k] in {-1,1}.
    # Hence C[i,j] = 4096*r[i]*c[j]. Binary fractions keep every product and
    # partial sum exact in FP32, including nonzero values of both signs.
    # Only the allocated NPU performs the full matrix product.
    rows, inner, columns = 8192, 4096, 1024
    row_coefficients = ((torch.arange(rows) % 17 - 8).float() / 8).to(torch.bfloat16)
    column_coefficients = ((torch.arange(columns) * 7 % 19 - 9).float() / 16).to(torch.bfloat16)
    signs = (1 - 2 * (torch.arange(inner) % 2)).to(torch.bfloat16)
    a = row_coefficients[:, None] * signs[None, :]
    b = signs[:, None] * column_coefficients[None, :]
    assert torch.unique(row_coefficients).numel() == 17
    assert torch.unique(column_coefficients).numel() == 19
    assert real_npu_ops.max_buffer_bytes == 256 << 20
    dispatch_rows = []
    dispatch = real_npu_ops._dispatch_into

    def record_dispatch(left, right, target):
        dispatch_rows.append(left.shape[0])
        return dispatch(left, right, target)

    monkeypatch.setattr(real_npu_ops, "_dispatch_into", record_dispatch)
    result = real_npu_ops.matmul(a, b)
    assert dispatch_rows == [1568] * 5 + [352]
    assert result.device.type == "cpu" and result.dtype == torch.float32
    assert result.shape == (rows, columns) and result.is_contiguous()
    # Compare every output, including every chunk boundary, with small oracle
    # buffers instead of retaining another full 32 MiB result or input clones.
    for start in range(0, rows, 256):
        expected = inner * row_coefficients[start : start + 256, None].float() * column_coefficients.float()
        torch.testing.assert_close(result[start : start + 256], expected, rtol=0, atol=0)
        assert torch.equal(
            a[start : start + 256], row_coefficients[start : start + 256, None] * signs[None, :]
        )
    assert torch.equal(b, signs[:, None] * column_coefficients[None, :])
