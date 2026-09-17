# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""K32 scale-order contracts, bridge lifecycle, and opt-in real Ascend dispatch.

Compiler checks use separate bounded processes without ptoas. Hardware checks
use PYPTO_V41_NPU_TESTS=1, PYPTO_V41_NPU_PLATFORM and the allocated TASK_DEVICE.
The 8K case additionally requires PYPTO_V41_NPU_8K_TESTS=1.
"""

import dataclasses
import importlib.util
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def scaled_module(monkeypatch):
    package = types.ModuleType("_v41_scaled_test")
    package.__path__ = [str(ROOT / "pypto_serving/model/deepseek_v41")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".pypto_ops"
    spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0]) / "pypto_ops.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _values(rows, columns, inner):
    a = ((torch.arange(rows * inner).reshape(rows, inner) % 11 - 5) / 8).bfloat16()
    b = ((torch.arange(inner * columns).reshape(inner, columns) % 7 - 3) / 4).bfloat16()
    a_scales = (torch.arange(rows * (inner // 32)).reshape(rows, -1) % 5 + 1).float() / 4
    b_scales = (torch.arange((inner // 32) * columns).reshape(-1, columns) % 7 + 1).float() / 8
    return a, b, a_scales, b_scales


def _scalar_reference(a, b, a_scales, b_scales):
    def fp32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    # Fixture dot products are exact binary fractions. Explicit scalar FP32
    # rounding then distinguishes sequential scale multiplication and addition.
    result = torch.zeros(a.shape[0], b.shape[1])
    for row in range(a.shape[0]):
        for column in range(b.shape[1]):
            total = 0.0
            for block in range(a.shape[1] // 32):
                partial = fp32(sum(float(a[row, k]) * float(b[k, column])
                                   for k in range(block * 32, block * 32 + 32)))
                partial = fp32(partial * float(a_scales[row, block]))
                partial = fp32(partial * float(b_scales[block, column]))
                total = fp32(total + partial)
            result[row, column] = total
    return result


@pytest.fixture
def scaled_dispatch(scaled_module, monkeypatch):
    @dataclasses.dataclass
    class RunConfig:
        platform: str
        device_id: int
        save_kernels: bool
        save_kernels_dir: str | None = None

    state = types.SimpleNamespace(calls=[], fail=False, nonfinite=False)

    def factory(scaled):
        def kernel(*args, config):
            state.calls.append((scaled, [tuple(t.shape) for t in args], config))
            (Path(config.save_kernels_dir) / "artifact.txt").write_text("test artifact")
            if state.fail:
                raise RuntimeError("synthetic dispatch failure")
            a, b, *rest = args
            if scaled:
                sa, sb, output = rest
                output.zero_()
                for block in range(a.shape[1] // 32):
                    partial = a[:, block * 32 : block * 32 + 32].float() @ b[
                        block * 32 : block * 32 + 32
                    ].float()
                    output.add_(partial * sa[block : block + 1].T * sb[block : block + 1])
            else:
                output = rest[0]
                output.copy_(a.float() @ b.float())
            if state.nonfinite:
                output[0, 0] = float("inf")

        return kernel

    runtime = types.ModuleType("pypto.runtime")
    runtime.RunConfig = RunConfig
    kernels = types.ModuleType("_v41_scaled_test.kernels")
    kernels.make_bf16_matmul_kernel = lambda: factory(False)
    kernels.make_block_scaled_matmul_kernel = lambda: factory(True)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    return state


def test_cpu_scaled_reference_keeps_fp32_scale_order(scaled_module):
    values = list(_values(3, 5, 96))
    # Non-power-of-two scales expose premature BF16 dequantization or fused scales.
    values[2] *= 1.003
    values[3] *= 0.999
    before = [value.clone() for value in values]
    expected = _scalar_reference(*values)
    actual = scaled_module.TorchMatmulOps().block_scaled_matmul(*values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32 and actual.is_contiguous() and not actual.requires_grad
    actual.zero_()
    assert all(torch.equal(value, old) for value, old in zip(values, before))


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_scaled_bridge_chunks_rows_and_keeps_k_groups_in_one_dispatch(
    scaled_module, scaled_dispatch, tmp_path, platform
):
    a, b, sa, sb = _values(65, 7, 96)
    # Preserve the same values through noncontiguous caller views.
    values = [value.T.contiguous().T for value in (a, b, sa, sb)]
    expected = _scalar_reference(*values)
    ops = scaled_module.PyptoMatmulOps(platform=platform, build_dir=tmp_path, max_buffer_bytes=175000)
    try:
        actual = ops.block_scaled_matmul(*values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert [shapes[0][0] for _, shapes, _ in scaled_dispatch.calls] == [32, 32, 16]
        for scaled, shapes, config in scaled_dispatch.calls:
            assert scaled and shapes[0][1] == 96
            assert shapes[2] == (3, shapes[0][0]) and shapes[3] == (3, 64)
            assert config.platform == platform
        repeated = ops.block_scaled_matmul(-a, b, sa, sb)
        torch.testing.assert_close(repeated, -expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        ops.close()
    assert not list(tmp_path.iterdir())


def test_scaled_and_plain_kernels_share_bounded_cache_without_collision(
    scaled_module, scaled_dispatch, tmp_path
):
    values = _values(16, 64, 64)
    ops = scaled_module.PyptoMatmulOps(build_dir=tmp_path, max_cached_shapes=1)
    try:
        ops.matmul(*values[:2])
        first = Path(scaled_dispatch.calls[-1][2].save_kernels_dir)
        expected = _scalar_reference(*values)
        torch.testing.assert_close(ops.block_scaled_matmul(*values), expected, rtol=0, atol=0)
        assert not first.exists()
        ops.matmul(*values[:2])
        assert [scaled for scaled, _, _ in scaled_dispatch.calls] == [False, True, False]
    finally:
        ops.close()
    with pytest.raises(RuntimeError, match="closed"):
        ops.block_scaled_matmul(*values)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("operand", range(4))
def test_scaled_bridge_rejects_late_nonfinite_inputs_before_dispatch(
    scaled_module, scaled_dispatch, tmp_path, operand
):
    values = list(_values(65, 7, 96))
    values[operand][-1, -1] = float("nan")
    ops = scaled_module.PyptoMatmulOps(build_dir=tmp_path, max_buffer_bytes=175000)
    try:
        with pytest.raises(ValueError, match="finite"):
            ops.block_scaled_matmul(*values)
        assert not scaled_dispatch.calls
    finally:
        ops.close()


@pytest.mark.parametrize("invalid", ["k", "scale_dtype", "scale_shape", "scale_device", "budget"])
def test_scaled_bridge_rejects_invalid_contract_before_allocation(
    scaled_module, scaled_dispatch, tmp_path, monkeypatch, invalid
):
    values = list(_values(2, 3, 64))
    if invalid == "k":
        values[:2] = values[0][:, :63], values[1][:63]
    elif invalid == "scale_dtype":
        values[2] = values[2].bfloat16()
    elif invalid == "scale_shape":
        values[3] = values[3].T
    elif invalid == "scale_device":
        values[2] = torch.empty_like(values[2], device="meta")
    ops = scaled_module.PyptoMatmulOps(
        build_dir=tmp_path, max_buffer_bytes=1 if invalid == "budget" else 1 << 20,
    )

    def no_allocation(*_args, **_kwargs):
        pytest.fail("invalid inputs must fail before output or padding allocation")

    monkeypatch.setattr(torch, "empty", no_allocation)
    monkeypatch.setattr(torch, "zeros", no_allocation)
    try:
        with pytest.raises(ValueError):
            ops.block_scaled_matmul(*values)
        assert not scaled_dispatch.calls
    finally:
        ops.close()


@pytest.mark.parametrize("failure", ["fail", "nonfinite"])
def test_scaled_dispatch_failure_propagates_and_retry_is_clean(
    scaled_module, scaled_dispatch, tmp_path, failure
):
    values = _values(3, 5, 96)
    ops = scaled_module.PyptoMatmulOps(build_dir=tmp_path)
    try:
        setattr(scaled_dispatch, failure, True)
        with pytest.raises((RuntimeError, ValueError)):
            ops.block_scaled_matmul(*values)
        setattr(scaled_dispatch, failure, False)
        torch.testing.assert_close(
            ops.block_scaled_matmul(*values), _scalar_reference(*values), rtol=0, atol=0,
        )
    finally:
        ops.close()
    assert not list(tmp_path.iterdir())


def _compile_scaled(platform, output_directory):
    from pypto import ir
    from pypto.pypto_core import DataType
    from pypto.runtime import RunConfig

    decorator = importlib.import_module("pypto.jit.decorator")

    spec = importlib.util.spec_from_file_location(
        "_v41_scaled_kernel", ROOT / "pypto_serving/model/deepseek_v41/kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(decorator, "_ptoas_available", lambda: False)
        a, b, sa, sb = _values(16, 64, 96)
        compiled = module.make_block_scaled_matmul_kernel().compile(
            a, b, sa.T.contiguous(), sb, torch.empty(16, 64),
            config=RunConfig(
                platform=platform, codegen_only=True, save_kernels=True, save_kernels_dir=output_directory
            ),
        )
    assert compiled.platform == platform and compiled.output_indices == [4]
    entry = next(
        fn for fn in compiled.program.functions.values() if fn.func_type == ir.FunctionType.Orchestration
    )
    assert [param.type.dtype for param in entry.params] == [
        DataType.BF16, DataType.BF16, DataType.FP32, DataType.FP32, DataType.FP32,
    ]
    assert [[dim.value for dim in param.type.shape] for param in entry.params] == [
        [16, 96], [96, 64], [3, 16], [3, 64], [16, 64],
    ]
    assert list(Path(output_directory).rglob("*.pto")), "real PTO codegen must emit artifacts"


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_scaled_kernel_real_compiler(platform, tmp_path):
    pytest.importorskip("pypto.pypto_core", reason="requires the real PyPTO frontend")
    child = "import runpy,sys; runpy.run_path(sys.argv[1])['_compile_scaled'](sys.argv[2],sys.argv[3])"
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-c", child, str(Path(__file__).resolve()), platform, str(tmp_path)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture
def scaled_npu(scaled_module, tmp_path, monkeypatch):
    if os.getenv("PYPTO_V41_NPU_TESTS") != "1":
        pytest.skip("requires an allocated task-submit device and PYPTO_V41_NPU_TESTS=1")
    device = os.getenv("TASK_DEVICE", "")
    assert device.isascii() and device.isdecimal(), "TASK_DEVICE must be the allocated single device"

    def no_cpu(*_args, **_kwargs):
        pytest.fail("the real NPU test must not use the CPU provider")

    monkeypatch.setattr(scaled_module.TorchMatmulOps, "block_scaled_matmul", no_cpu)
    ops = scaled_module.PyptoMatmulOps(
        platform=os.getenv("PYPTO_V41_NPU_PLATFORM", "a2a3"), device_id=int(device), build_dir=tmp_path,
    )
    try:
        yield ops
    finally:
        ops.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("rows,columns,inner", [(5, 7, 32), (17, 65, 96)])
def test_scaled_real_npu_matches_scalar_oracle(scaled_npu, rows, columns, inner):
    values = _values(rows, columns, inner)
    expected = _scalar_reference(*values)
    actual = scaled_npu.block_scaled_matmul(*values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu" and actual.is_contiguous()
    a, b, sa, sb = values
    torch.testing.assert_close(scaled_npu.block_scaled_matmul(-a, b, sa, sb), -expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.getenv("PYPTO_V41_NPU_8K_TESTS") != "1", reason="requires explicit 8K hardware opt-in")
def test_scaled_8k_projection_runs_complete_k_per_dispatch(scaled_npu, monkeypatch):
    rows, inner, columns = 8192, 6144, 128
    rc = ((torch.arange(rows) % 17 - 8) / 8).bfloat16()
    cc = ((torch.arange(columns) % 13 - 6) / 8).bfloat16()
    signs = (1 - 2 * (torch.arange(inner) % 2)).bfloat16()
    a, b = rc[:, None] * signs[None, :], signs[:, None] * cc[None, :]
    group_scales = torch.tensor([0.5, 2.0]).repeat(inner // 64)
    sa = group_scales.expand(rows, -1)
    sb = torch.full((inner // 32, columns), 0.5)
    calls = []
    dispatch = scaled_npu._dispatch_scaled_into

    def recorded(left, right, left_scales, right_scales, output):
        calls.append(left.shape)
        dispatch(left, right, left_scales, right_scales, output)

    monkeypatch.setattr(scaled_npu, "_dispatch_scaled_into", recorded)
    actual = scaled_npu.block_scaled_matmul(a, b, sa, sb)
    assert all(shape[1] == inner for shape in calls)
    assert len(calls) < inner // 32
    multiplier = 32 * float(group_scales.sum()) * 0.5
    for start in range(0, rows, 256):
        expected = multiplier * rc[start : start + 256, None].float() * cc[None, :].float()
        torch.testing.assert_close(actual[start : start + 256], expected, rtol=0, atol=0)
