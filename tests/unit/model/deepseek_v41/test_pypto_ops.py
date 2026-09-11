# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU numerics and dispatch-boundary tests; fake JIT is never NPU numerical evidence."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="matrix provider tests require CPU Torch")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def ops_module(monkeypatch):
    package_name = "_v41_ops_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "pypto_serving/model/deepseek_v41")]
    monkeypatch.setitem(sys.modules, package_name, package)
    name = package_name + ".pypto_ops"
    spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0]) / "pypto_ops.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_dispatch(ops_module, monkeypatch):
    @dataclasses.dataclass
    class RunConfig:
        platform: str
        device_id: int
        save_kernels: bool
        save_kernels_dir: str | None = None

    state = SimpleNamespace(created=0, calls=[], fail=False, nonfinite=False)

    def factory():
        state.created += 1
        identity = state.created

        def kernel(a, b, output, *, config):
            state.calls.append((identity, a.clone(), b.clone(), output.shape, config))
            # Simulate a compiler-owned artifact only to test its cleanup path.
            (Path(config.save_kernels_dir) / "artifact.txt").write_text("dispatch test")
            if state.fail:
                raise RuntimeError("synthetic dispatch failure")
            output.copy_(a.float() @ b.float())
            if state.nonfinite:
                output[0, 0] = float("inf")

        return kernel

    runtime = types.ModuleType("pypto.runtime")
    runtime.RunConfig = RunConfig
    kernels = types.ModuleType("_v41_ops_test.kernels")
    kernels.make_bf16_matmul_kernel = factory
    monkeypatch.setitem(sys.modules, "pypto.runtime", runtime)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    return state


def reference(a, b):
    # Small scalar dot products provide a reference independent of provider GEMM.
    a, b = a.detach(), b.detach()
    return torch.tensor(
        [
            [sum(float(a[i, k]) * float(b[k, j]) for k in range(a.shape[1])) for j in range(b.shape[1])]
            for i in range(a.shape[0])
        ],
        dtype=torch.float32,
    )


def test_torch_provider_uses_bf16_values_and_returns_fp32_without_aliasing(ops_module):
    a = torch.tensor([[1.0, 2.0, -3.0], [0.5, -0.25, 4.0]], dtype=torch.bfloat16, requires_grad=True)
    b = torch.tensor([[0.125, 2.0], [-4.0, 0.5], [2.0, -1.0]], dtype=torch.bfloat16)
    before = a.detach().clone(), b.clone()
    ops = ops_module.TorchMatmulOps()
    result = ops.matmul(a, b)
    assert result.dtype == torch.float32 and result.is_contiguous() and not result.requires_grad
    torch.testing.assert_close(result, reference(a, b), rtol=0, atol=0)
    result.zero_()
    assert torch.equal(a, before[0]) and torch.equal(b, before[1])
    ops.close()


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (5, 7, 3), (17, 65, 33), (16, 64, 64), (32, 128, 128)])
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_pypto_bridge_padding_crop_and_runconfig(ops_module, fake_dispatch, tmp_path, m, n, k, platform):
    # Strided inputs exercise layout normalization at the public bridge boundary.
    a = (torch.arange(m * k * 2).reshape(m, k * 2) % 9 - 4).to(torch.bfloat16)[:, ::2]
    b = (torch.arange(k * n).reshape(k, n) % 7 - 3).to(torch.bfloat16)
    before = a.clone(), b.clone()
    ops = ops_module.PyptoMatmulOps(platform=platform, device_id=3, build_dir=tmp_path)
    try:
        result = ops.matmul(a, b)
        assert result.shape == (m, n) and result.dtype == torch.float32 and result.is_contiguous()
        torch.testing.assert_close(result, reference(a, b), rtol=0, atol=0)
        _, lhs, rhs, shape, config = fake_dispatch.calls[-1]
        assert lhs.dtype == rhs.dtype == torch.bfloat16
        assert lhs.is_contiguous() and rhs.is_contiguous()
        assert lhs.shape[0] % 16 == lhs.shape[1] % 64 == rhs.shape[1] % 64 == 0
        assert shape == (lhs.shape[0], rhs.shape[1])
        assert torch.count_nonzero(lhs[m:]) == torch.count_nonzero(lhs[:, k:]) == 0
        assert torch.count_nonzero(rhs[k:]) == torch.count_nonzero(rhs[:, n:]) == 0
        assert config.platform == platform and config.device_id == 3 and config.save_kernels
        assert Path(config.save_kernels_dir).is_relative_to(tmp_path)
        result.zero_()
        assert torch.equal(a, before[0]) and torch.equal(b, before[1])
    finally:
        ops.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_shape_cache_lru_reuses_hot_shape_evicts_cold_shape_and_preserves_user_files(
    ops_module, fake_dispatch, tmp_path, platform
):
    marker = tmp_path / "user-file.txt"
    marker.write_text("preserve")
    ops = ops_module.PyptoMatmulOps(platform=platform, build_dir=tmp_path, max_cached_shapes=2)

    def run(rows):
        return ops.matmul(torch.ones(rows, 1, dtype=torch.bfloat16), torch.ones(1, 1, dtype=torch.bfloat16))

    try:
        run(1)
        run(17)
        cold_path = Path(fake_dispatch.calls[-1][-1].save_kernels_dir)
        run(2)  # Same padded shape as the first call, now most recently used.
        assert fake_dispatch.created == 2
        assert fake_dispatch.calls[0][0] == fake_dispatch.calls[-1][0]
        run(33)
        assert fake_dispatch.created == 3 and not cold_path.exists()
        assert len(list(Path(ops._artifacts.name).iterdir())) == 2
        run(17)
        assert fake_dispatch.created == 4
    finally:
        ops.close()
    ops.close()
    assert list(tmp_path.iterdir()) == [marker] and marker.read_text() == "preserve"
    with pytest.raises(RuntimeError, match="closed"):
        run(1)


@pytest.mark.parametrize(
    "provider,kwargs",
    [
        ("TorchMatmulOps", {}),
        ("PyptoMatmulOps", {"platform": "a2a3"}),
        ("PyptoMatmulOps", {"platform": "a5"}),
    ],
)
@pytest.mark.parametrize("bad", ["dtype", "rank", "empty", "inner", "nan"])
def test_provider_rejects_invalid_matrix_contract(ops_module, fake_dispatch, provider, kwargs, bad):
    a, b = torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(3, 2, dtype=torch.bfloat16)
    if bad == "dtype":
        a = a.float()
    elif bad == "rank":
        a = a.reshape(-1)
    elif bad == "empty":
        a = a[:0]
    elif bad == "inner":
        b = b[:2]
    else:
        a[0, 0] = float("nan")
    ops = getattr(ops_module, provider)(**kwargs)
    try:
        with pytest.raises(ValueError, match="BF16 matrix|dimensions|finite"):
            ops.matmul(a, b)
        assert fake_dispatch.created == 0
    finally:
        ops.close()


@pytest.mark.parametrize(
    "provider,kwargs",
    [
        ("TorchMatmulOps", {}),
        ("PyptoMatmulOps", {"platform": "a2a3"}),
        ("PyptoMatmulOps", {"platform": "a5"}),
    ],
)
def test_budget_rejects_before_padding_or_dispatch(ops_module, fake_dispatch, monkeypatch, provider, kwargs):
    ops = getattr(ops_module, provider)(max_buffer_bytes=1, **kwargs)
    a, b = torch.ones(1, 1, dtype=torch.bfloat16), torch.ones(1, 1, dtype=torch.bfloat16)

    def no_padding(*args, **kwargs):
        pytest.fail("over-budget dispatch must fail before allocating padding")

    monkeypatch.setattr(ops_module.PyptoMatmulOps, "_pad", no_padding)
    try:
        with pytest.raises(ValueError, match="budget"):
            ops.matmul(a, b)
        assert fake_dispatch.created == 0
    finally:
        ops.close()


@pytest.mark.parametrize("failure", ["fail", "nonfinite"])
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_dispatch_error_propagates_and_owned_artifacts_are_cleaned(
    ops_module, fake_dispatch, tmp_path, monkeypatch, failure, platform
):
    setattr(fake_dispatch, failure, True)

    def no_fallback(*args, **kwargs):
        pytest.fail("hardware dispatch failures must never select the CPU reference provider")

    monkeypatch.setattr(ops_module.TorchMatmulOps, "matmul", no_fallback)
    ops = ops_module.PyptoMatmulOps(platform=platform, build_dir=tmp_path)
    try:
        with pytest.raises((ValueError, RuntimeError), match="finite|synthetic dispatch failure"):
            ops.matmul(torch.ones(1, 1, dtype=torch.bfloat16), torch.ones(1, 1, dtype=torch.bfloat16))
    finally:
        ops.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs", [{"device_id": -1}, {"device_id": True}, {"max_cached_shapes": 0}, {"max_buffer_bytes": 0}]
)
def test_pypto_provider_rejects_invalid_resource_settings(ops_module, fake_dispatch, kwargs):
    with pytest.raises(ValueError):
        ops_module.PyptoMatmulOps(**kwargs)


@pytest.mark.parametrize("platform", ["a3", "a2a3sim", "a5sim", "", None, 1])
def test_pypto_provider_rejects_unsupported_platform_before_creating_artifacts(
    ops_module, fake_dispatch, tmp_path, platform
):
    with pytest.raises(ValueError, match="platform"):
        ops_module.PyptoMatmulOps(platform=platform, build_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [] and fake_dispatch.created == 0


def test_pypto_provider_defaults_to_a3_hardware_backend(ops_module, fake_dispatch, tmp_path):
    ops = ops_module.PyptoMatmulOps(build_dir=tmp_path)
    try:
        ops.matmul(torch.ones(1, 1, dtype=torch.bfloat16), torch.ones(1, 1, dtype=torch.bfloat16))
        assert fake_dispatch.calls[-1][-1].platform == "a2a3"
    finally:
        ops.close()


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_pypto_provider_rejects_runtime_platform_substitution(
    ops_module, fake_dispatch, tmp_path, monkeypatch, platform
):
    runtime = sys.modules["pypto.runtime"]
    original = runtime.RunConfig

    def substituted_config(**kwargs):
        kwargs["platform"] = "a5" if platform == "a2a3" else "a2a3"
        return original(**kwargs)

    monkeypatch.setattr(runtime, "RunConfig", substituted_config)
    with pytest.raises(RuntimeError, match="did not select the requested"):
        ops_module.PyptoMatmulOps(platform=platform, build_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [] and fake_dispatch.created == 0
