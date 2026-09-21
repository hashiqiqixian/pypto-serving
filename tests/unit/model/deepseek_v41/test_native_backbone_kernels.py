# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real compiler and opt-in A5 tests for the TP1 backbone entry points.

Random actual-shape kernel tests are not checkpoint-to-text acceptance. Run the
device cases under a single-device task-submit allocation, with TASK_DEVICE and
PYPTO_V41_NATIVE_NPU_TESTS=1. All seven compiler cases use isolated processes.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

RUN_NPU = os.environ.get("PYPTO_V41_NATIVE_NPU_TESTS") == "1"
if RUN_NPU:
    import torch
else:
    torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[4]
MODES = ("mixes", "pre", "post", "norm", "routing", "routed", "shared")


def load_entries():
    library_root = Path(os.environ.get("PYPTO_LIB_ROOT", str(ROOT / "pypto-lib")))
    sys.path.insert(0, str(library_root))
    from models.deepseek_v4_1_flash import local_attention
    constants = local_attention.load_kernel_configuration(tp_size=1, ep_size=2)
    spec = importlib.util.spec_from_file_location(
        "_native_backbone_kernel_entries", ROOT / "pypto_serving/model/deepseek_v41/native_backbone_kernels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return constants, module.load_kernels()


def compile_entry(mode, path):
    import pypto.language as pl
    from pypto.runtime import RunConfig
    _, kernels = load_entries()
    kwargs = {"num_tokens": pl.RUNTIME} if mode in ("norm", "routing", "routed", "shared") else {}
    compiled = kernels[mode].compile(config=RunConfig(platform="a5", codegen_only=True, save_kernels=True,
                                                      save_kernels_dir=path), **kwargs)
    assert Path(compiled.output_dir).is_dir()
    assert compiled.output_indices


@pytest.mark.parametrize("mode", MODES)
def test_native_backbone_a5_codegen(mode, tmp_path):
    pytest.importorskip("pypto.pypto_core", reason="real PyPTO compiler extension is required")
    command = "import runpy,sys; runpy.run_path(sys.argv[1])['compile_entry'](sys.argv[2],sys.argv[3])"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", command, str(Path(__file__).resolve()), mode, str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]


@pytest.mark.skipif(not RUN_NPU, reason="allocated physical A5 opt-in required")
@pytest.mark.parametrize("rows", [1, 3, 32])
def test_real_a5_mhc_norm_and_router(rows, tmp_path):
    import pypto.language as pl
    from pypto.runtime import ChipWorker, RunConfig
    C, kernels = load_entries()
    from models.deepseek_v4_1_flash.golden import hc_mixes, hc_pre, hc_post, rms_norm, gate
    device = os.environ.get("TASK_DEVICE", "")
    assert device.isdecimal(), "TASK_DEVICE must identify the allocated A5 device"
    config = RunConfig(platform="a5", device_id=int(device), save_kernels=True)
    worker = ChipWorker(config=config)
    generator = torch.Generator().manual_seed(7391)

    def run(mode, *args):
        from dataclasses import replace
        kwargs = {"num_tokens": pl.RUNTIME} if mode in ("norm", "routing") else {}
        compiled = kernels[mode].compile(config=replace(config, save_kernels_dir=str(tmp_path / mode)), **kwargs)
        worker.register(compiled)(*args)

    try:
        x = (torch.randn((rows, C.HC_MULT, C.D), generator=generator) * .2).bfloat16().float()
        function = torch.randn((C.MIX_HC, C.HC_DIM), generator=generator) * .005
        scale, base = torch.tensor([.2, .3, .1]), torch.randn(C.MIX_HC, generator=generator) * .1
        pre = torch.empty((rows, C.HC_MULT))
        post, comb = torch.empty_like(pre), torch.empty((rows, C.HC_MULT, C.HC_MULT))
        run("mixes", x, function, scale, base, pre, post, comb)
        expected = hc_mixes(x, function, scale, base)
        for actual, reference in zip((pre, post, comb), expected):
            torch.testing.assert_close(actual, reference, rtol=.003, atol=.003)
        collapsed = torch.empty((rows, C.D), dtype=torch.bfloat16)
        run("pre", x, pre, collapsed)
        torch.testing.assert_close(collapsed, hc_pre(x, pre).bfloat16(), rtol=.008, atol=.008)
        expanded = torch.empty_like(x)
        run("post", collapsed, x, post, comb, expanded)
        torch.testing.assert_close(expanded, hc_post(collapsed, x, post, comb).bfloat16().float(), rtol=.008, atol=.008)
        norm_weight = (torch.randn(C.D, generator=generator) * .01 + 1).bfloat16()
        normalized = torch.empty_like(collapsed)
        run("norm", collapsed, norm_weight, normalized, rows)
        torch.testing.assert_close(normalized, rms_norm(collapsed, norm_weight).bfloat16(), rtol=.008, atol=.008)
        padded = torch.zeros((32, C.D), dtype=torch.bfloat16)
        padded[:rows] = normalized
        gate_weight = torch.randn((C.N_EXPERTS, C.D), generator=generator) * .01
        bias = torch.randn(C.N_EXPERTS, generator=generator) * .03
        indices = torch.empty((32, C.TOPK), dtype=torch.int32)
        weights = torch.empty((32 * C.TOPK, C.AUX_WIDTH))
        run("routing", padded, gate_weight, bias, indices, weights, rows)
        reference_weights, reference_indices = gate(normalized, gate_weight, bias)
        assert torch.equal(indices[:rows].long(), reference_indices.long())
        torch.testing.assert_close(weights[:rows*C.TOPK, 0].reshape(rows, C.TOPK), reference_weights,
                                   rtol=.003, atol=.003)
    finally:
        worker.close()


@pytest.mark.skipif(not RUN_NPU, reason="allocated physical A5 opt-in required")
@pytest.mark.parametrize("mode", ["routed", "shared"])
@pytest.mark.parametrize("rows", [1, 3, 32])
def test_real_a5_expert_pipeline(mode, rows, tmp_path):
    import pypto.language as pl
    from pypto.runtime import ChipWorker, RunConfig
    C, kernels = load_entries()
    from models.deepseek_v4_1_flash.quantization import quantize_mxfp4_weight, dequantize_mxfp4
    from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale
    from models.deepseek_v4_1_flash.moe import _golden_expert
    device = os.environ.get("TASK_DEVICE", "")
    assert device.isdecimal(), "TASK_DEVICE must identify the allocated A5 device"
    generator = torch.Generator().manual_seed(5197)
    x = torch.zeros((32, C.D), dtype=torch.bfloat16)
    x[:rows] = torch.randn((rows, C.D), generator=generator).bfloat16() * .1
    packed, decoded = [], []
    for outputs, inner in ((C.MOE_INTER, C.D), (C.D, C.MOE_INTER), (C.MOE_INTER, C.D)):
        if mode == "routed":
            weight = torch.randn((outputs, inner), generator=generator) * .005
            payload, scales = quantize_mxfp4_weight(weight)
            decoded.append(dequantize_mxfp4(payload, scales))
            scales = scales.view(torch.float8_e8m0fnu)
        else:
            payload = (torch.randn((inner, outputs), generator=generator) * .7).to(torch.float8_e4m3fn)
            codes = torch.full((inner // 32, outputs), 120, dtype=torch.uint8)
            scales = pack_mx_b_scale(codes).view(torch.float8_e8m0fnu)
            decoded.append((payload.float() * 2**-7).T.contiguous())
        packed.extend((payload, scales))
    coefficients = torch.zeros((32, C.AUX_WIDTH))
    coefficients[:rows, 0] = 1 if mode == "shared" else .35
    expected = _golden_expert(x[:rows], *decoded, coefficients[:rows, 0])
    output = torch.empty_like(x)
    config = RunConfig(platform="a5", device_id=int(device), save_kernels=True, save_kernels_dir=str(tmp_path))
    worker = ChipWorker(config=config)
    try:
        compiled = kernels[mode].compile(num_tokens=pl.RUNTIME, config=config)
        worker.register(compiled)(x, *packed, coefficients, output, rows)
        assert bool(torch.isfinite(output[:rows]).all())
        torch.testing.assert_close(output[:rows].float(), expected.float(), rtol=.025, atol=.002)
    finally:
        worker.close()
