# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent dtype goldens and arithmetic-order tests; no real-model acceptance claim."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "pypto_serving/model/deepseek_v41/numerics.py"
    spec = importlib.util.spec_from_file_location("_v41_numerics", path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    return result


def test_all_e4m3_encodings_and_midpoint_rounding(module):
    bits = torch.arange(256, dtype=torch.uint8)
    native = bits.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(module.decode_e4m3(bits), native, rtol=0, atol=0, equal_nan=True)
    finite = bits[(bits & 127) != 127]
    assert torch.equal(module.encode_e4m3(finite.view(torch.float8_e4m3fn).float()), finite)
    positive = native[:127]
    midpoint = (positive[:-1] + positive[1:]) / 2
    values = torch.cat((midpoint, -midpoint, torch.tensor([0., -0., 448., -448.])))
    assert torch.equal(module.encode_e4m3(values), values.to(torch.float8_e4m3fn).view(torch.uint8))


def test_e2m1_low_nibble_signed_zero_and_ties(module):
    expected = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
    actual = module.decode_e2m1(packed)
    assert torch.equal(actual, expected) and torch.equal(actual.signbit(), expected.signbit())
    assert torch.equal(module.encode_e2m1(expected), packed)
    midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 100.])
    torch.testing.assert_close(module.decode_e2m1(module.encode_e2m1(midpoints)),
                               torch.tensor([0., 1., 1., 2., 2., 4., 4., 6.]), rtol=0, atol=0)


@pytest.mark.parametrize("fmt,block", [("fp8_e4m3_ue8m0", 32), ("fp4_e2m1_ue8m0", 32),
                                      ("fp4_e2m1_e4m3", 16)])
def test_quantizer_scale_contract(module, fmt, block):
    x = torch.linspace(-6, 6, 128).reshape(2, 64).to(torch.bfloat16)
    packed = module.quantize_rows(x, fmt, block)
    maximum = x.float().unflatten(-1, (-1, block)).abs().amax(-1)
    if fmt.endswith("e4m3"):
        expected = (maximum / 6).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        expected = (torch.ceil(torch.log2(maximum / (448 if fmt.startswith("fp8") else 6))) + 127).byte()
    assert torch.equal(packed.scales, expected)
    assert packed.values.shape == (2, 64 if fmt.startswith("fp8") else 32)
    assert packed.dequantize().shape == x.shape
    assert torch.isfinite(packed.dequantize()).all()


def test_main_kv_scale_overflow_is_reported(module):
    with pytest.raises(ValueError, match="overflows"):
        module.quantize_rows(torch.full((1, 32), 3000.), "fp4_e2m1_e4m3", 16)


def test_hc_residual_orientation(module):
    residual = torch.tensor([[[1., 2.], [3., 5.]]])
    output = torch.tensor([[7., 11.]])
    post = torch.tensor([[.5, 2.]])
    comb = torch.tensor([[[.1, .2], [.3, .4]]])
    expected = torch.einsum("bij,bid->bjd", comb, residual) + post[..., None] * output[:, None]
    torch.testing.assert_close(module.ModelMath.hc_post(output, residual, post, comb), expected)


def test_routing_weight_is_applied_before_second_quantized_projection(module):
    calls = []
    class Ops:
        def linear(self, x, name):
            calls.append((name, x.clone()))
            return {"e.w1": x * 2, "e.w3": x * 3}.get(name, x)
    math = module.ModelMath(SimpleNamespace(text_config={"hc_mult": 2, "rms_norm_eps": 1e-20,
                                                         "swiglu_limit": 10}), Ops())
    x = torch.tensor([[2., -3.]], dtype=torch.bfloat16)
    weight = torch.tensor([[.37]])
    math.expert(x, "e", weight)
    expected = (torch.nn.functional.silu((x * 2).float().clamp_max(10)) *
                (x * 3).float().clamp(-10, 10) * weight).to(x.dtype)
    assert calls[-1][0] == "e.w2"
    torch.testing.assert_close(calls[-1][1], expected, rtol=0, atol=0)


def test_fp32_inputs_are_preserved_for_routing_and_head(module):
    class Weights:
        def matrix_shape(self, name):
            return 1, 2
        def matrix_format(self, name):
            return "float32"
        def matrix_tiles(self, name):
            yield 0, 0, torch.tensor([[1.001, -1.]]), None
    ops = module.TensorOps(Weights())
    x = torch.tensor([[1.003, 1.]])
    torch.testing.assert_close(ops.linear(x, "head"), x @ torch.tensor([[1.001], [-1.]]), rtol=0, atol=0)
