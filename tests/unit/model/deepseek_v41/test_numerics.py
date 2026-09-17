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
import math
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("PYPTO_V41_NPU_TESTS") == "1":
    import torch
else:
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


def _fp4_golden(values):
    """Scalar nearest-distance oracle, independent of tensor threshold encoding."""
    table = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    codes = []
    for value in values:
        code = min(range(8), key=lambda index: (abs(abs(value) - table[index]), index % 2))
        codes.append(code | (8 if math.copysign(1., value) < 0 else 0))
    return [low | (high << 4) for low, high in zip(codes[::2], codes[1::2])]


def _assert_ieee_encoding_boundaries(module, device):
    # Native CPU FP8 is the independent oracle; no native FP8 tensor is sent to A3.
    raw = torch.arange(256, dtype=torch.uint8)
    native = raw.view(torch.float8_e4m3fn).float()
    finite = (raw & 127) != 127
    decoded = module.decode_e4m3(raw.to(device)).cpu()
    assert torch.equal(decoded[finite].view(torch.int32), native[finite].view(torch.int32))
    assert torch.isnan(decoded[~finite]).all()
    midpoint = (native[:126] + native[1:127]) * .5
    neighbors = torch.cat((torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf"))),
                           midpoint, torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))))
    values = torch.cat((native[finite], neighbors, -neighbors, torch.tensor([1000., -1000.])))
    # A stride exercises the bit reinterpretation's explicit contiguous boundary.
    strided = torch.stack((values, values), -1).to(device)[:, 0]
    actual = module.encode_e4m3(strided)
    assert actual.device == device
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(actual.cpu(), expected)

    midpoint = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    positive = torch.cat((torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., 1000.]),
                          torch.nextafter(midpoint, torch.zeros_like(midpoint)), midpoint,
                          torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))))
    values = torch.cat((positive, -positive))
    actual = module.encode_e2m1(values.to(device))
    assert actual.device == device
    assert torch.equal(actual.cpu(), torch.tensor(_fp4_golden(values.tolist()), dtype=torch.uint8))
    fp4_bytes = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
    expected = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    decoded = module.decode_e2m1(fp4_bytes.to(device)).cpu()
    assert torch.equal(decoded.view(torch.int32), expected.view(torch.int32))


def _assert_ue8m0_scale_bits(module, device):
    raw = torch.arange(256, dtype=torch.uint8)
    actual = module.decode_ue8m0(raw.to(device))
    expected = torch.tensor([math.ldexp(1., code - 127) for code in range(255)], dtype=torch.float32)
    assert actual.device == device
    assert torch.equal(actual.cpu()[:255].view(torch.int32), expected.view(torch.int32))
    assert torch.isnan(actual.cpu()[255])


def _assert_power_scale_midpoints(module, device, fmt):
    fp8 = fmt.startswith("fp8")
    maximum = 448 if fp8 else 6
    exponents = (-22, -8, 0, 7, 118, 119) if fp8 else (-126, -125, -124, -8, 0, 7, 124, 125)
    # -336 at scale=1 is the FP8 0xfa/0xfb midpoint seen failing on A3;
    # the FP4 5.0 midpoint similarly must choose magnitude code 6, not 7.
    midpoint = [0., -0., 1.0625, -1.0625, 336., -336., 432., -432.] if fp8 else [
        0., -0., .25, -.25, .75, -.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5., -5.]
    row = midpoint + [float(maximum)] + [0.] * (31 - len(midpoint))
    values = torch.tensor([[math.ldexp(value, exponent) for value in row] for exponent in exponents],
                          dtype=torch.bfloat16)
    expected_scales = torch.tensor([[exponent + 127] for exponent in exponents], dtype=torch.uint8)
    expected_powers = torch.tensor([math.ldexp(1., exponent) for exponent in exponents])[:, None]
    normalized = values.float() / expected_powers
    if fp8:
        expected_bytes = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        expected_bytes = torch.tensor([_fp4_golden(row) for row in normalized.tolist()], dtype=torch.uint8)
    packed = module.quantize_rows(values.to(device), fmt, 32)
    assert packed.values.device == packed.scales.device == device
    assert torch.equal(packed.scales.cpu(), expected_scales)
    assert torch.equal(packed.values.cpu(), expected_bytes)


def _assert_nonpower_main_kv_midpoints(module, device):
    scales = torch.tensor([1.375, 1.75, 3.5])
    midpoints = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)
    normalized = [value for midpoint in midpoints for value in (midpoint, -midpoint)] + [6., -0.]
    values = (scales[:, None] * torch.tensor(normalized)[None, :]).to(torch.bfloat16)
    packed = module.quantize_rows(values.to(device), "fp4_e2m1_e4m3", 16)
    expected = torch.tensor([_fp4_golden(normalized)] * len(scales), dtype=torch.uint8)
    expected_scales = scales.to(torch.float8_e4m3fn).view(torch.uint8)[:, None]
    assert packed.values.device == packed.scales.device == device
    assert torch.equal(packed.scales.cpu(), expected_scales)
    assert torch.equal(packed.values.cpu(), expected)


def test_ieee_encoders_match_independent_cpu_oracles_at_all_rounding_boundaries(module):
    _assert_ieee_encoding_boundaries(module, torch.device("cpu"))


def test_all_ue8m0_scales_have_exact_fp32_bits_including_subnormal_and_nan(module):
    _assert_ue8m0_scale_bits(module, torch.device("cpu"))


@pytest.mark.parametrize("fmt", ["fp8_e4m3_ue8m0", "fp4_e2m1_ue8m0"])
def test_power_scale_quantization_preserves_midpoint_ties_and_negative_zero(module, fmt):
    _assert_power_scale_midpoints(module, torch.device("cpu"), fmt)


def test_main_kv_nonpower_scales_preserve_midpoints_and_negative_zero(module):
    _assert_nonpower_main_kv_midpoints(module, torch.device("cpu"))


@pytest.mark.parametrize("maximum,fmt", [(448, "fp8_e4m3_ue8m0"), (6, "fp4_e2m1_ue8m0")])
def test_scale_rounding_matches_pinned_fp32_multiply_and_ieee_ceil(module, maximum, fmt):
    boundaries = torch.tensor([maximum * 2. ** exponent for exponent in (-8, 0, 7)])
    maxima = torch.cat((torch.nextafter(boundaries, torch.zeros_like(boundaries)), boundaries,
                        torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))))

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    expected = []
    for amax in maxima.tolist():
        # kernel.py fast_round_scale uses an FP32 reciprocal multiply, followed
        # by exponent/mantissa extraction, not approximate log2 or FP64 math.
        required = f32(amax * f32(1. / maximum))
        bits = struct.unpack("<I", struct.pack("<f", required))[0]
        expected.append((bits >> 23) + bool(bits & 0x7fffff))
    packed = module.quantize_rows(maxima[:, None].expand(-1, 32), fmt, 32)
    assert torch.equal(packed.scales[:, 0], torch.tensor(expected, dtype=torch.uint8))


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


@pytest.fixture
def npu_device():
    """Use only the explicitly allocated card; opt-in setup failures are errors."""
    if os.environ.get("PYPTO_V41_NPU_TESTS") != "1":
        pytest.skip("set PYPTO_V41_NPU_TESTS=1 and TASK_DEVICE to run real Ascend numerics")
    raw_device = os.environ.get("TASK_DEVICE")
    if raw_device is None or not raw_device.isdecimal():
        pytest.fail("TASK_DEVICE must explicitly select one nonnegative integer NPU device")
    importlib.import_module("torch_npu")
    assert torch.npu.is_available(), "explicit NPU numerical tests require an available device"
    device = torch.device(f"npu:{int(raw_device)}")
    torch.npu.set_device(device)
    return device


def test_real_npu_ieee_encoding_boundaries_match_cpu(module, npu_device):
    _assert_ieee_encoding_boundaries(module, npu_device)


def test_real_npu_all_ue8m0_scale_bits_match_cpu(module, npu_device):
    _assert_ue8m0_scale_bits(module, npu_device)


@pytest.mark.parametrize("fmt", ["fp8_e4m3_ue8m0", "fp4_e2m1_ue8m0"])
def test_real_npu_power_scale_midpoints_and_signed_zero_match_cpu(module, npu_device, fmt):
    _assert_power_scale_midpoints(module, npu_device, fmt)


def test_real_npu_nonpower_main_kv_midpoints_match_cpu(module, npu_device):
    _assert_nonpower_main_kv_midpoints(module, npu_device)


@pytest.mark.parametrize("fmt,block", [("fp8_e4m3_ue8m0", 32), ("fp4_e2m1_ue8m0", 32),
                                      ("fp4_e2m1_e4m3", 16)])
def test_real_npu_quantization_bytes_scales_and_dequantization_match_cpu(module, npu_device, fmt, block):
    # Tiny deterministic rows include signed zero, FP4 midpoint ties and several
    # scale exponents. Native checkpoint float8 tensors never move onto the NPU.
    values = torch.linspace(-6, 6, 256).reshape(4, 64)
    values[0, :16] = torch.tensor([0., -0., .25, -.25, .75, -.75, 1.25, -1.25,
                                   1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5., -5.])
    values *= torch.tensor([1., .03125, 8., .00390625])[:, None]
    values = values.to(torch.bfloat16)
    expected = module.quantize_rows(values, fmt, block)
    with torch.inference_mode():
        actual = module.quantize_rows(values.to(npu_device), fmt, block)
        decoded = actual.dequantize()
    assert actual.values.device == actual.scales.device == decoded.device == npu_device
    assert actual.values.dtype == actual.scales.dtype == torch.uint8
    assert torch.equal(actual.values.cpu(), expected.values)
    assert torch.equal(actual.scales.cpu(), expected.scales)
    # Comparing BF16 storage catches signed-zero differences as well as values.
    assert torch.equal(decoded.cpu().view(torch.int16), expected.dequantize().view(torch.int16))


def test_real_npu_rms_norm_and_hc_pre_post_match_independent_cpu_formulas(module, npu_device):
    x = torch.tensor([[[1., -2., .5, 3.], [-.5, 4., -1., 2.]],
                      [[2., 1., -3., .25], [1., -2., 2., -4.]]], dtype=torch.bfloat16)
    pre = torch.tensor([[.25, .75], [.5, .125]])
    post = torch.tensor([[.5, 1.5], [.25, .75]])
    comb = torch.tensor([[[.1, .2], [.3, .4]], [[.5, .25], [.125, .75]]])
    update = torch.tensor([[1., -.5, 2., 3.], [-1., 2., .5, -.25]], dtype=torch.bfloat16)
    weight = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.bfloat16)
    eps = 1e-6
    normalized = update.float() / (update.float().square().mean(-1, keepdim=True) + eps).sqrt()
    expected_norm = (normalized * weight.float()).to(torch.bfloat16)
    expected_pre = torch.einsum("bhd,bh->bd", x.float(), pre).to(torch.bfloat16)
    expected_post = (torch.einsum("bij,bid->bjd", comb, x.float()) +
                     post[..., None] * update.float()[:, None]).to(torch.bfloat16)
    with torch.inference_mode():
        actual_norm = module.rms_norm(update.to(npu_device), weight.to(npu_device), eps)
        actual_pre = module.ModelMath.hc_pre(x.to(npu_device), pre.to(npu_device))
        actual_post = module.ModelMath.hc_post(update.to(npu_device), x.to(npu_device),
                                              post.to(npu_device), comb.to(npu_device))
    for actual, expected in ((actual_norm, expected_norm), (actual_pre, expected_pre),
                             (actual_post, expected_post)):
        assert actual.device == npu_device and actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_real_npu_hc_mixes_match_independent_cpu_formula(module, npu_device):
    x = torch.tensor([[[.25, -.5, 1., 2.], [1.5, .75, -1., .5]],
                      [[-1., 2., .5, .25], [.75, -.5, 1.25, 1.]]])
    projection = torch.arange(64).reshape(8, 8).float() / 128 - .25
    scale = torch.tensor([.3, -.2, .4])
    base = torch.linspace(-.2, .3, 8)
    eps, hc_eps, iterations = 1e-6, 1e-5, 3

    class DeviceWeights:
        def linear(self, value, name):
            assert name == "hc_fn" and value.device == npu_device
            return torch.nn.functional.linear(value, projection.to(npu_device))

        def weight(self, name):
            return {"hc_scale": scale, "hc_base": base}[name].to(npu_device)

    config = SimpleNamespace(text_config={"hc_mult": 2, "rms_norm_eps": eps,
                                          "hc_eps": hc_eps, "hc_sinkhorn_iters": iterations})
    flat = x.flatten(-2)
    logits = (flat @ projection.T) / (flat.square().mean(-1, keepdim=True) + eps).sqrt()
    expected_pre = (logits[:, :2] * scale[0] + base[:2]).sigmoid() + hc_eps
    expected_post = 2 * (logits[:, 2:4] * scale[1] + base[2:4]).sigmoid()
    mixed = (logits[:, 4:] * scale[2] + base[4:]).reshape(2, 2, 2)
    exponentials = (mixed - mixed.amax(-1, keepdim=True)).exp()
    expected_comb = exponentials / exponentials.sum(-1, keepdim=True) + hc_eps
    expected_comb /= expected_comb.sum(-2, keepdim=True) + hc_eps
    for _ in range(iterations - 1):
        expected_comb /= expected_comb.sum(-1, keepdim=True) + hc_eps
        expected_comb /= expected_comb.sum(-2, keepdim=True) + hc_eps
    with torch.inference_mode():
        actual = module.ModelMath(config, DeviceWeights()).hc_mixes(x.to(npu_device), "hc")
    for observed, expected in zip(actual, (expected_pre, expected_post, expected_comb)):
        assert observed.device == npu_device and observed.dtype == torch.float32
        torch.testing.assert_close(observed.cpu(), expected, rtol=2e-5, atol=2e-6)
