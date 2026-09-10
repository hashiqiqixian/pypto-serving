# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU format and arithmetic goldens; these do not establish a device ABI or real-layer accuracy."""

from __future__ import annotations
import math
import importlib.util
from pathlib import Path
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[4] / "pypto_serving/model/deepseek_v41/quantization.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v41_quantization_reference", _MODULE_PATH)
quantization = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quantization)


def test_fp4_official_table_and_nibble_order():
    expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    packed = bytes([16, 50, 84, 118, 152, 186, 220, 254])
    assert [quantization.decode_fp4_e2m1(i) for i in range(16)] == expected
    assert quantization.unpack_fp4_e2m1(packed) == expected
    assert quantization.pack_fp4_e2m1(expected) == packed
    assert math.copysign(1, quantization.unpack_fp4_e2m1(packed)[8]) == -1


def test_all_packed_byte_patterns_round_trip():
    data = bytes(range(256))
    assert quantization.pack_fp4_e2m1(quantization.unpack_fp4_e2m1(data)) == data


def test_fp4_ties_to_even_and_saturation():
    values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 99.0]
    assert [quantization.encode_fp4_e2m1(value) for value in values] == [0, 2, 2, 4, 4, 6, 6, 7]
    assert quantization.encode_fp4_e2m1(-99.0) == 15
    assert quantization.encode_fp4_e2m1(-0.0) == 8


def test_ue8m0_boundaries_are_exponents_not_integer_scales():
    expected = {0: 2.0 ** (-127), 1: 2.0 ** (-126), 126: 0.5, 127: 1.0, 128: 2.0, 254: 2.0**127}
    for code, value in expected.items():
        assert quantization.decode_ue8m0(code) == value
        assert quantization.encode_ue8m0(value) == code
    with pytest.raises(ValueError, match="NaN"):
        quantization.decode_ue8m0(255)


def test_ue8m0_rejects_nonrepresentable_scales():
    for value in [0, -1, 1.5, 2.0 ** (-128), 2.0**128, math.nan, math.inf]:
        with pytest.raises(ValueError):
            quantization.encode_ue8m0(value)


def test_fp8_finite_extremes_subnormal_and_negative_zero():
    expected = {
        0: 0.0,
        1: 2.0 ** (-9),
        7: 7 / 512,
        8: 2.0 ** (-6),
        56: 1.0,
        120: 256.0,
        126: 448.0,
        254: -448.0,
    }
    for code, value in expected.items():
        assert quantization.decode_fp8_e4m3fn(code) == value
        assert quantization.encode_fp8_e4m3fn(value) == code
    assert math.copysign(1, quantization.decode_fp8_e4m3fn(128)) == -1
    assert quantization.encode_fp8_e4m3fn(-0.0) == 128
    for code in [127, 255]:
        with pytest.raises(ValueError, match="NaN"):
            quantization.decode_fp8_e4m3fn(code)


def test_fp8_rounding_and_finite_code_roundtrip():
    assert quantization.encode_fp8_e4m3fn(1.0625) == 56
    assert quantization.encode_fp8_e4m3fn(1.1875) == 58
    assert quantization.encode_fp8_e4m3fn(2.0 ** (-10)) == 0
    assert quantization.encode_fp8_e4m3fn(500) == 126
    for code in range(256):
        if code not in (127, 255):
            assert quantization.encode_fp8_e4m3fn(quantization.decode_fp8_e4m3fn(code)) == code


def test_rejects_invalid_scalar_types_and_codes():
    for decode, maximum in [
        (quantization.decode_fp4_e2m1, 15),
        (quantization.decode_fp8_e4m3fn, 255),
        (quantization.decode_ue8m0, 255),
    ]:
        for value in [-1, maximum + 1, True, 1.0, "1", None]:
            with pytest.raises(ValueError):
                decode(value)
    for encode in [quantization.encode_fp4_e2m1, quantization.encode_fp8_e4m3fn, quantization.encode_ue8m0]:
        for value in [math.nan, math.inf, -math.inf, True, "1", None, 10**1000]:
            with pytest.raises(ValueError):
                encode(value)


def test_pack_requires_pairs_and_raw_buffers():
    for values in [[1.0], [1.0, math.inf], "12", b"12"]:
        with pytest.raises(ValueError):
            quantization.pack_fp4_e2m1(values)
    with pytest.raises(ValueError, match="raw bytes"):
        quantization.unpack_fp4_e2m1([34])


def test_expert_scale_is_per_row_and_k_block():
    actual = quantization.dequantize_fp4_expert(bytes([34]) * 64, bytes([127, 128, 129, 130]), (2, 64))
    assert actual == [[1.0] * 32 + [2.0] * 32, [4.0] * 32 + [8.0] * 32]


def test_dense_scale_is_per_32_by_32_tile_including_row_tail():
    actual = quantization.dequantize_fp8_dense(bytes([56]) * (33 * 64), bytes([127, 128, 129, 130]), (33, 64))
    assert actual[:32] == [[1.0] * 32 + [2.0] * 32] * 32
    assert actual[32] == [4.0] * 32 + [8.0] * 32


def test_activation_quantization_scale_floor_and_block_changes():
    values = [[448.0] * 32 + [896.0] * 32, [224.0] * 32 + [0.0] * 32]
    (data, scales) = quantization.quantize_fp8_activation(values)
    assert scales == bytes([127, 128, 126, 105])
    assert data == bytes([126]) * 96 + bytes(32)
    assert quantization.dequantize_fp8_activation(data, scales, (2, 64)) == values


def test_activation_scale_ceil_at_power_of_two_boundary():
    exact = [[448.0] * 32]
    next_value = [[math.nextafter(448.0, math.inf)] * 32]
    assert quantization.quantize_fp8_activation(exact)[1] == bytes([127])
    assert quantization.quantize_fp8_activation(next_value)[1] == bytes([128])


def test_gemm_fp4_hand_computed_multiple_rows_and_blocks():
    actual = quantization.gemm_fp4_reference(
        bytes([56]) * 128,
        bytes([127, 128, 128, 129]),
        (2, 64),
        bytes([34]) * 64,
        bytes([127, 128, 129, 130]),
        (2, 64),
    )
    assert actual == [[160.0, 640.0], [320.0, 1280.0]]


def test_gemm_fp8_hand_computed_dense_tile_boundary():
    actual = quantization.gemm_fp8_reference(
        bytes([56]) * 128,
        bytes([127, 128, 128, 129]),
        (2, 64),
        bytes([56]) * (33 * 64),
        bytes([127, 128, 129, 130]),
        (33, 64),
    )
    assert actual == [[160.0] * 32 + [640.0], [320.0] * 32 + [1280.0]]


def test_gemm_fp4_signed_packed_dot_products():
    actual = quantization.gemm_fp4_reference(
        bytes([0x38, 0xB8]) * 16,
        bytes([127]),
        (1, 32),
        bytes([0xA2]) * 16 + bytes([0x2A]) * 16,
        bytes([127, 127]),
        (2, 32),
    )
    assert actual == [[32.0, -32.0]]


def test_fp4_to_fp8_official_max_scale_offset():
    (data, scales) = quantization.convert_fp4_expert_to_fp8_dense(
        bytes([119]) * 512, bytes([127]) * 32, (32, 32)
    )
    assert data == bytes([124]) * 1024
    assert scales == bytes([121])


def test_fp4_to_fp8_multi_tile_preserves_values_and_gemm():
    shape = (64, 64)
    data = bytes(range(256)) * 8
    scales = bytes((120 + (row + col) % 9 for row in range(64) for col in range(2)))
    (fp8, fp8_scales) = quantization.convert_fp4_expert_to_fp8_dense(data, scales, shape)
    assert quantization.dequantize_fp8_dense(fp8, fp8_scales, shape) == quantization.dequantize_fp4_expert(
        data, scales, shape
    )
    (activation, a_scales) = quantization.quantize_fp8_activation(
        [[float(index - 32) for index in range(64)]]
    )
    assert quantization.gemm_fp8_reference(
        activation, a_scales, (1, 64), fp8, fp8_scales, shape
    ) == quantization.gemm_fp4_reference(activation, a_scales, (1, 64), data, scales, shape)


def test_fp4_to_fp8_rejects_loss_and_unrepresentable_scale():
    for scales, message in [([0] * 32, "below"), ([80] + [127] * 31, "lose precision")]:
        with pytest.raises(ValueError, match=message):
            quantization.convert_fp4_expert_to_fp8_dense(bytes([17]) * 512, bytes(scales), (32, 32))
    with pytest.raises(ValueError, match="complete 32x32"):
        quantization.convert_fp4_expert_to_fp8_dense(bytes(16), bytes([127]), (1, 32))


def test_dequantizers_reject_invalid_shapes_and_payloads():
    for function, payload in [
        (quantization.dequantize_fp4_expert, bytes(16)),
        (quantization.dequantize_fp8_dense, bytes(32)),
        (quantization.dequantize_fp8_activation, bytes(32)),
    ]:
        for shape in [(0, 32), (1, 16), (-1, 32), (True, 32), (1, 32.0), [1, 32], (1, 32, 1)]:
            with pytest.raises(ValueError):
                function(payload, bytes([127]), shape)
        for data, scales in [
            (payload[:-1], bytes([127])),
            (payload, b""),
            (payload, bytes([127, 127])),
            (payload, bytes([255])),
            (list(payload), bytes([127])),
        ]:
            with pytest.raises(ValueError):
                function(data, scales, (1, 32))


def test_dense_nan_and_accidental_expert_scale_layout_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        quantization.dequantize_fp8_dense(bytes([127]) + bytes(31), bytes([127]), (1, 32))
    with pytest.raises(ValueError, match="byte count"):
        quantization.dequantize_fp8_dense(bytes(32 * 32), bytes([127]) * 32, (32, 32))


def test_activation_matrix_validation():
    for values in [
        [],
        [[]],
        [[1.0] * 31],
        [[1.0] * 32, [1.0] * 64],
        [[math.nan] * 32],
        [[2.0**150] * 32],
        [1],
        ["x" * 32],
    ]:
        with pytest.raises(ValueError):
            quantization.quantize_fp8_activation(values)


def test_gemm_rejects_k_mismatch_before_decoding():
    for function in [quantization.gemm_fp4_reference, quantization.gemm_fp8_reference]:
        with pytest.raises(ValueError, match="K dimensions must match"):
            function(b"", b"", (1, 32), b"", b"", (1, 64))
