# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small CPU references for V4.1 checkpoint quantization, without device dependencies.

The checkpoint contract comes from DeepSeek-V4.1-Flash inference/{model,kernel,convert}.py,
revision dba1be0a40aa45a94ad051997016db3960a90277. FP4 expert weights pack the first
K element into the low nibble, with one UE8M0 scale per row and 32 K elements.
Dense weights use E4M3FN and one UE8M0 scale per 32 by 32 tile. Activation scales
are per row and 32 K elements. Main KV uses a different block16/E4M3 scale contract
and must not be passed to these weight converters.

Scalar encodings follow https://onnx.ai/onnx/technical/float8.html and the OCP MX
specification. All buffers contain raw bytes, including UE8M0 exponent bytes, not
numeric integer scales. Matrices are row-major [out_features, in_features]; no
transpose, device packing, padding, or Ascend ABI is implied. These references
use Python floating point and do not emulate BF16 output or FP32 accumulation.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from numbers import Real


BLOCK_SIZE = 32
Shape = tuple[int, int]
Buffer = bytes | bytearray | memoryview
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _code(value: int, upper: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= upper:
        raise ValueError(f"{name} must be an integer in [0, {upper}]")
    return value


def _finite(value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("values must be finite real numbers")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("values must be finite real numbers") from exc
    if not math.isfinite(result):
        raise ValueError("values must be finite real numbers")
    return result


def _shape(shape: Shape) -> Shape:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be a (rows, columns) tuple")
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
        raise ValueError("shape dimensions must be positive integers")
    if shape[1] % BLOCK_SIZE:
        raise ValueError("columns must be divisible by 32; padding must be explicit")
    return shape


def _buffer(data: Buffer, length: int, name: str) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name} must contain raw bytes")
    if isinstance(data, memoryview) and (data.itemsize != 1 or data.ndim != 1):
        raise ValueError(f"{name} must be a one-dimensional byte buffer")
    if len(data) != length:
        raise ValueError(f"{name} byte count: expected {length}, got {len(data)}")
    return bytes(data)


def decode_ue8m0(code: int) -> float:
    """Decode a positive scale: byte 0 is 2**-127, 127 is 1, and 255 is invalid NaN."""
    code = _code(code, 255, "UE8M0 code")
    if code == 255:
        raise ValueError("UE8M0 NaN scale (255) is not a valid checkpoint scale")
    return math.ldexp(1.0, code - 127)


def encode_ue8m0(value: float) -> int:
    """Encode an exact positive power of two; never silently round or relabel a scale."""
    value = _finite(value)
    fraction, exponent = math.frexp(value)
    code = exponent - 1 + 127
    if fraction != 0.5 or not 0 <= code <= 254:
        raise ValueError("UE8M0 scale must be an exact power of two in [2**-127, 2**127]")
    return code


def decode_fp4_e2m1(code: int) -> float:
    """Decode one E2M1 nibble, including the sign of zero."""
    code = _code(code, 15, "E2M1 code")
    value = _FP4_MAGNITUDES[code & 7]
    return -value if code & 8 else value


def decode_fp8_e4m3fn(code: int) -> float:
    """Decode finite E4M3FN; reject the two NaN encodings 0x7f and 0xff."""
    code = _code(code, 255, "E4M3FN code")
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        raise ValueError("E4M3FN NaN is not a valid checkpoint value")
    value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return -value if code & 128 else value


_FP8_MAGNITUDES = tuple(decode_fp8_e4m3fn(code) for code in range(127))


def _nearest_code(value: float, magnitudes: tuple[float, ...]) -> int:
    upper = bisect_left(magnitudes, value)
    if upper == 0:
        return 0
    if upper == len(magnitudes):
        return upper - 1
    lower = upper - 1
    lower_distance, upper_distance = value - magnitudes[lower], magnitudes[upper] - value
    if lower_distance == upper_distance:
        return lower if lower % 2 == 0 else upper
    return lower if lower_distance < upper_distance else upper


def encode_fp4_e2m1(value: float) -> int:
    """Round a finite scalar to nearest, ties to even, saturating at +/-6."""
    value = _finite(value)
    sign = 8 if math.copysign(1.0, value) < 0 else 0
    return sign | _nearest_code(abs(value), _FP4_MAGNITUDES)


def encode_fp8_e4m3fn(value: float) -> int:
    """Round a finite scalar to nearest, ties to even, saturating at +/-448."""
    value = _finite(value)
    sign = 128 if math.copysign(1.0, value) < 0 else 0
    return sign | _nearest_code(abs(value), _FP8_MAGNITUDES)


def pack_fp4_e2m1(values: Sequence[float]) -> bytes:
    """Quantize and pack adjacent scalar pairs: first low nibble, second high nibble."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("values must be a sequence of real numbers")
    if len(values) % 2:
        raise ValueError("FP4 packing requires an even element count; padding must be explicit")
    return bytes(
        encode_fp4_e2m1(values[index]) | (encode_fp4_e2m1(values[index + 1]) << 4)
        for index in range(0, len(values), 2)
    )


def unpack_fp4_e2m1(data: Buffer) -> list[float]:
    """Unpack raw signed-int8/uint8 storage bytes in increasing K order."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("FP4 storage must contain raw bytes")
    data = _buffer(data, len(data), "FP4 storage")
    return [decode_fp4_e2m1(code) for value in data for code in (value & 15, value >> 4)]


def dequantize_fp4_expert(data: Buffer, scales: Buffer, shape: Shape) -> list[list[float]]:
    """Decode [N,K/2] FP4 bytes with [N,K/32] UE8M0 scale bytes into [N,K]."""
    rows, columns = _shape(shape)
    data = _buffer(data, rows * columns // 2, "FP4 weight")
    scales = _buffer(scales, rows * columns // BLOCK_SIZE, "FP4 scale")
    decoded_scales = [decode_ue8m0(code) for code in scales]
    values = unpack_fp4_e2m1(data)
    return [
        [
            values[row * columns + col] * decoded_scales[(row * columns + col) // BLOCK_SIZE]
            for col in range(columns)
        ]
        for row in range(rows)
    ]


def _dequantize_fp8(data: Buffer, scales: Buffer, shape: Shape, scale_rows: int) -> list[list[float]]:
    rows, columns = _shape(shape)
    data = _buffer(data, rows * columns, "FP8 values")
    columns_of_scales = columns // BLOCK_SIZE
    scales = _buffer(scales, ((rows + scale_rows - 1) // scale_rows) * columns_of_scales, "FP8 scale")
    decoded_scales = [decode_ue8m0(code) for code in scales]
    return [
        [
            decode_fp8_e4m3fn(data[row * columns + col])
            * decoded_scales[(row // scale_rows) * columns_of_scales + col // BLOCK_SIZE]
            for col in range(columns)
        ]
        for row in range(rows)
    ]


def dequantize_fp8_dense(data: Buffer, scales: Buffer, shape: Shape) -> list[list[float]]:
    """Decode [N,K] E4M3FN with [ceil(N/32),K/32] UE8M0; allow a final partial N tile."""
    return _dequantize_fp8(data, scales, shape, BLOCK_SIZE)


def dequantize_fp8_activation(data: Buffer, scales: Buffer, shape: Shape) -> list[list[float]]:
    """Decode [M,K] E4M3FN with [M,K/32] UE8M0 activation scales."""
    return _dequantize_fp8(data, scales, shape, 1)


def quantize_fp8_activation(values: Sequence[Sequence[float]]) -> tuple[bytes, bytes]:
    """CPU act_quant reference: ceil-power-of-two scale from max(amax, 1e-4)/448.

    Returns E4M3FN bytes and UE8M0 bytes, both row-major. Input values are not first
    rounded to BF16. Values that require an unrepresentable UE8M0 scale are rejected.
    """
    if not isinstance(values, Sequence) or not values:
        raise ValueError("activations must be a nonempty rectangular matrix")
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) for row in values):
        raise ValueError("activations must be a nonempty rectangular matrix")
    rows, columns = _shape((len(values), len(values[0])))
    if any(len(row) != columns for row in values):
        raise ValueError("activations must be a rectangular matrix")
    data, scales = bytearray(), bytearray()
    for row in range(rows):
        for begin in range(0, columns, BLOCK_SIZE):
            block = [_finite(value) for value in values[row][begin : begin + BLOCK_SIZE]]
            required = max(max(abs(value) for value in block), 1e-4) / 448.0
            fraction, exponent = math.frexp(required)
            scale = math.ldexp(1.0, exponent - 1 if fraction == 0.5 else exponent)
            scales.append(encode_ue8m0(scale))
            data.extend(encode_fp8_e4m3fn(value / scale) for value in block)
    return bytes(data), bytes(scales)


def convert_fp4_expert_to_fp8_dense(data: Buffer, scales: Buffer, shape: Shape) -> tuple[bytes, bytes]:
    """Convert 32x32 tiles using official max(scale)/64, requiring exact preservation.

    The reference converter calls this lossless. Extreme scale spreads can underflow
    E4M3FN values, or max(scale)/64 can be outside UE8M0. Reject such inputs instead
    of silently claiming lossless conversion. This is a CPU format conversion only.
    """
    rows, columns = _shape(shape)
    if rows % BLOCK_SIZE:
        raise ValueError("FP4 to FP8 conversion requires complete 32x32 tiles")
    data = _buffer(data, rows * columns // 2, "FP4 weight")
    scales = _buffer(scales, rows * columns // BLOCK_SIZE, "FP4 scale")
    for code in scales:
        decode_ue8m0(code)
    values = unpack_fp4_e2m1(data)
    result = bytearray(rows * columns)
    result_scales = bytearray()
    blocks = columns // BLOCK_SIZE
    for block_row in range(rows // BLOCK_SIZE):
        for block_col in range(blocks):
            row_start = block_row * BLOCK_SIZE
            tile_scales = [
                scales[row * blocks + block_col] for row in range(row_start, row_start + BLOCK_SIZE)
            ]
            output_scale = max(tile_scales) - 6
            if output_scale < 0:
                raise ValueError("max(FP4 scale)/64 is below the UE8M0 representable range")
            result_scales.append(output_scale)
            for local_row, input_scale in enumerate(tile_scales):
                row = row_start + local_row
                for col in range(block_col * BLOCK_SIZE, (block_col + 1) * BLOCK_SIZE):
                    value = math.ldexp(values[row * columns + col], input_scale - output_scale)
                    code = encode_fp8_e4m3fn(value)
                    if decode_fp8_e4m3fn(code) != value:
                        raise ValueError(f"FP4 to FP8 conversion would lose precision at ({row}, {col})")
                    result[row * columns + col] = code
    return bytes(result), bytes(result_scales)


def _gemm(activation: list[list[float]], weight: list[list[float]]) -> list[list[float]]:
    return [[math.fsum(x * y for x, y in zip(row, output)) for output in weight] for row in activation]


def gemm_fp4_reference(
    activation: Buffer,
    activation_scales: Buffer,
    activation_shape: Shape,
    weight: Buffer,
    weight_scales: Buffer,
    weight_shape: Shape,
) -> list[list[float]]:
    """Compute A[M,K] @ W[N,K].T from FP8 activations and FP4 expert bytes."""
    if _shape(activation_shape)[1] != _shape(weight_shape)[1]:
        raise ValueError("activation and weight K dimensions must match")
    return _gemm(
        dequantize_fp8_activation(activation, activation_scales, activation_shape),
        dequantize_fp4_expert(weight, weight_scales, weight_shape),
    )


def gemm_fp8_reference(
    activation: Buffer,
    activation_scales: Buffer,
    activation_shape: Shape,
    weight: Buffer,
    weight_scales: Buffer,
    weight_shape: Shape,
) -> list[list[float]]:
    """Compute A[M,K] @ W[N,K].T from FP8 activations and dense weight bytes."""
    if _shape(activation_shape)[1] != _shape(weight_shape)[1]:
        raise ValueError("activation and weight K dimensions must match")
    return _gemm(
        dequantize_fp8_activation(activation, activation_scales, activation_shape),
        dequantize_fp8_dense(weight, weight_scales, weight_shape),
    )
