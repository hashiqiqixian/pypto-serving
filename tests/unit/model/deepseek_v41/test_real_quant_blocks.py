# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real checkpoint block CPU checks, not full-layer, model, or NPU goldens.

Each fixture is only row zero's first 32 K values and its scale. PyTorch native
FP8/UE8M0 byte views and the pinned official FP4 table provide independent oracles.
Dequantization must match exactly; FP64 GEMM uses absolute tolerance 1e-12 and
zero relative tolerance. Missing local PyTorch is an explicit skip, never PASS.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", reason="real checkpoint block CPU goldens require PyTorch")
if not all(hasattr(torch, name) for name in ("float8_e4m3fn", "float8_e8m0fnu")):
    pytest.skip(
        "PyTorch lacks native E4M3FN/UE8M0 dtypes for the independent oracle", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parents[4]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "v41_real_blocks_quantization", ROOT / "pypto_serving/model/deepseek_v41/quantization.py"
)
quantization = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(quantization)
FIXTURE = json.loads((ROOT / "tests/fixtures/deepseek_v41/real_quant_blocks.json").read_text())


def native_bytes(raw, dtype):
    return torch.tensor(list(raw), dtype=torch.uint8).view(dtype).to(torch.float64)


def case_buffers(case):
    buffers = case["buffers"]
    return bytes.fromhex(buffers["weight"]["hex"]), bytes.fromhex(buffers["scale"]["hex"])


def native_weight(case):
    weight, scales = case_buffers(case)
    shape = tuple(case["logical_shape"])
    assert shape == (1, 32), "This fixture oracle only describes one K32 block"
    if case["format"] == "fp8_dense":
        values = native_bytes(weight, torch.float8_e4m3fn)
    else:
        assert case["format"] == "fp4_expert"
        # inference/convert.py:13-14, revision dba1be0. First K value is the low nibble.
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float64,
        )
        packed = torch.tensor(list(weight), dtype=torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten().long()
        values = table[codes]
    return values.reshape(shape) * native_bytes(scales, torch.float8_e8m0fnu).reshape(1, 1)


def test_fixture_provenance_and_buffer_integrity():
    assert FIXTURE["model_id"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert FIXTURE["revision"] == "dba1be0a40aa45a94ad051997016db3960a90277"
    assert FIXTURE["shard"] == "model-00003-of-00048.safetensors"
    assert FIXTURE["header_sha256"] == "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449"
    assert {case["tensor"] for case in FIXTURE["cases"]} == {
        "layers.0.attn.wq_a",
        "layers.0.ffn.experts.0.w1",
    }
    for case in FIXTURE["cases"]:
        assert tuple(case["logical_shape"]) == (1, 32)
        for buffer in case["buffers"].values():
            raw = bytes.fromhex(buffer["hex"])
            assert hashlib.sha256(raw).hexdigest() == buffer["sha256"]
            first, last = buffer["file_byte_range"]
            assert last - first + 1 == len(raw)
        expected_dtype = "I8" if case["format"] == "fp4_expert" else "F8_E4M3"
        assert case["buffers"]["weight"]["storage_dtype"] == expected_dtype
        assert case["buffers"]["scale"]["storage_dtype"] == "F8_E8M0"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["tensor"])
def test_real_block_dequantization_matches_independent_torch_oracle(case):
    weight, scales = case_buffers(case)
    decode = (
        quantization.dequantize_fp4_expert
        if case["format"] == "fp4_expert"
        else quantization.dequantize_fp8_dense
    )
    actual = torch.tensor(decode(weight, scales, tuple(case["logical_shape"])), dtype=torch.float64)
    torch.testing.assert_close(actual, native_weight(case), rtol=0, atol=0)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["tensor"])
def test_real_block_gemm_matches_native_float64_matmul(case):
    # These deterministic values are exactly representable in E4M3FN. Use native
    # encoding and two distinct scale bytes so activation decoding is independent.
    values = torch.tensor(
        [[((-1) ** (row + col)) * ((row + col) % 8) / 8 for col in range(32)] for row in range(2)],
        dtype=torch.float64,
    )
    activation = bytes(values.to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist())
    activation_scales = bytes([127, 126])
    decoded_activation = native_bytes(activation, torch.float8_e4m3fn).reshape(2, 32)
    decoded_activation *= native_bytes(activation_scales, torch.float8_e8m0fnu).reshape(2, 1)
    expected = decoded_activation @ native_weight(case).T
    weight, scales = case_buffers(case)
    gemm = (
        quantization.gemm_fp4_reference if case["format"] == "fp4_expert" else quantization.gemm_fp8_reference
    )
    actual = gemm(activation, activation_scales, (2, 32), weight, scales, tuple(case["logical_shape"]))
    torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), expected, rtol=0, atol=1e-12)
