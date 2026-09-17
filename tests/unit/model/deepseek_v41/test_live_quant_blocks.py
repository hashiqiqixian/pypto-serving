# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in live checkpoint K32 blocks through the real A3 scaled-matmul provider.

Requires PYPTO_V41_CHECKPOINT_DIR, PYPTO_V41_NPU_TESTS=1 and allocated TASK_DEVICE.
Only the first 32 output rows and first K32 block are read, never a whole tensor.
The independent oracle uses native FP8/UE8M0 and the pinned official E2M1 table.
"""

import importlib
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
CASES = (
    pytest.param("layers.0.attn.wq_a", "fp8", id="layers.0.attn.wq_a.weight[0:32,0:32]"),
    pytest.param("layers.0.ffn.experts.0.w1", "fp4", id="layers.0.ffn.experts.0.w1.weight[0:32,0:16bytes]"),
)


@pytest.fixture
def live_modules(monkeypatch):
    if not os.getenv("PYPTO_V41_CHECKPOINT_DIR"):
        pytest.skip("requires PYPTO_V41_CHECKPOINT_DIR pointing to a real V4.1 checkpoint")
    if os.getenv("PYPTO_V41_NPU_TESTS") != "1":
        pytest.skip("requires an allocated A3 device and PYPTO_V41_NPU_TESTS=1")
    device = os.getenv("TASK_DEVICE", "")
    assert device.isascii() and device.isdecimal(), "TASK_DEVICE must identify the allocated single A3 device"
    assert os.getenv("PYPTO_V41_NPU_PLATFORM", "a2a3") == "a2a3", "this live test requires the A3 backend"
    # Missing dependencies after explicit opt-in are failures, not successful skips.
    torch = importlib.import_module("torch")
    safetensors = importlib.import_module("safetensors")
    pypto = importlib.import_module("pypto")
    assert not getattr(pypto, "__pypto_stub__", False), "the live test requires the real PyPTO runtime"
    assert hasattr(torch, "float8_e8m0fnu"), "native UE8M0 is required for the independent oracle"

    # Match the unit suite's package isolation without importing the serving engine.
    before = {key: value for key, value in sys.modules.items()
              if key == "pypto_serving" or key.startswith("pypto_serving.")}
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    package = types.ModuleType("pypto_serving")
    package.__path__ = [str(ROOT / "pypto_serving")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    try:
        store_module = importlib.import_module("pypto_serving.model.deepseek_v41.tensor_store")
        ops_module = importlib.import_module("pypto_serving.model.deepseek_v41.pypto_ops")

        def no_cpu(*_args, **_kwargs):
            pytest.fail("live checkpoint NPU validation must not use a CPU matmul provider")

        monkeypatch.setattr(ops_module.TorchMatmulOps, "matmul", no_cpu)
        monkeypatch.setattr(ops_module.TorchMatmulOps, "block_scaled_matmul", no_cpu)
        yield torch, safetensors, store_module, ops_module, int(device)
    finally:
        for name in tuple(sys.modules):
            if name == "pypto_serving" or name.startswith("pypto_serving."):
                sys.modules.pop(name, None)
        sys.modules.update(before)


@pytest.mark.parametrize("prefix,weight_format", CASES)
def test_live_checkpoint_block_matches_native_oracle_on_a3(
    live_modules, tmp_path, record_property, prefix, weight_format,
):
    torch, safetensors, store_module, ops_module, device = live_modules
    model_dir = Path(os.environ["PYPTO_V41_CHECKPOINT_DIR"]).resolve()
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    weight_name, scale_name = prefix + ".weight", prefix + ".scale"
    ranges = {
        weight_name: (slice(0, 32), slice(0, 16 if weight_format == "fp4" else 32)),
        scale_name: (slice(0, 32 if weight_format == "fp4" else 1), slice(0, 1)),
    }
    record_property("checkpoint_tensor", weight_name)
    record_property("logical_range", "rows[0:32], K[0:32]")
    record_property("weight_storage_range", "rows[0:32], columns[0:16]" if weight_format == "fp4"
                    else "rows[0:32], columns[0:32]")
    record_property("weight_shard", weight_map[weight_name])
    record_property("scale_tensor", scale_name)
    record_property("scale_range", "rows[0:32], K-block[0:1]" if weight_format == "fp4"
                    else "N-block[0:1], K-block[0:1]")
    record_property("scale_shard", weight_map[scale_name])

    # Independent payload reads, bypassing all serving quantization helpers.
    raw = {}
    for name, selection in ranges.items():
        path = (model_dir / weight_map[name]).resolve()
        assert path.parent == model_dir, "checkpoint shard must stay within the selected model directory"
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as reader:
            raw[name] = reader.get_slice(name)[selection].clone()
    assert raw[scale_name].dtype == torch.float8_e8m0fnu
    native_scales = raw[scale_name].to(torch.float64).reshape(-1)
    if weight_format == "fp8":
        assert raw[weight_name].dtype == torch.float8_e4m3fn
        native_values = raw[weight_name].to(torch.float64)
        native_scales = native_scales.expand(32)
    else:
        assert raw[weight_name].dtype == torch.int8
        # Official inference/convert.py:13-14 at dba1be0a40aa45a94ad051997016db3960a90277.
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float64,
        )
        packed = raw[weight_name].view(torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
        native_values = table[codes]
    assert native_values.shape == (32, 32) and native_scales.shape == (32,)
    assert bool(torch.isfinite(native_values).all() & torch.isfinite(native_scales).all())

    reads = []

    @contextmanager
    def bounded_open(path, device):
        assert device == "cpu"
        with safetensors.safe_open(str(path), framework="pt", device=device) as source:
            class Reader:
                def get_tensor(self, name):
                    pytest.fail(f"live block validation must never load a full tensor: {name}")

                def get_slice(self, name):
                    assert name in ranges, f"unexpected checkpoint payload read: {name}"
                    original = source.get_slice(name)

                    class Slice:
                        def get_shape(self):
                            return original.get_shape()

                        def __getitem__(self, selection):
                            assert selection == ranges[name], (
                                f"oversized or shifted read for {name}: {selection}"
                            )
                            reads.append(name)
                            return original[selection]

                    return Slice()

            yield Reader()

    store = store_module.DeepSeekV41TensorStore(
        model_dir, raw_config, rank=0, world_size=1, out_tile_rows=32,
        max_load_bytes=1 << 20, row_cache_bytes=0, placement="uncached", safe_open_fn=bounded_open,
    )
    try:
        assert store.matrix_format(weight_name) == weight_format
        blocks = store.matrix_tiles(weight_name)
        try:
            row, column, values, scales = next(blocks)
        finally:
            blocks.close()
        assert (row, column) == (0, 0) and values.shape == (32, 32) and scales.shape == (32,)
        assert values.dtype == torch.bfloat16 and scales.dtype == torch.float32
        torch.testing.assert_close(values.double(), native_values, rtol=0, atol=0, msg=weight_name)
        torch.testing.assert_close(scales.double(), native_scales, rtol=0, atol=0, msg=scale_name)
        assert reads == [weight_name, scale_name], "only one weight block and its scales may be read"
    finally:
        store.close()

    # Signed powers of two keep each 32-term E4M3/E2M1 dot product exact in FP32.
    columns = torch.arange(32)
    activation = torch.stack((1 - 2 * (columns % 2), (1 - 2 * (columns % 3 == 0).int()) / 2)).bfloat16()
    activation_scales = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    expected = (activation.double() @ native_values.T).float()
    expected = expected * activation_scales
    expected = expected * native_scales.float()[None, :]
    ops = ops_module.PyptoMatmulOps(
        platform="a2a3", device_id=device, max_buffer_bytes=1 << 20, max_cached_shapes=1, build_dir=tmp_path,
    )
    try:
        actual = ops.block_scaled_matmul(
            activation, values.T.contiguous(), activation_scales, scales[None, :].contiguous(),
        )
        assert actual.shape == (2, 32) and actual.dtype == torch.float32 and actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=weight_name + " rows[0:32], K[0:32]")
    finally:
        ops.close()
    assert not list(tmp_path.iterdir()), "the provider must release its private build artifacts"
