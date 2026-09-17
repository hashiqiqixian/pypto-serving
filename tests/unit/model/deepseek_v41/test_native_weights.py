# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native parameter bytes and layout through the real streaming checkpoint store."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

# Reuse the existing real-safetensors fixture and bounded-read instrumentation.
_spec = importlib.util.spec_from_file_location(
    "_native_weight_store_fixtures", Path(__file__).with_name("test_tensor_store.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
module = _fixtures.module
raw_config = _fixtures.raw_config
case = _fixtures.case
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def packing(module, monkeypatch):
    # Load the actual library packing implementation; its only configuration
    # dependency is MX_GROUP, so this CPU test needs no compiled PyPTO frontend.
    for name in ("models", "models.deepseek_v4_1_flash"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    config = ModuleType("models.deepseek_v4_1_flash.config")
    config.MX_GROUP = 32
    monkeypatch.setitem(sys.modules, config.__name__, config)
    name = "models.deepseek_v4_1_flash.quantization"
    path = ROOT / "pypto-lib/models/deepseek_v4_1_flash/quantization.py"
    spec = importlib.util.spec_from_file_location(name, path)
    quantization = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, quantization)
    spec.loader.exec_module(quantization)
    return importlib.import_module("pypto_serving.model.deepseek_v41.native_weights")


def _mx_bytes(logical):
    """Physical order: output block, K-group pair, output lane, pair lane."""
    groups, columns = logical.shape
    return torch.tensor([
        int(logical[k_pair * 2 + pair_lane, n_block * 16 + n_lane])
        for n_block in range(columns // 16)
        for k_pair in range(groups // 2)
        for n_lane in range(16)
        for pair_lane in range(2)
    ], dtype=torch.uint8).reshape(groups, columns)


@pytest.mark.parametrize("projection", ["wq_a", "wq_b", "wkv", "wo_b", "index_wq_b"])
def test_fp8_native_bytes_preserve_payload_scales_and_streaming(case, packing, projection):
    suffix = "indexer.wq_b" if projection == "index_wq_b" else projection
    name = f"layers.0.attn.{suffix}.weight"
    shape = case.tensors[name].shape
    finite_codes = torch.arange(256, dtype=torch.uint8)
    finite_codes = finite_codes[(finite_codes & 127) != 127]
    payload = finite_codes[torch.arange(shape.numel()) % len(finite_codes)].reshape(shape)
    case.tensors[name] = payload.view(torch.float8_e4m3fn)
    scale_name = name.removesuffix(".weight") + ".scale"
    scale_shape = case.tensors[scale_name].shape
    boundary_codes = torch.tensor([0, 1, 126, 127, 128, 129, 253, 254], dtype=torch.uint8)
    scales = boundary_codes[torch.arange(scale_shape.numel()) % len(boundary_codes)].reshape(scale_shape)
    case.tensors[scale_name] = scales.view(torch.float8_e8m0fnu)
    _fixtures.write_checkpoint(case.root, case.tensors)
    weights = _fixtures.store(case, out_tile_rows=32)
    try:
        result = packing.pack_native_attention_weights(weights, 0, 1, [projection, projection + "_scale"])
        assert set(result) == {projection, projection + "_scale"}
        assert result[projection].dtype == torch.float8_e4m3fn
        assert result[projection + "_scale"].dtype == torch.float8_e8m0fnu
        assert all(value.device.type == "cpu" and value.is_contiguous() for value in result.values())
        assert torch.equal(result[projection].view(torch.uint8), payload.T.contiguous())
        logical_scales = scales.repeat_interleave(32, dim=0)[:shape[0]].T
        assert torch.equal(result[projection + "_scale"].view(torch.uint8), _mx_bytes(logical_scales))
        assert {source for source, _ in case.reads} == {name, scale_name}
        for source, ranges in case.reads:
            if source == name:
                assert ranges[0].stop - ranges[0].start <= 32
                assert ranges[1].stop - ranges[1].start == 32
        # Native buffers own their bytes; mutating one cannot alter the mmap.
        result[projection].view(torch.uint8).zero_()
        reread = packing.pack_native_attention_weights(weights, 0, 1, [projection])
        assert set(reread) == {projection}
        assert torch.equal(reread[projection].view(torch.uint8), payload.T.contiguous())
    finally:
        weights.close()


@pytest.mark.parametrize("ratio", [1, 2])
def test_dense_parameter_mapping_keeps_native_precision_and_orientation(packing, ratio):
    tensors = {
        "q_norm.weight": torch.arange(64).bfloat16(),
        "kv_norm.weight": torch.arange(32).bfloat16(),
        "attn_sink": torch.tensor([1.0001, 2.0003], dtype=torch.float32),
        "wo_a.weight": torch.arange(2 * 32 * 64).reshape(2, 32, 64).bfloat16(),
        "compressor.wkv.weight": torch.arange(32 * 64).reshape(32, 64).bfloat16(),
        "compressor.wgate.weight": torch.arange(32 * 64).reshape(32, 64).bfloat16(),
        "compressor.norm.weight": torch.arange(32).bfloat16(),
        "indexer.wk.weight": torch.arange(32 * 64).reshape(32, 64).bfloat16(),
        "indexer.k_norm.weight": torch.arange(32).bfloat16(),
        "indexer.weights_proj.weight": torch.arange(2 * 64).reshape(2, 64).bfloat16(),
    }
    mapping = {
        "q_norm_weight": "q_norm.weight", "kv_norm_weight": "kv_norm.weight",
        "attn_sink": "attn_sink", "wo_a": "wo_a.weight", "compressor_wkv": "compressor.wkv.weight",
        "compressor_wgate": "compressor.wgate.weight", "compressor_norm_weight": "compressor.norm.weight",
        "index_wk": "indexer.wk.weight", "index_norm_weight": "indexer.k_norm.weight",
        "index_weights_proj": "indexer.weights_proj.weight",
    }

    class Store:
        world_size = 1

        def weight(self, name):
            assert name.startswith("layers.7.attn.")
            return tensors[name.removeprefix("layers.7.attn.")].clone()

    result = packing.pack_native_attention_weights(Store(), 7, ratio, list(mapping))
    for name, source in mapping.items():
        expected = tensors[source]
        if expected.ndim == 2:
            expected = expected.T
        dtype = torch.float32 if name == "attn_sink" or (
            ratio == 2 and name in ("compressor_wkv", "compressor_wgate")
        ) else torch.bfloat16
        assert result[name].dtype == dtype
        assert result[name].is_contiguous() and result[name].device.type == "cpu"
        torch.testing.assert_close(result[name], expected.to(dtype), rtol=0, atol=0)


def test_unknown_names_and_parallel_shards_fail_before_checkpoint_access(packing):
    class Store:
        world_size = 1

        def weight(self, name):
            pytest.fail("invalid native parameters must not read weights")

        def matrix_format(self, name):
            pytest.fail("invalid native parameters must not read projections")

    weights = Store()
    with pytest.raises(ValueError, match="unknown native"):
        packing.pack_native_attention_weights(weights, 0, 1, ["wq_a", "unused"])
    with pytest.raises(ValueError, match="unknown native"):
        packing.pack_native_attention_weights(weights, 0, 1, "wq_a")
    weights.world_size = 2
    with pytest.raises(ValueError, match="TP1"):
        packing.pack_native_attention_weights(weights, 0, 1, ["wq_a"])


def test_scale_only_selection_does_not_return_a_payload(case, packing):
    weights = _fixtures.store(case)
    try:
        result = packing.pack_native_attention_weights(weights, 0, 1, ["wq_a_scale"])
        assert set(result) == {"wq_a_scale"}
        assert result["wq_a_scale"].dtype == torch.float8_e8m0fnu
    finally:
        weights.close()
