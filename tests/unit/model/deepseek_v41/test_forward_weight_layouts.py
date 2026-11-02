# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native projection bytes and source axes for the V4.1 forward ABI."""

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41 import weight_loader
from pypto_serving.model.deepseek_v41.weight_loader import (
    DecodeWeightLayout, ResidentFullDecodeWeights, V41WeightLoader,
)
from pypto_serving.model.deepseek_v41.weight_spec import TensorSpec, backbone_weight_specs


def _loader(name, value, *, groups=2):
    loader = V41WeightLoader.__new__(V41WeightLoader)
    loader.tp_size = loader.ep_size = 1
    loader.tp_rank = loader.ep_rank = 0
    loader.text = {"o_groups": groups}
    loader.max_load_bytes = 1 << 24
    dtype = "F8_E4M3" if value.dtype == torch.float8_e4m3fn else "BF16"
    loader.specs = {name: TensorSpec(tuple(value.shape), dtype, name, "identity")}
    tensors = {name: value}
    if dtype == "F8_E4M3":
        scale_name = name.removesuffix(".weight") + ".scale"
        scale = torch.full((value.shape[0] // 32, value.shape[1] // 32), 127,
                           dtype=torch.uint8).view(torch.float8_e8m0fnu)
        tensors[scale_name] = scale
        loader.specs[scale_name] = TensorSpec(tuple(scale.shape), "F8_E8M0", scale_name, "identity")
    loader._read = lambda key, ranges, session=None: tensors[key][ranges].contiguous()
    return loader


def _nz_reference(logical):
    rows, columns = logical.shape[-2:]
    c0 = 32 // logical.element_size()
    matrices = logical.reshape(-1, rows, columns)
    blocks = [matrix[row:row + 16, column:column + c0].contiguous().view(torch.uint8).flatten()
              for matrix in matrices
              for column in range(0, columns, c0)
              for row in range(0, rows, 16)]
    return torch.cat(blocks).view(logical.dtype).reshape(logical.shape)


@pytest.mark.parametrize("suffix", [
    "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b", "attn.indexer.wq_b",
])
def test_native_fp8_projection_has_nz_bytes_after_input_major_packing(suffix):
    name = "layers.0." + suffix + ".weight"
    rows = 128 if suffix == "attn.wo_a" else 64
    payload = (torch.arange(rows * 64) % 120).to(torch.uint8).reshape(rows, 64)
    payload = payload.view(torch.float8_e4m3fn)
    logical = (payload.reshape(2, 64, 64).transpose(-1, -2).contiguous()
               if suffix == "attn.wo_a" else payload.t().contiguous())

    bundle = _loader(name, payload).load(name)

    assert bundle.weight.shape == logical.shape
    assert torch.equal(bundle.weight.view(torch.uint8), _nz_reference(logical).view(torch.uint8))
    assert not torch.equal(bundle.weight.view(torch.uint8), logical.view(torch.uint8))
    assert bundle.scale.dtype == torch.float8_e8m0fnu
    assert torch.all(bundle.scale.view(torch.uint8) == 127)


@pytest.mark.parametrize("suffix", [
    "attn.compressor.wkv", "attn.compressor.wgate", "attn.indexer.wk", "attn.indexer.weights_proj",
])
def test_native_dense_projection_preserves_bf16_and_packs_nz(suffix):
    name = "layers.0." + suffix + ".weight"
    payload = torch.arange(32 * 64).reshape(32, 64).to(torch.bfloat16)
    bundle = _loader(name, payload).load(name)

    assert bundle.weight.dtype == torch.bfloat16
    assert bundle.weight.shape == (64, 32)
    assert torch.equal(bundle.weight, _nz_reference(payload.t().contiguous()))


def test_shared_expert_projection_keeps_its_nd_payload():
    name = "layers.0.ffn.shared_experts.w1.weight"
    payload = (torch.arange(64 * 64) % 120).to(torch.uint8).reshape(64, 64)
    bundle = _loader(name, payload.view(torch.float8_e4m3fn)).load(name)

    assert torch.equal(bundle.weight.view(torch.uint8), payload.t().contiguous())


def test_c2a_checkpoint_spec_keeps_native_bf16():
    text = dict(hidden_size=64, moe_intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=2, n_routed_experts=2, n_shared_experts=1,
                head_dim=32, q_lora_rank=64, o_lora_rank=32, o_groups=2,
                hc_mult=4, vocab_size=64, index_n_heads=2, index_head_dim=32,
                compress_ratios=[2], kv_source_layer_ids=[0], index_source_layer_ids=[0])
    specs = backbone_weight_specs({"text_config": text, "quantization_config": {
        "quant_method": "fp8", "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0", "expert_dtype": "fp4"}})

    for projection in ("wkv", "wgate"):
        name = f"layers.0.attn.compressor.{projection}.weight"
        assert specs[name].dtype == "BF16"
        assert "cast_" not in specs[name].conversion


def test_index_key_bank_preserves_source_axis_and_slot_offsets():
    from pypto.runtime import DeviceTensor

    class Worker:
        def __init__(self):
            self.views = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            return DeviceTensor(10000 + worker_id * 10000, shape, dtype)

        def copy_to(self, *args, **kwargs):
            pass

        def view_tensor(self, parent, shape, *, byte_offset, worker_id):
            self.views.append((worker_id, byte_offset))
            return DeviceTensor(parent.data_ptr + byte_offset, shape, parent.dtype)

    worker = Worker()
    bank = SimpleNamespace(
        tensors={"index_wk": torch.zeros((2, 2, 32, 32), dtype=torch.bfloat16)},
        layout=DecodeWeightLayout(1, 1, 2, 1, world=2), finish=lambda names: None)
    resident = ResidentFullDecodeWeights(worker, bank, ("index_wk",))

    assert resident.full("index_wk").shape == (2, 2, 32, 32)
    assert resident.view("index_wk", 1).shape == (2, 32, 32)
    assert worker.views == [(0, 32 * 32 * 2), (1, 32 * 32 * 2)]


def test_legacy_nd_sidecar_is_rejected_before_mapping(tmp_path, monkeypatch):
    import safetensors

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def metadata(self):
            return {"format": "pypto-deepseek-v41-decode-slot-major-v1"}

    path = tmp_path / "old.safetensors"
    path.write_bytes(b"legacy")
    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: Reader())
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda *args: 1.0)

    assert weight_loader.load_prepacked_bank(
        tmp_path, DecodeWeightLayout(1, 1, 1, 1), (), path=path) is None
