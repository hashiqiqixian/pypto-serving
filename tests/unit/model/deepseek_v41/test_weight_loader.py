# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual small safetensors loads, reference conversions, partitioning, and preflight budgets.

Synthetic tensor shapes exercise the loader; one test embeds the published real K32
bytes in these shapes. This is not a full real layer or a device pack/upload golden.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="CPU weight-loader tests require Torch")
safetensors = pytest.importorskip(
    "safetensors.torch", reason="actual weight-loader tests require safetensors"
)
if not hasattr(torch, "float8_e8m0fnu"):
    pytest.skip("weight-loader tests require native UE8M0 dtype", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def loader_module(monkeypatch):
    # Use normal absolute imports while avoiding unrelated package-root NPU/runtime
    # initialization. Every temporary module entry is restored after this test.
    packages = (
        "pypto_serving",
        "pypto_serving.model",
        "pypto_serving.model.common",
        "pypto_serving.model.common.weights",
        "pypto_serving.model.deepseek_v41",
    )
    for name in packages:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    modules = {}
    for suffix in (
        "common.weights.store",
        "deepseek_v41.checkpoint",
        "deepseek_v41.config",
        "deepseek_v41.weight_spec",
        "deepseek_v41.weight_loader",
    ):
        name = "pypto_serving.model." + suffix
        spec = importlib.util.spec_from_file_location(
            name, ROOT.joinpath(*name.split(".")).with_suffix(".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[suffix] = module
    return modules["deepseek_v41.weight_loader"]


@pytest.fixture
def raw_config():
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32,
        vocab_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_n_heads=2,
        index_head_dim=32,
        compress_ratios=[2, 1],
        kv_source_layer_ids=[0, 1],
        index_source_layer_ids=[0, 1],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        engram_layer_ids=[1],
        engram_num_embeddings=[3],
        engram_head_dim=32,
        engram_max_ngram_size=2,
        engram_n_heads=1,
        num_nextn_predict_layers=0,
        dspark_target_layer_ids=[],
    )
    return raw


def write_checkpoint(directory, tensors):
    directory.mkdir(parents=True, exist_ok=True)
    safetensors.save_file(tensors, directory / "model.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )


@pytest.fixture
def case(tmp_path, loader_module, raw_config):
    specs = loader_module.backbone_weight_specs(raw_config)
    tensors = {}
    for name, spec in specs.items():
        if spec.dtype == "F8_E8M0":
            tensors[name] = torch.full(spec.shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        elif spec.dtype == "F8_E4M3":
            tensors[name] = torch.ones(spec.shape).to(torch.float8_e4m3fn)
        elif spec.dtype == "I8":
            tensors[name] = torch.full(spec.shape, 0x22, dtype=torch.int8)
        else:
            dtype = torch.bfloat16 if spec.dtype == "BF16" else torch.float32
            tensors[name] = torch.full(spec.shape, 0.25, dtype=dtype)
    for layer in range(2):
        for projection in ("wq_b", "wo_a"):
            tensors[f"layers.{layer}.attn.{projection}.scale"].view(torch.uint8)[1, 0] = 128
        tensors[f"layers.{layer}.attn.wo_b.scale"].view(torch.uint8)[0, 1] = 129
    tensors["layers.1.engram.embed.weight"] = torch.stack(
        [torch.full((32,), float(row + 1)) for row in range(3)]
    ).to(torch.float8_e4m3fn)
    tensors["layers.1.engram.embed.scale"].view(torch.uint8)[:, 0] = torch.tensor([126, 127, 128])
    write_checkpoint(tmp_path, tensors)
    return SimpleNamespace(root=tmp_path, module=loader_module, config=raw_config, tensors=tensors)


def load(case, **kwargs):
    return case.module.DeepSeekV41WeightLoader(case.root, case.config, **kwargs)


def test_layer_load_reuses_store_selects_only_owned_experts_and_preserves_fp4(case, monkeypatch):
    loader = load(case)
    assert isinstance(loader.store, case.module.LazySafetensorsStore)
    original = loader.store.load_many
    selected = []

    def capture(names):
        selected.extend(names)
        return original(names)

    monkeypatch.setattr(loader.store, "load_many", capture)
    result = loader.load_layer(0, rank=1, world_size=2)
    assert result.parallel_mode == "ep_only"
    assert set(selected) == set(result.source_names)
    assert all(name.startswith("layers.0.") for name in selected)
    assert not any(".experts.0." in name or ".experts.1." in name for name in selected)
    assert any(".experts.2." in name for name in selected)
    assert any(".experts.3." in name for name in selected)
    weight = "layers.0.ffn.experts.2.w1.weight"
    assert result.tensors[weight].dtype == torch.int8
    assert result.tensors[weight].shape == (32, 16)
    assert torch.equal(result.tensors[weight], case.tensors[weight])
    assert result.formats[weight] == "packed_e2m1_i8_low_nibble_first"
    assert result.tensors["layers.0.attn.wq_b.weight"].shape == (64, 32)
    assert result.tensors["layers.0.attn.wo_a.weight"].shape == (2, 32, 32)
    assert result.tensors["layers.0.attn.compressor.wkv.weight"].dtype == torch.float32
    assert result.tensors["layers.0.ffn.gate.weight"].dtype == torch.float32
    assert result.estimated_peak_bytes <= loader.max_load_bytes


def test_reference_tp_dense_scales_grouped_wo_a_and_compressor(case):
    result = load(case).load_layer(0, rank=1, world_size=2, parallel_mode="reference_tp")
    assert result.tensors["layers.0.attn.wq_b.weight"].shape == (32, 32)
    assert result.tensors["layers.0.attn.wq_b.scale"].view(torch.uint8).tolist() == [[128]]
    assert result.tensors["layers.0.attn.wo_b.weight"].shape == (32, 32)
    assert result.tensors["layers.0.attn.wo_b.scale"].view(torch.uint8).tolist() == [[129]]
    output = result.tensors["layers.0.attn.wo_a.weight"]
    assert output.dtype == torch.bfloat16
    assert output.shape == (1, 32, 32)
    assert torch.equal(output, torch.full((1, 32, 32), 2.0, dtype=torch.bfloat16))
    assert "layers.0.attn.wo_a.scale" not in result.tensors
    assert "layers.0.attn.wo_a.scale" in result.source_names
    ratio_one = load(case).load_layer(1, tensor_names=["layers.1.attn.compressor.wkv.weight"])
    assert ratio_one.tensors["layers.1.attn.compressor.wkv.weight"].dtype == torch.bfloat16


def test_global_tp_embedding_and_float32_head_preserve_reference_rows(case):
    loader = load(case)
    replicated = loader.load_globals(rank=1, world_size=2)
    sharded = loader.load_globals(rank=1, world_size=2, parallel_mode="reference_tp")
    assert replicated.tensors["embed.weight"].shape == (64, 32)
    assert sharded.tensors["embed.weight"].shape == (32, 32)
    assert sharded.tensors["embed.weight"].dtype == torch.bfloat16
    assert sharded.tensors["head.weight"].dtype == torch.float32
    assert torch.equal(sharded.tensors["head.weight"], case.tensors["head.weight"][32:].float())
    assert sharded.tensors["norm.weight"].shape == (32,)


def test_engram_reference_row_sharding_pads_weight_zero_and_e8m0_one(case):
    name = "layers.1.engram.embed.weight"
    scale_name = "layers.1.engram.embed.scale"
    loader = load(case)
    result = loader.load_layer(1, rank=1, world_size=2, parallel_mode="reference_tp", tensor_names=[name])
    assert result.tensors[name].shape == (2, 32)
    assert result.tensors[name].float().tolist() == [[3.0] * 32, [0.0] * 32]
    assert result.tensors[scale_name].view(torch.uint8).tolist() == [[128], [127]]
    replicated = loader.load_layer(1, rank=1, world_size=2, tensor_names=[name])
    assert replicated.tensors[name].shape == (3, 32)
    assert replicated.tensors[scale_name].view(torch.uint8).tolist() == [[126], [127], [128]]


def test_selection_adds_quantization_companion_and_does_not_mutate_inputs(case, monkeypatch):
    name = "layers.0.attn.wo_a.weight"
    companion = "layers.0.attn.wo_a.scale"
    loader = load(case)
    source = {name: case.tensors[name], companion: case.tensors[companion]}
    before = {key: value.view(torch.uint8).clone() for key, value in source.items()}
    monkeypatch.setattr(loader.store, "load_many", lambda names: {key: source[key] for key in names})
    result = loader.load_layer(0, tensor_names=[companion])
    assert set(result.source_names) == {name, companion}
    result.tensors[name].zero_()
    assert all(torch.equal(value.view(torch.uint8), before[key]) for key, value in source.items())


@pytest.mark.parametrize(
    "names",
    [
        [],
        "layers.0.attn.wq_a.weight",
        ["layers.1.attn.wq_a.weight"],
        ["layers.0.ffn.experts.0.w1.weight"],
        ["not_a_weight"],
    ],
)
def test_invalid_or_foreign_rank_selection_is_rejected(case, names):
    with pytest.raises(ValueError, match="tensor_names"):
        load(case).load_layer(0, rank=1, world_size=2, tensor_names=names)


@pytest.mark.parametrize(
    "rank,world_size,mode",
    [(0, 0, "ep_only"), (2, 2, "ep_only"), (0, 3, "ep_only"), (0, 4, "reference_tp"), (0, 1, "dp8")],
)
def test_invalid_parallel_contract_is_rejected(case, rank, world_size, mode):
    with pytest.raises(ValueError):
        load(case).load_layer(0, rank, world_size, parallel_mode=mode)


def test_budget_is_checked_before_any_payload_read_even_for_tp(case, monkeypatch):
    loader = load(case, max_load_bytes=1)
    calls = []
    monkeypatch.setattr(loader.store, "load_many", lambda names: calls.append(names))
    with pytest.raises(case.module.WeightLoadBudgetError, match="full source tensor"):
        loader.load_layer(
            1,
            rank=1,
            world_size=2,
            parallel_mode="reference_tp",
            tensor_names=["layers.1.engram.embed.weight"],
        )
    assert calls == []


def test_real_layer_all_experts_cannot_bypass_default_budget(tmp_path, loader_module):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    names = [name for name in loader_module.backbone_weight_specs(raw) if name.startswith("layers.0.")]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "not-materialized.safetensors" for name in names}})
    )
    loader = loader_module.DeepSeekV41WeightLoader(tmp_path, raw)
    with pytest.raises(loader_module.WeightLoadBudgetError):
        loader.load_layer(0)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "scale_nan", "fp8_nan", "bf16_nan"])
def test_corrupt_source_contract_or_nonfinite_values_are_rejected(case, corruption):
    weight, scale = "layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"
    selected = [weight]
    if corruption == "shape":
        case.tensors[weight] = case.tensors[weight][:16].contiguous()
    elif corruption == "dtype":
        case.tensors[weight] = case.tensors[weight].view(torch.int8)
    elif corruption == "scale_nan":
        case.tensors[scale].view(torch.uint8)[0, 0] = 255
    elif corruption == "fp8_nan":
        case.tensors[weight].view(torch.uint8)[0, 0] = 127
    else:
        selected = ["layers.0.attn_norm.weight"]
        case.tensors[selected[0]][0] = float("nan")
    write_checkpoint(case.root, case.tensors)
    with pytest.raises(ValueError, match="contract mismatch|non-finite"):
        load(case).load_layer(0, tensor_names=selected)


def test_wo_a_overflow_is_not_returned_as_valid_bf16(case):
    name = "layers.0.attn.wo_a.weight"
    case.tensors[name] = torch.full((64, 32), 448.0).to(torch.float8_e4m3fn)
    case.tensors["layers.0.attn.wo_a.scale"].view(torch.uint8).fill_(254)
    write_checkpoint(case.root, case.tensors)
    with pytest.raises(ValueError, match="overflows"):
        load(case).load_layer(0, tensor_names=[name])


def test_tp_must_not_split_through_a_quantization_scale_block(case):
    case.config["text_config"]["o_lora_rank"] = 24
    name = "layers.0.attn.wo_b.weight"
    tensors = {
        name: torch.ones((32, 48)).to(torch.float8_e4m3fn),
        "layers.0.attn.wo_b.scale": torch.full((1, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    write_checkpoint(case.root, tensors)
    with pytest.raises(ValueError, match="32-element scale blocks"):
        load(case).load_layer(0, rank=1, world_size=2, parallel_mode="reference_tp", tensor_names=[name])


def test_missing_scale_is_an_error_instead_of_unscaled_weight(case):
    name = "layers.0.attn.wq_a.weight"
    del case.tensors["layers.0.attn.wq_a.scale"]
    write_checkpoint(case.root, case.tensors)
    with pytest.raises(KeyError, match="missing required"):
        load(case).load_layer(0, tensor_names=[name])


def test_real_k32_bytes_survive_loading_without_format_renaming(case):
    fixture = json.loads((ROOT / "tests/fixtures/deepseek_v41/real_quant_blocks.json").read_text())
    requested = []
    for block in fixture["cases"]:
        name = block["tensor"] + ".weight"
        scale = block["tensor"] + ".scale"
        requested.append(name)
        raw_weight = bytes.fromhex(block["buffers"]["weight"]["hex"])
        raw_scale = bytes.fromhex(block["buffers"]["scale"]["hex"])
        case.tensors[name].view(torch.uint8)[0].copy_(torch.tensor(list(raw_weight), dtype=torch.uint8))
        case.tensors[scale].view(torch.uint8)[0, 0] = raw_scale[0]
    write_checkpoint(case.root, case.tensors)
    result = load(case).load_layer(0, tensor_names=requested)
    for block in fixture["cases"]:
        name = block["tensor"] + ".weight"
        scale = block["tensor"] + ".scale"
        assert (
            bytes(result.tensors[name].view(torch.uint8)[0].tolist()).hex()
            == block["buffers"]["weight"]["hex"]
        )
        assert (
            bytes(result.tensors[scale].view(torch.uint8)[0].tolist()).hex()
            == block["buffers"]["scale"]["hex"]
        )
