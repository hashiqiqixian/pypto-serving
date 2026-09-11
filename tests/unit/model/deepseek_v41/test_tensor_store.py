# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual safetensors slices, scaled matrix blocks, TP/EP and bounded sparse row reads."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="tensor-store tests require CPU Torch")
safetensors = pytest.importorskip("safetensors")
serialization = pytest.importorskip("safetensors.torch")
if not hasattr(torch, "float8_e8m0fnu"):
    pytest.skip("native UE8M0 dtype is required", allow_module_level=True)
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def module(monkeypatch):
    before = {
        name: value
        for name, value in sys.modules.items()
        if name == "pypto_serving" or name.startswith("pypto_serving.")
    }
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    package = types.ModuleType("pypto_serving")
    package.__path__ = [str(ROOT / "pypto_serving")]
    monkeypatch.setitem(sys.modules, "pypto_serving", package)
    try:
        yield importlib.import_module("pypto_serving.model.deepseek_v41.tensor_store")
    finally:
        for name in tuple(sys.modules):
            if name == "pypto_serving" or name.startswith("pypto_serving."):
                sys.modules.pop(name, None)
        sys.modules.update(before)


@pytest.fixture
def raw_config():
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=64,
        vocab_size=64,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_n_heads=2,
        index_head_dim=32,
        compress_ratios=[1, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        engram_layer_ids=[0],
        engram_num_embeddings=[5],
        engram_head_dim=32,
        engram_max_ngram_size=2,
        engram_n_heads=1,
        num_nextn_predict_layers=1,
        dspark_target_layer_ids=[0],
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dspark_markov_rank=8,
    )
    raw["vision_config"].update(
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        num_hidden_layers=1,
        patch_size=2,
        downsample_ratio=2,
        max_image_tokens=64,
        min_pixels=0,
    )
    return raw


def write_checkpoint(path, tensors):
    serialization.save_file(tensors, path / "weights.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )


@pytest.fixture
def case(tmp_path, module, raw_config):
    specs = module.backbone_weight_specs(raw_config) | module.draft_weight_specs(raw_config)
    specs.update(
        {
            name: module.TensorSpec(shape, "BF16", name, "identity")
            for name, shape in module.vision_weight_shapes(
                module.VisionConfig.from_config(raw_config)
            ).items()
        }
    )
    tensors = {}
    for name, spec in specs.items():
        if spec.dtype == "F8_E4M3":
            value = torch.ones(spec.shape).to(torch.float8_e4m3fn)
        elif spec.dtype == "F8_E8M0":
            value = torch.full(spec.shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            if len(spec.shape) == 2:
                rows, columns = spec.shape
                value.view(torch.uint8).copy_(
                    (127 + torch.arange(rows)[:, None] % 2 + torch.arange(columns)[None, :] % 2).to(
                        torch.uint8
                    )
                )
        elif spec.dtype == "I8":
            value = (
                (torch.arange(spec.shape[1])[None, :].expand(spec.shape) % 256)
                .to(torch.uint8)
                .view(torch.int8)
                .clone()
            )
        else:
            value = torch.full(
                spec.shape, 0.25, dtype=torch.bfloat16 if spec.dtype == "BF16" else torch.float32
            )
        tensors[name] = value
    embed = "layers.0.engram.embed.weight"
    tensors[embed] = torch.arange(1, 6).float()[:, None].expand(5, 32).to(torch.float8_e4m3fn).contiguous()
    write_checkpoint(tmp_path, tensors)
    reads = []

    class Slice:
        def __init__(self, name, source):
            self.name, self.source = name, source

        def get_shape(self):
            return self.source.get_shape()

        def __getitem__(self, ranges):
            reads.append((self.name, ranges))
            return self.source[ranges]

    class Reader:
        def __init__(self, path, device):
            self.context = safetensors.safe_open(str(path), framework="pt", device=device)

        def __enter__(self):
            self.source = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_slice(self, name):
            return Slice(name, self.source.get_slice(name))

        def get_tensor(self, name):
            pytest.fail("the streaming store must never read a full tensor")

    return SimpleNamespace(
        root=tmp_path, module=module, raw=raw_config, tensors=tensors, reads=reads, opener=Reader
    )


def store(case, **kwargs):
    return case.module.DeepSeekV41TensorStore(case.root, case.raw, safe_open_fn=case.opener, **kwargs)


def materialize_tiles(weights, name):
    result = torch.zeros(weights.matrix_shape(name), dtype=torch.float32)
    for row, column, values, scales in weights.matrix_tiles(name):
        assert values.dtype in (torch.bfloat16, torch.float32)
        block = values.float() if scales is None else values.float() * scales[:, None]
        result[row : row + values.shape[0], column : column + values.shape[1]] = block
    return result


def test_fp8_tp_tiles_use_source_scale_offsets_and_never_materialize_full_source(case):
    weights = store(case, rank=1, world_size=2, out_tile_rows=32)
    name = "layers.0.attn.wq_b"
    assert weights.matrix_shape(name) == (64, 64)
    assert weights.matrix_format(name) == "fp8"
    result = materialize_tiles(weights, name)
    expected = torch.ones(64, 64)
    expected[:32, 32:] *= 2
    expected[32:, :32] *= 2
    expected[32:, 32:] *= 4
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for source, ranges in case.reads:
        if source.endswith(".weight"):
            assert ranges[0].start >= 64 and ranges[0].stop - ranges[0].start <= 32
            assert ranges[1].stop - ranges[1].start == 32


def test_fp4_tiles_decode_both_nibbles_with_per_row_k32_scales_and_ep_selection(case):
    weights = store(case, rank=1, world_size=2, out_tile_rows=32)
    name = "layers.0.ffn.experts.2.w1"
    result = materialize_tiles(weights, name)
    lookup = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    raw = case.tensors[name + ".weight"].view(torch.uint8).long()
    expected = torch.stack((lookup[raw & 15], lookup[raw >> 4]), -1).flatten(-2)
    scale_bits = case.tensors[name + ".scale"].view(torch.uint8).float()
    expected *= torch.pow(2, scale_bits - 127).repeat_interleave(32, -1)
    assert weights.matrix_format(name) == "fp4" and result.shape == (64, 64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert all(
        ranges[1].stop - ranges[1].start == 16 for source, ranges in case.reads if source.endswith(".weight")
    )
    with pytest.raises(ValueError, match="not owned"):
        weights.matrix_shape("layers.0.ffn.experts.0.w1")
    with pytest.raises(ValueError, match="not owned"):
        weights.matrix_shape("mtp.0.ffn.experts.0.w1")


def test_wo_a_is_streamed_to_grouped_bf16_and_reference_fp32_promotions_are_kept(case):
    weights = store(case, rank=1, world_size=2)
    for prefix in ("layers.0", "mtp.0"):
        result = weights.weight(prefix + ".attn.wo_a.weight")
        expected = torch.cat((torch.full((32, 32), 2), torch.full((32, 32), 4)), -1).to(torch.bfloat16)
        assert result.shape == (1, 32, 64) and result.dtype == torch.bfloat16
        assert torch.equal(result[0], expected)
        assert weights.matrix_format(prefix + ".ffn.gate") == "float32"
        assert weights.weight(prefix + ".ffn.gate.weight").dtype == torch.float32
    assert weights.matrix_format("head") == "float32"
    assert weights.matrix_format("mtp.0.markov_head.head") == "float32"
    assert weights.matrix_format("mtp.0.confidence_head.proj") == "float32"
    assert weights.matrix_format("vision.blocks.0.attn.wqkv") == "bf16"
    assert torch.equal(weights.weight("image_start"), case.tensors["image_start"])
    assert weights.matrix_shape("mtp.0.head") == weights.matrix_shape("head") == (32, 64)


def test_tp_axis_one_and_ep_only_layout_are_distinct(case):
    tp = store(case, rank=1, world_size=2)
    ep = store(case, rank=1, world_size=2, parallel_mode="ep_only")
    assert tp.matrix_shape("layers.0.attn.wo_b") == (64, 32)
    assert ep.matrix_shape("layers.0.attn.wo_b") == (64, 64)
    result = materialize_tiles(tp, "layers.0.attn.wo_b")
    assert result[:32].eq(2).all() and result[32:].eq(4).all()


def test_engram_sparse_rank_rows_padding_duplicates_and_cache_ownership(case):
    weights = store(case, rank=1, world_size=2, prefetch_rows=1, row_cache_bytes=128)
    name = "layers.0.engram.embed"
    ids = torch.tensor([[0, 3, 4, 3]], dtype=torch.int64)
    result = weights.embedding(name, ids)
    assert result.shape == (1, 4, 32) and result.dtype == torch.bfloat16
    assert result[0, 0].eq(0).all()
    assert result[0, 1].eq(8).all() and result[0, 2].eq(5).all() and result[0, 3].eq(8).all()
    assert len(case.reads) == 4  # Two unique owned rows, with one weight and one scale each.
    result.zero_()
    again = weights.embedding(name, ids)
    assert again[0, 1].eq(8).all() and len(case.reads) == 4
    with pytest.raises(ValueError, match="unpadded"):
        weights.embedding(name, torch.tensor([5]))
    assert weights._cached_bytes <= 128
    weights.close()
    assert weights._cached_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        weights.embedding(name, ids)


def test_prefetch_lru_evicts_old_rows_and_uncached_mode_reloads(case):
    name = "layers.0.engram.embed.weight"
    weights = store(case, prefetch_rows=1, row_cache_bytes=64)
    weights.prefetch(name, torch.tensor([0]))
    assert len(case.reads) == 2
    weights.embedding(name, torch.tensor([0]))
    assert len(case.reads) == 2
    weights.prefetch(name, torch.tensor([1]))
    weights.embedding(name, torch.tensor([0]))
    assert len(case.reads) == 6 and weights._cached_bytes == 64
    uncached = store(case, placement="uncached", prefetch_rows=1)
    uncached.embedding(name, torch.tensor([0]))
    uncached.embedding(name, torch.tensor([0]))
    assert len(case.reads) == 10 and uncached._cached_bytes == 0


def test_global_and_shared_draft_embedding_return_only_rank_owned_rows(case):
    weights = store(case, rank=1, world_size=2)
    ids = torch.tensor([0, 32, 63])
    actual = weights.embedding("mtp.0.embed", ids)
    assert actual[0].eq(0).all() and actual[1:].eq(0.25).all()
    assert all(source == "embed.weight" for source, _ in case.reads)
    with pytest.raises(ValueError, match="absent stage"):
        weights.embedding("mtp.1.embed", ids)


def test_table_larger_than_budget_is_read_by_selected_slice(case):
    name = "layers.0.engram.embed.weight"
    case.raw["text_config"]["engram_num_embeddings"] = [5000]
    case.tensors[name] = torch.ones(5000, 32).to(torch.float8_e4m3fn)
    case.tensors[name.replace(".weight", ".scale")] = torch.full((5000, 1), 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    write_checkpoint(case.root, case.tensors)
    weights = store(case, max_load_bytes=8192, prefetch_rows=1)
    actual = weights.embedding(name, torch.tensor([4999]))
    assert actual.eq(1).all()
    assert all(ranges[0] == slice(4999, 5000) for _, ranges in case.reads)
    assert case.tensors[name].numel() > weights.max_load_bytes


@pytest.mark.parametrize("kind", ["matrix", "weight", "embedding"])
def test_load_budget_is_checked_before_slice_payload_reads(case, kind):
    weights = store(case, max_load_bytes=1)
    with pytest.raises(case.module.WeightLoadBudgetError):
        if kind == "matrix":
            next(weights.matrix_tiles("layers.0.attn.wq_b"))
        elif kind == "weight":
            weights.weight("layers.0.attn.wo_a.weight")
        else:
            weights.embedding("embed", torch.tensor([0]))
    assert case.reads == []


@pytest.mark.parametrize("corruption", ["shape", "dtype", "fp8_nan", "scale_nan", "bf16_nan"])
def test_header_and_payload_corruption_are_rejected(case, corruption):
    name = "layers.0.attn.wq_b.weight"
    if corruption == "shape":
        case.tensors[name] = case.tensors[name][:32].contiguous()
    elif corruption == "dtype":
        case.tensors[name] = case.tensors[name].float()
    elif corruption == "fp8_nan":
        case.tensors[name].view(torch.uint8)[0, 0] = 127
    elif corruption == "scale_nan":
        case.tensors[name.replace(".weight", ".scale")].view(torch.uint8)[0, 0] = 255
    else:
        name = "norm.weight"
        case.tensors[name][0] = float("nan")
    write_checkpoint(case.root, case.tensors)
    weights = store(case)
    with pytest.raises(ValueError, match="mismatch|non-finite"):
        if corruption == "bf16_nan":
            weights.weight(name)
        else:
            next(weights.matrix_tiles(name))


def test_shared_slice_api_preserves_dimensions_and_validates_bounds(case):
    shared = store(case).store
    result = shared.load_slice("embed.weight", (slice(2, 4), slice(5, 9)))
    assert result.shape == (2, 4)
    assert torch.equal(result, case.tensors["embed.weight"][2:4, 5:9])
    before = len(case.reads)
    for ranges in (
        (slice(-1, 2), slice(0, 1)),
        (slice(0, 65), slice(0, 1)),
        (slice(0, 2, 2), slice(0, 1)),
        (slice(0, 2, True), slice(0, 1)),
        (slice(0, 2),),
    ):
        with pytest.raises(ValueError, match="slice"):
            shared.load_slice("embed.weight", ranges)
    assert len(case.reads) == before


@pytest.mark.parametrize("name", ["layers.0.attn.wo_a.weight", "layers.0.engram.embed.weight"])
def test_finite_quantized_source_that_overflows_bf16_is_rejected(case, name):
    case.tensors[name].view(torch.uint8).fill_(126)  # Finite E4M3 maximum, 448.
    case.tensors[name.replace(".weight", ".scale")].view(torch.uint8).fill_(254)
    write_checkpoint(case.root, case.tensors)
    weights = store(case)
    with pytest.raises(ValueError, match="overflows"):
        if ".engram." in name:
            weights.embedding(name, torch.tensor([0]))
        else:
            weights.weight(name)


def test_tp_split_must_preserve_quantization_block_boundaries(case):
    case.raw["text_config"]["o_lora_rank"] = 24
    # Metadata-only rejection occurs before inspecting the now-incompatible shard.
    weights = store(case, rank=1, world_size=2)
    with pytest.raises(ValueError, match="32-value quantization block"):
        next(weights.matrix_tiles("layers.0.attn.wo_b"))
    assert case.reads == []
