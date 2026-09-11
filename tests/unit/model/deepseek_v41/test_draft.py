# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark actual CPU tensor math, stage-window updates, and transactional sampling tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


torch = pytest.importorskip("torch", reason="DSpark numerical execution requires Torch")
F = torch.nn.functional
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    for name in ("pypto_serving", "pypto_serving.model", "pypto_serving.model.deepseek_v41"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    result = {}
    for suffix in ("config", "weight_spec", "draft_spec", "numerics", "attention", "draft"):
        name = f"pypto_serving.model.deepseek_v41.{suffix}"
        spec = importlib.util.spec_from_file_location(
            name, ROOT.joinpath(*name.split(".")).with_suffix(".py")
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[suffix] = module
    return result


class NumericalOps:
    """Unquantized small test weights, actual linear algebra and reference activation quantization."""

    rank, world_size, device, dtype = 0, 1, torch.device("cpu"), torch.float32

    def __init__(self, raw, modules):
        self.values, self.calls, self.fail_name = {}, [], None
        self.quantize_rows = modules["numerics"].quantize_rows
        generator = torch.Generator().manual_seed(165)
        for name, spec in modules["draft_spec"].draft_weight_specs(raw).items():
            if name.endswith(".scale"):
                continue
            shape = spec.shape
            if spec.dtype == "I8":
                shape = (*shape[:-1], shape[-1] * 2)
            self.values[name] = torch.randn(shape, generator=generator) * 0.02
            if "norm.weight" in name:
                self.values[name].fill_(1)
        text = raw["text_config"]
        for name in ("embed.weight", "head.weight"):
            self.values[name] = (
                torch.randn((text["vocab_size"], text["hidden_size"]), generator=generator) * 0.1
            )

    def weight(self, name):
        return self.values[name]

    def linear(self, x, name, bias=False):
        self.calls.append(name)
        if name == self.fail_name:
            raise RuntimeError("injected operator failure")
        key = name if name in self.values else name + ".weight"
        return F.linear(x, self.values[key], self.values[name + ".bias"] if bias else None)

    def matmul(self, a, b):
        return a @ b

    def embedding(self, name, ids):
        return F.embedding(ids, self.values[name])

    def all_reduce(self, value):
        return value

    def all_gather(self, value):
        return value

    def quantize(self, value, fmt, block):
        return self.quantize_rows(value, fmt, block).dequantize(value.dtype)


@pytest.fixture
def model(modules):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32,
        vocab_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=1,
        sliding_window=4,
        max_position_embeddings=128,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        compress_ratios=[2, 2, 2, 1, 1, 0, 0, 0],
        kv_source_layer_ids=[0, 3],
        index_source_layer_ids=[0, 2, 3, 4],
        candidate_source_layer_id=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=[1],
        engram_num_embeddings=[3],
        engram_head_dim=32,
        engram_max_ngram_size=2,
        engram_n_heads=1,
        num_nextn_predict_layers=3,
        dspark_target_layer_ids=[2, 3, 4],
        dspark_block_size=3,
        dspark_noise_token_id=63,
        dspark_markov_rank=8,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
    )
    config = modules["config"].DeepSeekV41Config.from_dict(raw)
    ops = NumericalOps(raw, modules)
    return modules["draft"].DSparkDrafter(config, ops)


def test_markov_chain_uses_preceding_output_and_raw_confidence(model):
    ops = model.ops
    ops.values["head.weight"].zero_()
    embed = ops.values["mtp.2.markov_head.embed.weight"]
    embed.zero_()
    embed[1, 0], embed[2, 1], embed[3, 2] = 1, 1, 1
    projection = ops.values["mtp.2.markov_head.head.weight"]
    projection.zero_()
    projection[2, 0], projection[3, 1], projection[4, 2] = 5, 7, 6
    confidence = ops.values["mtp.2.confidence_head.proj.weight"]
    confidence.zero_()
    confidence[0, 32:35] = torch.tensor([2.0, -3.0, 4.0])
    hidden = torch.arange(96).float().reshape(3, 32) / 100
    before = hidden.clone()
    result = model.forward_head(hidden, 1)
    assert result.output_ids.tolist() == [1, 2, 3, 4]
    expected_logits = torch.zeros(3, 64)
    expected_logits[0, 2], expected_logits[1, 3], expected_logits[2, 4] = 5, 7, 6
    torch.testing.assert_close(result.logits, expected_logits, atol=0, rtol=0)
    torch.testing.assert_close(result.confidence, torch.tensor([2.0, -3.0, 4.0]), atol=0, rtol=0)
    torch.testing.assert_close(hidden, before, atol=0, rtol=0)


def test_three_real_stages_seed_then_draft_without_committing_noise_tokens(model):
    state = model.new_state(128)
    other = model.new_state(128)
    main = torch.linspace(-1, 1, 3 * 96).reshape(3, 96)
    assert model.forward(5, main, 0, state) is None
    assert state.position == 3 and other.position == 0
    assert all(layer.window.shape == (3, 32) for layer in state.attention)
    model.ops.calls.clear()
    output = model.forward(7, main[-1:], 3, state)
    assert output.output_ids.shape == (4,) and output.output_ids[0] == 7
    assert output.logits.shape == (3, 64) and output.confidence.shape == (3,)
    assert state.position == 4
    assert all(layer.window.shape == (4, 32) for layer in state.attention)
    for stage in range(3):
        assert f"mtp.{stage}.attn.wq_a" in model.ops.calls
        assert f"mtp.{stage}.ffn.shared_experts.w2" in model.ops.calls
    assert all(torch.isfinite(value).all() for value in (output.logits, output.confidence))


def test_seed_chunks_equal_full_prefill_and_run_only_main_projections(model):
    first, second = model.new_state(128), model.new_state(128)
    main = torch.linspace(-1, 1, 6 * 96).reshape(6, 96)
    model.seed(main, 0, first)
    model.ops.calls.clear()
    model.seed(main[:2], 0, second)
    model.seed(main[2:5], 2, second)
    model.seed(main[5:], 5, second)
    assert first.position == second.position == 6
    for left, right in zip(first.attention, second.attention):
        torch.testing.assert_close(left.window, right.window, atol=0, rtol=0)
    assert all(name == "mtp.0.main_proj" or name.endswith(".attn.wkv") for name in model.ops.calls)
    with pytest.raises(ValueError, match="boundary|bounded"):
        second.rollback(3)


def test_seed_then_propose_does_not_duplicate_main_cache(model):
    state = model.new_state(128)
    main = torch.linspace(-1, 1, 3 * 96).reshape(3, 96)
    model.seed(main, 0, state)
    before = [layer.window.clone() for layer in state.attention]
    first = model.propose(7, state)
    second = model.propose(7, state)
    assert state.position == 3
    assert len(state._history) == 1
    torch.testing.assert_close(first.output_ids, second.output_ids, atol=0, rtol=0)
    for layer, expected in zip(state.attention, before):
        torch.testing.assert_close(layer.window, expected, atol=0, rtol=0)


def test_draft_failure_restores_all_stage_windows_and_sampling_rng(model):
    state = model.new_state(128, seed=986)
    main = torch.linspace(-1, 1, 2 * 96).reshape(2, 96)
    model.forward(5, main, 0, state)
    before = state.snapshot()
    windows = [layer.window.clone() for layer in state.attention]
    model.ops.fail_name = "mtp.2.confidence_head.proj"
    with pytest.raises(RuntimeError, match="operator failure"):
        model.forward(7, main[-1:], 2, state, temperature=0.8)
    assert state.position == 2
    torch.testing.assert_close(state.generator.get_state(), before.rng_state, atol=0, rtol=0)
    for layer, expected in zip(state.attention, windows):
        torch.testing.assert_close(layer.window, expected, atol=0, rtol=0)
    model.ops.fail_name = None
    first = model.forward(7, main[-1:], 2, state, temperature=0.8)
    state.restore(before)
    second = model.forward(7, main[-1:], 2, state, temperature=0.8)
    torch.testing.assert_close(first.output_ids, second.output_ids, atol=0, rtol=0)
    torch.testing.assert_close(first.logits, second.logits, atol=0, rtol=0)


def test_rollback_pending_boundaries_rejects_discarded_history(model):
    state = model.new_state(128, rollback_window=3)
    main = torch.linspace(-1, 1, 4 * 96).reshape(4, 96)
    model.forward(5, main, 0, state)
    saved = None
    for position in range(4, 9):
        model.forward(7, main[-1:], position, state)
        if state.position == 7:
            saved = [layer.window.clone() for layer in state.attention]
    assert len(state._history) <= 4
    with pytest.raises(ValueError, match="bounded"):
        state.rollback(4)
    state.rollback(7)
    assert state.position == 7
    for layer, expected in zip(state.attention, saved):
        torch.testing.assert_close(layer.window, expected, atol=0, rtol=0)
    other = model.new_state(128)
    with pytest.raises(ValueError, match="different request"):
        other.restore(state.snapshot())
    with pytest.raises(ValueError, match="one committed"):
        model.forward(7, main[-2:], 7, state)
