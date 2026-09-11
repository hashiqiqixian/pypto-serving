# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small numerical attention tests: actual projections, quantization and stateful math."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest


torch = pytest.importorskip("torch", reason="attention numerical tests require Torch")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    for name in ("pypto_serving", "pypto_serving.model", "pypto_serving.model.deepseek_v41"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    result = {}
    for suffix in ("config", "attention"):
        name = f"pypto_serving.model.deepseek_v41.{suffix}"
        spec = importlib.util.spec_from_file_location(name, ROOT.joinpath(*name.split(".")).with_suffix(".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result[suffix] = module
    return result


@pytest.fixture
def config(modules):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32, vocab_size=64, moe_intermediate_size=32, num_hidden_layers=5,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, qk_rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=1, sliding_window=4,
        max_position_embeddings=128, hc_mult=2, n_routed_experts=4, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=32, index_topk=2,
        compress_ratios=[2, 2, 2, 1, 1], kv_source_layer_ids=[0, 3],
        index_source_layer_ids=[0, 2, 3, 4], candidate_source_layer_id=3,
        candidate_topk_blocks=2, candidate_block_size=2,
        engram_layer_ids=[1], engram_num_embeddings=[3], engram_head_dim=32,
        engram_max_ngram_size=2, engram_n_heads=1, num_nextn_predict_layers=0,
        dspark_target_layer_ids=[],
    )
    return modules["config"].DeepSeekV41Config.from_dict(raw)


def quantize_values(values, fmt, block):
    """Independent vector implementation of the reference activation quantizers."""
    rows = values.float().reshape(-1, values.shape[-1] // block, block)
    maximum = rows.abs().amax(-1, keepdim=True)
    if fmt == "fp8_e4m3_ue8m0":
        scales = 2.0 ** torch.ceil(torch.log2(maximum.clamp_min(1e-4) / 448.0))
        quantized = (rows / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        if fmt == "fp4_e2m1_e4m3":
            scales = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
        elif fmt == "fp4_e2m1_ue8m0":
            scales = 2.0 ** torch.ceil(torch.log2(maximum.clamp_min(6 * 2**-126) / 6))
        else:
            raise ValueError(f"unknown numerical quantizer: {fmt}")
        normalized = (rows / scales).clamp(-6, 6)
        levels = rows.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        differences = (normalized.abs()[..., None] - levels).abs()
        nearest = differences.argmin(-1)
        upper = (nearest + 1).clamp_max(7)
        tied = differences.gather(-1, nearest[..., None]) == differences.gather(-1, upper[..., None])
        nearest = torch.where(tied.squeeze(-1) & (nearest % 2 == 1), upper, nearest)
        quantized = levels[nearest] * normalized.sign()
    return (quantized * scales).reshape_as(values).to(values.dtype)


class NumericalOps:
    """Real CPU tensor arithmetic; no synthetic logits or substituted attention outputs."""

    def __init__(self, config, dtype=torch.float32):
        self.values = {}
        self.quantized = []
        self.matmul_shapes = []
        self.fail_suffix = None
        generator = torch.Generator().manual_seed(873)
        for layer, plan in enumerate(config.layer_plan):
            base = f"layers.{layer}.attn"
            shapes = {"wq_a": (32, 32), "wq_b": (64, 32), "wkv": (32, 32),
                      "wo_a": (1, 32, 64), "wo_b": (32, 32)}
            norms = ["q_norm", "kv_norm"]
            if plan.owns_main_kv:
                shapes["compressor.wkv"] = (32, 32)
                if plan.compress_ratio > 1:
                    shapes["compressor.wgate"] = (32, 32)
                shapes["indexer.wk"] = (32, 32)
                norms += ["compressor.norm", "indexer.k_norm"]
            if plan.owns_index_results:
                shapes["indexer.wq_b"] = (64, 32)
                shapes["indexer.weights_proj"] = (2, 32)
            for name, shape in shapes.items():
                value = torch.randn(shape, generator=generator) * 0.08
                self.values[f"{base}.{name}.weight"] = value.to(dtype)
            for name in norms:
                self.values[f"{base}.{name}.weight"] = torch.ones(32, dtype=dtype)
            self.values[f"{base}.attn_sink"] = torch.tensor([-0.4, 0.6])

    def weight(self, name):
        return self.values[name]

    def linear(self, x, name):
        if self.fail_suffix and name.endswith(self.fail_suffix):
            raise RuntimeError("injected projection failure after cache writes")
        return torch.nn.functional.linear(x, self.values[name].to(x.dtype))

    def all_reduce(self, values):
        return values

    def matmul(self, a, b):
        self.matmul_shapes.append((a.shape, b.shape))
        return torch.matmul(a, b)

    def quantize(self, values, fmt, block):
        result = quantize_values(values, fmt, block)
        self.quantized.append((fmt, block, tuple(values.shape)))
        return result


def run_chunks(module, config, ops, x, chunks):
    layers = [module.Attention(config, ops, i, index_key_tile=3) for i in range(config.num_hidden_layers)]
    states = [layer.new_state(128, cache_page_size=3) for layer in layers]
    start, outputs = 0, []
    for width in chunks:
        shared = module.SharedAttention(start, start + width)
        hidden = x[start:start + width]
        for layer, state in zip(layers, states):
            hidden = layer.forward(hidden, start, state, shared)
        outputs.append(hidden)
        start += width
    assert start == len(x)
    return torch.cat(outputs), states


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_reindex_reuse_are_numerical_and_chunk_contiguous(modules, config, dtype):
    module = modules["attention"]
    x = torch.randn((11, 32), generator=torch.Generator().manual_seed(12)).to(dtype)
    ops = NumericalOps(config, dtype)
    whole, whole_states = run_chunks(module, config, ops, x, [11])
    chunked, states = run_chunks(module, config, ops, x, [1, 2, 1, 3, 4])
    tolerance = 0.035 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(chunked, whole, rtol=tolerance, atol=tolerance)
    assert torch.isfinite(chunked).all() and chunked.abs().sum() > 0
    for state, expected in zip(states, whole_states):
        assert state.position == 11
        assert len(state.window) == config.sliding_window
        assert state.window.untyped_storage().nbytes() == 4 * 32 * state.window.element_size()
        torch.testing.assert_close(state.window, expected.window, rtol=tolerance, atol=tolerance)
        if state.main is not None:
            assert state.main.length == 11 // config.compress_ratios[state.layer_id]
            torch.testing.assert_close(state.main.read(0, state.main.length),
                                       expected.main.read(0, expected.main.length),
                                       rtol=tolerance, atol=tolerance)
    assert len(states[0].partial_kv) == 1
    assert states[1].main is states[2].main is states[4].main is None
    assert {entry[:2] for entry in ops.quantized} == {
        (module.FP8_SWA, 32), (module.FP4_INDEX, 32), (module.FP4_MAIN, 16)
    }
    # Index scores are tiled by keys, not a [tokens,heads,history] allocation.
    assert any(a == torch.Size([2, 32]) and b == torch.Size([32, 3]) for a, b in ops.matmul_shapes)


@pytest.mark.parametrize("padding", [0, 1, 7, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topk_cutoff_ties_use_absolute_position_independent_of_future_padding(modules, padding, dtype):
    select = modules["attention"]._topk_positions
    # The observed position-3 failure chose different pairs from four exact
    # ReLU zero scores when torch.topk saw width 4 versus width 11.
    scores = torch.tensor([0, 0, 0, 0] + [-torch.inf] * padding, dtype=dtype)
    assert select(scores, 2).tolist() == [0, 1]
    # Strictly greater scores keep priority; only the cutoff tie is ordered.
    mixed = torch.tensor([1, 7, 1, 4, 1] + [-torch.inf] * padding, dtype=dtype)
    assert select(mixed, 3).tolist() == [0, 1, 3]


@pytest.mark.parametrize("padding", [0, 2, 18])
def test_candidate_block_cutoff_ties_keep_latest_and_earliest_absolute_block(modules, padding):
    select = modules["attention"].select_candidate_blocks
    # Three reachable zero-score blocks; the newest is pinned and the remaining
    # slot deterministically selects block zero, including with masked future blocks.
    scores = torch.tensor([0.0] * 6 + [-torch.inf] * padding)
    assert select(scores, reachable=6, topk_blocks=2, block_size=2).tolist() == [0, 2]
    # Underfilled candidate sets still discard the reference's -inf filler picks.
    assert select(scores.masked_fill(torch.arange(len(scores)) >= 1, -torch.inf),
                  reachable=1, topk_blocks=2, block_size=2).tolist() == [0]


def test_topk_position_selection_handles_empty_and_rejects_nan(modules):
    select = modules["attention"]._topk_positions
    assert select(torch.empty(0), 0).numel() == 0
    with pytest.raises(ValueError, match="NaN"):
        select(torch.tensor([0.0, torch.nan]), 1)


def test_compressor_pooling_carries_the_actual_partial_group(modules, config):
    module, ops = modules["attention"], NumericalOps(config)
    layer = module.Attention(config, ops, 0)
    state = layer.new_state(20)
    x = torch.randn((3, 32), generator=torch.Generator().manual_seed(65))
    layer.forward(x[:1], 0, state, module.SharedAttention(0, 1))
    assert state.main.length == state.index.length == 0
    assert state.partial_kv.shape == (1, 32)
    layer.forward(x[1:], 1, state, module.SharedAttention(1, 3))
    kv = torch.nn.functional.linear(x[:2], ops.values["layers.0.attn.compressor.wkv.weight"])
    gate = torch.nn.functional.linear(x[:2], ops.values["layers.0.attn.compressor.wgate.weight"])
    pooled = (kv * gate.softmax(0)).sum(0, keepdim=True)
    pooled *= torch.rsqrt(pooled.square().mean(-1, keepdim=True) + config.text_config["rms_norm_eps"])
    # The first group's RoPE position is zero; its main quantization uses E4M3 scales.
    expected = quantize_values(pooled, module.FP4_MAIN, 16)
    torch.testing.assert_close(state.main.read(0, 1), expected)
    tail = torch.nn.functional.linear(x[2:], ops.values["layers.0.attn.compressor.wkv.weight"])
    torch.testing.assert_close(state.partial_kv, tail)


def test_sparse_attention_sink_and_invalid_slots_match_direct_arithmetic(modules, config):
    module, ops = modules["attention"], NumericalOps(config)
    q = torch.tensor([[1.0, 0.5], [-0.5, 1.0]], dtype=torch.bfloat16)
    values = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [999.0, -999.0]], dtype=torch.bfloat16)
    mask, sink = torch.tensor([True, True, False]), torch.tensor([0.3, -1.0])
    actual = module.sparse_attention(q, values, sink, mask, ops, 2**-0.5)
    logits = torch.einsum("hd,nd->hn", q.float(), values[:2].float()) * 2**-0.5
    maximum = logits.max(-1, keepdim=True).values
    weights = (logits - maximum).exp()
    expected = weights.to(torch.bfloat16).float() @ values[:2].float()
    expected /= weights.sum(-1, keepdim=True) + (sink[:, None] - maximum).exp()
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))
    empty = module.sparse_attention(q, values, sink, torch.zeros(3, dtype=torch.bool), ops, 1.0)
    assert torch.equal(empty, torch.zeros_like(q))


def test_sparse_attention_crosses_the_64_slot_online_softmax_boundary(modules, config):
    module, ops = modules["attention"], NumericalOps(config)
    q, values = torch.zeros((2, 4), dtype=torch.bfloat16), torch.ones((65, 4), dtype=torch.bfloat16)
    output = module.sparse_attention(q, values, torch.zeros(2), torch.ones(65, dtype=torch.bool), ops, 0.5)
    torch.testing.assert_close(output, torch.full_like(q, 65 / 66))


def test_hierarchical_candidates_pin_newest_partial_and_drop_unreachable(modules):
    choose = modules["attention"].select_candidate_blocks
    logits = torch.tensor([8.0, 9.0, 2.0, 3.0, -4.0, -torch.inf, -torch.inf, -torch.inf])
    assert set(choose(logits, 5, 2, 2).tolist()) == {0, 2}
    assert choose(torch.full((4,), -torch.inf), 0, 2, 2).numel() == 0


def test_real_rope_rotation_inverse_and_base_frequency(modules, config):
    module = modules["attention"]
    frequencies = module.rotary_frequencies(config, False, torch.device("cpu"))
    assert frequencies[0] == 1
    x = torch.randn((3, 2, 32), generator=torch.Generator().manual_seed(61))
    positions = torch.tensor([0, 3, 71])
    rotated = module.apply_rotary(x, positions, frequencies)
    torch.testing.assert_close(rotated[..., :-16], x[..., :-16])
    torch.testing.assert_close(rotated[0], x[0])
    restored = module.apply_rotary(rotated, positions, frequencies, inverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)
    yarn = module.rotary_frequencies(config, True, torch.device("cpu"))
    assert not torch.equal(yarn, frequencies)


def test_projection_failure_restores_cache_partial_and_publications(modules, config):
    module, ops = modules["attention"], NumericalOps(config)
    attention = module.Attention(config, ops, 0)
    state = attention.new_state(16, cache_page_size=2)
    x = torch.randn((5, 32), generator=torch.Generator().manual_seed(15))
    attention.forward(x[:3], 0, state, module.SharedAttention(0, 3))
    before, main = state.snapshot(), state.main.read(0, 1).clone()
    shared = module.SharedAttention(3, 5)
    ops.fail_suffix = ".wo_b.weight"
    with pytest.raises(RuntimeError, match="injected projection"):
        attention.forward(x[3:], 3, state, shared)
    assert state.position == 3 and state.main.length == state.index.length == 1
    assert state.window is before.window and state.partial_kv is before.partial_kv
    assert shared.sources == shared.indexes == shared.candidates == {}
    torch.testing.assert_close(state.main.read(0, 1), main)
    ops.fail_suffix = None
    output = attention.forward(x[3:], 3, state, shared)
    assert output.shape == (2, 32) and state.position == 5
    state.restore(before)
    assert state.position == 3 and state.main.length == 1


def test_reuse_requires_current_chunk_producer_and_rejects_stale_positions(modules, config):
    module, ops = modules["attention"], NumericalOps(config)
    reuse = module.Attention(config, ops, 1)
    state = reuse.new_state(10)
    x = torch.ones((2, 32))
    with pytest.raises(ValueError, match="source has not executed"):
        reuse.forward(x, 0, state, module.SharedAttention(0, 2))
    assert state.position == 0 and state.window is None
    with pytest.raises(ValueError, match="position mismatch"):
        reuse.forward(x, 1, state, module.SharedAttention(1, 3))
    alien = reuse.new_state(10)
    with pytest.raises(ValueError, match="different request"):
        state.restore(alien.snapshot())


def test_tensor_rows_cross_page_gather_rollback_and_reuse(modules):
    rows = modules["attention"].TensorRows(3, 8, page_size=2)
    values = torch.arange(18).reshape(6, 3).float()
    rows.append(values[:3])
    rows.append(values[3:])
    torch.testing.assert_close(rows.gather(torch.tensor([5, 0, 3, 5])), values[[5, 0, 3, 5]])
    torch.testing.assert_close(rows.read(1, 5), values[1:5])
    rows.truncate(3)
    rows.append(-values[3:5])
    torch.testing.assert_close(rows.read(0, 5), torch.cat((values[:3], -values[3:5])))
    with pytest.raises(ValueError, match="uncommitted"):
        rows.gather(torch.tensor([5]))
    with pytest.raises(ValueError, match="capacity"):
        rows.append(values)


def test_injected_swa_store_matches_tensor_tail_without_duplicate_storage(modules, config):
    module, ops = modules["attention"], NumericalOps(config)
    attention = module.Attention(config, ops, 0)
    supplied = {}

    def factory(kind, layer_id, dim, fmt):
        rows = module.TensorRows(dim, 20, page_size=3)
        supplied[kind] = rows
        assert layer_id == 0
        assert fmt == {"swa": module.FP8_SWA, "main_kv": module.FP4_MAIN, "index_k": module.FP4_INDEX}[kind]
        return rows

    custom = attention.new_state(20, rows_factory=factory)
    ordinary = attention.new_state(20)
    x = torch.randn((8, 32), generator=torch.Generator().manual_seed(791))
    for start, end in ((0, 3), (3, 5), (5, 8)):
        expected = attention.forward(x[start:end], start, ordinary, module.SharedAttention(start, end))
        actual = attention.forward(x[start:end], start, custom, module.SharedAttention(start, end))
        torch.testing.assert_close(actual, expected)
        assert custom.window is None and custom.swa.length == end
    snapshot = custom.snapshot()
    ops.fail_suffix = ".wo_b.weight"
    with pytest.raises(RuntimeError, match="injected projection"):
        attention.forward(x[:1], 8, custom, module.SharedAttention(8, 9))
    assert custom.swa.length == snapshot.swa_length == 8
    assert custom.position == 8


def draft_case(module, config):
    config = replace(config, num_nextn_predict_layers=1, compress_ratios=config.compress_ratios + (0,))
    ops = NumericalOps(config)
    for name, tensor in list(ops.values.items()):
        if name.startswith("layers.1.attn."):
            ops.values[name.replace("layers.1.attn.", "mtp.0.attn.")] = tensor
    return module.DraftAttention(config, ops, 0), ops


def test_draft_attention_seeds_main_ring_and_attends_to_future_draft_tokens(modules, config):
    module = modules["attention"]
    attention, _ = draft_case(module, config)
    state, other = attention.new_state(24), attention.new_state(24)
    main = torch.randn((6, 32), generator=torch.Generator().manual_seed(14))
    draft = torch.randn((3, 32), generator=torch.Generator().manual_seed(16))
    assert attention.forward(draft, main[:5], 0, state) is draft
    attention.forward(draft, main[:5], 0, other)
    assert state.position == 5 and len(state.window) == 4
    output = attention.forward(draft, main[5:], 5, state)
    changed = draft.clone()
    changed[-1] += torch.linspace(-3, 4, 32)
    future = attention.forward(changed, main[5:], 5, other)
    assert output.shape == draft.shape and torch.isfinite(output).all()
    assert not torch.allclose(output[0], future[0], atol=1e-6)
    # Draft KV never pollutes the committed main-token window.
    assert state.position == other.position == 6
    torch.testing.assert_close(state.window, other.window)
    assert state.main is state.index is state.swa is None


def test_draft_projection_failure_preserves_main_ring(modules, config):
    attention, ops = draft_case(modules["attention"], config)
    state = attention.new_state(16)
    main, draft = torch.ones((3, 32)), torch.ones((2, 32))
    attention.forward(draft, main, 0, state)
    before = state.snapshot()
    ops.fail_suffix = ".wo_b.weight"
    with pytest.raises(RuntimeError, match="injected projection"):
        attention.forward(draft, main[:1], 3, state)
    assert state.position == 3 and state.window is before.window
    with pytest.raises(ValueError, match="one main token"):
        attention.forward(draft, main[:2], 3, state)


def test_draft_seed_accepts_chunked_main_hidden_without_draft_projection(modules, config):
    attention, ops = draft_case(modules["attention"], config)
    whole, chunked = attention.new_state(24), attention.new_state(24)
    main = torch.randn((11, 32), generator=torch.Generator().manual_seed(391))
    ops.fail_suffix = ".wo_b.weight"
    attention.seed_main(main, 0, whole)
    for start, end in ((0, 2), (2, 5), (5, 11)):
        attention.seed_main(main[start:end], start, chunked)
    assert whole.position == chunked.position == 11
    torch.testing.assert_close(whole.window, chunked.window)
    before = chunked.snapshot()
    ops.fail_suffix = ".wkv.weight"
    with pytest.raises(RuntimeError, match="injected projection"):
        attention.seed_main(main[:2], 11, chunked)
    assert chunked.position == 11 and chunked.window is before.window


def test_draft_propose_does_not_append_main_tokens_again(modules, config):
    attention, _ = draft_case(modules["attention"], config)
    state, combined = attention.new_state(24), attention.new_state(24)
    main = torch.randn((5, 32), generator=torch.Generator().manual_seed(190))
    draft = torch.randn((3, 32), generator=torch.Generator().manual_seed(191))
    attention.seed_main(main, 0, state)
    before = state.snapshot()
    actual = attention.propose(draft, state)
    again = attention.propose(draft, state)
    attention.seed_main(main[:4], 0, combined)
    expected = attention.forward(draft, main[4:], 4, combined)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(again, actual)
    assert state.position == 5 and state.window is before.window
