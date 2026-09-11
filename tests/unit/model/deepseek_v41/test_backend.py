# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete miniature checkpoint numerical/transaction tests, not A5 acceptance.

The CPU matrix provider evaluates real randomly initialized FP8/FP4/BF16 weights.
The same target graph is exercised through different scheduling and storage paths;
these invariance checks supplement the independent per-operation reference goldens.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="backend numerical tests require CPU Torch")
serialization = pytest.importorskip("safetensors.torch")
pytest.importorskip("numpy")
if not hasattr(torch, "float8_e8m0fnu"):
    pytest.skip("native UE8M0 dtype is required", allow_module_level=True)
ROOT = Path(__file__).resolve().parents[4]
TOKENS = (3, 9, 12, 4, 18, 23, 8, 17, 11)


@pytest.fixture
def modules(monkeypatch):
    before = {name: value for name, value in sys.modules.items()
              if name == "pypto_serving" or name.startswith("pypto_serving.")}
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    package = types.ModuleType("pypto_serving")
    package.__path__ = [str(ROOT / "pypto_serving")]
    monkeypatch.setitem(sys.modules, "pypto_serving", package)
    try:
        yield SimpleNamespace(**{
            suffix: importlib.import_module("pypto_serving.model.deepseek_v41." + suffix)
            for suffix in ("config", "cache", "engram", "tensor_store", "numerics", "pypto_ops",
                           "backend", "npu_runner", "attention", "packed_cache")
        })
    finally:
        for name in tuple(sys.modules):
            if name == "pypto_serving" or name.startswith("pypto_serving."):
                sys.modules.pop(name, None)
        sys.modules.update(before)


@pytest.fixture
def checkpoint(tmp_path, modules):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32, vocab_size=64, moe_intermediate_size=32, num_hidden_layers=5,
        num_attention_heads=2, head_dim=32, qk_rope_head_dim=16, q_lora_rank=32,
        o_lora_rank=32, o_groups=1, hc_mult=2, hc_sinkhorn_iters=4, rms_norm_eps=1e-6,
        n_routed_experts=4, num_experts_per_tok=2, index_n_heads=2, index_head_dim=32,
        index_topk=2, sliding_window=4, max_position_embeddings=64,
        compress_ratios=[2, 2, 2, 1, 1, 0, 0, 0], kv_source_layer_ids=[0, 3],
        index_source_layer_ids=[0, 2, 3, 4], candidate_source_layer_id=3,
        candidate_topk_blocks=2, candidate_block_size=2,
        engram_layer_ids=[1, 3], engram_vocab_size=5, engram_num_embeddings=[5, 7],
        engram_head_dim=32, engram_max_ngram_size=2, engram_n_heads=1,
        engram_compressed_vocab_size=64, engram_pad_token_id=2,
        num_nextn_predict_layers=3, dspark_target_layer_ids=[1, 3],
        dspark_n_routed_experts=4, dspark_num_experts_per_tok=2, dspark_markov_rank=8,
        dspark_block_size=3, dspark_noise_token_id=63,
    )
    raw["vision_config"].update(
        hidden_size=32, num_attention_heads=2, intermediate_size=32, num_hidden_layers=1,
        patch_size=2, downsample_ratio=2, max_image_tokens=64, min_pixels=0,
    )
    store = modules.tensor_store
    specs = store.backbone_weight_specs(raw) | store.draft_weight_specs(raw)
    specs.update({name: store.TensorSpec(shape, "BF16", name, "identity")
                  for name, shape in store.vision_weight_shapes(store.VisionConfig.from_config(raw)).items()})
    generator = torch.Generator().manual_seed(48137)
    tensors = {}
    for name, spec in specs.items():
        if spec.dtype == "F8_E4M3":
            value = (torch.randn(spec.shape, generator=generator) * 1.2).to(torch.float8_e4m3fn)
        elif spec.dtype == "F8_E8M0":
            value = torch.randint(121, 124, spec.shape, generator=generator, dtype=torch.uint8)
            value = value.view(torch.float8_e8m0fnu)
        elif spec.dtype == "I8":
            value = torch.randint(0, 256, spec.shape, generator=generator, dtype=torch.int16).to(torch.uint8)
            value = value.view(torch.int8)
        else:
            value = torch.randn(spec.shape, generator=generator) * .08
            if "norm.weight" in name or name.endswith(("q_weight", "k_weight")):
                value += 1
            if name.endswith("_scale"):
                value += .2
            value = value.to(torch.bfloat16 if spec.dtype == "BF16" else torch.float32)
        tensors[name] = value.contiguous()
    serialization.save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "weights.safetensors" for name in tensors}
    }))
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path, raw


@pytest.fixture
def factory(modules, checkpoint):
    path, raw = checkpoint
    created = []

    def make(*, max_seq_len=32, max_chunk=16, speculative=True):
        config = modules.config.DeepSeekV41Config.from_dict(raw)
        cache = modules.cache.V41CacheState(config, page_size=2, max_seq_len=max_seq_len,
                                            max_chunk_tokens=max_chunk)
        weights = modules.tensor_store.DeepSeekV41TensorStore(
            path, raw, max_load_bytes=8 << 20, row_cache_bytes=4096, prefetch_rows=4,
            out_tile_rows=32, dense_k_tile=32,
        )
        provider = modules.pypto_ops.TorchMatmulOps(max_buffer_bytes=8 << 20)
        ops = modules.numerics.TensorOps(weights, matmul_provider=provider)
        runtime = SimpleNamespace(
            max_seq_len=max_seq_len, max_batch_size=2, total_kv_pages=cache.groups[0].max_blocks_per_seq * 2,
            max_prefill_tokens_per_request=max_chunk, max_num_batched_tokens=max_chunk * 2,
            num_speculative_tokens=3 if speculative else 0,
        )
        backend = modules.backend.DeepSeekV41Backend(config, runtime, cache.groups, ops, platform="cpu")
        layout = modules.engram.EngramLayout.from_config(raw)

        def history():
            return modules.engram.EngramHashState(layout, tuple(range(64)), compressed_vocab_size=64,
                                                 pad_token_id=2, rollback_window=8)

        runner = modules.npu_runner.DeepSeekV41Runner(config, cache, backend, history, max_requests=2)

        def work(rid="a", tokens=TOKENS, start=0, slot=0, mode="prefill"):
            pages = {group.name: tuple(range(slot * group.max_blocks_per_seq,
                                            (slot + 1) * group.max_blocks_per_seq)) for group in cache.groups}
            return modules.npu_runner.V41WorkItem(rid, tuple(tokens), start, pages, mode=mode)

        def execute(*items):
            return runner.execute(items, validate_outputs=lambda outputs: outputs)

        harness = SimpleNamespace(config=config, cache=cache, backend=backend, runner=runner,
                                  ops=ops, weights=weights, history=history, work=work, execute=execute)
        created.append(harness)
        return harness

    yield make
    for harness in created:
        harness.runner.close()


def assert_logits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float32
    assert bool(torch.isfinite(actual).all())
    # BF16 rounding can straddle a representable value when the BLAS batch shape changes.
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    assert actual.argmax().item() == expected.argmax().item()


def page_snapshot(backend):
    return {key: value.clone() for key, value in backend.pool.pages.items()}


def assert_pages(backend, snapshot):
    assert backend.pool.pages.keys() == snapshot.keys()
    for name, value in snapshot.items():
        assert torch.equal(backend.pool.pages[name], value)


def unpaged_forward(harness, modules, tokens):
    """Same model operations through append-only BF16 rows, without packed pages or a runner."""
    cfg, ops = harness.config, harness.ops
    ids = torch.tensor(tokens)
    hashes = torch.tensor(harness.history().advance(tokens, start_pos=0))
    hidden = ops.embedding("embed.weight", ids).unsqueeze(1).repeat(1, cfg.text_config["hc_mult"], 1)
    pre = torch.zeros((len(tokens), cfg.text_config["hc_mult"]), dtype=torch.float32)
    pre[:, 0] = 1
    math = modules.numerics.ModelMath(cfg, ops)
    shared = modules.attention.SharedAttention(0, len(tokens))
    targets = []
    for plan in cfg.layer_plan:
        if plan.requires_engram:
            index = cfg.engram_layer_ids.index(plan.layer_id)
            hidden = math.engram(hidden, hashes[:, index], f"layers.{plan.layer_id}.engram")
        if plan.layer_id in cfg.dspark_target_layer_ids:
            targets.append(hidden.mean(-2))
        attention = modules.attention.Attention(cfg, ops, plan.layer_id)
        state = attention.new_state(32)
        hidden, pre = math.block(hidden, pre, f"layers.{plan.layer_id}",
                                 lambda x: attention.forward(x, 0, state, shared))
    collapsed = math.hc_pre(hidden, pre)[-1:]
    normalized = modules.numerics.rms_norm(collapsed, ops.weight("norm.weight"), math.eps)
    return ops.linear(normalized.float(), "head.weight")[0], torch.cat(targets, -1)


def test_complete_checkpoint_whole_chunks_decode_and_unpaged_storage(factory, modules):
    whole = factory()
    expected = whole.execute(whole.work())[0]
    golden, targets = unpaged_forward(whole, modules, TOKENS)
    assert_logits(expected, golden)
    request = next(iter(whole.backend.requests.values()))
    assert torch.equal(request.last_hidden, targets[-1:])
    assert len(request.draft.attention) == 3
    assert request.draft.position == len(TOKENS)
    assert expected.std().item() > .01

    for chunks in ((1, 3, 2, 3), (2, 1, 1, 5), (1,) * len(TOKENS)):
        chunked = factory()
        start = 0
        for length in chunks:
            actual = chunked.execute(chunked.work(tokens=TOKENS[start:start + length], start=start))[0]
            start += length
        assert_logits(actual, expected)
        assert chunked.runner.position("a") == len(TOKENS)
    decoded = factory()
    actual = decoded.execute(decoded.work(tokens=TOKENS[:3]))[0]
    decoded.runner.finalize_prefill(["a"])
    for pos in range(3, len(TOKENS)):
        actual = decoded.execute(decoded.work(tokens=(TOKENS[pos],), start=pos, mode="decode"))[0]
    assert_logits(actual, expected)


@pytest.mark.parametrize("failure", ["layer", "validation", "commit"])
def test_two_requests_abort_restores_pages_partial_compressor_engram_and_retry(factory, monkeypatch, failure):
    run, baseline = factory(), factory()
    for harness in (run, baseline):
        harness.execute(harness.work(tokens=TOKENS[:3]), harness.work("b", TOKENS[3:6], slot=1))
    pages = page_snapshot(run.backend)
    targets = {key: state.last_hidden.clone() for key, state in run.backend.requests.items()}
    original_layer, original_commit = run.backend.layer, run.backend.commit_batch

    def fail_layer(ticket, plan, state, context):
        result = original_layer(ticket, plan, state, context)
        if context.work.request_id == "b" and plan.layer_id == 3:
            raise ArithmeticError("injected after packed KV and compressor writes")
        return result

    def fail_commit(ticket):
        original_commit(ticket)
        raise ArithmeticError("injected after backend commit")

    def validate(outputs):
        if failure == "validation":
            raise ArithmeticError("injected output validation failure")
        return outputs

    if failure == "layer":
        monkeypatch.setattr(run.backend, "layer", fail_layer)
    if failure == "commit":
        monkeypatch.setattr(run.backend, "commit_batch", fail_commit)
    works = (run.work(tokens=TOKENS[3:6], start=3), run.work("b", TOKENS[:3], start=3, slot=1))
    with pytest.raises(ArithmeticError):
        run.runner.execute(works, validate_outputs=validate)
    assert_pages(run.backend, pages)
    assert run.runner.position("a") == run.runner.position("b") == 3
    for key, state in run.backend.requests.items():
        assert torch.equal(state.last_hidden, targets[key])
        assert state.layers[0].partial_kv.shape == (1, 32)
        assert state.layers[0].main.length == 1
        assert state.draft.position == 3
    monkeypatch.setattr(run.backend, "layer", original_layer)
    monkeypatch.setattr(run.backend, "commit_batch", original_commit)
    expected = baseline.execute(baseline.work(tokens=TOKENS[3:6], start=3),
                                baseline.work("b", TOKENS[:3], start=3, slot=1))
    actual = run.execute(*works)
    for output, golden in zip(actual, expected):
        assert_logits(output, golden)


def test_release_reuses_physical_pages_with_new_generation_and_no_engram_leak(factory):
    run, fresh = factory(), factory()
    run.execute(run.work(), run.work("b", TOKENS[::-1], slot=1))
    b_key = next(key for key in run.backend.requests if key[0] == "b")
    b_pages = {key: value.clone() for key, value in run.backend.pool.pages.items()
               if key[1] >= run.cache.layouts[key[0]].max_blocks_per_seq}
    old_generation = run.runner._requests["a"].generation
    run.runner.release(["a"])
    actual = run.execute(run.work(tokens=TOKENS[::-1]))[0]
    expected = fresh.execute(fresh.work(tokens=TOKENS[::-1]))[0]
    assert_logits(actual, expected)
    assert run.runner._requests["a"].generation > old_generation
    assert run.backend.requests[b_key].position == len(TOKENS)
    for key, value in b_pages.items():
        assert torch.equal(run.backend.pool.pages[key], value)


def prefill_decode(harness, tokens=TOKENS[:5]):
    logits = harness.execute(harness.work(tokens=tokens))[0]
    harness.runner.finalize_prefill(["a"])
    return int(logits.argmax().item())


def test_real_dspark_verified_prefix_equals_greedy_baseline_and_committed_stats(factory):
    run, baseline = factory(), factory(speculative=False)
    pending = prefill_decode(run)
    assert pending == prefill_decode(baseline)
    observed = []
    for _ in range(3):
        start = run.runner.position("a")
        _, emitted = run.runner.speculate(
            [run.work(tokens=(pending,), start=start, mode="decode")],
            extra_tokens=3, validate_outputs=lambda outputs: outputs,
        )
        for expected in emitted[0]:
            actual = baseline.execute(baseline.work(tokens=(pending,), start=baseline.runner.position("a"),
                                                    mode="decode"))[0]
            assert int(actual.argmax().item()) == expected
            pending = expected
            observed.append(expected)
        assert run.runner.position("a") == baseline.runner.position("a")
    stats = run.runner.speculation_stats
    assert stats["rounds"] == 3 and stats["proposed"] == 9
    assert stats["verified"] == len(observed) - 3
    assert stats["accepted"] + stats["rejected"] == stats["verified"]
    assert stats["rejected"] <= stats["rounds"]


def test_confidence_threshold_and_sequence_tail_leave_pending_target_unconsumed(factory, monkeypatch):
    run = factory(max_seq_len=8)
    pending = prefill_decode(run)
    # Position 6 leaves two slots: the full three-token draft cannot be evaluated.
    monkeypatch.setattr(run.backend, "propose", lambda *args: pytest.fail("tail must skip the full draft graph"))
    logits, emitted = run.runner.speculate([run.work(tokens=(pending,), start=5, mode="decode")],
                                          extra_tokens=3, validate_outputs=lambda rows: rows)
    assert emitted == [[int(logits[0].argmax().item())]]
    assert run.runner.position("a") == 6
    assert all(value == 0 for value in run.runner.speculation_stats.values())

    other = factory()
    pending = prefill_decode(other)
    logits, emitted = other.runner.speculate([other.work(tokens=(pending,), start=5, mode="decode")],
                                             extra_tokens=3, minimum_score=1e20,
                                             validate_outputs=lambda rows: rows)
    assert emitted == [[int(logits[0].argmax().item())]]
    assert other.runner.position("a") == 6
    assert other.runner.speculation_stats == dict(rounds=1, proposed=3, verified=0, accepted=0, rejected=0)


def test_failed_speculation_restores_inner_commits_rng_cache_and_statistics(factory, monkeypatch):
    run, baseline = factory(max_chunk=1), factory(max_chunk=1)
    for harness in (run, baseline):
        for start, token in enumerate(TOKENS):
            harness.execute(harness.work(tokens=(token,), start=start))
        harness.runner.finalize_prefill(["a"])
    pages = page_snapshot(run.backend)
    request = next(iter(run.backend.requests.values()))
    old_rng = request.draft.generator.get_state().clone()
    old_history = run.runner._requests["a"].history.snapshot()
    pending = 7
    calls = 0
    # Force a matching candidate prefix obtained from real target forwards so the
    # failure occurs after several committed inner steps and a ring overwrite.
    # The draft graph still runs; only its candidate IDs are controlled here.
    probe = factory(max_chunk=1, speculative=False)
    for start, token in enumerate(TOKENS):
        probe.execute(probe.work(tokens=(token,), start=start))
    probe.runner.finalize_prefill(["a"])
    verified = []
    token = pending
    for _ in range(4):
        value = probe.execute(probe.work(tokens=(token,), start=probe.runner.position("a"), mode="decode"))[0]
        token = int(value.argmax().item())
        verified.append(token)
    original = run.backend.propose

    def matching_proposal(*args):
        output = original(*args)
        return type(output)(torch.tensor(verified), output.logits, output.confidence)

    monkeypatch.setattr(run.backend, "propose", matching_proposal)

    def validate(outputs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise ArithmeticError("failure during target verification after initial target commit")
        return outputs

    work = run.work(tokens=(pending,), start=len(TOKENS), mode="decode")
    with pytest.raises(ArithmeticError, match="target verification"):
        run.runner.speculate([work], extra_tokens=3, validate_outputs=validate)
    assert_pages(run.backend, pages)
    assert run.runner.position("a") == len(TOKENS)
    assert run.runner._requests["a"].history.snapshot() == old_history
    assert torch.equal(request.draft.generator.get_state(), old_rng)
    assert all(value == 0 for value in run.runner.speculation_stats.values())
    monkeypatch.setattr(run.backend, "propose", original)
    actual, emitted = run.runner.speculate([work], extra_tokens=3, validate_outputs=lambda rows: rows)
    expected, expected_tokens = baseline.runner.speculate(
        [baseline.work(tokens=(pending,), start=len(TOKENS), mode="decode")],
        extra_tokens=3, validate_outputs=lambda rows: rows,
    )
    assert_logits(actual[0], expected[0])
    assert emitted == expected_tokens


@pytest.mark.parametrize("reject_second", [False, True])
def test_prefix_accounting_uses_conditional_target_predictions(factory, monkeypatch, reject_second):
    run, baseline = factory(), factory(speculative=False)
    pending = prefill_decode(run)
    assert pending == prefill_decode(baseline)
    tokens = []
    for _ in range(4):
        result = baseline.execute(baseline.work(tokens=(pending,), start=baseline.runner.position("a"),
                                                mode="decode"))[0]
        pending = int(result.argmax().item())
        tokens.append(pending)
    original = run.backend.propose

    def controlled_proposal(*args):
        result = original(*args)
        ids = torch.tensor(tokens)
        if reject_second:
            ids[2] = (ids[2] + 1) % run.config.vocab_size
        return type(result)(ids, result.logits, result.confidence)

    monkeypatch.setattr(run.backend, "propose", controlled_proposal)
    # Reconstruct the initial pending token from the same committed prefix.
    initial = factory(speculative=False)
    pending = prefill_decode(initial)
    _, emitted = run.runner.speculate([run.work(tokens=(pending,), start=5, mode="decode")],
                                      extra_tokens=3, validate_outputs=lambda rows: rows)
    compared, accepted = (2, 1) if reject_second else (3, 3)
    assert emitted == [tokens[:compared + 1]]
    assert run.runner.position("a") == 6 + compared
    assert run.runner.speculation_stats == dict(rounds=1, proposed=3, verified=compared,
                                                accepted=accepted, rejected=int(reject_second))


def test_nested_page_journal_preserves_outer_snapshot_after_inner_abort_retry(factory):
    run = factory()
    run.execute(run.work(tokens=TOKENS[:3]))
    pool = run.backend.pool
    before = page_snapshot(run.backend)
    key = next(iter(before))
    checkpoint = pool.checkpoint()
    pool.begin()
    # Production cache writes run in the backend's inference-mode layer call;
    # preserve that context when injecting byte mutations directly into pages.
    with torch.inference_mode():
        pool.page(*key, write=True).fill_(91)
    pool.abort()
    assert_pages(run.backend, before)
    pool.begin()
    with torch.inference_mode():
        pool.page(*key, write=True).fill_(173)
    pool.finish_checkpoint(checkpoint, restore=True)
    assert_pages(run.backend, before)


def test_cpu_diagnostics_report_only_actual_allocations(factory):
    run = factory()
    empty = run.backend.diagnostics()
    assert empty["cache_bytes"] == empty["active_requests"] == 0
    run.execute(run.work(tokens=TOKENS[:3]))
    report = run.backend.diagnostics()
    assert report["device"] == "cpu" and report["rank"] == 0 and report["world_size"] == 1
    assert report["active_requests"] == 1
    assert report["logical_cache_capacity_bytes"] == sum(
        layout.page_size_bytes * run.backend.pool.limits[layout.name] for layout in run.cache.groups
    )
    assert report["memory_preflight_estimate"] is None
    assert report["cache_bytes"] == sum(t.numel() * t.element_size() for t in run.backend.pool.pages.values())
    assert 0 < report["weight_row_cache_bytes"] <= 4096 and report["weight_placement"] == "cpu"
    assert all(report[name] is None for name in ("npu_memory_allocated", "npu_memory_reserved",
                                               "npu_max_memory_allocated"))
