# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host framework integration with an explicitly synthetic recording backend.

These tests do not open an NPU, run model arithmetic, or claim HTTP/model goldens.
The on-disk index and empty shard only exercise metadata and lazy loader construction;
test_weight_loader.py separately exercises actual safetensors payloads and conversions.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="V4.1 serving adapter tests require CPU Torch")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    # Import the actual production modules, bypassing only package-root eager
    # engine startup. Restore every package module after each isolated test.
    before = {
        name: module
        for name, module in sys.modules.items()
        if name == "pypto_serving" or name.startswith("pypto_serving.")
    }
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    package = types.ModuleType("pypto_serving")
    package.__path__ = [str(ROOT / "pypto_serving")]
    monkeypatch.setitem(sys.modules, "pypto_serving", package)
    try:
        yield SimpleNamespace(
            format=importlib.import_module("pypto_serving.model.deepseek_v41.format_loader"),
            family=importlib.import_module("pypto_serving.model.model_family"),
            registry=importlib.import_module("pypto_serving.model.model_loader"),
            config=importlib.import_module("pypto_serving.config.types"),
            cli=importlib.import_module("pypto_serving.cli.main"),
            executor=importlib.import_module("pypto_serving.model.deepseek_v41.npu_executor"),
            engram=importlib.import_module("pypto_serving.model.deepseek_v41.engram"),
            store=importlib.import_module("pypto_serving.model.common.weights.store"),
        )
    finally:
        for name in tuple(sys.modules):
            if name == "pypto_serving" or name.startswith("pypto_serving."):
                sys.modules.pop(name, None)
        sys.modules.update(before)


@pytest.fixture
def case(tmp_path, modules, monkeypatch):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32,
        vocab_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
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
        sliding_window=4,
        max_position_embeddings=8,
        compress_ratios=[1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        engram_layer_ids=[0],
        engram_num_embeddings=[2],
        engram_head_dim=32,
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_vocab_size=2,
        engram_compressed_vocab_size=64,
        num_nextn_predict_layers=0,
        dspark_target_layer_ids=[],
    )
    (tmp_path / "config.json").write_text(json.dumps(raw))
    names = modules.format.backbone_weight_specs(raw)
    weight_map = {name: "metadata-placeholder.safetensors" for name in names}
    # Presence alone is sufficient for this metadata registration path. No payload
    # reader is allowed to consume this deliberately non-tensor placeholder.
    (tmp_path / "metadata-placeholder.safetensors").write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    tokenizer = SimpleNamespace(bos_token_id=55, eos_token_id=56, pad_token_id=57)
    monkeypatch.setattr(modules.format, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(modules.executor, "build_compressed_token_map", lambda value: (tuple(range(64)), 64))
    monkeypatch.setattr(modules.engram, "compute_hash_multipliers", lambda layout, size: ((1, 3),))

    def no_payloads(*args, **kwargs):
        pytest.fail("metadata registration must not materialize model weights")

    monkeypatch.setattr(modules.store.LazySafetensorsStore, "load_many", no_payloads)
    parsed = modules.format.DeepSeekV41Config.from_dict(raw)
    groups = modules.executor.build_v41_cache_group_specs(parsed, 4, 8, 4, max_chunk_tokens=4)
    runtime = modules.config.RuntimeConfig(
        page_size=4,
        max_seq_len=8,
        max_batch_size=2,
        total_kv_pages=4,
        max_num_batched_tokens=4,
        max_prefill_tokens_per_request=4,
        kv_cache_groups=groups,
    )
    loaded = modules.registry.ModelLoader().load("tiny-v41", str(tmp_path), runtime_config=runtime)
    record = modules.config.ModelRecord(
        loaded.config, runtime, loaded.tokenizer, loaded.layer_specs, loaded.runtime_model
    )
    return SimpleNamespace(
        root=tmp_path,
        raw=raw,
        weight_map=weight_map,
        modules=modules,
        runtime=runtime,
        loaded=loaded,
        record=record,
    )


class RecordingBackend:
    """Synthetic protocol backend: event ordering and logits validation only."""

    def __init__(self, capabilities, vocab=64):
        self.capabilities = capabilities
        self.num_pages = 4
        self.vocab = vocab
        self.events = []
        self.invalid_logits = False
        self.closed = False

    def begin_batch(self, contexts):
        self.events.append(("begin", tuple(context.work.request_id for context in contexts)))
        return object()

    def embed(self, ticket, context):
        self.events.append(("embed", context.work.request_id))
        return context.work.token_ids

    def engram(self, ticket, layer, state, context):
        self.events.append(("engram", layer.layer_id))
        return state

    def layer(self, ticket, layer, state, context):
        self.events.append(("layer", layer.layer_id))
        return state

    def head(self, ticket, state, context):
        self.events.append(("head", context.work.request_id))
        logits = torch.zeros(self.vocab, dtype=torch.float32)
        logits[state[-1]] = float("nan") if self.invalid_logits else 1.0
        return logits

    def commit_batch(self, ticket):
        self.events.append(("commit",))

    def abort_batch(self, ticket):
        self.events.append(("abort",))

    def release_request(self, request_id, generation):
        self.events.append(("release", request_id, generation))

    def close(self):
        self.closed = True


def executor_case(case, *, capability_changes=None):
    module = case.modules.executor
    caps = module.V41BackendCapabilities(
        world_size=1,
        max_chunk_tokens=4,
        max_batch_size=2,
        max_seq_len=8,
        num_layers=1,
        parallel_mode="reference_tp",
    )
    if capability_changes:
        caps = dataclasses.replace(caps, **capability_changes)
    backend = RecordingBackend(caps)
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return backend

    executor = module.DeepSeekV41PyptoExecutor(kernel_factory=factory, device_ids=(0,))
    return executor, backend, factory_calls


def prefill(case, *, request="request-a", tokens=(3, 4), start=0):
    pages = {group.name: list(range(group.max_blocks_per_seq)) for group in case.runtime.kv_cache_groups}
    return case.modules.config.PrefillBatch(
        request_ids=[request],
        token_ids=torch.tensor(tokens, dtype=torch.int64),
        input_embeddings=None,
        seq_lens=[start + len(tokens)],
        chunk_lens=[len(tokens)],
        chunk_offsets=[0],
        chunk_starts=[start],
        block_ids_by_group=[pages],
    )


def test_directory_loader_is_lazy_and_uses_outer_special_token_ids(case):
    model = case.loaded.runtime_model
    assert model.extra["family"] == "deepseek_v41"
    assert model.extra["weights_materialized"] is False
    assert model.embed_tokens.numel() == model.lm_head.numel() == model.final_norm_weight.numel() == 0
    assert (
        case.loaded.config.bos_token_id,
        case.loaded.config.eos_token_id,
        case.loaded.config.pad_token_id,
    ) == (0, 1, 2)
    assert len(case.loaded.layer_specs) == 1
    assert model.extra["weight_map"] == case.weight_map


def test_family_and_format_dispatch_keep_v4_and_v41_distinct(case):
    family, registry = case.modules.family, case.modules.registry
    assert family.detect_model_family(case.raw) == "deepseek_v41"
    assert not family.is_deepseek_v4_config(case.raw)
    v4 = {"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}
    assert family.detect_model_family(v4) == "deepseek_v4"
    assert not family.is_deepseek_v41_config(v4)
    conflict = dict(case.raw, architectures=["DeepseekV4ForCausalLM"])
    with pytest.raises(ValueError, match="Conflicting"):
        family.detect_model_family(conflict)
    for alias in (None, "deepseek_v41", "deepseek-v41", "dsv41"):
        request = registry.ModelLoadRequest("tiny-v41", str(case.root), model_format=alias)
        assert isinstance(
            registry.ModelLoader()._select_loader(request), case.modules.format.DeepSeekV41DirectoryLoader
        )
    (case.root / "config.json").write_text(json.dumps(v4))
    selected = registry.ModelLoader()._select_loader(registry.ModelLoadRequest("v4", str(case.root)))
    assert isinstance(selected, registry.DeepSeekV4W8A8DirectoryLoader)


@pytest.mark.parametrize("change", ["missing", "unknown", "shard"])
def test_directory_loader_rejects_incomplete_or_unknown_text_weights(case, change):
    if change == "missing":
        case.weight_map.pop("layers.0.attn.wq_a.scale")
    elif change == "unknown":
        case.weight_map["layers.0.unknown.weight"] = "metadata-placeholder.safetensors"
    else:
        (case.root / "metadata-placeholder.safetensors").unlink()
    (case.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": case.weight_map}))
    with pytest.raises((ValueError, FileNotFoundError), match="index mismatch|shard missing"):
        case.modules.registry.ModelLoader().load("bad", str(case.root), runtime_config=case.runtime)


def test_directory_loader_classifies_vision_and_draft_without_loading_them(case):
    case.weight_map.update(
        {
            name: "metadata-placeholder.safetensors"
            for name in ("vision.weight", "aligner.weight", "image_start", "mtp.0.weight")
        }
    )
    (case.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": case.weight_map}))
    loaded = case.modules.registry.ModelLoader().load("tiny-v41", str(case.root), runtime_config=case.runtime)
    assert loaded.runtime_model.extra["weights_materialized"] is False
    assert loaded.runtime_model.extra["weight_map"] == case.weight_map


def test_executor_without_implemented_factory_fails_before_device_start(modules):
    with pytest.raises(RuntimeError, match="not bundled"):
        modules.executor.DeepSeekV41PyptoExecutor()
    with pytest.raises(ValueError, match="platform='a5'"):
        modules.executor.DeepSeekV41PyptoExecutor(platform="a2a3", kernel_factory=lambda **kwargs: None)


def test_executor_registers_lazy_loader_and_translates_prefill_decode_release(case):
    executor, backend, calls = executor_case(case)
    try:
        assert executor.register_model("tiny-v41", case.record) == 4
        assert len(calls) == 1
        assert isinstance(calls[0]["weight_loader"], case.modules.executor.DeepSeekV41WeightLoader)
        assert calls[0]["weight_loader"].max_load_bytes == 256 << 20
        assert calls[0]["device_ids"] == (0,)
        assert executor.supports_device_embedding
        batch = prefill(case)
        result = executor.run_prefill(case.loaded.runtime_model, batch)
        assert result.logits.shape == (1, 64)
        assert result.logits.argmax(-1).tolist() == [4]
        assert executor._runner.position("request-a") == 2
        executor.finalize_prefill(case.loaded.runtime_model, ["request-a"], [4])
        decode = case.modules.config.DecodeBatch(
            ["request-a"],
            torch.tensor([4], dtype=torch.int64),
            None,
            torch.tensor([3], dtype=torch.int64),
            block_ids_by_group=batch.block_ids_by_group,
        )
        decoded = executor.run_decode(case.loaded.runtime_model, decode)
        assert decoded.logits.shape == (1, 64)
        assert executor._runner.position("request-a") == 3
        assert sum(event[0] == "engram" for event in backend.events) == 2
        assert sum(event[0] == "layer" for event in backend.events) == 2
        executor.release_finished_requests(["request-a"])
        assert sum(event[0] == "release" for event in backend.events) == 1
    finally:
        executor.close()
    assert backend.closed


def test_executor_invalid_logits_abort_without_advancing_request(case):
    executor, backend, _ = executor_case(case)
    try:
        executor.register_model("tiny-v41", case.record)
        backend.invalid_logits = True
        with pytest.raises(ValueError, match="nonfinite logits"):
            executor.run_prefill(case.loaded.runtime_model, prefill(case))
        assert ("abort",) in backend.events
        assert ("commit",) not in backend.events
        backend.invalid_logits = False
        executor.run_prefill(case.loaded.runtime_model, prefill(case))
        assert executor._runner.position("request-a") == 2
    finally:
        executor.close()


def test_executor_allows_three_short_requests_in_two_full_length_slots(case):
    runtime = dataclasses.replace(case.runtime, max_batch_size=3)
    record = dataclasses.replace(case.record, runtime=runtime)
    executor, backend, _ = executor_case(case, capability_changes={"max_batch_size": 3})
    try:
        assert executor.register_model("tiny-v41", record) == 4
        # Four physical pages hold two maximum-length sequences, but three
        # one-token requests need only three pages in each independent pool.
        assert backend.num_pages // runtime.kv_cache_groups[0].max_blocks_per_seq == 2
        request_ids = [f"short-{i}" for i in range(3)]
        batch = case.modules.config.PrefillBatch(
            request_ids=request_ids,
            token_ids=torch.tensor([3, 4, 5], dtype=torch.int64),
            input_embeddings=None,
            seq_lens=[1, 1, 1],
            chunk_lens=[1, 1, 1],
            chunk_offsets=[0, 1, 2],
            chunk_starts=[0, 0, 0],
            block_ids_by_group=[
                {group.name: [page] for group in runtime.kv_cache_groups} for page in range(3)
            ],
        )
        result = executor.run_prefill(case.loaded.runtime_model, batch)
        assert result.logits.shape == (3, 64)
        assert result.logits.argmax(-1).tolist() == [3, 4, 5]
        assert [executor._runner.position(request_id) for request_id in request_ids] == [1, 1, 1]
        assert backend.events.count(("begin", tuple(request_ids))) == 1
        assert backend.events.count(("commit",)) == 1
        executor.release_finished_requests(request_ids)
        assert sum(event[0] == "release" for event in backend.events) == 3
    finally:
        executor.close()


@pytest.mark.parametrize("group_name", ["main_kv.0", "index_k.0", "swa.0"])
def test_executor_rejects_physical_page_at_pool_upper_bound_before_dispatch(case, group_name):
    executor, backend, _ = executor_case(case)
    try:
        executor.register_model("tiny-v41", case.record)
        batch = prefill(case)
        batch.block_ids_by_group[0][group_name] = [backend.num_pages]
        with pytest.raises(ValueError, match="page ID exceeds the backend allocation"):
            executor.run_prefill(case.loaded.runtime_model, batch)
        assert backend.events == []
        # Rejection must leave the request available for a valid first dispatch.
        batch.block_ids_by_group[0][group_name] = [backend.num_pages - 1]
        executor.run_prefill(case.loaded.runtime_model, batch)
        assert executor._runner.position("request-a") == 2
        assert backend.events.count(("commit",)) == 1
    finally:
        executor.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"parallel_mode": "ep_only"},
        {"world_size": 2},
        {"supports_engram": False},
        {"transactional": False},
        {"max_chunk_tokens": 1},
        {"cache_format": "v4"},
    ],
)
def test_executor_rejects_incomplete_backend_contract_and_closes_backend(case, changes):
    executor, backend, _ = executor_case(case, capability_changes=changes)
    with pytest.raises(ValueError, match="contract|supported limits"):
        executor.register_model("tiny-v41", case.record)
    assert backend.closed


def test_executor_rejects_tokenizer_compressed_vocabulary_mismatch(case, monkeypatch):
    monkeypatch.setattr(case.modules.executor, "build_compressed_token_map", lambda value: ((0,) * 64, 1))
    executor, backend, _ = executor_case(case)
    with pytest.raises(ValueError, match="compressed|tokenizer"):
        executor.register_model("tiny-v41", case.record)
    assert backend.closed


def test_executor_rejects_scheduler_cache_layout_mismatch_before_factory(case):
    executor, _, calls = executor_case(case)
    bad = dataclasses.replace(case.record, runtime=dataclasses.replace(case.runtime, kv_cache_groups=()))
    with pytest.raises(ValueError, match="cache groups"):
        executor.register_model("tiny-v41", bad)
    assert calls == []


@pytest.mark.parametrize("chunk_limit,expected_chunk", [(None, 4), (2, 2)])
def test_programmatic_engine_resolves_same_cache_topology_as_directory_loader(
    case, monkeypatch, chunk_limit, expected_chunk
):
    engine = importlib.import_module("pypto_serving.serving.engine.async_engine")

    def no_worker(*args, **kwargs):
        pytest.fail("runtime configuration must not launch a worker")

    monkeypatch.setattr(engine, "spawn_worker", no_worker)
    runtime = dataclasses.replace(
        case.runtime,
        kv_cache_groups=(),
        max_prefill_tokens_per_request=chunk_limit,
        supports_chunked_prefill_with_speculation=True,
    )
    config = engine.EngineConfig(
        model_id="tiny-v41",
        model_dir=str(case.root),
        platform="a5",
        executor_cls="PyptoDeepSeekV41Executor",
        runtime_config=runtime,
        enable_prefix_cache=False,
        max_num_running_reqs=runtime.max_batch_size,
        max_num_scheduled_tokens=runtime.max_num_batched_tokens,
    )
    resolved = config.resolve_runtime_config()
    loaded = case.modules.registry.ModelLoader().load("tiny-v41", str(case.root), runtime_config=runtime)
    assert resolved == loaded.runtime_model.runtime
    assert {group.name for group in resolved.kv_cache_groups} == {"main_kv.0", "index_k.0", "swa.0"}
    assert resolved.max_prefill_tokens_per_request == expected_chunk
    assert resolved.supports_chunked_prefill_with_speculation is False
    assert runtime.kv_cache_groups == ()


def test_programmatic_engine_rejects_v41_prefix_cache(case):
    engine = importlib.import_module("pypto_serving.serving.engine.async_engine")
    config = engine.EngineConfig(
        model_dir=str(case.root),
        executor_cls="PyptoDeepSeekV41Executor",
        runtime_config=case.runtime,
        enable_prefix_cache=True,
    )
    with pytest.raises(ValueError, match="enable_prefix_cache=False"):
        config.resolve_runtime_config()


@pytest.mark.parametrize("field,value", [("max_num_running_reqs", 3), ("max_num_scheduled_tokens", 5)])
def test_programmatic_engine_rejects_scheduler_limits_above_backend_runtime(case, field, value):
    engine = importlib.import_module("pypto_serving.serving.engine.async_engine")
    options = {"max_num_running_reqs": 2, "max_num_scheduled_tokens": 4, field: value}
    config = engine.EngineConfig(
        model_dir=str(case.root),
        executor_cls="PyptoDeepSeekV41Executor",
        runtime_config=case.runtime,
        enable_prefix_cache=False,
        **options,
    )
    with pytest.raises(ValueError, match="exceeds runtime"):
        config.resolve_runtime_config()


def engine_config(case, monkeypatch, *extra):
    # Exercise the actual CLI builder without importing or launching the worker.
    engine_module = types.ModuleType("pypto_serving.serving.engine.async_engine")
    engine_module.EngineConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)
    args = case.modules.cli.build_parser().parse_args(
        [
            "--model",
            str(case.root),
            "--platform",
            "a5",
            "--v41-kernel-factory",
            "test_backend:factory",
            "--max-model-len",
            "8",
            "--max-num-batched-tokens",
            "4",
            "--max-num-seqs",
            "2",
            "--block-size",
            "4",
            *extra,
        ]
    )
    return case.modules.cli.build_serving_engine_config(args)


def test_cli_selects_v41_executor_tp_ep_and_matching_cache_groups(case, monkeypatch):
    result = engine_config(case, monkeypatch, "--devices", "0,1", "--tp", "2", "--ep", "2")
    assert result.executor_cls == "PyptoDeepSeekV41Executor"
    assert result.executor_kwargs["kernel_factory"] == "test_backend:factory"
    assert result.parallel_config.data_parallel_size == 1
    assert result.parallel_config.tensor_parallel_size == result.parallel_config.expert_parallel_size == 2
    assert result.enable_prefix_cache is False
    assert result.runtime_config.max_prefill_tokens_per_request == 4
    assert result.runtime_config.kv_cache_groups
    assert {group.name for group in result.runtime_config.kv_cache_groups} == {
        "main_kv.0",
        "index_k.0",
        "swa.0",
    }
    assert all(group.num_partitions == 1 for group in result.runtime_config.kv_cache_groups)
    assert case.modules.cli._executor_cls_for_model_family("deepseek_v4") == "PyptoDeepSeekV4Executor"


@pytest.mark.parametrize(
    "options,message",
    [
        (("--platform", "a2a3"), "platform a5"),
        (("--devices", "0,1", "--dp", "2", "--ep", "2"), "--dp 1"),
        (("--num-speculative-tokens", "1"), "only supported"),
    ],
)
def test_cli_rejects_unsupported_v41_modes(case, monkeypatch, options, message):
    with pytest.raises(ValueError, match=message):
        engine_config(case, monkeypatch, *options)
