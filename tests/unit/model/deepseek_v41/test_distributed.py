# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rank transport and mirrored transaction tests; CPU/Gloo is not A5 acceptance.

The miniature rank backend isolates IPC/collectives and transaction ordering from
model arithmetic, which is covered by test_backend.py and per-operation goldens.
No production factory can select this test backend.
"""

from __future__ import annotations

import importlib
import json
import pickle
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="rank IPC and collective tests require CPU Torch")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    before = {name: value for name, value in sys.modules.items()
              if name == "pypto_serving" or name.startswith("pypto_serving.")}
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    package = ModuleType("pypto_serving")
    package.__path__ = [str(ROOT / "pypto_serving")]
    monkeypatch.setitem(sys.modules, "pypto_serving", package)
    try:
        yield SimpleNamespace(**{suffix: importlib.import_module("pypto_serving.model.deepseek_v41." + suffix)
                                 for suffix in ("config", "cache", "npu_runner", "distributed")})
    finally:
        for name in tuple(sys.modules):
            if name == "pypto_serving" or name.startswith("pypto_serving."):
                sys.modules.pop(name, None)
        sys.modules.update(before)


@pytest.fixture
def settings(modules, tmp_path):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32, vocab_size=64, num_hidden_layers=2, num_attention_heads=2,
        head_dim=32, qk_rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=2,
        moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
        index_n_heads=2, index_head_dim=32, index_topk=2, sliding_window=4,
        max_position_embeddings=64, compress_ratios=[2, 2], kv_source_layer_ids=[0],
        index_source_layer_ids=[0], candidate_source_layer_id=-1,
        candidate_topk_blocks=0, candidate_block_size=0, engram_layer_ids=[],
        engram_num_embeddings=[], num_nextn_predict_layers=0, dspark_target_layer_ids=[],
    )
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = modules.config.DeepSeekV41Config.from_dict(raw)
    cache = modules.cache.V41CacheState(config, page_size=2, max_seq_len=32, max_chunk_tokens=4)
    runtime = SimpleNamespace(page_size=2, max_seq_len=32, max_batch_size=2,
                              max_prefill_tokens_per_request=4, max_num_batched_tokens=8,
                              total_kv_pages=32, num_speculative_tokens=0)
    return SimpleNamespace(config=config, cache=cache, runtime=runtime, path=tmp_path)


class ArithmeticRank:
    """Small real collective arithmetic used only to test process protocol behavior."""

    def __init__(self, config, rank, world_size, *, collectives=False):
        self.config, self.rank, self.world_size = config, rank, world_size
        self.collectives = collectives
        self.num_pages = 32
        self.capabilities = SimpleNamespace(world_size=world_size, platform="cpu-test")
        self.values = {}
        self.fail = False

    def begin_batch(self, contexts):
        return {key: value.clone() for key, value in self.values.items()}

    def embed(self, ticket, context):
        token = context.work.token_ids[-1]
        if token == 63:
            threading.Event().wait(30)  # Parent deadline must kill this task-owned rank.
        if self.fail or token == 62:
            raise ValueError("rank arithmetic injected failure")
        value = torch.tensor(context.work.token_ids, dtype=torch.float32).sum().reshape(1, 1)
        return value @ torch.tensor([[self.rank + 1.0]])

    def engram(self, ticket, layer, value, context):
        return value

    def layer(self, ticket, layer, value, context):
        if self.collectives:
            from pypto_serving.model.deepseek_v41.distributed import TorchCollective
            value = TorchCollective().all_reduce(value) / self.world_size
        return value + layer.layer_id + 1

    def head(self, ticket, value, context):
        key = context.work.request_id, context.generation
        value = value.flatten() + self.values.get(key, torch.zeros(1))
        self.values[key] = value.clone()
        local = value + torch.tensor([self.rank * 2.0, self.rank * 2.0 + 1])
        if self.collectives:
            from pypto_serving.model.deepseek_v41.distributed import TorchCollective
            return TorchCollective().all_gather(local)
        return local

    def commit_batch(self, ticket):
        pass

    def abort_batch(self, ticket):
        self.values = ticket

    def checkpoint(self):
        return {key: value.clone() for key, value in self.values.items()}

    def finish_checkpoint(self, checkpoint, *, restore=False):
        if restore:
            self.values = checkpoint

    def release_request(self, request_id, generation):
        self.values.pop((request_id, generation), None)

    def propose(self, request_id, generation, anchor):
        value = self.values[request_id, generation]
        return SimpleNamespace(output_ids=torch.tensor([anchor, anchor + 1]),
                               logits=torch.cat((value, value + 1)).reshape(1, 2),
                               confidence=torch.tensor([.5]))

    def close(self):
        self.values.clear()

    def diagnostics(self):
        return {"rank": self.rank, "requests": len(self.values)}


def cpu_rank_factory(settings, rank, world_size):
    """Top-level spawn-pickleable factory, never selected by production callers."""
    from pypto_serving.model.deepseek_v41.config import DeepSeekV41Config
    config = DeepSeekV41Config.from_json(Path(settings["model_dir"]) / "config.json")
    return ArithmeticRank(config, rank, world_size, collectives=True)


def record(settings, *, rid="a", generation=1, start=0, tokens=(1, 2), slot=0):
    pages = {group.name: tuple(range(slot * group.max_blocks_per_seq,
                                    (slot + 1) * group.max_blocks_per_seq)) for group in settings.cache.groups}
    return {"request_id": rid, "generation": generation, "start_pos": start, "end_pos": start + len(tokens),
            "token_ids": tuple(tokens), "pages": pages, "partition": 0, "mode": "prefill",
            "engram_hashes": (), "multimodal": None}


def context(modules, settings, **kwargs):
    item = record(settings, **kwargs)
    # Only the serializer reads end_pos here; real RankSession creates its own
    # transaction using the scheduler page tables and generation in the record.
    work = modules.npu_runner.V41WorkItem(item["request_id"], item["token_ids"], item["start_pos"], item["pages"])
    return modules.npu_runner.V41ExecutionContext(work, item["generation"],
                                                 SimpleNamespace(end_pos=item["end_pos"]), ())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int64, torch.bool])
def test_tensor_ipc_owns_bytes_and_handles_noncontiguous_and_empty(modules, dtype):
    codec = modules.distributed
    original = torch.arange(12).reshape(3, 4).to(dtype).T
    payload = codec._pack_tensor(original)
    actual = codec._unpack_tensor(payload)
    assert actual.dtype == dtype and torch.equal(actual, original)
    original.zero_()
    assert torch.equal(actual, codec._unpack_tensor(payload))
    actual.zero_()
    assert not torch.equal(actual, codec._unpack_tensor(payload))
    assert codec._unpack_tensor(codec._pack_tensor(torch.empty((0, 3), dtype=dtype))).shape == (0, 3)


def test_tensor_ipc_rejects_corrupt_size_and_unbounded_budget(modules, monkeypatch):
    codec = modules.distributed
    payload = codec._pack_tensor(torch.ones(3))
    with pytest.raises(ValueError, match="byte count"):
        codec._unpack_tensor({**payload, "shape": (4,)})
    with pytest.raises(ValueError, match="shape"):
        codec._unpack_tensor({**payload, "shape": (-1,)})
    monkeypatch.setattr(codec, "_MAX_TENSOR_BYTES", 8)
    with pytest.raises(ValueError, match="budget"):
        codec._pack_tensor(torch.ones(3))
    with pytest.raises(ValueError, match="byte count"):
        codec._unpack_tensor(payload)


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan"])
def test_rpc_deadline_must_be_finite_and_positive(modules, monkeypatch, value):
    monkeypatch.setenv("PYPTO_V41_RPC_TIMEOUT", value)
    with pytest.raises(ValueError, match="finite positive"):
        modules.distributed._timeout("PYPTO_V41_RPC_TIMEOUT", 600)


def test_context_serialization_has_plain_owned_page_tables(modules, settings):
    original = context(modules, settings)
    encoded = modules.distributed.serialize_context(original)
    assert pickle.loads(pickle.dumps(encoded)) == encoded
    original.work.block_ids_by_group.clear()
    assert encoded["pages"] and encoded["generation"] == 1


def test_rank_session_commit_abort_ownership_and_generation(modules, settings):
    rank = ArithmeticRank(settings.config, 0, 1)
    session = modules.distributed.RankSession(rank, settings.cache, 0)
    first = record(settings)
    output = session.begin((first,))
    lease = session.leases["a", 1]
    assert settings.cache.valid_length(lease) == 0
    expected = modules.distributed._unpack_tensor(output[0])
    session.commit()
    assert settings.cache.valid_length(lease) == 2
    before = rank.values["a", 1].clone()
    session.begin((record(settings, start=2, tokens=(3,)),))
    session.abort()
    session.abort()  # Recovery is idempotent before another batch is created.
    assert settings.cache.valid_length(lease) == 2 and torch.equal(rank.values["a", 1], before)
    # A late failure must unwind the earlier request and release a newly created lease.
    bad = record(settings, rid="b", generation=2, tokens=(62,), slot=1)
    with pytest.raises(ValueError, match="injected failure"):
        session.begin((record(settings, start=2, tokens=(4,)), bad))
    assert session.pending is None and set(session.leases) == {("a", 1)}
    assert torch.equal(rank.values["a", 1], before)
    session.release(("a", 1))
    output = session.begin((record(settings, generation=3),))
    assert torch.equal(modules.distributed._unpack_tensor(output[0]), expected)
    session.commit()
    with pytest.raises(ValueError, match="already registered"):
        session.begin((record(settings, generation=1),))


def test_rank_checkpoint_restores_ring_visibility_and_new_requests(modules, settings):
    rank = ArithmeticRank(settings.config, 0, 1)
    session = modules.distributed.RankSession(rank, settings.cache, 0)
    session.begin((record(settings),))
    session.commit()
    before = rank.values["a", 1].clone()
    session.checkpoint(7)
    for start in (2, 6, 10):
        session.begin((record(settings, start=start, tokens=(2, 3, 4, 5)),))
        session.commit()
    session.begin((record(settings, rid="b", generation=2, slot=1),))
    session.commit()
    lease = session.leases["a", 1]
    assert settings.cache.snapshot_bindings(lease).retained_start > 0
    session.finish_checkpoint(7, True)
    assert set(session.leases) == {("a", 1)}
    assert settings.cache.valid_length(lease) == 2
    assert settings.cache.snapshot_bindings(lease).retained_start == 0
    assert torch.equal(rank.values["a", 1], before)
    session.begin((record(settings, start=2, tokens=(3,)),))
    session.commit()


@pytest.fixture
def distributed(modules, settings, monkeypatch):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the explicit CPU rank-transport test")
    monkeypatch.setenv("PYPTO_V41_INIT_TIMEOUT", "45")
    monkeypatch.setenv("PYPTO_V41_RPC_TIMEOUT", "15")
    backend = modules.distributed.DistributedV41Backend(
        config=settings.config, runtime=settings.runtime, cache_layouts=settings.cache.groups,
        weight_loader=SimpleNamespace(model_dir=settings.path, max_load_bytes=1 << 20), device_ids=[0, 1],
        _rank_factory=cpu_rank_factory, _collective_backend="gloo",
    )
    try:
        yield backend
    finally:
        backend.close()


def test_two_gloo_ranks_execute_collectives_commit_abort_checkpoint_and_release(distributed, modules, settings):
    backend = distributed
    assert backend.diagnostics() == {"ranks": [{"rank": 0, "requests": 0}, {"rank": 1, "requests": 0}]}
    first = context(modules, settings)
    ticket = backend.begin_batch((first,))
    # Average of rank-local sums is 4.5, then layers add 1 and 2.
    expected = torch.tensor([7.5, 8.5, 9.5, 10.5])
    torch.testing.assert_close(backend.head(ticket, backend.embed(ticket, first), first), expected)
    with pytest.raises(ValueError, match="different request"):
        backend.head(ticket, 1, first)
    backend.commit_batch(ticket)
    checkpoint = backend.checkpoint()
    for start in (2, 6, 10):
        later = context(modules, settings, start=start, tokens=(2, 3, 4, 5))
        ticket = backend.begin_batch((later,))
        backend.commit_batch(ticket)
    backend.finish_checkpoint(checkpoint, restore=True)
    later = context(modules, settings, start=2, tokens=(3,))
    ticket = backend.begin_batch((later,))
    actual = backend.head(ticket, backend.embed(ticket, later), later)
    torch.testing.assert_close(actual, expected + 7.5)
    backend.abort_batch(ticket)
    ticket = backend.begin_batch((later,))
    torch.testing.assert_close(backend.head(ticket, backend.embed(ticket, later), later), actual)
    backend.commit_batch(ticket)
    proposal = backend.propose("a", 1, 4)
    assert proposal.output_ids.tolist() == [4, 5]
    assert proposal.logits.device.type == "cpu"
    backend.release_request("a", 1)
    again = context(modules, settings, generation=3)
    ticket = backend.begin_batch((again,))
    torch.testing.assert_close(backend.head(ticket, backend.embed(ticket, again), again), expected)
    backend.commit_batch(ticket)
    processes = tuple(backend._processes)
    backend.close()
    assert all(not process.is_alive() for process in processes)


@pytest.mark.parametrize("token,exception,pattern", [(62, RuntimeError, "injected failure"),
                                                    (63, TimeoutError, "timed out")])
def test_failed_or_unresponsive_rank_poison_group_and_bound_cleanup(distributed, modules, settings,
                                                                  token, exception, pattern):
    backend = distributed
    backend._rpc_timeout = .5
    started = time.monotonic()
    with pytest.raises(exception, match=pattern):
        backend.begin_batch((context(modules, settings, tokens=(token,)),))
    assert time.monotonic() - started < 8
    assert all(not process.is_alive() for process in backend._processes)
    with pytest.raises(RuntimeError, match="closed"):
        backend.begin_batch((context(modules, settings),))


@pytest.mark.parametrize("ids", [[], [0], [0, 0], [0, -1], [0, True], list(range(17))])
def test_device_validation_precedes_any_spawn(modules, settings, ids):
    with pytest.raises(ValueError, match="2..16 distinct"):
        modules.distributed.DistributedV41Backend(
            config=settings.config, runtime=settings.runtime, cache_layouts=settings.cache.groups,
            weight_loader=None, device_ids=ids,
        )
