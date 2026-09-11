# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual host state with a recording backend: lifecycle tests, not model inference goldens."""

import importlib
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    name = "_v41_runner_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "pypto_serving/model/deepseek_v41")]
    monkeypatch.setitem(sys.modules, name, package)
    result = {}
    for suffix in ("cache", "config", "engram", "npu_runner"):
        full = name + "." + suffix
        spec = importlib.util.find_spec(full)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
        result[suffix] = module
    return types.SimpleNamespace(**result)


class RecordingBackend:
    def __init__(self):
        self.events = []
        self.persisted = {}
        self.fail_layer = None
        self.fail_abort = False
        self.fail_commit = False
        self.closed = False

    def begin_batch(self, contexts):
        self.events.append(("begin", len(contexts)))
        return dict(self.persisted)

    def embed(self, ticket, context):
        self.persisted[context.work.request_id] = context.work.start_pos + len(context.work.token_ids)
        return context.work.token_ids

    def engram(self, ticket, layer, state, context):
        self.events.append(("engram", context.work.request_id, layer.layer_id, context.engram_hashes))
        return state

    def layer(self, ticket, layer, state, context):
        self.events.append(("layer", context.work.request_id, layer.layer_id))
        if layer.owns_index_results:
            context.cache.publish_index(layer.layer_id, (context.work.request_id, layer.layer_id))
        if layer.compress_ratio:
            assert context.cache.index_for(layer.layer_id)[0] == context.work.request_id
        if layer.layer_id == self.fail_layer:
            raise ArithmeticError("injected device failure")
        return state

    def head(self, ticket, state, context):
        return (context.work.request_id, context.work.start_pos, state[-1])

    def commit_batch(self, ticket):
        if self.fail_commit:
            raise ArithmeticError("injected commit failure")
        self.events.append(("commit",))

    def abort_batch(self, ticket):
        if self.fail_abort:
            raise RuntimeError("device cannot recover")
        self.persisted = ticket
        self.events.append(("abort",))

    def release_request(self, request_id, generation):
        self.persisted.pop(request_id, None)
        self.events.append(("release", request_id, generation))

    def close(self):
        self.persisted.clear()
        self.closed = True


@pytest.fixture
def setup(modules):
    config = modules.config.DeepSeekV41Config.from_json(ROOT / "tests/fixtures/deepseek_v41/config.json")
    cache = modules.cache.V41CacheState(config, page_size=4, max_seq_len=512, max_chunk_tokens=256)
    engram = modules.engram
    layout = engram.EngramLayout.from_config(config.text_config)

    def history():
        return engram.EngramHashState(
            layout,
            tuple(range(32)),
            compressed_vocab_size=32,
            pad_token_id=0,
            multipliers=((3, 5, 7, 9), (11, 13, 17, 19)),
        )

    backend = RecordingBackend()
    runner = modules.npu_runner.DeepSeekV41Runner(config, cache, backend, history, max_requests=2)

    def work(rid, start=0, tokens=(1, 2), slot=0, mode="prefill"):
        pages = {
            group.name: tuple(range(slot * 512, slot * 512 + group.max_blocks_per_seq))
            for group in cache.groups
        }
        return modules.npu_runner.V41WorkItem(rid, tokens, start, pages, mode=mode)

    return types.SimpleNamespace(runner=runner, backend=backend, cache=cache, work=work)


def execute(setup, *items):
    return setup.runner.execute(items, validate_outputs=lambda rows: rows)


def test_chunked_prefill_decode_and_all_layers(setup):
    a = execute(setup, setup.work("a"))
    assert a == [("a", 0, 2)]
    execute(setup, setup.work("a", 2, (3,)))
    setup.runner.finalize_prefill(["a"])
    execute(setup, setup.work("a", 3, (4,), mode="decode"))
    assert setup.runner.position("a") == 4
    assert setup.backend.persisted == {"a": 4}
    layers = [event[2] for event in setup.backend.events if event[0] == "layer"]
    assert layers == list(range(40)) * 3
    engrams = [event[2] for event in setup.backend.events if event[0] == "engram"]
    assert engrams == [1, 14] * 3


def test_batched_histories_are_request_local(setup):
    execute(setup, setup.work("a", tokens=(1, 2, 3)), setup.work("b", tokens=(7,), slot=1))
    execute(setup, setup.work("a", 3, (4,)), setup.work("b", 1, (4,), slot=1))
    records = [event for event in setup.backend.events if event[0] == "engram" and event[2] == 1]
    assert records[-2][3] != records[-1][3]
    assert [setup.runner.position(rid) for rid in ("a", "b")] == [4, 2]


@pytest.mark.parametrize("failure", ["layer", "commit", "output"])
def test_failed_batch_restores_existing_and_releases_new_request(setup, failure):
    execute(setup, setup.work("a"))
    if failure == "layer":
        setup.backend.fail_layer = 20
    if failure == "commit":
        setup.backend.fail_commit = True

    def validate(rows):
        if failure == "output":
            raise ArithmeticError("invalid logits")
        return rows

    with pytest.raises(ArithmeticError):
        setup.runner.execute((setup.work("a", 2), setup.work("b", slot=1)), validate_outputs=validate)
    assert setup.runner.position("a") == 2
    assert setup.backend.persisted == {"a": 2}
    assert setup.cache.active_request_count == 1
    setup.backend.fail_layer = None
    setup.backend.fail_commit = False
    execute(setup, setup.work("a", 2), setup.work("b", slot=1))
    assert setup.runner.position("b") == 2


def test_second_request_validation_failure_leaves_first_unchanged(setup):
    execute(setup, setup.work("a"))
    with pytest.raises(ValueError, match="another live request"):
        execute(setup, setup.work("a", 2), setup.work("b"))
    assert setup.runner.position("a") == 2
    assert setup.backend.persisted == {"a": 2}
    execute(setup, setup.work("a", 2))


@pytest.mark.parametrize("fail_before_begin", [True, False])
def test_failed_page_extension_is_reusable_by_another_request(setup, fail_before_begin):
    a = setup.work("a")
    first = {name: (0,) for name in a.block_ids_by_group}
    execute(setup, replace(a, block_ids_by_group=first))
    expanded = {name: (0, 100) for name in first}
    if not fail_before_begin:
        setup.backend.fail_layer = 20
    with pytest.raises((ValueError, ArithmeticError)):
        execute(setup, replace(a, start_pos=4 if fail_before_begin else 2, block_ids_by_group=expanded))
    setup.backend.fail_layer = None
    b = replace(setup.work("b"), block_ids_by_group={name: (100,) for name in first})
    execute(setup, b)
    assert setup.runner.position("a") == 2
    assert setup.runner.position("b") == 2


def test_generation_reuse_release_and_capacity(setup):
    execute(setup, setup.work("a"), setup.work("b", slot=1))
    with pytest.raises(ValueError, match="capacity"):
        execute(setup, setup.work("c", slot=2))
    setup.runner.release(["a", "a", "unknown"])
    execute(setup, setup.work("a"))
    setup.runner.release(["a"])
    generations = [event[2] for event in setup.backend.events if event[:2] == ("release", "a")]
    assert len(generations) == 2 and generations[0] < generations[1]


def test_decode_phase_and_contiguous_positions(setup):
    with pytest.raises(ValueError, match="unknown request"):
        execute(setup, setup.work("a", mode="decode", tokens=(1,)))
    execute(setup, setup.work("a"))
    with pytest.raises(ValueError, match="phase mismatch"):
        execute(setup, setup.work("a", 2, (1,), mode="decode"))
    with pytest.raises(ValueError, match="contiguously"):
        execute(setup, setup.work("a", 4))
    setup.runner.finalize_prefill(["a"])
    with pytest.raises(ValueError, match="exactly once"):
        setup.runner.finalize_prefill(["a"])
    with pytest.raises(ValueError, match="phase mismatch"):
        execute(setup, setup.work("a", 2))


def test_abort_failure_poisoned_runner_still_closes_backend(setup):
    setup.backend.fail_layer = 0
    setup.backend.fail_abort = True
    with pytest.raises(RuntimeError, match="unusable"):
        execute(setup, setup.work("a"))
    with pytest.raises(RuntimeError, match="recreate"):
        execute(setup, setup.work("b", slot=1))
    setup.runner.close()
    assert setup.backend.closed


def test_empty_duplicate_and_oversize_chunks_do_not_reach_backend(setup):
    for items in ((), (setup.work("a"), setup.work("a")), (setup.work("a", tokens=(1,) * 257),)):
        with pytest.raises(ValueError):
            execute(setup, *items)
    assert not [e for e in setup.backend.events if e[0] == "begin"]
    assert setup.cache.active_request_count == 0
