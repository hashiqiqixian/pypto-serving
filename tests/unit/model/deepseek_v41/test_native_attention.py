# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native adapter transport/lifetime tests using real scheduler transactions.

An in-memory worker substitutes only device allocation and dispatch. These tests
validate the host ABI, ownership and recovery, not PyPTO attention mathematics;
the library's native harness supplies numerical and hardware acceptance.
"""

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
if not hasattr(torch, "float8_e8m0fnu"):
    pytest.skip("native cache ABI requires the UE8M0 Torch dtype", allow_module_level=True)
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    before = {name: value for name, value in sys.modules.items()
              if name == "pypto_serving" or name.startswith("pypto_serving.")}
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    for name in ("pypto_serving", "pypto_serving.model", "pypto_serving.model.deepseek_v41"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    try:
        yield SimpleNamespace(**{suffix: importlib.import_module("pypto_serving.model.deepseek_v41." + suffix)
                                for suffix in ("config", "cache", "backend", "attention", "native_attention")})
    finally:
        for name in tuple(sys.modules):
            if name == "pypto_serving" or name.startswith("pypto_serving."):
                sys.modules.pop(name, None)
        sys.modules.update(before)


class MemoryWorker:
    """Byte-preserving stand-in for ChipWorker memory; addresses are real CPU pointers."""

    def __init__(self):
        self.storage = {}

    def alloc_tensor(self, shape, dtype, *, init=None):
        tensor = torch.empty(shape, dtype=dtype) if init is None else init.clone()
        pointer = tensor.data_ptr()
        self.storage[pointer] = tensor
        return SimpleNamespace(data_ptr=pointer, shape=tuple(shape), dtype=dtype,
                               nbytes=tensor.numel() * tensor.element_size())

    def free_tensor(self, tensor):
        del self.storage[tensor.data_ptr]

    def zero(self, tensor):
        self.storage[tensor.data_ptr].view(torch.uint8).zero_()

    def read(self, tensor, offset, size):
        return self.storage[tensor.data_ptr].view(torch.uint8).flatten()[offset:offset + size].clone()

    def write(self, tensor, offset, host):
        self.storage[tensor.data_ptr].view(torch.uint8).flatten()[offset:offset + host.numel()].copy_(host.flatten())

    def bytes(self, tensor):
        return self.storage[tensor.data_ptr].view(torch.uint8).reshape(tensor.shape[0] * 128, -1)

    def close(self):
        self.storage.clear()

    def clear(self):
        pass


@pytest.fixture
def setup(modules):
    raw = json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())
    raw["text_config"].update(
        hidden_size=32, num_hidden_layers=6, num_nextn_predict_layers=0,
        num_attention_heads=2, head_dim=32, qk_rope_head_dim=16, q_lora_rank=32,
        o_lora_rank=32, o_groups=1, sliding_window=128, max_position_embeddings=1024,
        index_n_heads=2, index_head_dim=32, index_topk=2,
        compress_ratios=[0, 2, 2, 1, 1, 1], kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4], candidate_source_layer_id=3,
        candidate_topk_blocks=2, candidate_block_size=2,
        engram_layer_ids=[], engram_num_embeddings=[], dspark_target_layer_ids=[],
    )
    config = modules.config.DeepSeekV41Config.from_dict(raw)
    cache = modules.cache.V41CacheState(config, page_size=256, max_seq_len=1024, max_chunk_tokens=256)
    runtime = SimpleNamespace(max_seq_len=1024, max_batch_size=2, total_kv_pages=8,
                              max_prefill_tokens_per_request=256, max_num_batched_tokens=512,
                              num_speculative_tokens=0)
    weights = SimpleNamespace(close=lambda: None)
    reductions = []
    def reduce(value):
        reductions.append(value.clone())
        return value
    ops = SimpleNamespace(device=torch.device("cpu"), world_size=1, rank=0, weights=weights,
                          matmul_provider=None, all_reduce=reduce)
    backend = modules.backend.DeepSeekV41Backend(config, runtime, cache.groups, ops)
    worker = MemoryWorker()
    modes = ("swa", "c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse")
    common = ("x", "window_cache", "window_cache_scale", "window_slots", "window_indices")
    args = {}
    calls = []
    fail = {"enabled": False}
    for mode in modes:
        names = list(common)
        if mode != "swa":
            names += ["compressed_cache", "compressed_cache_scale"]
        if mode.endswith("full"):
            names += ["compressed_slots", "index_cache", "index_cache_scale", "index_block_table", "topk_indices"]
        elif mode.endswith("reuse"):
            names += ["compressed_indices"]
        else:
            if mode == "c1a_reindex":
                names += ["index_block_table", "candidate_mask", "topk_indices"]
        if mode == "c2a_full":
            names += ["compressor_state", "position_ids"]
        if mode == "c1a_full":
            names += ["candidate_mask"]
        names += ["partial" if mode.startswith("c2a") else "output", "num_tokens"]
        args[mode] = tuple(names)

    def dispatch(mode, *values):
        entry = dict(zip(args[mode], values))
        calls.append((mode, entry))
        for prefix, slots in (("window", entry["window_slots"]),
                              ("compressed", entry.get("compressed_slots", ())),
                              ("index", entry.get("compressed_slots", ()))):
            if not len(slots):
                continue
            for row in slots:
                if int(row) >= 0:
                    worker.bytes(entry[prefix + "_cache"])[int(row)].fill_(29)
                    worker.bytes(entry[prefix + "_cache_scale"])[int(row)].fill_(91)
        if "compressor_state" in entry:
            worker.storage[entry["compressor_state"].data_ptr][0].add_(3)
        if fail["enabled"]:
            raise RuntimeError("injected device failure after cache writes")
        if "topk_indices" in entry:
            entry["topk_indices"][:, 0] = int(entry["index_block_table"][0, 0]) * 128
        if mode == "c1a_full":
            entry["candidate_mask"][:, 0] = 1
        output = "partial" if mode.startswith("c2a") else "output"
        entry[output].copy_(entry["x"].float() + 0.125)

    provider = modules.native_attention.NativeAttention(
        backend, worker=worker, kernels={mode: object() for mode in modes}, arguments=args,
        run_config=None, pack_weights=lambda *args: {}, weight_budget_bytes=1024, cache_io=worker)
    provider.handles = {mode: (lambda *values, mode=mode: dispatch(mode, *values)) for mode in modes}
    backend.attention_provider = provider
    yield SimpleNamespace(config=config, cache=cache, backend=backend, worker=worker, provider=provider,
                          calls=calls, fail=fail, reductions=reductions, modules=modules)
    backend.close()


def prepare(setup, lease, start, end, *, request_slot=0):
    tables = {}
    for group in setup.cache.groups:
        needed = min((end + 255) // 256, group.max_blocks_per_seq)
        # Index pages use a different order from main pages in the same slot.
        pages = list(range(request_slot * group.max_blocks_per_seq,
                           (request_slot + 1) * group.max_blocks_per_seq))
        if group.kind == "index_k":
            pages.reverse()
        tables[group.name] = tuple(pages[:needed])
    setup.cache.bind(lease, tables)
    tx = setup.cache.prepare(lease, start, end)
    context = SimpleNamespace(cache=tx, generation=lease.generation,
                              work=SimpleNamespace(request_id=lease.request_id, start_pos=start))
    ticket = setup.backend.begin_batch((context,))
    return tx, context, ticket


def run_layers(setup, context):
    states = setup.backend.requests[(context.work.request_id, context.generation)].layers
    count = context.cache.end_pos - context.cache.start_pos
    x = torch.arange(count * 32, dtype=torch.float32).reshape(count, 32).remainder(23).to(torch.bfloat16)
    for layer in range(6):
        result = setup.provider.forward(layer, x, context, states)
        torch.testing.assert_close(result, (x.float() + 0.125).to(torch.bfloat16), rtol=0, atol=0)
    setup.backend.requests[(context.work.request_id, context.generation)].position = context.cache.end_pos


def commit(setup, tx, ticket):
    setup.backend.commit_batch(ticket)
    setup.cache.commit(tx)


def test_real_transactions_reach_all_dispatch_modes_with_distinct_index_pages(setup):
    lease = setup.cache.create("first", generation=7)
    tx, context, ticket = prepare(setup, lease, 0, 3)
    assert tx.map_position("main_kv.3", 0).page_id != tx.map_position("index_k.3", 0).page_id
    run_layers(setup, context)
    assert [mode for mode, _ in setup.calls] == ["swa", "c2a_full", "c2a_reuse", "c1a_full",
                                               "c1a_reindex", "c1a_reuse"]
    for mode, args in setup.calls:
        if mode.endswith("full"):
            assert args["index_block_table"][0, 0].item() == 0
        assert args["window_indices"][0, 1:].eq(-1).all()
    assert all(value.dtype == torch.float32 for value in setup.reductions)
    assert tx.index_for(5) is setup.calls[4][1]["topk_indices"]
    assert tx.candidates_for(4) is setup.calls[3][1]["candidate_mask"]
    commit(setup, tx, ticket)


def test_failure_restores_payload_scales_and_partial_state_at_same_addresses(setup):
    lease = setup.cache.create("request", generation=2)
    tx, context, ticket = prepare(setup, lease, 0, 3)
    run_layers(setup, context)
    commit(setup, tx, ticket)
    before = {pointer: tensor.clone().view(torch.uint8) for pointer, tensor in setup.worker.storage.items()}
    tx, context, ticket = prepare(setup, lease, 3, 4)
    setup.fail["enabled"] = True
    states = setup.backend.requests[("request", 2)].layers
    with pytest.raises(RuntimeError, match="injected device failure"):
        setup.provider.forward(1, torch.ones((1, 32), dtype=torch.bfloat16), context, states)
    setup.backend.abort_batch(ticket)
    setup.cache.abort(tx)
    assert set(setup.worker.storage) == set(before)
    for pointer, old in before.items():
        assert torch.equal(setup.worker.storage[pointer].view(torch.uint8), old)
    assert states[1].position == 3


def test_checkpoint_and_generation_reuse_restore_visibility(setup):
    first = setup.cache.create("same-id", generation=1)
    tx, context, ticket = prepare(setup, first, 0, 3)
    run_layers(setup, context)
    commit(setup, tx, ticket)
    snapshot = setup.backend.checkpoint()
    before = {pointer: tensor.clone().view(torch.uint8) for pointer, tensor in setup.worker.storage.items()}
    tx, context, ticket = prepare(setup, first, 3, 5)
    run_layers(setup, context)
    setup.backend.commit_batch(ticket)
    setup.cache.abort(tx)
    setup.backend.finish_checkpoint(snapshot, restore=True)
    for pointer, old in before.items():
        assert torch.equal(setup.worker.storage[pointer].view(torch.uint8), old)
    setup.backend.release_request("same-id", 1)
    setup.cache.release(first)
    second = setup.cache.create("same-id", generation=2)
    tx, context, ticket = prepare(setup, second, 0, 2, request_slot=1)
    run_layers(setup, context)
    full = setup.calls[-3][1]
    assert full["compressed_slots"].tolist() == [1024, 1025]
    assert full["index_block_table"][0, 0].item() == 8
    commit(setup, tx, ticket)
    with pytest.raises(RuntimeError, match="active cache lease"):
        setup.provider.forward(0, torch.ones((1, 32)), SimpleNamespace(
            cache=SimpleNamespace(lease=first), generation=1, work=SimpleNamespace(request_id="same-id")), [])


def test_page_boundary_and_concurrent_request_keep_distinct_physical_rows(setup):
    first = setup.cache.create("first", generation=1)
    tx, context, ticket = prepare(setup, first, 0, 256)
    run_layers(setup, context)
    assert setup.calls[3][1]["index_block_table"].tolist() == [[0, 1]]
    assert setup.calls[0][1]["window_indices"][-1].tolist() == list(range(128, 256))
    commit(setup, tx, ticket)
    first_main = setup.worker.bytes(setup.provider.buffers["main_kv.3"][0])[:256].clone()
    second = setup.cache.create("second", generation=1)
    tx, context, ticket = prepare(setup, second, 0, 2, request_slot=1)
    run_layers(setup, context)
    assert setup.calls[-3][1]["compressed_slots"].tolist() == [1024, 1025]
    commit(setup, tx, ticket)
    assert torch.equal(setup.worker.bytes(setup.provider.buffers["main_kv.3"][0])[:256], first_main)
    tx, context, ticket = prepare(setup, first, 256, 257)
    run_layers(setup, context)
    assert setup.calls[-3][1]["index_block_table"].tolist() == [[0, 1, 2]]
    assert setup.calls[-6][1]["window_indices"][0].tolist() == list(range(129, 257))
    assert setup.calls[-5][1]["compressed_slots"].tolist() == [-1]
    commit(setup, tx, ticket)


def test_failed_worker_close_can_be_retried_without_discarding_ownership(setup, monkeypatch):
    lease = setup.cache.create("close", generation=1)
    tx, context, ticket = prepare(setup, lease, 0, 2)
    run_layers(setup, context)
    commit(setup, tx, ticket)
    calls = []
    close = setup.worker.close
    def close_once():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("retryable worker cleanup")
        close()
    monkeypatch.setattr(setup.worker, "close", close_once)
    with pytest.raises(RuntimeError, match="retryable worker cleanup"):
        setup.provider.close()
    assert setup.provider.buffers
    assert setup.worker.storage
    setup.provider.close()
    assert len(calls) == 2
    assert not setup.provider.buffers
    assert not setup.worker.storage


@pytest.mark.parametrize("platform,devices", [("a2a3", [0]), ("a5", [0, 1])])
def test_factory_rejects_unsupported_topology_before_loading_library(modules, platform, devices, monkeypatch):
    def forbidden():
        raise AssertionError("unsupported topology reached library initialization")
    monkeypatch.setattr(modules.native_attention, "_load_library", forbidden)
    with pytest.raises(ValueError, match="A5 TP1"):
        modules.native_attention.create_backend(config=None, runtime=None, cache_layouts=(),
                                                weight_loader=None, device_ids=devices, platform=platform)


@pytest.mark.parametrize("mode", ["zero", "read", "write"])
def test_native_cache_transfer_codegen(modules, tmp_path, mode):
    pl = pytest.importorskip("pypto.language", reason="actual PyPTO frontend is required for codegen")
    if not hasattr(pl, "jit"):
        pytest.skip("actual PyPTO frontend is unavailable")
    from pypto.runtime import RunConfig
    library = modules.native_attention._load_library()
    kernel = library.load_local_cache_kernels()[mode]
    values = {"size": pl.RUNTIME}
    if mode != "zero":
        values["offset"] = pl.RUNTIME
    platform = os.environ.get("PYPTO_V41_NPU_PLATFORM", "a2a3")
    compiled = kernel.compile(config=RunConfig(platform=platform, codegen_only=True, save_kernels=True,
                                               save_kernels_dir=str(tmp_path / mode)), **values)
    assert Path(compiled.output_dir).is_dir()


@pytest.mark.skipif(os.environ.get("PYPTO_V41_NPU_TESTS") != "1", reason="allocated NPU opt-in required")
def test_real_npu_cache_transfer_preserves_owner_and_nonzero_offset_bytes(modules, tmp_path):
    """Exercise real ChipWorker Buffer identity and offset transfers, not a memory substitute."""
    import pypto.language as pl  # noqa: F401 - an opted-in missing frontend is an error
    from pypto.runtime import ChipWorker, RunConfig
    device = os.environ.get("TASK_DEVICE", "")
    if not device.isdecimal():
        pytest.fail("TASK_DEVICE must be an allocated nonnegative device index")
    platform = os.environ.get("PYPTO_V41_NPU_PLATFORM", "a2a3")
    config = RunConfig(platform=platform, device_id=int(device), save_kernels=True)
    library = modules.native_attention._load_library()
    worker = ChipWorker(config=config)
    io = modules.native_attention.NativeCacheIO(worker, library.load_local_cache_kernels(), config,
                                                SimpleNamespace(name=str(tmp_path)))
    try:
        pool = worker.alloc_tensor((3, 4096), torch.float32)
        alias = io._view(pool)
        assert alias.data_ptr == pool.data_ptr
        assert alias.buffer is pool.buffer
        io.zero(pool)
        values = torch.arange(1031, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(1, -1)
        io.write(pool, 16376, values)
        actual = io.read(pool, 16000, 2048)
        expected = torch.zeros((1, 2048), dtype=torch.uint8)
        expected[:, 376:376 + 1031] = values
        assert torch.equal(actual, expected)
        io.write(pool, 16376, torch.zeros_like(values))
        assert torch.count_nonzero(io.read(pool, 16000, 2048)) == 0
        worker.free_tensor(pool)
    finally:
        worker.close()
