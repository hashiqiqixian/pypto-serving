# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for the complete V4.1 decode dispatch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from pypto.runtime import DeviceTensor, StackedDeviceTensor
from pypto_serving.model.deepseek_v41.npu_runner import (
    DecodeForwardSegment, DecodeFullCachePools, LayerState, SegmentTopology,
)
from pypto_serving.model.deepseek_v41.weight_loader import ResidentFullDecodeWeights
from pypto_serving.model.deepseek_v41.weight_loader import _ALL


def test_selected_lib_full_decode_abi_is_fully_bound():
    source = Path(__file__).resolve().parents[4] / "pypto-lib/models/deepseek_v4_1_flash/decode_fwd.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    entry = next(node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == "l3_decode_fwd")
    parameters = {argument.arg for argument in entry.args.args}
    metadata = set(DecodeForwardSegment._METADATA) | {
        "window_slots", "window_indices", "c2a_compressed_lens", "c1a_compressed_lens",
        "c2a_compressed_slots", "c1a_compressed_slots", "moe_num_tokens",
    }
    caches = {
        "window_cache_pool", "window_cache_scale_pool", "compressed_cache_pool",
        "compressed_cache_scale_pool", "index_cache_pool", "index_cache_scale_pool",
        "state_cache_pool",
    }
    assert parameters == _ALL | metadata | caches | {
        "x_hc", "pre_mix", "attention_num_tokens"}


def test_weight_views_alias_one_full_rank_allocation():
    class Worker:
        def __init__(self):
            self.next_ptr = 1000
            self.copies = []
            self.freed = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.next_ptr += 1000
            return DeviceTensor(self.next_ptr, shape, dtype)

        def copy_to(self, base, source, nbytes, *, dst_offset, worker_id):
            self.copies.append((base, dst_offset, nbytes, worker_id))

        def view_tensor(self, parent, shape, *, byte_offset, worker_id):
            assert parent.shape[0] == 4
            assert worker_id in (0, 1)
            return DeviceTensor(parent.data_ptr, shape, parent.dtype, byte_offset=byte_offset)

        def free_tensor(self, tensor, *, worker_id):
            self.freed.append((tensor.data_ptr, worker_id))

        def free_stacked_tensor(self, tensor):
            for worker_id, shard in enumerate(tensor.shards):
                self.free_tensor(shard, worker_id=worker_id)

    host = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3).share_memory_()
    bank = SimpleNamespace(
        tensors={"hc_attn_fn": host},
        layout=SimpleNamespace(world=2, slots=lambda name: 2),
        finish=lambda names: None,
    )
    worker = Worker()
    resident = ResidentFullDecodeWeights(worker, bank, ("hc_attn_fn",))
    full = resident.full("hc_attn_fn")
    second = resident.view("hc_attn_fn", 1)
    assert full.shape == (2, 4, 3)
    assert second.shape == (2, 2, 3)
    assert second.shards[0].data_ptr == full.shards[0].data_ptr
    assert second.shards[1].data_ptr == full.shards[1].data_ptr
    assert second.shards[0].byte_offset == second.shards[1].byte_offset == 2 * 3 * 4
    assert len(worker.copies) == 4
    resident.close()
    assert len(worker.freed) == 2


def test_full_prefill_cache_views_borrow_decode_pools_without_copy():
    class Worker:
        def view_tensor(self, parent, shape, *, byte_offset, worker_id):
            assert byte_offset == 0
            return DeviceTensor(parent.data_ptr, shape, parent.dtype)

    pool = DecodeFullCachePools.__new__(DecodeFullCachePools)
    pool.worker = Worker()
    pool.plan = SimpleNamespace(
        family_groups=SimpleNamespace(window="window", c2a="c2a", c1a="c1a"),
        physical_group_blocks={"window": 2, "c2a": 2, "c1a": 2},
        physical_request_slots=2,
    )
    sources = {
        "window_cache_pool": 40, "window_cache_scale_pool": 40,
        "compressed_cache_pool": 4, "compressed_cache_scale_pool": 4,
        "index_cache_pool": 8, "index_cache_scale_pool": 8,
        "state_cache_pool": 3,
    }
    pool.source_counts = sources
    pool._full = {}
    for offset, (name, count) in enumerate(sources.items()):
        shape = (count * 2, 128, 1, 4)
        shards = tuple(DeviceTensor(10000 + offset * 100 + rank, shape, torch.uint8)
                       for rank in range(2))
        pool._full[name] = StackedDeviceTensor(shards, (2, *shape), (0, 1))
    views = pool.prefill_full_views()
    assert views["window_cache"].shape == (2, 40, 2, 128, 1, 4)
    assert views["index_cache"].shape == (2, 4, 2, 128, 1, 4)
    assert views["state_cache"].shape == (2, 3, 2, 128, 1, 4)
    assert all(views[target].shards[rank].data_ptr == pool.full(source).shards[rank].data_ptr
               for target, source in (
                   ("window_cache", "window_cache_pool"),
                   ("compressed_cache", "compressed_cache_pool"),
                   ("index_cache", "index_cache_pool"),
                   ("state_cache", "state_cache_pool"),
               ) for rank in range(2))


def test_full_decode_binds_distinct_c2a_and_c1a_metadata():
    fields = {source: object() for source in DecodeForwardSegment._METADATA.values()}
    for name in (
        "window_slots", "window_indices", "c2a_compressed_lens", "c1a_compressed_lens",
        "c2a_compressed_slots", "c1a_compressed_slots", "moe_counts",
    ):
        fields[name] = object()
    metadata = SimpleNamespace(fields=fields)
    weight = object()
    cache_tensor = object()
    weights = SimpleNamespace(layout_names=("hc_attn_fn",), full=lambda name: weight)
    cache = SimpleNamespace(_full={"window_cache_pool": cache_tensor},
                            full=lambda name: cache_tensor)
    names = (
        "x_hc", "pre_mix", "attention_num_tokens", "hc_attn_fn", "window_cache_pool",
        "moe_num_tokens", *DecodeForwardSegment._METADATA,
        "window_slots", "window_indices", "c2a_compressed_lens", "c1a_compressed_lens",
        "c2a_compressed_slots", "c1a_compressed_slots",
    )
    calls = []
    worker = SimpleNamespace(_persistent=True, _reset_persistent_windows=False,
                             run=lambda *args, **kwargs: calls.append(args))
    program = SimpleNamespace(compiled=object())
    segment = DecodeForwardSegment(worker, program, names, object(),
                                   SegmentTopology.for_decode(tp=1, dp=8),
                                   False, weights, cache, metadata)
    state = LayerState(object(), object(), "tp_local_token")
    assert segment.run(state, group_counts=(1,) * 8) is state
    bound = dict(zip(names, calls[0][1:]))
    assert bound["c2a_compressed_lens"] is not bound["c1a_compressed_lens"]
    assert bound["c2a_index_block_table"] is not bound["c1a_index_block_table"]
    assert bound["x_hc"] is state.residual
    assert bound["pre_mix"] is state.pre_mix
    assert bound["hc_attn_fn"] is weight
    assert worker._reset_persistent_windows is False

    calls.clear()
    tensor_segment = DecodeForwardSegment(
        worker, program, names, object(), SegmentTopology.for_decode(tp=1, dp=8),
        True, weights, cache, metadata)
    assert tensor_segment.run(state, group_counts=(1,) * 8) is state
    tensor_bound = dict(zip(names, calls[0][1:]))
    assert tensor_bound["attention_num_tokens"] is fields["moe_counts"]
