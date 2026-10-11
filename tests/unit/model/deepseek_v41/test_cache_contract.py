# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-backbone cache layout checks before a worker opens device resources."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.model.deepseek_v41.cache_contract import (
    V41CacheGroups, build_cache_groups, validate_cache_groups,
)
from pypto_serving.model.deepseek_v41.cache_resources import (
    DecodeCachePools, allocate_cache_with_retry, cache_bytes_per_rank,
    make_cache_page_reset_buffers, make_post_resident_cache_planner,
    make_state_reset_buffer, plan_decode_cache_pools,
)
from pypto_serving.model.deepseek_v41.composite import CompositeBindings
from pypto_serving.model.deepseek_v41.execution_plan import RankPlacement, plan_layers
from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner


@pytest.fixture
def contract():
    path = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"
    layers = plan_layers(json.loads(path.read_text(encoding="utf-8")))
    groups = (
        KVCacheGroupSpec("window", tuple(range(40)), KVCacheSpec(128, 16),
                         65, num_blocks=65, num_partitions=2),
        KVCacheGroupSpec("c2a", (2, 8, 14), KVCacheSpec(256, 16, 2),
                         33, num_blocks=33, num_partitions=2),
        KVCacheGroupSpec("c1a", (20,), KVCacheSpec(128, 16),
                         65, num_blocks=65, num_partitions=2),
    )
    return layers, groups


def test_full_backbone_resolves_three_independent_cache_families(contract):
    layers, groups = contract
    assert validate_cache_groups(layers, groups, 8320) == V41CacheGroups(
        window="window", c2a="c2a", c1a="c1a")


def test_default_groups_match_the_full_checkpoint_schedule(contract):
    layers, _ = contract
    groups = build_cache_groups(layers, 8320)
    assert tuple(group.name for group in groups) == ("window", "c2a", "c1a")
    assert tuple(group.max_blocks_per_seq for group in groups) == (65, 33, 65)
    assert all(group.num_blocks is None and group.num_partitions == 2 for group in groups)
    assert validate_cache_groups(layers, groups, 8320) == V41CacheGroups("window", "c2a", "c1a")


def test_cache_family_names_follow_producers_not_labels(contract):
    layers, groups = contract
    renamed = tuple(replace(group, name=f"pool_{index}") for index, group in enumerate(groups))
    assert validate_cache_groups(layers, renamed, 8320) == V41CacheGroups(
        window="pool_0", c2a="pool_1", c1a="pool_2")


@pytest.mark.parametrize("change,match", [
    (lambda groups: groups[:2], "separate"),
    (lambda groups: (groups[0], replace(groups[1], layer_indices=(2, 8, 14, 20)), groups[2]),
     "producer"),
    (lambda groups: (groups[0], groups[1], replace(groups[2], spec=KVCacheSpec(256, 16, 2))),
     "c1a.*layout"),
    (lambda groups: (replace(groups[0], max_blocks_per_seq=64), *groups[1:]),
     "window.*max_seq_len"),
    (lambda groups: (replace(groups[0], num_blocks=64), *groups[1:]),
     "window.*max_seq_len"),
    (lambda groups: (replace(groups[0], num_partitions=1), *groups[1:]),
     "window.*layout"),
    (lambda groups: (groups[0], replace(groups[1], max_blocks_per_seq=32), groups[2]),
     "c2a.*max_seq_len"),
])
def test_mismatched_cache_family_rejected(contract, change, match):
    layers, groups = contract
    with pytest.raises(ValueError, match=match):
        validate_cache_groups(layers, change(groups), 8320)


def test_runner_rejects_bad_layout_before_device_allocation(contract):
    layers, groups = contract
    calls = []
    bindings = CompositeBindings(
        revision="recording-only", entries={("prefill", layer.mode): lambda *a: None
                                            for layer in layers},
        initialize=lambda *a: None, output=lambda *a: None,
        allocate=lambda *a: calls.append("allocate"), prepare_weights=lambda *a: None,
        reset_request=lambda *a: None, wait=lambda *a: None, close=lambda *a: None,
        cache_groups=groups[:2], decode_backbone=lambda *a: None,
    )
    plan = SimpleNamespace(placement=RankPlacement(0), layers=layers)
    with pytest.raises(ValueError, match="separate"):
        V41ModelRunner(plan, bindings, device_ids=range(8),
                       runtime=SimpleNamespace(max_seq_len=8320))
    assert calls == []


def decode_abi():
    return SimpleNamespace(
        N_LAYERS=40, KV_SOURCE_COUNT=4, C2A_SOURCE_COUNT=3,
        INDEX_SOURCE_COUNT=8, BLOCK_SIZE=128, HEAD_DIM=512, INDEX_DIM=128,
        EP_SIZE=8, STATE_CAPACITY=4, STATE_WIDTH=1024,
        WINDOW_CACHE_GROUP=32, COMPRESSED_CACHE_GROUP=16, INDEX_CACHE_GROUP=32,
    )


def test_decode_cache_pools_preserve_layer_and_source_namespaces(contract):
    layers, groups = contract
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8320, primary_num_blocks=65,
        request_slots=2, abi=decode_abi(),
    )
    buffers = {spec.name: spec for spec in plan.buffers}
    assert plan.group_blocks == {"window": 65, "c2a": 33, "c1a": 65}
    assert plan.physical_group_blocks == {"window": 66, "c2a": 34, "c1a": 66}
    assert plan.request_slots == 2 and plan.physical_request_slots == 3
    assert plan.group_sources["c2a"]["compressed"] == (0, 1, 2)
    assert plan.group_sources["c1a"]["compressed"] == (3,)
    assert buffers["window_cache_pool"].shape == (8, 40 * 66, 128, 1, 512)
    assert buffers["compressed_cache_pool"].shape == (8, 4 * 66, 128, 1, 256)
    assert buffers["index_cache_pool"].shape == (8, 8 * 66, 128, 1, 64)
    assert buffers["state_cache_pool"].shape == (8, 3 * 3, 4, 1024)
    assert buffers["window_cache_scale_pool"].dtype == torch.float8_e8m0fnu


def test_prefill_cache_views_borrow_the_decode_producer_pools(contract):
    layers, groups = contract
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8320, primary_num_blocks=65,
        request_slots=2, abi=decode_abi(),
    )

    class Worker:
        def __init__(self):
            self.next_ptr = 100000
            self.freed = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.next_ptr += 100000000
            return DeviceTensor(self.next_ptr, shape, dtype)

        def free_stacked_tensor(self, tensor):
            self.freed.append(tensor)

    worker = Worker()
    pools = DecodeCachePools(worker, plan, make_state_reset_buffer(plan))
    sources = dict(kv_sources=(2, 8, 14, 20), index_sources=(2, 8, 14, 20, 24, 28, 32, 36),
                   c2a_sources=(2, 8, 14))
    c1a = pools.prefill_views(layers[24], **sources)
    assert c1a["window_cache"].shape[1] == 66
    assert c1a["compressed_cache"].shape[1] == 66
    assert c1a["index_cache"].shape[1] == 66
    assert c1a["compressed_cache"] is pools.view("compressed_cache_pool", 3)
    assert c1a["index_cache"] is pools.view("index_cache_pool", 4)
    c2a = pools.prefill_views(layers[2], **sources)
    assert c2a["state_cache"].shape == (8, 3, 4, 1024)
    assert set(pools.prefill_views(layers[0], **sources)) == {
        "window_cache", "window_cache_scale"}
    with pytest.raises(ValueError, match="outside"):
        pools.view("index_cache_pool", 8)
    pools.close()
    assert len(worker.freed) == sum(pools.source_counts.values())


def test_decode_cache_plan_rejects_scheduler_capacity_mismatch(contract):
    layers, groups = contract
    with pytest.raises(ValueError, match="conflicts with the scheduler"):
        plan_decode_cache_pools(
            layers, groups, max_seq_len=8320, primary_num_blocks=130,
            request_slots=2, abi=decode_abi(),
        )


def test_post_resident_budget_uses_actual_pools_and_limiting_rank(contract):
    layers, fixed_groups = contract
    groups = tuple(replace(group, num_blocks=None) for group in fixed_groups)
    one = plan_decode_cache_pools(layers, groups, max_seq_len=8320,
                                  primary_num_blocks=65, request_slots=1, abi=decode_abi())
    two = plan_decode_cache_pools(layers, groups, max_seq_len=8320,
                                  primary_num_blocks=130, request_slots=2, abi=decode_abi())
    slot_bytes = cache_bytes_per_rank(two) - cache_bytes_per_rank(one)
    fixed_bytes = cache_bytes_per_rank(one) - slot_bytes
    reserve = 128 << 20
    total = 16 << 30
    limiting_free = reserve + fixed_bytes + 2 * slot_bytes

    class Worker:
        def device_memory_info(self, rank):
            return (limiting_free if rank == 7 else limiting_free + slot_bytes, total)

    planner = make_post_resident_cache_planner(
        layers, groups, max_seq_len=8320, abi=decode_abi(), utilization=1.0,
        max_request_slots=32, reserved_bytes_per_rank=reserve)
    slots, plan_for_slots = planner(Worker())
    assert slots == 2
    assert plan_for_slots(slots).primary_num_blocks == 130
    assert cache_bytes_per_rank(plan_for_slots(slots)) == fixed_bytes + 2 * slot_bytes
    with pytest.raises(MemoryError, match="one V4.1 cache request slot"):
        make_post_resident_cache_planner(
            layers, groups, max_seq_len=8320, abi=decode_abi(), utilization=1.0,
            max_request_slots=32, reserved_bytes_per_rank=reserve)(
                SimpleNamespace(device_memory_info=lambda rank: (
                    reserve + fixed_bytes + slot_bytes - 1, total)))


def test_post_resident_cache_retry_halves_slots_after_complete_cleanup(contract, monkeypatch):
    layers, fixed_groups = contract
    groups = tuple(replace(group, num_blocks=None) for group in fixed_groups)
    attempts = []

    def plan_for_slots(slots):
        return plan_decode_cache_pools(
            layers, groups, max_seq_len=8320,
            primary_num_blocks=slots * 65, request_slots=slots, abi=decode_abi())

    def allocate(worker, plan, state_reset, page_resets):
        attempts.append(plan.request_slots)
        if plan.request_slots > 2:
            raise MemoryError("simulated atomic cache OOM")
        return SimpleNamespace(plan=plan)

    monkeypatch.setattr("pypto_serving.model.deepseek_v41.cache_resources.DecodeCachePools", allocate)
    result = allocate_cache_with_retry(object(), 5, plan_for_slots, object(), object())
    assert attempts == [5, 2]
    assert result.plan.group_blocks == {"window": 130, "c2a": 66, "c1a": 130}


def test_decode_cache_allocation_releases_completed_pools_after_failure(contract):
    layers, groups = contract
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8320, primary_num_blocks=65,
        request_slots=2, abi=decode_abi(),
    )

    class Worker:
        def __init__(self):
            self.allocated = 0
            self.freed = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.allocated += 1
            if self.allocated == 10:
                raise RuntimeError("device allocation failed")
            return DeviceTensor(self.allocated, shape, dtype)

        def free_tensor(self, tensor, *, worker_id):
            self.freed.append(tensor.data_ptr)

        def free_stacked_tensor(self, tensor):
            self.freed.extend(shard.data_ptr for shard in tensor.shards)

    worker = Worker()
    with pytest.raises(RuntimeError, match="device allocation failed"):
        DecodeCachePools(worker, plan, make_state_reset_buffer(plan))
    assert set(worker.freed) == set(range(1, 10))


def test_reset_only_targets_one_dp_request_state_slot(contract):
    layers, groups = contract
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8320, primary_num_blocks=65,
        request_slots=2, abi=decode_abi(),
    )

    class Worker:
        def __init__(self):
            self.next_ptr = 0
            self.copies = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.next_ptr += 1
            return DeviceTensor(self.next_ptr, shape, dtype)

        def free_stacked_tensor(self, tensor):
            pass

        def copy_to(self, destination, source, count, *, dst_offset, worker_id):
            self.copies.append((worker_id, destination, dst_offset, count))

    worker = Worker()
    pools = DecodeCachePools(worker, plan, make_state_reset_buffer(plan))
    pools.reset_state_slot(1, 1)
    row_bytes = 4 * 1024 * 4
    assert worker.copies == [
        (rank, pools.view("state_cache_pool", source).shards[rank].data_ptr,
         row_bytes, row_bytes)
        for rank in range(4, 8) for source in range(3)
    ]
    with pytest.raises(ValueError, match="invalid compressor"):
        pools.reset_state_slot(1, 3)
    pools.close()


def test_request_reset_keeps_other_dp_and_compression_family_pages(contract):
    layers, groups = contract
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8320, primary_num_blocks=65,
        request_slots=2, abi=decode_abi(),
    )

    class Worker:
        def __init__(self):
            self.next_ptr = 0
            self.copies = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.next_ptr += 1
            return DeviceTensor(self.next_ptr, shape, dtype)

        def free_stacked_tensor(self, tensor):
            pass

        def copy_to(self, destination, source, count, *, dst_offset, worker_id):
            self.copies.append((worker_id, destination, dst_offset, count))

    worker = Worker()
    zeros = make_cache_page_reset_buffers(plan)
    assert all(tensor.is_shared() for tensor in zeros.values())
    assert torch.all(zeros["compressed_cache_scale_pool"].view(torch.uint8) == 0x38)
    pools = DecodeCachePools(worker, plan, make_state_reset_buffer(plan), zeros)
    owner = SimpleNamespace(partition=0, slot=1,
                            pages={"window": (1,), "c2a": (1,), "c1a": ()})
    pools.reset_request(owner)
    page_bytes = zeros["compressed_cache_pool"][0].numel()
    for rank in range(8):
        for source in range(4):
            compressed = pools.view("compressed_cache_pool", source)
            offsets = {offset for worker_id, ptr, offset, _ in worker.copies
                       if worker_id == rank and ptr == compressed.shards[rank].data_ptr}
            assert offsets == ({page_bytes} if rank < 4 and source < 3 else set())
    with pytest.raises(ValueError, match="outside its cache family"):
        pools.reset_request(SimpleNamespace(partition=0, slot=2, pages=owner.pages))
    with pytest.raises(ValueError, match="outside its cache family"):
        pools.reset_request(SimpleNamespace(partition=0, slot=1,
                                            pages={"window": (65,), "c2a": (), "c1a": ()}))
    worker.copies.clear()
    pools.reset_padding(1)
    for rank in range(8):
        for source in range(4):
            compressed = pools.view("compressed_cache_pool", source)
            offsets = {offset for worker_id, ptr, offset, _ in worker.copies
                       if worker_id == rank and ptr == compressed.shards[rank].data_ptr}
            expected = {(33 if source < 3 else 65) * page_bytes} if rank >= 4 else set()
            assert offsets == expected
    pools.close()
