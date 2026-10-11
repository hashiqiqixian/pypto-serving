# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A decode filler has private physical resources and no ledger ownership."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.decode_padding import pad_single_decode
from pypto_serving.model.deepseek_v41.final_output import prepare_final_output_rows
from pypto_serving.model.deepseek_v41.prefill_metadata_resources import PrefillMetadataResources
from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.segment_inputs import prepare_segment_inputs
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


@pytest.mark.parametrize("active_partition", (0, 1))
def test_single_decode_fills_only_the_other_partition(active_partition):
    groups = {"ori": 65, "cmp2": 33, "cmp1": 65}
    plan = SimpleNamespace(group_blocks=groups, physical_group_blocks={name: blocks + 1
                           for name, blocks in groups.items()}, group_sources=dict.fromkeys(groups),
                           request_slots=2, physical_request_slots=3)
    request = RequestSlice("real", active_partition, 0, 8192, (42,), 8192,
                           MappingProxyType({name: (0,) for name in groups}))
    step = ForwardStep("decode", (request,), 7)
    embeddings = torch.ones((1, 5120), dtype=torch.bfloat16)
    padded, rows, partition = pad_single_decode(step, embeddings, plan)

    assert step.requests == (request,)
    assert partition == 1 - active_partition
    assert padded.requests[0] is request
    assert padded.requests[1].partition == partition
    assert padded.requests[1].state_slot == plan.request_slots
    assert padded.requests[1].pages == {name: (blocks,) for name, blocks in groups.items()}
    assert padded.requests[1].start == 0
    assert padded.epoch == step.epoch
    assert rows.shape == (2, 5120) and torch.equal(rows[0], rows[1])


def test_equal_decode_counts_do_not_allocate_padding():
    requests = tuple(RequestSlice(str(partition), partition, 0, 2, (42,), 2,
                                  MappingProxyType({})) for partition in range(2))
    step = ForwardStep("decode", requests, 3)
    embeddings = torch.ones((2, 16), dtype=torch.bfloat16)
    unchanged, rows, partition = pad_single_decode(step, embeddings, None)
    assert unchanged is step and rows is embeddings and partition is None


def test_multiple_asymmetric_requests_still_fail_closed():
    requests = tuple(RequestSlice(str(index), 0, index, 2, (42,), 2,
                                  MappingProxyType({})) for index in range(2))
    step = ForwardStep("decode", requests, 3)
    with pytest.raises(ValueError, match="one-request asymmetric"):
        pad_single_decode(step, torch.ones((2, 16), dtype=torch.bfloat16), None)


def test_missing_physical_reservation_fails_before_padding():
    request = RequestSlice("real", 0, 0, 2, (42,), 2, MappingProxyType({}))
    step = ForwardStep("decode", (request,), 3)
    groups = {"window": 65, "c2a": 33, "c1a": 65}
    plan = SimpleNamespace(group_blocks=groups, physical_group_blocks=groups,
                           group_sources=dict.fromkeys(groups),
                           request_slots=2, physical_request_slots=2)
    with pytest.raises(ValueError, match="reserved cache families"):
        pad_single_decode(step, torch.ones((1, 16), dtype=torch.bfloat16), plan)


def test_padding_flows_through_all_cache_metadata_but_not_output_rows():
    topology = SegmentTopology.for_decode()
    groups = {"window": 65, "c2a": 33, "c1a": 65}
    plan = SimpleNamespace(group_blocks=groups, physical_group_blocks={name: blocks + 1
                           for name, blocks in groups.items()}, group_sources=dict.fromkeys(groups),
                           request_slots=2, physical_request_slots=3)
    real = RequestSlice("real", 0, 0, 2, (42,), 2,
                        MappingProxyType({name: (0,) for name in groups}))
    step = ForwardStep("decode", (real,), 4)
    padded, embeddings, partition = pad_single_decode(
        step, torch.ones((1, 5120), dtype=torch.bfloat16), plan)
    table = torch.ones((16, 64), dtype=torch.float32)
    metadata = PrefillMetadataResources(
        topology, max_seq_len=16, max_requests=3, window_pages=66,
        c2a_pages=34, c1a_pages=66, rope_tables=(table, table),
        compressed_rope_tables=(table, table))
    assert metadata.stage(padded) == (1, 1)
    rank = partition * topology.tp
    assert metadata.fields["window_slots"][rank, 0] == 65 * 128
    assert metadata.fields["c2a_compressed_slots"][rank, 0] == -1
    assert metadata.fields["c1a_compressed_slots"][rank, 0] == 65 * 128
    assert metadata.fields["c2a_state_block_table"][rank, 0, 0] == 2
    assert metadata.fields["c1a_index_block_table"][rank, 0, 0] == 65

    inputs = prepare_segment_inputs(embeddings, padded, topology)
    outputs = replace(inputs, request_last_rows=inputs.request_last_rows[:1])
    selected = prepare_final_output_rows(
        outputs, torch.full((8, 16), -1, dtype=torch.int32).share_memory_(),
        local_capacity=topology.local_capacity)
    assert len(selected.locations) == 1
    assert selected.locations[0][0] < topology.tp
