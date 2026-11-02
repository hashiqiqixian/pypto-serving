# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check official host Engram ownership alongside resident DSpark resources."""

from types import SimpleNamespace

import torch

from pypto_serving.model.deepseek_v41.npu_runner import (
    HostEngramResources,
    V41ModelRunner,
    V41ResidentResources,
)
from pypto_serving.model.deepseek_v41.weight_loader import DecodeWeightLayout, _ALL


def test_engram_rows_keep_request_offsets_and_tp_replicas():
    calls = []

    def lookup(token_ids, capacity, start_pos, *, request_id):
        calls.append((request_id, tuple(token_ids), capacity, start_pos))
        rows = torch.zeros((2, capacity, 3), dtype=torch.bfloat16)
        rows[:, :len(token_ids)] = torch.tensor(token_ids).reshape(1, -1, 1)
        return rows

    topology = SimpleNamespace(world=4, dp=2, tp=2, capacity=4)
    host = HostEngramResources.__new__(HostEngramResources)
    host.host = SimpleNamespace(lookup=lookup)
    host.lookup = torch.empty((4, 2, 4, 3), dtype=torch.bfloat16)
    target = torch.full_like(host.lookup, 99)
    requests = tuple(SimpleNamespace(request_id=key, partition=rank, token_ids=ids, start=start)
                     for key, rank, ids, start in (
                         ("a", 0, (11, 12), 0), ("b", 0, (13,), 7), ("c", 1, (21,), 3)))

    host.stage_lookup(target, SimpleNamespace(requests=requests), topology)

    assert calls == [("a", (11, 12), 4, 0), ("b", (13,), 4, 7), ("c", (21,), 4, 3)]
    assert target[:, 0, :, 0].tolist() == [
        [11, 12, 13, 0], [11, 12, 13, 0], [21, 0, 0, 0], [21, 0, 0, 0]]
    assert torch.equal(target[:, 0], target[:, 1])


def test_engram_fixed_request_stride_does_not_lookup_inactive_tail_rows():
    calls = []

    def lookup(token_ids, capacity, start_pos, *, request_id):
        calls.append((request_id, tuple(token_ids), start_pos))
        rows = torch.zeros((2, capacity, 1), dtype=torch.bfloat16)
        rows[:, :len(token_ids), 0] = torch.tensor(token_ids)
        return rows

    host = HostEngramResources.__new__(HostEngramResources)
    host.host = SimpleNamespace(lookup=lookup)
    host.lookup = torch.empty((1, 2, 12, 1), dtype=torch.bfloat16)
    target = torch.full_like(host.lookup, 99)
    requests = tuple(SimpleNamespace(request_id=key, partition=0, token_ids=ids, start=start)
                     for key, ids, start in (("tail", (11,), 127), ("full", (20, 21, 22, 23, 24, 25), 3)))

    host.stage_lookup(target, SimpleNamespace(requests=requests),
                      SimpleNamespace(dp=1, tp=1, capacity=12), row_stride=6)

    assert calls == [("tail", (11,), 127), ("full", (20, 21, 22, 23, 24, 25), 3)]
    assert target[0, 0, :, 0].tolist() == [11, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25]


def test_request_release_keeps_device_weights_and_waits_before_host_cleanup():
    events = []
    resources = V41ResidentResources.__new__(V41ResidentResources)
    resources.cache = SimpleNamespace(reset_request=lambda owner: events.append(("cache", owner)))
    resources.dspark = SimpleNamespace(release=lambda keys: events.append(("dspark", keys)))
    resources.engram = object()
    resources.engram_host = SimpleNamespace(
        release_request=lambda key: events.append(("engram", key)))
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner.resources = resources
    runner.bindings = SimpleNamespace(
        reset_request=lambda resident, key, owner: resident.reset_request(key, owner),
        wait=lambda resident: events.append(("wait",)),
    )

    runner._reset_request("request", "owner")

    assert events == [("cache", "owner"), ("dspark", ("request",)), ("wait",), ("engram", "request")]


def test_backbone_bank_excludes_separately_owned_engram_weights():
    abi = SimpleNamespace(N_LAYERS=40, KV_SOURCE_COUNT=4, INDEX_SOURCE_COUNT=8,
                          C2A_SOURCE_COUNT=3, EP_SIZE=4, ENGRAM_K=6144)
    layout = DecodeWeightLayout.from_lib(abi, tp_size=1)

    assert layout.world == 4
    assert not {"engram_wkv", "engram_wkv_scale", "engram_weight"} & _ALL
