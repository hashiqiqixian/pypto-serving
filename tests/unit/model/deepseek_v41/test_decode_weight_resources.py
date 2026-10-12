# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode weight banking must preserve source axes and packed payload bits."""

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.model.deepseek_v41.decode_weight_resources import (
    DecodeWeightBank, DecodeWeightLayout, ResidentDecodeWeights, _ALL,
    estimate_decode_weight_bytes,
)
from pypto_serving.model.deepseek_v41.decode_weights import DecodeWeightSlice
from pypto_serving.model.deepseek_v41.execution_plan import LayerPlan


def test_decode_bank_places_flat_axis_and_singleton_slots():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    for slot in range(2):
        bits = torch.arange(12, dtype=torch.uint8).reshape(2, 2, 3) + slot * 32
        bank.append(DecodeWeightSlice("routed_w1", slot, bits))
        bank.append(DecodeWeightSlice("wq_a", slot, bits.view(torch.float8_e4m3fn)))
    bank.append(DecodeWeightSlice("mxfp4_pair_lut", 0,
                                  torch.arange(8, dtype=torch.int16).reshape(2, 2, 2)))

    assert bank.tensors["routed_w1"].shape == (2, 2, 2, 3)
    assert torch.equal(bank.tensors["routed_w1"][1],
                       torch.arange(12, dtype=torch.uint8).reshape(2, 2, 3) + 32)
    assert bank.tensors["wq_a"].shape == (2, 2, 2, 3)
    assert torch.equal(bank.tensors["wq_a"][1].view(torch.uint8),
                       torch.arange(12, dtype=torch.uint8).reshape(2, 2, 3) + 32)
    assert bank.tensors["wq_a"].is_shared()


def test_decode_bank_rejects_duplicate_and_budget_before_allocation():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    value = torch.ones((2, 2, 3), dtype=torch.uint8)
    bank = DecodeWeightBank(layout, max_host_bytes=24)
    item = DecodeWeightSlice("routed_w1", 0, value)
    bank.append(item)
    with pytest.raises(ValueError, match="duplicate"):
        bank.append(item)
    with pytest.raises(MemoryError, match="budget"):
        bank.append(DecodeWeightSlice("wq_a", 0, value))


def test_decode_bank_requires_every_lib_weight_slot():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    with pytest.raises(ValueError, match="missing"):
        bank.finish(_ALL)
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bank.append(DecodeWeightSlice(name, slot, torch.full((2, 1), slot, dtype=torch.uint8)))
    assert set(bank.finish(_ALL)) == _ALL
    with pytest.raises(ValueError, match="omits"):
        bank.finish(_ALL - {"wq_a"})


def test_decode_bank_accepts_c2a_storage_names_mapped_to_layer_abi():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bank.append(DecodeWeightSlice(name, slot, torch.zeros((2, 1), dtype=torch.uint8)))
    layer_names = (_ALL - {"c2a_compressor_wkv", "c2a_compressor_wgate"}
                   | {"compressor_wkv", "compressor_wgate"})
    assert set(bank.finish(layer_names)) == _ALL
    with pytest.raises(ValueError, match="c2a_compressor_wgate"):
        bank.finish(layer_names - {"compressor_wgate"})


def test_decode_bank_estimate_matches_allocations():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    geometry = {name: ((2, 1), torch.uint8) for name in _ALL}
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bank.append(DecodeWeightSlice(name, slot, torch.zeros((2, 1), dtype=torch.uint8)))
    assert estimate_decode_weight_bytes(layout, geometry) == bank.allocated_bytes


def test_resident_upload_views_preserve_every_slot_payload():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    expected = {}
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bits = (torch.arange(12, dtype=torch.uint8).reshape(2, 2, 3)
                    + slot * 32 + len(name))
            value = bits.view(torch.float8_e8m0fnu) if name == "wq_a_scale" else bits
            bank.append(DecodeWeightSlice(name, slot, value))
            expected[name, slot] = bits

    class Worker:
        def __init__(self):
            self.uploaded = {}
            self.next_id = 0

        def alloc_stacked_tensor(self, tensor):
            assert tensor.is_contiguous() and tensor.shape[0] == layout.world
            self.next_id += 1
            self.uploaded[self.next_id] = tensor
            return self.next_id

        def free_stacked_tensor(self, handle):
            del self.uploaded[handle]

    worker = Worker()
    resident = ResidentDecodeWeights(worker, bank, _ALL)
    for name, slot in expected:
        uploaded = worker.uploaded[resident.view(name, slot)]
        assert torch.equal(uploaded.view(torch.uint8), expected[name, slot])
        assert uploaded.data_ptr() == (bank.tensors[name] if name in {"mxfp4_pair_lut",
                                                                    "c1a_compressor_wkv"}
                                       else bank.tensors[name][slot]).data_ptr()
    resident.close()
    assert not worker.uploaded


def test_decode_bank_upload_is_released_on_partial_failure():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bank.append(DecodeWeightSlice(name, slot, torch.zeros((2, 1), dtype=torch.uint8)))

    class Worker:
        def __init__(self):
            self.handles = []
            self.freed = []

        def alloc_stacked_tensor(self, tensor):
            if len(self.handles) == 2:
                raise RuntimeError("upload failed")
            handle = object()
            self.handles.append(handle)
            return handle

        def free_stacked_tensor(self, handle):
            self.freed.append(handle)

    worker = Worker()
    with pytest.raises(RuntimeError, match="upload failed"):
        ResidentDecodeWeights(worker, bank, _ALL)
    assert worker.freed == list(reversed(worker.handles))


def test_resident_views_share_layer_storage_without_uploading_again():
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bank.append(DecodeWeightSlice(name, slot, torch.zeros((2, 2), dtype=torch.uint8)))

    class Worker:
        def __init__(self):
            self.next_ptr = 10000
            self.freed = []
            self.hosts = []

        def alloc_stacked_tensor(self, tensor):
            assert tensor.is_contiguous()
            self.hosts.append(tensor)
            self.next_ptr += 1000
            return StackedDeviceTensor(
                tuple(DeviceTensor(self.next_ptr + rank * 100, tensor.shape[1:], tensor.dtype)
                      for rank in range(2)), tensor.shape, (0, 1),
            )

        def free_stacked_tensor(self, tensor):
            self.freed.append(tensor)

    worker = Worker()
    resident = ResidentDecodeWeights(worker, bank, _ALL)
    assert any(host.data_ptr() == bank.tensors["wq_a"][1].data_ptr()
               for host in worker.hosts)
    flat = resident.view("routed_w1", 1)
    axis = resident.view("wq_a", 1)
    assert flat.shape == axis.shape == (2, 2)
    assert flat is resident.tensors["routed_w1", 1]
    assert axis is resident.tensors["wq_a", 1]
    assert resident.view("mxfp4_pair_lut", 0) is resident.tensors["mxfp4_pair_lut", 0]
    attention, moe = resident.prefill_weights(
        LayerPlan(0, "c1a_full", 0, 0, 0),
        kv_sources=(0,), index_sources=(0, 1), c2a_sources=(1,),
    )
    assert attention["compressor_wkv"] is resident.tensors["c1a_compressor_wkv", 0]
    assert attention["index_wq_b"].shape == (2, 2)
    assert moe["norm_weight"].shape == (2, 2)
    assert moe["mxfp4_pair_lut"] is resident.tensors["mxfp4_pair_lut", 0]
    with pytest.raises(ValueError, match="outside"):
        resident.view("wq_a", 2)
    resident.close()
    assert len(worker.freed) == sum(layout.slots(name) for name in _ALL)
