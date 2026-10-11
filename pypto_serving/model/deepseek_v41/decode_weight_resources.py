# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack selective checkpoint slices into the lib's resident decode weight ABI."""

from dataclasses import dataclass

import torch

from .decode_weights import DecodeWeightSlice, _ATTENTION, _MOE


_FLAT_LAYER = frozenset({
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight", "wq_a_scale",
    "q_norm_weight", "wq_b_scale", "wkv_scale", "kv_norm_weight", "attn_sink",
    "wo_a", "wo_b_scale", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "ffn_norm_weight", "gate_weight", "correction_bias", "routed_w1",
    "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3",
    "routed_w3_scale", "shared_w1_scale", "shared_w2_scale", "shared_w3_scale",
})
_AXIS_LAYER = frozenset({
    "wq_a", "wq_b", "wkv", "wo_b", "shared_w1", "shared_w2", "shared_w3",
})
_FLAT_KV = frozenset({"compressor_norm_weight"})
_FLAT_INDEX = frozenset({"index_wk", "index_norm_weight", "index_wq_b_scale"})
_AXIS_INDEX = frozenset({"index_wq_b", "index_weights_proj"})
_AXIS_C2A = frozenset({"c2a_compressor_wkv", "c2a_compressor_wgate"})
_SINGLE = frozenset({"c1a_compressor_wkv", "mxfp4_pair_lut"})
_ALL = (_FLAT_LAYER | _AXIS_LAYER | _FLAT_KV | _FLAT_INDEX |
        _AXIS_INDEX | _AXIS_C2A | _SINGLE)
_LAYER_ABI_ALIASES = {
    "c2a_compressor_wkv": "compressor_wkv",
    "c2a_compressor_wgate": "compressor_wgate",
}


def _destination_shape(layout, name, source_shape):
    if len(source_shape) < 2 or source_shape[0] != layout.world or any(dim <= 0 for dim in source_shape):
        raise ValueError("decode weight geometry requires nonempty EP-stacked slices")
    slots = layout.slots(name)
    if name in _SINGLE:
        return tuple(source_shape)
    if name in _FLAT_LAYER | _FLAT_KV | _FLAT_INDEX:
        return (layout.world, slots * source_shape[1], *source_shape[2:])
    return (layout.world, slots, *source_shape[1:])


def estimate_decode_weight_bytes(layout, geometry):
    """Budget all 40-layer shared tensors before reading checkpoint payloads."""
    if set(geometry) != _ALL:
        raise ValueError("decode weight geometry must cover exactly the lib weight fields")
    total = 0
    for name, (shape, dtype) in geometry.items():
        destination = _destination_shape(layout, name, shape)
        total += torch.Size(destination).numel() * torch.empty((), dtype=dtype).element_size()
    return total


@dataclass(frozen=True)
class DecodeWeightLayout:
    layers: int
    kv_sources: int
    index_sources: int
    c2a_sources: int
    world: int = 8

    @classmethod
    def from_lib(cls, abi):
        return cls(abi.N_LAYERS, abi.KV_SOURCE_COUNT, abi.INDEX_SOURCE_COUNT,
                   abi.C2A_SOURCE_COUNT, abi.EP_SIZE)

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (
            self.layers, self.kv_sources, self.index_sources, self.c2a_sources, self.world,
        )):
            raise ValueError("decode weight dimensions must be positive integers")

    def slots(self, name):
        if name in _FLAT_LAYER | _AXIS_LAYER:
            return self.layers
        if name in _FLAT_KV:
            return self.kv_sources
        if name in _FLAT_INDEX | _AXIS_INDEX:
            return self.index_sources
        if name in _AXIS_C2A:
            return self.c2a_sources
        if name in _SINGLE:
            return 1
        raise ValueError(f"unknown decode weight ABI field: {name}")


class DecodeWeightBank:
    """Shared host tensors created before the persistent worker forks.

    One slot is written exactly once. Float8 and packed FP4 tensors are copied
    through byte views so no arithmetic conversion can change payload bits.
    """

    def __init__(self, layout: DecodeWeightLayout, *, max_host_bytes=350 << 30):
        if type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError("decode host weight budget must be positive")
        self.layout = layout
        self.max_host_bytes = max_host_bytes
        self.allocated_bytes = 0
        self.tensors = {}
        self._seen = set()

    def append(self, item: DecodeWeightSlice):
        if not isinstance(item, DecodeWeightSlice) or item.name not in _ALL:
            raise ValueError("unknown decode weight slice")
        value = item.value
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or not value.is_contiguous()):
            raise ValueError("decode weight slice must be contiguous CPU EP-stacked data")
        slots = self.layout.slots(item.name)
        if type(item.slot) is not int or not 0 <= item.slot < slots:
            raise ValueError("decode weight slot is outside its source axis")
        key = (item.name, item.slot)
        if key in self._seen:
            raise ValueError("duplicate decode weight slot")
        shape = _destination_shape(self.layout, item.name, value.shape)
        destination = self.tensors.get(item.name)
        if destination is None:
            size = value.element_size() * torch.Size(shape).numel()
            if size > self.max_host_bytes - self.allocated_bytes:
                raise MemoryError("decode weight bank exceeds the shared host budget")
            destination = torch.empty(shape, dtype=value.dtype).share_memory_()
            self.tensors[item.name] = destination
            self.allocated_bytes += size
        if destination.dtype != value.dtype or tuple(destination.shape) != shape:
            raise ValueError("decode weight slices disagree on shape or dtype")
        if item.name in _SINGLE:
            target = destination
        elif item.name in _FLAT_LAYER | _FLAT_KV | _FLAT_INDEX:
            width = value.shape[1]
            target = destination[:, item.slot * width:(item.slot + 1) * width]
        else:
            target = destination[:, item.slot]
        target.view(torch.uint8).copy_(value.view(torch.uint8))
        self._seen.add(key)

    def finish(self, parameter_names):
        expected = {(name, slot) for name in _ALL for slot in range(self.layout.slots(name))}
        missing = expected - self._seen
        if missing:
            name, slot = sorted(missing)[0]
            raise ValueError(f"decode weight bank is missing {name}[{slot}]")
        self.validate_abi(parameter_names)
        return self.tensors

    @staticmethod
    def validate_abi(parameter_names):
        names = set(parameter_names)
        absent = {name for name in _ALL
                  if name not in names and _LAYER_ABI_ALIASES.get(name) not in names}
        if absent:
            raise ValueError("lib decode ABI omits checkpoint weight fields: " + ", ".join(sorted(absent)))


class ResidentDecodeWeights:
    """Upload the verified shared bank once and release every handle on close."""

    def __init__(self, worker, bank: DecodeWeightBank, parameter_names):
        if not isinstance(bank, DecodeWeightBank):
            raise TypeError("decode weight upload requires a completed host bank")
        bank.finish(parameter_names)
        if any(not tensor.is_shared() for tensor in bank.tensors.values()):
            raise ValueError("decode weights must be shared before worker startup")
        self.worker = worker
        self.layout = bank.layout
        self.tensors = {}
        try:
            for name, tensor in bank.tensors.items():
                slots = self.layout.slots(name)
                for slot in range(slots):
                    if name in _SINGLE:
                        host = tensor
                    elif name in _FLAT_LAYER | _FLAT_KV | _FLAT_INDEX:
                        width = tensor.shape[1] // slots
                        host = tensor[:, slot * width:(slot + 1) * width].contiguous()
                    else:
                        host = tensor[:, slot].contiguous()
                    self.tensors[name, slot] = worker.alloc_stacked_tensor(host)
        except BaseException:
            self.close()
            raise

    def view(self, name, slot):
        """Borrow one layer/source slice without a second device allocation."""
        slots = self.layout.slots(name)
        if type(slot) is not int or not 0 <= slot < slots:
            raise ValueError("decode weight view slot is outside its source axis")
        return self.tensors[name, slot]

    def prefill_weights(self, layer, *, kv_sources, index_sources, c2a_sources):
        """Bind real producer weights to the existing half-layer ABI names."""
        if not 0 <= layer.layer_id < self.layout.layers:
            raise ValueError("prefill layer is outside the resident decode bank")
        attention = {source: self.view(target, layer.layer_id)
                     for target, source in _ATTENTION.items()}
        moe = {source: self.view(target, layer.layer_id)
               for target, source in _MOE.items()}
        moe["mxfp4_pair_lut"] = self.view("mxfp4_pair_lut", 0)
        if layer.layer_id in kv_sources:
            source = kv_sources.index(layer.layer_id)
            attention["compressor_norm_weight"] = self.view("compressor_norm_weight", source)
            attention["index_wk"] = self.view("index_wk", index_sources.index(layer.layer_id))
            attention["index_norm_weight"] = self.view("index_norm_weight", index_sources.index(layer.layer_id))
            if layer.mode == "c2a_full":
                slot = c2a_sources.index(layer.layer_id)
                attention["compressor_wkv"] = self.view("c2a_compressor_wkv", slot)
                attention["compressor_wgate"] = self.view("c2a_compressor_wgate", slot)
            elif layer.mode == "c1a_full":
                attention["compressor_wkv"] = self.view("c1a_compressor_wkv", 0)
            else:
                raise ValueError("KV producer must be a C1A/C2A Full layer")
        if layer.layer_id in index_sources:
            slot = index_sources.index(layer.layer_id)
            for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                attention[name] = self.view(name, slot)
        return attention, moe

    def close(self):
        for tensor in reversed(tuple(self.tensors.values())):
            self.worker.free_stacked_tensor(tensor)
        self.tensors.clear()
