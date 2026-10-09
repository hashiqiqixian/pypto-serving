# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device cache-pool layout for the lib's 40-layer decode composite."""

from dataclasses import dataclass

import torch

from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

from .cache_contract import validate_cache_groups


@dataclass(frozen=True)
class CacheBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class DecodeCachePlan:
    primary_num_blocks: int
    group_blocks: dict[str, int]
    buffers: tuple[CacheBufferSpec, ...]


def plan_decode_cache_pools(layers, groups, *, max_seq_len, primary_num_blocks,
                            request_slots, abi) -> DecodeCachePlan:
    """Match scheduler page capacities to decode_fwd's source-sliced pools.

    ``abi`` is the imported decode_fwd module from the selected lib revision.
    A single primary block count is returned to the grouped scheduler, which
    scales each group by its own max_blocks_per_seq. All shapes include the EP
    rank axis; each rank has a private copy of its DP partition's page IDs.
    """
    resolved = validate_cache_groups(tuple(layers), tuple(groups), max_seq_len)
    by_name = {group.name: group for group in groups}
    window = by_name[resolved.window]
    if groups[0].name != resolved.window:
        raise ValueError("window must be the primary scheduler cache group")
    if (type(primary_num_blocks) is not int or primary_num_blocks <= 0
            or primary_num_blocks % window.max_blocks_per_seq):
        raise ValueError("primary cache blocks must contain whole request slots")
    if type(request_slots) is not int or request_slots <= 0:
        raise ValueError("request slot capacity must be positive")
    slots = primary_num_blocks // window.max_blocks_per_seq
    group_blocks = {}
    for group in groups:
        blocks = slots * group.max_blocks_per_seq
        if group.num_blocks is not None and group.num_blocks != blocks:
            raise ValueError(f"{group.name} cache capacity conflicts with the scheduler")
        group_blocks[group.name] = blocks

    kv_sources = sum(layer.kv_source == layer.layer_id for layer in layers)
    c2a_sources = sum(layer.mode == "c2a_full" for layer in layers)
    index_sources = sum(layer.index_source == layer.layer_id for layer in layers)
    if (abi.N_LAYERS, abi.KV_SOURCE_COUNT, abi.C2A_SOURCE_COUNT,
            abi.INDEX_SOURCE_COUNT, abi.BLOCK_SIZE) != (
                len(layers), kv_sources, c2a_sources, index_sources, 128):
        raise ValueError("decode cache source axes disagree with the layer schedule")
    head, index, world = abi.HEAD_DIM, abi.INDEX_DIM, abi.EP_SIZE
    if (world != 8 or any(type(value) is not int or value <= 0 for value in
                       (head, index, abi.STATE_CAPACITY, abi.STATE_WIDTH))
            or head % abi.WINDOW_CACHE_GROUP or head % abi.COMPRESSED_CACHE_GROUP
            or index % abi.INDEX_CACHE_GROUP):
        raise ValueError("decode cache dimensions disagree with TP4/DP2/EP8")
    block = abi.BLOCK_SIZE
    window_pages = group_blocks[resolved.window]
    # One lib source axis covers both compression families, so its stride must
    # fit the larger physical page namespace without remapping scheduler IDs.
    compressed_pages = max(group_blocks[resolved.c2a], group_blocks[resolved.c1a])
    buffers = (
        CacheBufferSpec("window_cache_pool", (world, len(layers) * window_pages, block, 1, head),
                        torch.float8_e4m3fn),
        CacheBufferSpec("window_cache_scale_pool", (
            world, len(layers) * window_pages, block, 1, head // abi.WINDOW_CACHE_GROUP),
                        torch.float8_e8m0fnu),
        CacheBufferSpec("compressed_cache_pool", (
            world, kv_sources * compressed_pages, block, 1, head // 2), torch.uint8),
        CacheBufferSpec("compressed_cache_scale_pool", (
            world, kv_sources * compressed_pages, block, 1, head // abi.COMPRESSED_CACHE_GROUP),
                        torch.float8_e4m3fn),
        CacheBufferSpec("index_cache_pool", (
            world, index_sources * compressed_pages, block, 1, index // 2), torch.uint8),
        CacheBufferSpec("index_cache_scale_pool", (
            world, index_sources * compressed_pages, block, 1, index // abi.INDEX_CACHE_GROUP),
                        torch.float8_e8m0fnu),
        CacheBufferSpec("state_cache_pool", (
            world, c2a_sources * request_slots, abi.STATE_CAPACITY, abi.STATE_WIDTH), torch.float32),
    )
    return DecodeCachePlan(primary_num_blocks, group_blocks, buffers)


class DecodeCachePools:
    """Own cache handles; the caller must wait for all launches before close."""

    def __init__(self, worker, plan: DecodeCachePlan):
        self.worker = worker
        self.tensors = {}
        try:
            for spec in plan.buffers:
                self.tensors[spec.name] = alloc_device_buffer(
                    worker, spec.shape, spec.dtype, stacked=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        for tensor in reversed(tuple(self.tensors.values())):
            self.worker.free_stacked_tensor(tensor)
        self.tensors.clear()
