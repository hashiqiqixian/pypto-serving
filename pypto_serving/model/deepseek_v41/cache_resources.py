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

from .cache_contract import V41CacheGroups, validate_cache_groups


@dataclass(frozen=True)
class CacheBufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class DecodeCachePlan:
    primary_num_blocks: int
    family_groups: V41CacheGroups
    group_blocks: dict[str, int]
    physical_group_blocks: dict[str, int]
    group_sources: dict[str, dict[str, tuple[int, ...]]]
    buffers: tuple[CacheBufferSpec, ...]
    request_slots: int
    physical_request_slots: int


def cache_bytes_per_rank(plan: DecodeCachePlan) -> int:
    """Count the physical pool, including pages hidden from the scheduler."""
    return sum(torch.Size(spec.shape[1:]).numel() * torch.empty((), dtype=spec.dtype).element_size()
               for spec in plan.buffers)


def make_post_resident_cache_planner(layers, groups, *, max_seq_len, abi,
                                     utilization, max_request_slots,
                                     reserved_bytes_per_rank, requested_pages=None):
    """Build a V4-style post-resident cache planner for one persistent worker.

    The caller must charge every layer scratch allocation that follows cache
    allocation in ``reserved_bytes_per_rank``. Group capacities remain unset
    until the worker reports HBM; the scheduler sees only non-padding pages.
    """
    if (not isinstance(utilization, (float, int)) or not 0 < utilization <= 1
            or type(max_request_slots) is not int or max_request_slots <= 0
            or type(reserved_bytes_per_rank) is not int or reserved_bytes_per_rank <= 0
            or requested_pages is not None and (type(requested_pages) is not int or requested_pages <= 0)):
        raise ValueError("invalid post-resident cache budget")
    if any(group.num_blocks is not None for group in groups):
        raise ValueError("dynamic cache groups must not fix scheduler block counts")
    window_pages = groups[0].max_blocks_per_seq

    def plan_for_slots(slots):
        return plan_decode_cache_pools(
            layers, groups, max_seq_len=max_seq_len,
            primary_num_blocks=slots * window_pages,
            request_slots=slots, abi=abi)

    one, two = plan_for_slots(1), plan_for_slots(2)
    one_bytes, two_bytes = cache_bytes_per_rank(one), cache_bytes_per_rank(two)
    slot_bytes = two_bytes - one_bytes
    fixed_bytes = one_bytes - slot_bytes
    if slot_bytes <= 0 or fixed_bytes <= 0:
        raise ValueError("cache shape does not scale linearly with request slots")

    def choose(worker):
        budgets = []
        for rank in range(one.buffers[0].shape[0]):
            free_bytes, total_bytes = worker.device_memory_info(rank)
            if not 0 <= free_bytes <= total_bytes:
                raise ValueError("device memory report is inconsistent")
            used_bytes = total_bytes - free_bytes
            budgets.append(int(total_bytes * utilization) - used_bytes - reserved_bytes_per_rank)
        available = min(budgets)
        if available < fixed_bytes + slot_bytes:
            raise MemoryError("post-resident HBM cannot hold one V4.1 cache request slot")
        slots = min(max_request_slots, (available - fixed_bytes) // slot_bytes)
        if requested_pages is not None:
            slots = min(slots, requested_pages // window_pages)
        if slots < 1:
            raise MemoryError("configured cache pages cannot hold one V4.1 request slot")
        return slots, plan_for_slots

    return choose


def allocate_cache_with_retry(worker, requested_slots, plan_for_slots, state_reset, page_resets):
    """Release every cache family before retrying a smaller shared slot count."""
    if type(requested_slots) is not int or requested_slots <= 0:
        raise ValueError("requested cache slots must be positive")
    slots = requested_slots
    while True:
        try:
            return DecodeCachePools(worker, plan_for_slots(slots), state_reset, page_resets)
        except (MemoryError, RuntimeError) as exc:
            if slots == 1:
                raise RuntimeError("V4.1 cache allocation failed at the one-slot minimum") from exc
            slots = max(slots // 2, 1)


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
    physical_group_blocks = {name: blocks + 1 for name, blocks in group_blocks.items()}

    kv_sources = sum(layer.kv_source == layer.layer_id for layer in layers)
    c2a_sources = sum(layer.mode == "c2a_full" for layer in layers)
    index_sources = sum(layer.index_source == layer.layer_id for layer in layers)
    if (abi.N_LAYERS, abi.KV_SOURCE_COUNT, abi.C2A_SOURCE_COUNT,
            abi.INDEX_SOURCE_COUNT, abi.BLOCK_SIZE) != (
                len(layers), kv_sources, c2a_sources, index_sources, 128):
        raise ValueError("decode cache source axes disagree with the layer schedule")
    kv_layers = tuple(layer for layer in layers if layer.kv_source == layer.layer_id)
    index_layers = tuple(layer for layer in layers if layer.index_source == layer.layer_id)
    group_sources = {
        resolved.window: {"window": tuple(range(len(layers)))},
        resolved.c2a: {
            "compressed": tuple(i for i, layer in enumerate(kv_layers) if layer.mode.startswith("c2a_")),
            "index": tuple(i for i, layer in enumerate(index_layers) if layer.mode.startswith("c2a_")),
        },
        resolved.c1a: {
            "compressed": tuple(i for i, layer in enumerate(kv_layers) if layer.mode.startswith("c1a_")),
            "index": tuple(i for i, layer in enumerate(index_layers) if layer.mode.startswith("c1a_")),
        },
    }
    head, index, world = abi.HEAD_DIM, abi.INDEX_DIM, abi.EP_SIZE
    if (world != 8 or any(type(value) is not int or value <= 0 for value in
                       (head, index, abi.STATE_CAPACITY, abi.STATE_WIDTH))
            or head % abi.WINDOW_CACHE_GROUP or head % abi.COMPRESSED_CACHE_GROUP
            or index % abi.INDEX_CACHE_GROUP):
        raise ValueError("decode cache dimensions disagree with TP4/DP2/EP8")
    block = abi.BLOCK_SIZE
    window_pages = physical_group_blocks[resolved.window]
    # One lib source axis covers both compression families, so its stride must
    # fit the larger physical page namespace without remapping scheduler IDs.
    compressed_pages = max(physical_group_blocks[resolved.c2a],
                           physical_group_blocks[resolved.c1a])
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
            world, c2a_sources * (request_slots + 1), abi.STATE_CAPACITY, abi.STATE_WIDTH), torch.float32),
    )
    return DecodeCachePlan(primary_num_blocks, resolved, group_blocks, physical_group_blocks,
                           group_sources, buffers, request_slots, request_slots + 1)


def make_state_reset_buffer(plan: DecodeCachePlan) -> torch.Tensor:
    """Create the worker-inherited zero row used when a request slot is freed."""
    shape = next(spec.shape for spec in plan.buffers if spec.name == "state_cache_pool")
    return torch.zeros((shape[0], *shape[2:]), dtype=torch.float32).share_memory_()


def make_cache_page_reset_buffers(plan: DecodeCachePlan) -> dict[str, torch.Tensor]:
    """Allocate one pre-fork zero page per cache payload/scale field."""
    result = {}
    for spec in plan.buffers:
        if spec.name == "state_cache_pool":
            continue
        page = torch.empty((spec.shape[0], *spec.shape[2:]), dtype=spec.dtype).share_memory_()
        page.view(torch.uint8).zero_()
        if spec.name.endswith("scale_pool"):
            page.view(torch.uint8).fill_(0x38 if spec.name == "compressed_cache_scale_pool" else 127)
        result[spec.name] = page
    return result


class DecodeCachePools:
    """Own cache handles; the caller must wait for all launches before close."""

    def __init__(self, worker, plan: DecodeCachePlan, state_reset: torch.Tensor,
                 page_resets: dict[str, torch.Tensor] | None = None):
        state_spec = next(spec for spec in plan.buffers if spec.name == "state_cache_pool")
        if (not isinstance(state_reset, torch.Tensor) or state_reset.device.type != "cpu"
                or state_reset.dtype != torch.float32 or not state_reset.is_shared()
                or not state_reset.is_contiguous()
                or tuple(state_reset.shape) != (state_spec.shape[0], *state_spec.shape[2:])
                or not bool((state_reset == 0).all())):
            raise ValueError("state reset rows must be shared zero FP32 buffers created before worker startup")
        self.worker = worker
        self.plan = plan
        self.state_reset = state_reset
        self.page_resets = page_resets
        if page_resets is not None:
            expected = {spec.name for spec in plan.buffers if spec.name != "state_cache_pool"}
            if set(page_resets) != expected:
                raise ValueError("cache page reset buffers must cover every cache field")
            for spec in plan.buffers:
                if spec.name == "state_cache_pool":
                    continue
                value = page_resets[spec.name]
                if (not isinstance(value, torch.Tensor) or not value.is_shared()
                        or not value.is_contiguous() or value.dtype != spec.dtype
                        or tuple(value.shape) != (spec.shape[0], *spec.shape[2:])):
                    raise ValueError(f"invalid pre-fork reset page for {spec.name}")
        self.tensors = {}
        self.source_counts = {}
        try:
            for spec in plan.buffers:
                if spec.name.startswith("window_cache_"):
                    group = plan.family_groups.window
                    stride = plan.physical_group_blocks[group]
                    sources = len(plan.group_sources[group]["window"])
                elif spec.name.startswith(("compressed_cache_", "index_cache_")):
                    family = "compressed" if spec.name.startswith("compressed_") else "index"
                    stride = max(plan.physical_group_blocks[group] for group, families in
                                 plan.group_sources.items() if family in families)
                    ids = {slot for families in plan.group_sources.values()
                           for slot in families.get(family, ())}
                    sources = len(ids)
                    if ids != set(range(sources)):
                        raise ValueError(f"{family} cache source IDs must be contiguous")
                else:
                    stride = plan.physical_request_slots
                    sources = spec.shape[1] // stride
                if spec.shape[1] != sources * stride:
                    raise ValueError("cache source axis disagrees with the resident pool")
                self.source_counts[spec.name] = sources
                for source in range(sources):
                    self.tensors[spec.name, source] = alloc_device_buffer(
                        worker, (spec.shape[0], stride, *spec.shape[2:]), spec.dtype,
                        stacked=True)
        except BaseException:
            self.close()
            raise

    def view(self, name: str, source: int):
        """Borrow one layer/producer pool with the same page IDs as decode."""
        if name not in self.source_counts:
            raise ValueError(f"unknown V4.1 cache pool: {name}")
        if type(source) is not int or not 0 <= source < self.source_counts[name]:
            raise ValueError("cache source is outside the resident pool")
        return self.tensors[name, source]

    def prefill_views(self, layer, *, kv_sources, index_sources, c2a_sources):
        """Resolve a layer's physical producer pools for its lib prefill entry."""
        result = {
            "window_cache": self.view("window_cache_pool", layer.layer_id),
            "window_cache_scale": self.view("window_cache_scale_pool", layer.layer_id),
        }
        if layer.mode == "swa":
            return result
        kv_slot = kv_sources.index(layer.kv_source)
        index_slot = index_sources.index(layer.index_source)
        for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale"):
            slot = kv_slot if name.startswith("compressed_") else index_slot
            result[name] = self.view(name + "_pool", slot)
        if layer.mode == "c2a_full":
            result["state_cache"] = self.view("state_cache_pool", c2a_sources.index(layer.layer_id))
        return result

    def reset_state_slot(self, partition: int, slot: int):
        """Clear the three C2A pending-pair rows for one DP request slot."""
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("invalid DP partition")
        if type(slot) is not int or not 0 <= slot < self.plan.physical_request_slots:
            raise ValueError("invalid compressor state slot")
        row = self.state_reset[0]
        row_bytes = row.numel() * row.element_size()
        for rank in range(partition * 4, (partition + 1) * 4):
            source = self.state_reset[rank]
            for index in range(self.source_counts["state_cache_pool"]):
                state = self.view("state_cache_pool", index)
                self.worker.copy_to(
                    state.shards[rank].data_ptr, source.data_ptr(), row_bytes,
                    dst_offset=slot * row_bytes,
                    worker_id=rank,
                )

    def reset_request(self, owner, *, padding=False):
        """Clear only this request's physical pages and C2A pending state."""
        if self.page_resets is None:
            raise ValueError("request reset requires pre-fork cache page buffers")
        partition = owner.partition
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("invalid DP partition")
        if set(owner.pages) != set(self.plan.group_sources):
            raise ValueError("request page families disagree with the cache plan")
        if (type(owner.slot) is not int or
                (padding and owner.slot != self.plan.request_slots) or
                (not padding and not 0 <= owner.slot < self.plan.request_slots)):
            raise ValueError("request state slot is outside its cache family")
        fields = {
            "window": ("window_cache_pool", "window_cache_scale_pool"),
            "compressed": ("compressed_cache_pool", "compressed_cache_scale_pool"),
            "index": ("index_cache_pool", "index_cache_scale_pool"),
        }
        specs = {spec.name: spec for spec in self.plan.buffers}
        for group, pages in owner.pages.items():
            limit = self.plan.physical_group_blocks[group] if padding else self.plan.group_blocks[group]
            if any(type(page) is not int or not 0 <= page < limit
                   for page in pages):
                raise ValueError("request page is outside its cache family")
            if padding and tuple(pages) != (self.plan.group_blocks[group],):
                raise ValueError("padding must use the dedicated cache page")
            for family, sources in self.plan.group_sources[group].items():
                for name in fields[family]:
                    spec = specs[name]
                    zero = self.page_resets[name]
                    page_bytes = zero[0].numel() * zero.element_size()
                    for rank in range(partition * 4, (partition + 1) * 4):
                        for source in sources:
                            target = self.view(name, source)
                            for page in pages:
                                self.worker.copy_to(
                                    target.shards[rank].data_ptr, zero[rank].data_ptr(), page_bytes,
                                    dst_offset=page * page_bytes, worker_id=rank,
                                )
        self.reset_state_slot(partition, owner.slot)

    def reset_padding(self, partition):
        from types import SimpleNamespace

        owner = SimpleNamespace(
            partition=partition, slot=self.plan.request_slots,
            pages={name: (blocks,) for name, blocks in self.plan.group_blocks.items()},
        )
        self.reset_request(owner, padding=True)

    def close(self):
        for tensor in reversed(tuple(self.tensors.values())):
            self.worker.free_stacked_tensor(tensor)
        self.tensors.clear()
