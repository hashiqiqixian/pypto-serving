# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 cache addresses and request transactions over scheduler-owned pages.

No page allocator or tensor storage lives here. The existing KvCacheManager owns
physical pages; a backend owns tensor writes and snapshots of compressor/Engram
state. Commit/abort change visibility and invalidate temporary index results.
The project-defined ``v41_packed_v1`` payload/scale layout is not the reference's
physical allocation: the reference fake-quantizes values held in BF16 buffers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypto_serving.config.types import KVCacheGroupSpec, RuntimeConfig

    from .config import DeepSeekV41Config


CACHE_FORMAT = "v41_packed_v1"


class CacheStateError(ValueError):
    """An address, request lifetime, or cache transaction is invalid."""


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CacheStateError(f"{name} must be an integer >= {minimum}")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class CacheGroupLayout:
    """One V4.1 namespace, with page capacity measured in source tokens."""

    name: str
    layer_id: int
    kind: str
    page_size: int
    compress_ratio: int
    row_bytes: int
    max_blocks_per_seq: int
    sliding_window: int | None = None

    @property
    def rows_per_page(self) -> int:
        return self.page_size // self.compress_ratio

    @property
    def page_size_bytes(self) -> int:
        return self.rows_per_page * self.row_bytes


def build_v41_cache_layouts(
    config: DeepSeekV41Config,
    page_size: int,
    max_seq_len: int,
    *,
    max_chunk_tokens: int = 8192,
) -> tuple[CacheGroupLayout, ...]:
    """Define the project's v41_packed_v1 storage contract from config.

    Main KV scales are E4M3 per 16 values; index-key scales are UE8M0 per
    32 values. SWA uses FP8 with UE8M0 scales per 32 values. Index results
    and candidates belong to a transaction and have no persistent page pool.
    These byte counts are layout requirements, not measured HBM use or the
    reference's BF16 fake-quant buffers. Backends must explicitly support this ABI.
    """
    _integer(page_size, "page_size", 1)
    _integer(max_seq_len, "max_seq_len", 1)
    _integer(max_chunk_tokens, "max_chunk_tokens", 1)
    if max_seq_len > config.max_position_embeddings:
        raise CacheStateError("max_seq_len exceeds config.max_position_embeddings")
    if config.head_dim % 32 or config.index_head_dim % 32:
        raise CacheStateError("cache vector dimensions must fit their 16/32-value quantization blocks")
    full_blocks = _ceil_div(max_seq_len, page_size)
    # Keep the old attention tail and the complete pending chunk in separate
    # ring positions, including a page for arbitrary alignment at its boundary.
    chunk = min(max_chunk_tokens, max_seq_len)
    ring_blocks = min(full_blocks, _ceil_div(config.sliding_window - 1 + chunk, page_size) + 1)
    window = _ceil_div(min(config.sliding_window, max_seq_len), page_size) * page_size
    layouts = []
    for source in config.kv_source_layer_ids:
        ratio = config.compress_ratios[source]
        if page_size % ratio:
            raise CacheStateError(f"page_size must be divisible by compress_ratio={ratio} at layer {source}")
        layouts.extend(
            (
                CacheGroupLayout(
                    f"main_kv.{source}",
                    source,
                    "main_kv",
                    page_size,
                    ratio,
                    config.head_dim // 2 + config.head_dim // 16,
                    full_blocks,
                ),
                CacheGroupLayout(
                    f"index_k.{source}",
                    source,
                    "index_k",
                    page_size,
                    ratio,
                    config.index_head_dim // 2 + config.index_head_dim // 32,
                    full_blocks,
                ),
            )
        )
    layouts.extend(
        CacheGroupLayout(
            f"swa.{layer_id}",
            layer_id,
            "swa",
            page_size,
            1,
            config.head_dim + config.head_dim // 32,
            ring_blocks,
            window,
        )
        for layer_id in range(config.num_hidden_layers)
    )
    return tuple(layouts)


def build_v41_cache_group_specs(
    config: DeepSeekV41Config,
    page_size: int,
    max_seq_len: int,
    total_pages: int | None = None,
    *,
    max_chunk_tokens: int = 8192,
    num_partitions: int = 1,
) -> tuple[KVCacheGroupSpec, ...]:
    """Build existing scheduler cache specs, importing runtime types only on demand.

    ``total_pages`` is the first group's capacity in each partition. Every
    pool uses the same number of complete request slots, matching the existing
    allocator's ``primary_num_blocks`` scaling. None delegates sizing to it.
    """
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec

    _integer(num_partitions, "num_partitions", 1)
    layouts = build_v41_cache_layouts(config, page_size, max_seq_len, max_chunk_tokens=max_chunk_tokens)
    slots = None
    if total_pages is not None:
        _integer(total_pages, "total_pages", 1)
        stride = layouts[0].max_blocks_per_seq
        if total_pages % stride:
            raise CacheStateError("total_pages must be a multiple of the primary group's max_blocks_per_seq")
        slots = total_pages // stride
    return tuple(
        KVCacheGroupSpec(
            name=layout.name,
            layer_indices=(layout.layer_id,),
            spec=KVCacheSpec(layout.page_size, layout.page_size_bytes, layout.compress_ratio),
            max_blocks_per_seq=layout.max_blocks_per_seq,
            num_blocks=None if slots is None else slots * layout.max_blocks_per_seq,
            num_partitions=num_partitions,
            sliding_window=layout.sliding_window,
        )
        for layout in layouts
    )


def configure_v41_runtime(config: DeepSeekV41Config, runtime: RuntimeConfig) -> RuntimeConfig:
    """Resolve the same cache/chunk contract before scheduler and worker initialization."""
    chunk = runtime.max_prefill_tokens_per_request or min(
        8192, runtime.max_num_batched_tokens, runtime.max_seq_len
    )
    groups = build_v41_cache_group_specs(
        config, runtime.page_size, runtime.max_seq_len, runtime.total_kv_pages, max_chunk_tokens=chunk
    )
    if runtime.kv_cache_groups and runtime.kv_cache_groups != groups:
        raise ValueError("V4.1 runtime cache groups disagree with checkpoint topology")
    if runtime.num_speculative_tokens:
        raise ValueError("V4.1 speculative serving kernels are not implemented")
    return replace(
        runtime,
        kv_cache_groups=groups,
        max_prefill_tokens_per_request=chunk,
        supports_chunked_prefill_with_speculation=False,
    )


@dataclass(frozen=True)
class CacheLease:
    """One worker registration; object identity distinguishes reused request IDs."""

    request_id: str
    generation: int
    partition: int


@dataclass(frozen=True)
class CacheBindingSnapshot:
    """Bindings before a dispatch, excluding tensor contents and committed positions."""

    lease: CacheLease
    pages: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class PhysicalSlot:
    """A row in the backend's group-specific physical page storage."""

    page_id: int
    row_offset: int
    logical_row: int
    group_name: str


@dataclass(frozen=True)
class CacheTransaction:
    """An active contiguous chunk; all methods reject use after commit/abort."""

    lease: CacheLease
    start_pos: int
    end_pos: int
    _cache: V41CacheState = field(repr=False, compare=False)
    _indexes: dict[int, Any] = field(default_factory=dict, repr=False, compare=False)
    _candidates: dict[int, Any] = field(default_factory=dict, repr=False, compare=False)
    _status: str = field(default="pending", repr=False, compare=False)

    def map_position(
        self, group_name: str, source_token_position: int, *, write: bool = False
    ) -> PhysicalSlot:
        return self._cache._map_position(self, group_name, source_token_position, write=write)

    def write_slots(self, group_name: str) -> tuple[PhysicalSlot, ...]:
        """Return exactly the rows completed in this chunk, including a carried partial group."""
        self._cache._active_transaction(self)
        group = self._cache._group(group_name)
        return tuple(
            self.map_position(group_name, row * group.compress_ratio, write=True)
            for row in range(self.start_pos // group.compress_ratio, self.end_pos // group.compress_ratio)
        )

    def visible_rows(self, group_name: str, query_position: int) -> range:
        """Return stored logical rows visible under this query's causal/window boundary."""
        self._cache._active_transaction(self)
        group = self._cache._group(group_name)
        _integer(query_position, "query_position")
        if not self.start_pos <= query_position < self.end_pos:
            raise CacheStateError("query_position is outside the pending chunk")
        start = max(0, query_position + 1 - self._cache.config.sliding_window) if group.kind == "swa" else 0
        return (
            range(start, query_position + 1)
            if group.kind == "swa"
            else range((query_position + 1) // group.compress_ratio)
        )

    def publish_index(self, layer_id: int, value: Any) -> None:
        self._cache._active_transaction(self)
        plan = self._cache._plan(layer_id)
        if not plan.owns_index_results:
            raise CacheStateError(f"layer {layer_id} does not own index results")
        if layer_id in self._indexes:
            raise CacheStateError(f"layer {layer_id} already published index results")
        self._indexes[layer_id] = value

    def index_for(self, layer_id: int) -> Any:
        self._cache._active_transaction(self)
        source = self._cache._plan(layer_id).index_source_layer_id
        if source not in self._indexes:
            raise CacheStateError(f"index source {source} has not published for layer {layer_id}")
        return self._indexes[source]

    def publish_candidates(self, layer_id: int, value: Any) -> None:
        self._cache._active_transaction(self)
        self._cache._plan(layer_id)
        if layer_id != self._cache.config.candidate_source_layer_id:
            raise CacheStateError(f"layer {layer_id} does not own hierarchical candidates")
        if layer_id in self._candidates:
            raise CacheStateError("candidates have already been published in this transaction")
        self._candidates[layer_id] = value

    def candidates_for(self, layer_id: int) -> Any:
        self._cache._active_transaction(self)
        source = self._cache._plan(layer_id).candidate_source_layer_id
        if source not in self._candidates:
            raise CacheStateError(f"candidate source {source} has not published for layer {layer_id}")
        return self._candidates[source]


@dataclass
class _RequestCache:
    lease: CacheLease
    pages: dict[str, tuple[int, ...]] = field(default_factory=dict)
    valid_length: int = 0
    retained_start: int = 0
    transaction: CacheTransaction | None = None


class V41CacheState:
    """Validate ownership and causal visibility; reuse the engine's page allocator.

    A single worker drives this object serially. A backend must finish or cancel
    device writes before abort/release, and restore its own auxiliary state on
    rollback. Prefix sharing is intentionally rejected until it has a separate
    immutable-page/COW contract.
    """

    cache_format = CACHE_FORMAT

    def __init__(
        self,
        config: DeepSeekV41Config,
        *,
        page_size: int,
        max_seq_len: int,
        max_chunk_tokens: int = 8192,
        num_partitions: int = 1,
    ) -> None:
        self.config = config
        self.max_seq_len = max_seq_len
        self.max_chunk_tokens = min(max_chunk_tokens, max_seq_len)
        self.num_partitions = _integer(num_partitions, "num_partitions", 1)
        groups = build_v41_cache_layouts(config, page_size, max_seq_len, max_chunk_tokens=max_chunk_tokens)
        self.groups = groups
        self.layouts = MappingProxyType({group.name: group for group in groups})
        self.layer_plan = config.layer_plan
        self._requests: dict[str, _RequestCache] = {}
        self._owners: dict[tuple[int, str, int], CacheLease] = {}

    def create(self, request_id: str, *, generation: int, partition: int = 0) -> CacheLease:
        if not isinstance(request_id, str) or not request_id:
            raise CacheStateError("request_id must be a nonempty string")
        _integer(generation, "generation")
        _integer(partition, "partition")
        if partition >= self.num_partitions:
            raise CacheStateError("partition exceeds cache namespace count")
        if request_id in self._requests:
            raise CacheStateError(f"request {request_id!r} is already registered")
        lease = CacheLease(request_id, generation, partition)
        self._requests[request_id] = _RequestCache(lease)
        return lease

    def _request(self, lease: CacheLease) -> _RequestCache:
        state = self._requests.get(lease.request_id)
        if state is None or state.lease is not lease:
            raise CacheStateError("stale or unregistered cache lease")
        return state

    def _active_transaction(self, transaction: CacheTransaction) -> _RequestCache:
        state = self._request(transaction.lease)
        if state.transaction is not transaction:
            raise CacheStateError("stale or inactive cache transaction")
        return state

    def _group(self, name: str) -> CacheGroupLayout:
        if name not in self.layouts:
            raise CacheStateError(f"unknown V4.1 cache group: {name}")
        return self.layouts[name]

    def _plan(self, layer_id: int):
        _integer(layer_id, "layer_id")
        if layer_id >= len(self.layer_plan):
            raise CacheStateError("layer_id is outside the backbone")
        return self.layer_plan[layer_id]

    def group_for_layer(self, layer_id: int, kind: str) -> str:
        plan = self._plan(layer_id)
        if kind == "swa":
            return f"swa.{layer_id}"
        if kind not in ("main_kv", "index_k") or plan.kv_source_layer_id is None:
            raise CacheStateError(f"layer {layer_id} has no {kind} cache")
        return f"{kind}.{plan.kv_source_layer_id}"

    def valid_length(self, lease: CacheLease) -> int:
        return self._request(lease).valid_length

    def snapshot_bindings(self, lease: CacheLease) -> CacheBindingSnapshot:
        state = self._request(lease)
        if state.transaction is not None:
            raise CacheStateError("cannot snapshot bindings during a transaction")
        return CacheBindingSnapshot(lease, MappingProxyType(dict(state.pages)))

    def restore_bindings_many(self, snapshots: Sequence[CacheBindingSnapshot]) -> None:
        """Restore ownership atomically after abort, including pages released within a batch.

        Keep the conservative retained floor: pending SWA writes may have destroyed
        older rows even though the committed attention tail was protected.
        """
        states = [self._request(snapshot.lease) for snapshot in snapshots]
        if len({id(state) for state in states}) != len(states) or any(
            state.transaction is not None for state in states
        ):
            raise CacheStateError("binding restore requires distinct, inactive request transactions")
        participants = {id(state.lease) for state in states}
        restored = {}
        for snapshot in snapshots:
            for name, pages in snapshot.pages.items():
                self._group(name)
                for page in pages:
                    key = (snapshot.lease.partition, name, page)
                    owner = self._owners.get(key)
                    if key in restored or (owner is not None and id(owner) not in participants):
                        raise CacheStateError("binding snapshot conflicts with another request's pages")
                    restored[key] = snapshot.lease
        for state in states:
            for name, pages in state.pages.items():
                for page in pages:
                    self._owners.pop((state.lease.partition, name, page), None)
        self._owners.update(restored)
        for state, snapshot in zip(states, snapshots):
            state.pages = dict(snapshot.pages)

    @property
    def active_request_count(self) -> int:
        return len(self._requests)

    def _live_slots(self, group: CacheGroupLayout, length: int) -> set[int]:
        if not length:
            return set()
        if group.kind != "swa":
            completed_rows = length // group.compress_ratio
            return set(range(_ceil_div(completed_rows, group.rows_per_page)))
        start = max(0, length - self.config.sliding_window)
        return {
            block % group.max_blocks_per_seq
            for block in range(start // group.page_size, _ceil_div(length, group.page_size))
        }

    def bind(self, lease: CacheLease, block_ids_by_group: Mapping[str, Sequence[int]]) -> None:
        """Atomically accept scheduler tables without replacing any committed live page."""
        state = self._request(lease)
        if state.transaction is not None:
            raise CacheStateError("cannot rebind pages during a pending transaction")
        if set(block_ids_by_group) != set(self.layouts):
            raise CacheStateError("scheduler page tables must contain exactly the V4.1 cache groups")
        pages = {}
        floor = state.retained_start
        for name, group in self.layouts.items():
            incoming = tuple(block_ids_by_group[name])
            if len(incoming) > group.max_blocks_per_seq:
                raise CacheStateError(f"page table exceeds per-request capacity: {name}")
            for page_id in incoming:
                _integer(page_id, f"{name} page_id")
            if len(set(incoming)) != len(incoming):
                raise CacheStateError(f"duplicate physical page within group {name}")
            old = state.pages.get(name, ())
            for slot in self._live_slots(group, state.valid_length):
                if slot >= len(incoming) or slot >= len(old) or incoming[slot] != old[slot]:
                    raise CacheStateError(f"cannot replace committed live page in group {name}, slot {slot}")
            if group.kind == "swa" and state.valid_length:
                last_block = (state.valid_length - 1) // group.page_size
                for slot, old_page in enumerate(old):
                    if slot >= len(incoming) or incoming[slot] != old_page:
                        old_block = last_block - (last_block - slot) % group.max_blocks_per_seq
                        if old_block >= 0:
                            floor = max(floor, min(state.valid_length, (old_block + 1) * group.page_size))
            for page_id in incoming:
                owner = self._owners.get((lease.partition, name, page_id))
                if owner is not None and owner is not lease:
                    raise CacheStateError(f"page {page_id} in {name} is owned by another live request")
            pages[name] = incoming
        for name, old in state.pages.items():
            for page_id in old:
                self._owners.pop((lease.partition, name, page_id), None)
        for name, incoming in pages.items():
            for page_id in incoming:
                self._owners[lease.partition, name, page_id] = lease
        state.pages = pages
        state.retained_start = floor

    def prepare(self, lease: CacheLease, start_pos: int, end_pos: int) -> CacheTransaction:
        state = self._request(lease)
        _integer(start_pos, "start_pos")
        _integer(end_pos, "end_pos", 1)
        if state.transaction is not None:
            raise CacheStateError("request already has a pending transaction")
        if start_pos != state.valid_length or end_pos <= start_pos:
            raise CacheStateError("chunk must append contiguously to the committed valid length")
        if end_pos > self.max_seq_len or end_pos - start_pos > self.max_chunk_tokens:
            raise CacheStateError("chunk exceeds max sequence length or max_chunk_tokens")
        for name, group in self.layouts.items():
            required = _ceil_div(end_pos, group.page_size)
            if group.kind == "swa":
                required = min(required, group.max_blocks_per_seq)
            if len(state.pages.get(name, ())) < required:
                raise CacheStateError(f"scheduler has not supplied enough pages for {name}")
        # Treat every planned write as potentially performed, including on abort.
        # The ring reservation guarantees these overwrites cannot destroy the old
        # committed attention tail. Older rollback targets may become unavailable.
        for group in self.groups:
            if group.kind == "swa":
                state.retained_start = max(
                    state.retained_start, end_pos - group.max_blocks_per_seq * group.page_size
                )
        transaction = CacheTransaction(lease, start_pos, end_pos, self)
        state.transaction = transaction
        return transaction

    def _map_position(
        self, transaction: CacheTransaction, name: str, position: int, *, write: bool
    ) -> PhysicalSlot:
        state = self._active_transaction(transaction)
        group = self._group(name)
        _integer(position, "source_token_position")
        row = position // group.compress_ratio
        complete_at = (row + 1) * group.compress_ratio
        if complete_at > transaction.end_pos:
            raise CacheStateError("cache row is incomplete or beyond the pending chunk")
        if write and complete_at <= transaction.start_pos:
            raise CacheStateError("write would overwrite a previously committed cache row")
        if group.kind == "swa" and position < max(
            state.retained_start, transaction.start_pos - self.config.sliding_window
        ):
            raise CacheStateError("SWA row is outside the retained attention tail")
        logical_block, offset = divmod(row, group.rows_per_page)
        table_slot = logical_block % group.max_blocks_per_seq if group.kind == "swa" else logical_block
        page_ids = state.pages[name]
        if table_slot >= len(page_ids):
            raise CacheStateError(f"unallocated logical cache page in {name}")
        return PhysicalSlot(page_ids[table_slot], offset, row, name)

    def _close(self, state: _RequestCache, status: str) -> None:
        transaction = state.transaction
        if transaction is not None:
            object.__setattr__(transaction, "_status", status)
            transaction._indexes.clear()
            transaction._candidates.clear()
        state.transaction = None

    def commit(self, transaction: CacheTransaction) -> None:
        self.commit_many((transaction,))

    def validate_commit_many(self, transactions: Sequence[CacheTransaction]) -> None:
        """Prevalidate a batch before backend commit; no visibility is changed."""
        seen = set()
        for transaction in transactions:
            state = self._active_transaction(transaction)
            identity = id(state)
            if identity in seen:
                raise CacheStateError("batch contains a duplicate request transaction")
            seen.add(identity)

    def commit_many(self, transactions: Sequence[CacheTransaction]) -> None:
        """Validate the entire batch, then publish all host endpoints together.

        The serial worker must prevent host mutations between its prevalidation,
        backend commit, and this call. No backend work occurs here.
        """
        transactions = tuple(transactions)
        self.validate_commit_many(transactions)
        states = tuple(self._request(transaction.lease) for transaction in transactions)
        for state, transaction in zip(states, transactions):
            state.valid_length = transaction.end_pos
        for state in states:
            self._close(state, "committed")

    def abort(self, transaction: CacheTransaction) -> None:
        """Discard pending visibility after the backend has cancelled/completed writes."""
        if transaction._cache is self and transaction._status == "aborted":
            return
        state = self._active_transaction(transaction)
        self._close(state, "aborted")

    def rollback(self, lease: CacheLease, end_pos: int) -> None:
        """Hide a committed suffix, requiring the target's SWA tail to still exist.

        This never restores backend tensor or auxiliary state. The runner must
        restore its compressor/Engram snapshots before using the new endpoint.
        Zero explicitly resets visibility and permits full recomputation.
        """
        state = self._request(lease)
        _integer(end_pos, "end_pos")
        if state.transaction is not None:
            raise CacheStateError("abort the pending transaction before rollback")
        if end_pos > state.valid_length:
            raise CacheStateError("rollback cannot extend the committed sequence")
        if end_pos and max(0, end_pos - self.config.sliding_window) < state.retained_start:
            raise CacheStateError("rollback target requires SWA rows that are no longer retained")
        state.valid_length = end_pos
        if not end_pos:
            state.retained_start = 0

    def release(self, lease: CacheLease) -> bool:
        """Drop worker metadata; an old completion cannot release a new registration.

        The runner first stops backend writes. Actual physical pages remain
        owned by the engine and are returned through its existing release path.
        """
        state = self._requests.get(lease.request_id)
        if state is None or state.lease is not lease:
            return False
        self._close(state, "released")
        for name, page_ids in state.pages.items():
            for page_id in page_ids:
                self._owners.pop((lease.partition, name, page_id), None)
        del self._requests[lease.request_id]
        return True
