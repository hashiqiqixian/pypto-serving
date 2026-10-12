# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 npu runner."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Callable

import torch

from pypto_serving.config.types import DecodeResult, KVCacheGroupSpec, PrefillResult

from .task_args import RequestLedger, validate_cache_groups
from .weight_loader import LayerPlan, RankPlacement, V41ExecutionPlan, V41WeightLoader


# composite
class MissingCompositeInterface(NotImplementedError):
    """The selected lib revision cannot execute the requested serving segment."""


@dataclass(frozen=True)
class BuildOptions:
    """Worker-selected compilation settings passed unchanged to the adapter."""
    platform: str = "a5"
    pypto_build_dir: str | None = None
    use_compile_cache: bool = False


@dataclass(frozen=True)
class LayerState:
    """Opaque device state carried between composites, never converted on host."""
    residual: object
    pre_mix: object
    layout: str = "tp_replicated"


@dataclass(frozen=True)
class CompositeBindings:
    """Explicit adapter seam for a verified lib revision.

    Entries consume (LayerPlan, LayerState, step, resources, weights) and return
    LayerState. initialize consumes (embeddings, step, resources); output consumes
    (state, step, resources) and returns host logits in original request order.
    The adapter owns device uploads and lib ABI binding. Serving never expands
    packed FP4 or calls the sub-operators of a complete layer.
    allocate consumes (plan, device_ids, runtime, BuildOptions), including the
    worker's build directory and compile-cache choice, and returns (resources,
    num_pages). It also prepares global weights needed by initialize/output.

    No default implementation fabricates results. A supplied adapter must handle
    all DP partitions collectively, including empty partitions, and retain input
    objects until wait() completes. reset_request must clear every persistent
    cache and compressor slot for that request before returning.
    """
    revision: str
    entries: Mapping[tuple[str, str], Callable]
    initialize: Callable
    output: Callable
    allocate: Callable
    prepare_weights: Callable
    reset_request: Callable
    wait: Callable
    close: Callable
    cache_groups: tuple = ()
    input_layout: str = "tp_replicated"
    output_layout: str = "tp_replicated"
    decode_backbone: Callable | None = None
    prefill_backbone: Callable | None = None

    def require(self, layers: tuple[LayerPlan, ...], placement: RankPlacement) -> None:
        if not self.revision:
            raise ValueError("composite bindings must identify the validated lib revision")
        phases = tuple(phase for phase, backbone in (
            ("prefill", self.prefill_backbone), ("decode", self.decode_backbone),
        ) if not callable(backbone))
        missing = sorted({f"{phase}/{layer.mode}" for phase in phases
                          for layer in layers if not callable(self.entries.get((phase, layer.mode)))})
        if missing:
            raise MissingCompositeInterface("missing complete layer composites: " + ", ".join(missing))
        for name in ("initialize", "output", "allocate", "prepare_weights", "reset_request", "wait", "close"):
            if not callable(getattr(self, name)):
                raise MissingCompositeInterface(f"missing composite resource/output operation: {name}")
        if self.input_layout not in ("tp_replicated", "tp_local_token") or self.output_layout != self.input_layout:
            raise MissingCompositeInterface("layer entries must preserve their declared residual/pre_mix layout")
        if (placement.tp_size, placement.dp_size, placement.ep_size) != (4, 2, 8):
            raise ValueError("V4.1 serving currently targets TP4/DP2/EP8")


def load_composite_bindings(model_dir=None, max_seq_len=None) -> CompositeBindings:
    """Resolve the model-specific contract without opening a device."""
    if model_dir is None or max_seq_len is None:
        raise ValueError("V4.1 composite bindings require a model directory and context length")
    return make_production_bindings(model_dir, max_seq_len)


# input_preparation
def lookup_token_embeddings(
    loaders: Sequence[V41WeightLoader],
    token_ids: torch.Tensor,
    *,
    max_prepare_bytes: int = 256 << 20,
    max_rows_per_read: int = 64,
) -> torch.Tensor:
    """Return owned CPU BF16 rows with shape ``(*token_ids.shape, hidden_size)``.

    Supply one complete set of TP vocabulary shards from the same checkpoint,
    in any order. DP groups share this checkpoint table; token order and all
    leading dimensions are preserved, without duplicating rows for TP ranks.
    IDs must be CPU int32/int64 with at least one dimension. Empty inputs are
    allowed, but padding sentinels such as -1 are not vocabulary indices.

    Repeated IDs are read once. Adjacent unique rows in the same TP shard are
    read together, with at most max_rows_per_read rows per load_rows call.
    max_prepare_bytes bounds conservative preparation tensor storage, including
    the result, unique rows, indexing scratch and a read buffer; the loader's
    separate max_load_bytes still bounds its own read/conversion operation.
    Neither budget represents total process memory or previously returned data.
    """
    for value, name in ((max_prepare_bytes, "max_prepare_bytes"),
                        (max_rows_per_read, "max_rows_per_read")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    shards = tuple(loaders)
    if not shards or any(not isinstance(loader, V41WeightLoader) for loader in shards):
        raise ValueError("embedding lookup requires V41WeightLoader TP shards")
    first = shards[0]
    if len(shards) != first.tp_size or {loader.tp_rank for loader in shards} != set(range(first.tp_size)):
        raise ValueError("embedding lookup requires exactly one loader per TP shard")
    if any(loader.model_dir != first.model_dir or loader.config != first.config
           or loader.tp_size != first.tp_size or loader.ep_size != first.ep_size for loader in shards):
        raise ValueError("embedding TP shards must use the same checkpoint and parallel sizes")
    if not isinstance(token_ids, torch.Tensor) or token_ids.layout != torch.strided:
        raise ValueError("token_ids must be a strided CPU tensor")
    if token_ids.device.type != "cpu" or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a CPU int32/int64 tensor")
    if token_ids.ndim == 0:
        raise ValueError("token_ids must have at least one dimension")

    hidden = first.config.hidden_size
    count = token_ids.numel()
    if count == 0:
        return torch.empty((*token_ids.shape, hidden), dtype=torch.bfloat16)
    # Bound allocations before flattening or sorting. Worst case every ID is
    # unique; count*128 allows conservative int64 indexing/sorting scratch.
    row_bytes = hidden * 2
    estimate = (2 * count + min(count, max_rows_per_read)) * row_bytes + count * 128
    if estimate > max_prepare_bytes:
        raise ValueError(
            f"embedding preparation requires estimated {estimate} bytes; budget={max_prepare_bytes}"
        )
    if bool(((token_ids < 0) | (token_ids >= first.config.vocab_size)).any()):
        raise ValueError("token_ids must be inside the checkpoint vocabulary")

    flat = token_ids.reshape(-1).to(torch.int64)
    unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    unique_rows = torch.empty((unique.numel(), hidden), dtype=torch.bfloat16)
    by_rank = {loader.tp_rank: loader for loader in shards}
    shard_size = first.config.vocab_size // first.tp_size
    start = 0
    while start < unique.numel():
        token = int(unique[start])
        rank, local_row = divmod(token, shard_size)
        stop = start + 1
        limit = min(unique.numel(), start + max_rows_per_read, start + shard_size - local_row)
        while stop < limit and int(unique[stop]) == token + stop - start:
            stop += 1
        bundle = by_rank[rank].load_rows("embed.weight", local_row, local_row + stop - start)
        unique_rows[start:stop].copy_(bundle.weight)
        del bundle
        start = stop
    return unique_rows.index_select(0, inverse).reshape(*token_ids.shape, hidden)


# metadata
_MAX_REQUESTS_PER_DP = 32
_MAX_PREFILL_TOKENS_PER_DP = 4096


def _validate_dp_capacity(partitions, token_counts, *, phase):
    requests = [0, 0]
    tokens = [0, 0]
    for partition, count in zip(partitions, token_counts):
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("requests require an explicit DP cache partition in [0, 2)")
        requests[partition] += 1
        tokens[partition] += count
    if any(count > _MAX_REQUESTS_PER_DP for count in requests):
        raise ValueError("V4.1 lib supports at most 32 requests per DP partition")
    if phase == "prefill" and any(count > _MAX_PREFILL_TOKENS_PER_DP for count in tokens):
        raise ValueError("V4.1 lib supports at most 4096 prefill tokens per DP partition")


def validate_page_tables(rows, partitions, ends, groups):
    """Copy private full-history page tables and reject active request aliases.

    Physical IDs are local to each DP partition. Without an explicit num_blocks
    the device allocator must check its actual capacity before launching work.
    Rolling groups require a position-to-ring-slot ABI and are not guessed here.
    """
    if not groups or any(not isinstance(group, KVCacheGroupSpec) for group in groups):
        raise MissingCompositeInterface("V4.1 requires explicit grouped cache specifications")
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("cache group names must be unique")
    if any(group.num_partitions != 2 for group in groups):
        raise ValueError("V4.1 cache groups require two DP partitions")
    if any(group.sliding_window is not None for group in groups):
        raise MissingCompositeInterface("rolling cache page lowering requires a verified composite contract")
    if len(rows) != len(partitions) or len(rows) != len(ends):
        raise ValueError("page tables, partitions and request extents must have matching rows")
    required = {group.name for group in groups}
    owners, result = set(), []
    for pages, partition, end in zip(rows, partitions, ends):
        if type(partition) is not int or not 0 <= partition < 2:
            raise ValueError("requests require an explicit DP cache partition in [0, 2)")
        if type(end) is not int or end <= 0:
            raise ValueError("cache request extent must be a positive integer")
        if not isinstance(pages, Mapping) or set(pages) != required:
            raise ValueError("request page tables must match the declared cache groups")
        copied = {}
        for group in groups:
            ids = pages[group.name]
            if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
                raise ValueError("a cache page table must be a sequence of physical IDs")
            ids = tuple(ids)
            if any(type(page) is not int or page < 0 for page in ids):
                raise ValueError("physical page IDs must be nonnegative integers")
            if group.num_blocks is not None and any(page >= group.num_blocks for page in ids):
                raise ValueError("physical page ID exceeds the configured cache pool")
            required_pages = math.ceil(end / group.spec.token_capacity)
            if not required_pages <= len(ids) <= group.max_blocks_per_seq:
                raise ValueError("cache page table does not cover the request extent within its capacity")
            if len(set(ids)) != len(ids):
                raise ValueError("a full-history request must not alias its own cache pages")
            addresses = {(partition, group.name, page) for page in ids}
            if owners.intersection(addresses):
                raise ValueError("active requests must not share writable cache pages")
            owners.update(addresses)
            copied[group.name] = ids
        result.append(copied)
    return tuple(result)


def prefill_requests(batch, config, runtime, groups):
    """Build RequestLedger.begin_prefill items from the shared packed batch.

    seq_lens is the end of the current chunk; prompt_lens is the original total
    prompt length. Confusing them would mark the first chunk as terminal and
    prevent subsequent chunks from continuing their persistent state.
    """
    count = len(batch.request_ids)
    if not count or count > runtime.max_batch_size:
        raise ValueError("prefill request count exceeds the configured batch capacity")
    if any(not isinstance(key, str) or not key for key in batch.request_ids):
        raise ValueError("request IDs must be nonempty strings")
    if len(set(batch.request_ids)) != count:
        raise ValueError("prefill requests must have distinct IDs")
    for name in ("chunk_lens", "chunk_offsets", "chunk_starts", "seq_lens", "prompt_lens",
                 "block_ids_by_group", "cache_partitions"):
        if len(getattr(batch, name)) != count:
            raise ValueError(f"prefill {name} must contain one entry per request")
    tokens = batch.token_ids
    if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu" or tokens.ndim != 1:
        raise ValueError("prefill token IDs must be a flat CPU tensor")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("prefill token IDs must use int32 or int64")
    if tokens.numel() > runtime.max_num_batched_tokens:
        raise ValueError("prefill tokens exceed the configured dispatch capacity")
    token_ids = tokens.tolist()
    if any(not 0 <= token < config.vocab_size for token in token_ids):
        raise ValueError("prefill token ID is outside the vocabulary")

    cursor, chunks = 0, []
    for size, offset, start, end, prompt in zip(
        batch.chunk_lens, batch.chunk_offsets, batch.chunk_starts, batch.seq_lens, batch.prompt_lens,
    ):
        if any(type(value) is not int for value in (size, offset, start, end, prompt)):
            raise ValueError("prefill offsets and lengths must be integers")
        if size <= 0 or offset != cursor or offset + size > len(token_ids):
            raise ValueError("prefill chunks must cover consecutive packed token spans")
        if not 0 <= start < end == start + size <= prompt <= min(
            runtime.max_seq_len, config.max_position_embeddings,
        ):
            raise ValueError("prefill chunk extent or total prompt length is invalid")
        if runtime.max_prefill_tokens_per_request is not None and size > runtime.max_prefill_tokens_per_request:
            raise ValueError("prefill chunk exceeds the per-request capacity")
        chunks.append(tuple(token_ids[offset:offset + size]))
        cursor += size
    if cursor != len(token_ids):
        raise ValueError("prefill chunk spans leave unclaimed packed tokens")
    _validate_dp_capacity(batch.cache_partitions, batch.chunk_lens, phase="prefill")
    pages = validate_page_tables(
        batch.block_ids_by_group, batch.cache_partitions, batch.seq_lens, groups,
    )
    return [
        (key, partition, start, chunk, prompt, table)
        for key, partition, start, chunk, prompt, table in zip(
            batch.request_ids, batch.cache_partitions, batch.chunk_starts, chunks, batch.prompt_lens, pages,
        )
    ]


def decode_requests(batch, config, runtime, groups):
    """Decode consumes exactly one supplied token per request; no padding rows."""
    count = len(batch.request_ids)
    if not count or count > min(runtime.max_batch_size, runtime.max_num_batched_tokens):
        raise ValueError("decode request count exceeds the configured batch capacity")
    if len(set(batch.request_ids)) != count or any(not isinstance(k, str) or not k for k in batch.request_ids):
        raise ValueError("decode requires distinct nonempty request IDs")
    tokens, lengths = batch.token_ids, batch.seq_lens
    for value in (tokens, lengths):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in (
            torch.int32, torch.int64,
        ):
            raise ValueError("decode tokens and lengths must be CPU integer tensors")
    if tuple(tokens.shape) not in ((count,), (count, 1)) or tuple(lengths.shape) != (count,):
        raise ValueError("decode requires one token and one sequence length per request")
    ids, ends = tokens.reshape(-1).tolist(), lengths.tolist()
    if any(not 0 <= token < config.vocab_size for token in ids):
        raise ValueError("decode token ID is outside the vocabulary")
    if any(not 1 <= end <= min(runtime.max_seq_len, config.max_position_embeddings) for end in ends):
        raise ValueError("decode sequence length exceeds model capacity")
    if len(batch.cache_partitions) != count or len(batch.block_ids_by_group) != count:
        raise ValueError("decode requires one partition and grouped page table per request")
    _validate_dp_capacity(batch.cache_partitions, (1 for _ in range(count)), phase="decode")
    pages = validate_page_tables(batch.block_ids_by_group, batch.cache_partitions, ends, groups)
    return [(key, partition, end - 1, token, table)
            for key, partition, end, token, table in zip(
                batch.request_ids, batch.cache_partitions, ends, ids, pages)]


# npu_runner
class V41ModelRunner:
    """Own one collective session; missing entries fail before resource creation.

    This follows the shared runner lifecycle but deliberately does not inherit
    its generic K/V allocator: V4.1 pools include index and compressor state.
    """
    def __init__(self, plan: V41ExecutionPlan, bindings: CompositeBindings, *, device_ids, runtime,
                 build_options: BuildOptions = BuildOptions()):
        self.plan, self.bindings = plan, bindings
        self.device_ids = tuple(device_ids)
        if len(self.device_ids) != plan.placement.ep_size or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("one distinct physical device is required for each logical EP rank")
        if any(type(i) is not int or i < 0 for i in self.device_ids):
            raise ValueError("device IDs must be nonnegative integers")
        bindings.require(plan.layers, plan.placement)
        if len(plan.layers) == 40:
            self.cache_groups = validate_cache_groups(
                plan.layers, bindings.cache_groups, runtime.max_seq_len)
        else:
            self.cache_groups = None
        self.runtime = runtime
        self.build_options = build_options
        self.resources = None
        self.num_pages = None
        self.closed = False
        self.failed = False
        self.ledger = None
        self._lock = RLock()
        self._plans = None

    def preflight(self):
        if self.failed:
            raise RuntimeError("runner initialization failed; close the session before retrying")
        if self.closed:
            raise RuntimeError("runner is closed")
        if self.resources is not None:
            return self.num_pages
        try:
            resources, pages = self.bindings.allocate(
                self.plan, self.device_ids, self.runtime, self.build_options)
            self.resources = resources
            if resources is None or type(pages) is not int or pages <= 0:
                raise ValueError("composite allocator must return resources and a positive page capacity")
            self.bindings.wait(resources)
            self.num_pages = pages
        except Exception:
            self.failed = True
            self.close()
            raise
        return self.num_pages

    def close(self):
        if self.closed:
            return
        if self.resources is not None:
            # Do not free or reuse storage if completion itself fails.
            self.bindings.wait(self.resources)
            self.bindings.close(self.resources)
            self.resources = None
        self.closed = True

    def _request_ledger(self):
        self.preflight()
        if self.ledger is None:
            self.ledger = RequestLedger(max_requests=self.runtime.max_batch_size,
                                        max_seq_len=self.runtime.max_seq_len)
        return self.ledger

    def _reset_request(self, key, owner):
        self.bindings.reset_request(self.resources, key, owner)
        self.bindings.wait(self.resources)

    def run_prefill(self, model, batch):
        with self._lock:
            ledger = self._request_ledger()
            requests = prefill_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            step = ledger.begin_prefill(requests)
            return self._run_transaction(step, batch.input_embeddings)

    def _run_transaction(self, step, embeddings):
        try:
            result = self._execute_step(step, embeddings)
            self.bindings.wait(self.resources)
            self.ledger.commit(step)
            return result
        except Exception:
            try:
                self.bindings.wait(self.resources)
            except Exception:
                self.failed = True
                self.ledger.poisoned = True
                # Keep pending state and buffers owned: completion is unknown.
                raise
            self.ledger.abort(step, self._reset_request)
            raise

    def _rank_plans(self):
        if self._plans is None:
            self._plans = tuple(self.plan.for_rank(rank) for rank in range(self.plan.placement.ep_size))
        return self._plans

    def lookup_embeddings(self, token_ids):
        # A complete TP vocabulary exists in each DP group; read it once on host.
        plans = self._rank_plans()[:self.plan.placement.tp_size]
        return lookup_token_embeddings([plan.weights for plan in plans], token_ids)

    def _check_state(self, state):
        if not isinstance(state, LayerState) or state.residual is None or state.pre_mix is None:
            raise ValueError("composite must return both residual and delayed pre_mix")
        if state.layout != self.bindings.output_layout:
            raise ValueError("composite output token layout differs from the next layer input")
        return state

    def _run_layers(self, step, embeddings):
        config = self.plan.weights.config
        if embeddings is None:
            embeddings = self.lookup_embeddings(torch.tensor(step.token_ids, dtype=torch.int64))
        if (not isinstance(embeddings, torch.Tensor) or embeddings.device.type != "cpu"
                or embeddings.dtype != torch.bfloat16
                or tuple(embeddings.shape) != (len(step.token_ids), config.hidden_size)):
            raise ValueError("input embeddings must be packed CPU BF16 [active_tokens, hidden_size]")
        plans = self._rank_plans()
        state = self.bindings.initialize(embeddings, step, self.resources)
        self.bindings.wait(self.resources)
        self._check_state(state)
        backbone = (self.bindings.prefill_backbone if step.phase == "prefill"
                    else self.bindings.decode_backbone)
        if backbone is not None:
            state = backbone(state, step, self.resources, plans)
            self.bindings.wait(self.resources)
            return self._check_state(state)
        for layer in self.plan.layers:
            # The adapter chooses bounded staging or residency; payloads stay packed.
            weights = self.bindings.prepare_weights(plans, layer, self.resources)
            self.bindings.wait(self.resources)
            state = self.bindings.entries[(step.phase, layer.mode)](
                layer, state, step, self.resources, weights)
            self.bindings.wait(self.resources)
            self._check_state(state)
        return state

    def _execute_step(self, step, embeddings):
        state = self._run_layers(step, embeddings)
        return self._finish_step(state, step)

    def _finish_step(self, state, step):
        logits = self.bindings.output(state, step, self.resources)
        self.bindings.wait(self.resources)
        shape = (len(step.requests), self.plan.weights.config.vocab_size)
        if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
                or tuple(logits.shape) != shape or logits.dtype not in (
                    torch.float32, torch.bfloat16, torch.float16)):
            raise ValueError("output composite must return CPU logits [requests, vocabulary] in request order")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("output composite returned non-finite logits")
        # Shared workers may sample after the next dispatch reuses device output
        # scratch. Keep returned host logits independent of adapter-owned buffers.
        owned_logits = logits.detach().clone()
        if step.phase == "prefill":
            return PrefillResult(last_hidden=None, logits=owned_logits)
        return DecodeResult(hidden_states=None, logits=owned_logits)

    def run_decode(self, model, batch):
        with self._lock:
            ledger = self._request_ledger()
            requests = decode_requests(batch, model.config, self.runtime, self.bindings.cache_groups)
            step = ledger.begin_decode(requests)
            return self._run_transaction(step, batch.hidden_states)

    def release_finished_requests(self, request_ids):
        with self._lock:
            if self.ledger is not None:
                self.bindings.wait(self.resources)
                self.ledger.release(request_ids, self._reset_request)
