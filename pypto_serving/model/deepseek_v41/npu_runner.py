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

import ctypes
import importlib
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable

import torch

from pypto_serving.config.types import DecodeResult, KVCacheGroupSpec, PrefillResult
from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

from .task_args import ForwardStep, RequestLedger, validate_cache_groups
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


# swa_segment
ATTENTION_ARGS = (
    "x_hc incoming_pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale rope_cos rope_sin window_slots window_indices "
    "window_cache window_cache_scale output next_pre_mix hidden attn_out num_tokens attention_epoch"
).split()
MOE_ARGS = (
    "x_hc pre_mix hc_ffn_fn hc_ffn_scale hc_ffn_base norm_weight gate_weight correction_bias "
    "routed_w1 routed_w1_scale routed_w2 routed_w2_scale routed_w3 routed_w3_scale mxfp4_pair_lut "
    "shared_w1 shared_w1_scale shared_w2 shared_w2_scale shared_w3 shared_w3_scale "
    "next_pre_mix x_mixed x_next num_tokens moe_epoch"
).split()


@dataclass(frozen=True)
class SegmentTopology:
    tp: int = 4
    dp: int = 2
    local_capacity: int = 16

    @classmethod
    def for_decode(cls, *, tp: int = 4, dp: int = 2):
        """Match lib's padded local MoE extent for a full decode batch."""
        if type(tp) is not int or tp not in (1, 2, 4, 8):
            raise ValueError("unsupported decode TP size")
        decode_tokens = 32 * (5 + 1)
        row_tile = 16
        local_rows = (decode_tokens + tp - 1) // tp
        return cls(tp=tp, dp=dp, local_capacity=(local_rows + row_tile - 1) // row_tile * row_tile)

    def __post_init__(self):
        if self.tp not in (1, 2, 4, 8) or type(self.tp) is not int:
            raise ValueError("unsupported TP size")
        if type(self.dp) is not int or self.dp <= 0 or self.world not in (2, 4, 8):
            raise ValueError("TP * DP must be a supported EP size")
        if type(self.local_capacity) is not int or self.local_capacity <= 0:
            raise ValueError("local capacity must be positive")

    @property
    def world(self):
        return self.tp * self.dp

    @property
    def capacity(self):
        return self.tp * self.local_capacity

    def counts(self, group_counts):
        """Contiguous SP slabs; padded/empty ranks still participate in collectives."""
        if len(group_counts) != self.dp or any(
            type(n) is not int or not 0 <= n <= self.capacity for n in group_counts
        ):
            raise ValueError("one bounded active-token count is required per DP group")
        attention = tuple(n for n in group_counts for _ in range(self.tp))
        local = tuple(max(0, min(self.local_capacity, n - r * self.local_capacity))
                      for n in group_counts for r in range(self.tp))
        return attention, local


def load_segment_modules(lib_root, topology):
    """Use V4's explicit lib path and import-time topology, in a model worker.

    Never reload an already imported model with another topology/revision.
    Lib currently captures dimensions from argv when modules are imported.
    """
    root = Path(lib_root).resolve()
    package = "models.deepseek_v4_1_flash"
    existing = sys.modules.get(package + ".config")
    expected = root / "models/deepseek_v4_1_flash/config.py"
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise RuntimeError("V41 lib from another checkout is already imported; use a fresh worker")
    old_argv, old_path = sys.argv[:], sys.path[:]
    try:
        sys.path.insert(0, str(root))
        sys.argv = [old_argv[0], "--tp", str(topology.tp), "--ep", str(topology.world)]
        config = importlib.import_module(package + ".config")
        if Path(config.__file__).resolve() != expected:
            raise RuntimeError("import resolved to a different lib checkout")
        if (config.TP_SIZE, config.EP_SIZE, config.MOE_TOKENS) != (
            topology.tp, topology.world, topology.local_capacity
        ):
            raise ValueError("lib topology/MOE_TOKENS disagrees with the segment capacity")
        attention = importlib.import_module(package + ".prefill_swa")
        moe = importlib.import_module(package + ".moe")
        if moe.SKIP_SHARED_TEST or moe.SKIP_TRANSPORT_TEST:
            raise ValueError("cannot run serving with lib test-only MoE bypasses")
        return attention, moe
    finally:
        sys.argv, sys.path = old_argv, old_path


def compile_segment(compiler, lib_root, topology):
    """Compile the two existing lib composites with the shared V4 compiler."""
    import pypto.language as pl

    attention, moe = load_segment_modules(lib_root, topology)
    swa = compiler.compile(
        "v41_swa_segment", attention.make_hc_program(topology.capacity, topology.world, epochs=1),
        attention_epoch=pl.RUNTIME,
    )
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return swa, ffn


def make_segment_worker(programs, run_config, inherited_host_tensors=()):
    """Retain both programs' communication windows as in the V4 runner.

    Attention waits for (epoch - 1) * 2 before publishing the next phase.
    The runtime's default reset-on-reuse policy would erase that completion
    signal and stall the second layer. Each compiled program owns its windows.
    """
    from pypto.runtime import DistributedWorker

    return DistributedWorker(
        [program.compiled for program in programs], config=run_config,
        persistent=True, reset_persistent_windows=False,
        inherited_host_tensors=inherited_host_tensors,
    )


class SwaSegment:
    """Synchronous half-layer dispatcher on one persistent DistributedWorker.

    Construct the worker with make_segment_worker(). Metadata must be shared
    CPU tensors created before worker startup:
    attention_counts [world, 1] and moe_counts [world], both int32.
    Per-layer argument maps contain all ABI tensors except input state, counts
    and epoch. They own separate output/scratch/cache buffers. No host copies
    occur between half-layers or between consecutive calls to run_layer().
    """

    def __init__(self, worker, programs, topology, attention_counts, moe_counts, run_config):
        import torch

        for value, shape in ((attention_counts, (topology.world, 1)),
                             (moe_counts, (topology.world,))):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or tuple(value.shape) != shape or value.dtype != torch.int32 or not value.is_shared()):
                raise ValueError("counts must be shared CPU int32 metadata allocated before worker startup")
        self.worker, self.programs, self.topology = worker, programs, topology
        self.attention_counts, self.moe_counts = attention_counts, moe_counts
        self.run_config = run_config
        self._epoch = 0
        self._attention_epochs = {}
        self._failed = False

    def _check_state(self, state):
        import torch
        from pypto.runtime import StackedDeviceTensor

        if state.layout != "tp_local_token":
            raise ValueError("SWA segment requires contiguous TP-local token state")
        prefix = (self.topology.world, self.topology.local_capacity)
        for value, shape in ((state.residual, (*prefix, 4, 5120)), (state.pre_mix, (*prefix, 4))):
            if (not isinstance(value, StackedDeviceTensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32
                    or tuple(value.worker_ids) != tuple(range(self.topology.world))):
                raise ValueError("state must be FP32 device shards with canonical rank placement and capacity")

    def run_layer(self, state, attention, moe, *, group_counts):
        """Return MoE outputs directly for the following layer's Attention input.

        Request packing, RoPE, page mapping and cache initialization are owned
        by the caller and must use the same contiguous group/slab ordering.
        """
        return self._run_layer(state, attention, moe, group_counts=group_counts,
                               attention_program=self.programs[0], argument_names=ATTENTION_ARGS,
                               input_mix="incoming_pre_mix", output_residual="output")

    def _run_layer(self, state, attention, moe, *, group_counts, attention_program,
                   argument_names, input_mix, output_residual):
        """Dispatch an audited TP-local Attention ABI and the shared MoE entry."""
        if self._failed:
            raise RuntimeError("segment dispatch failed; close this worker before attempting recovery")
        self._check_state(state)
        group, local = self.topology.counts(group_counts)
        a = dict(attention, x_hc=state.residual, num_tokens=self.attention_counts)
        a[input_mix] = state.pre_mix
        attention_state = LayerState(a[output_residual], a["next_pre_mix"], "tp_local_token")
        m = dict(moe, x_hc=attention_state.residual, pre_mix=attention_state.pre_mix,
                 num_tokens=self.moe_counts)
        result = LayerState(m["x_next"], m["next_pre_mix"], "tp_local_token")
        self._check_state(attention_state)
        self._check_state(result)
        # Reject aliases before launching either kernel, including distinct
        # handles pointing at the same allocation. Reuse across layers is fine.
        buffers = (state.residual, state.pre_mix, attention_state.residual,
                   attention_state.pre_mix, result.residual, result.pre_mix)
        seen = set()
        for buffer in buffers:
            for rank, shard in zip(buffer.worker_ids, buffer.shards):
                key = (rank, shard.data_ptr)
                if key in seen:
                    raise ValueError("input and half-layer outputs must not alias")
                seen.add(key)
        moe_reset = self._epoch == 512
        epoch = 1 if moe_reset else self._epoch + 1
        program_key = id(attention_program.compiled)
        prior_attention_epoch = self._attention_epochs.get(program_key, 0)
        attention_reset = prior_attention_epoch == 512
        attention_epoch = 1 if attention_reset else prior_attention_epoch + 1
        if max(epoch, attention_epoch) > (2**31 - 1) // 2:
            raise OverflowError("communication epoch exhausted; recreate worker")
        # Each compiled Attention owns separate retained signal windows. A
        # mode's first call must use epoch 1 even after another mode ran.
        a["attention_epoch"] = ctypes.c_int32(attention_epoch)
        m["moe_epoch"] = ctypes.c_int32(epoch)
        # Resolve all arguments before the first collective: a missing MoE
        # weight must not leave an already-mutated Attention cache behind.
        a_args = [a[name] for name in argument_names]
        m_args = [m[name] for name in MOE_ARGS]
        self._check_weights(m)
        for rank, (global_n, local_n) in enumerate(zip(group, local)):
            self.attention_counts[rank, 0] = global_n
            self.moe_counts[rank] = local_n
        self._epoch = epoch
        self._attention_epochs[program_key] = attention_epoch
        if attention_reset or moe_reset:
            if not getattr(self.worker, "_persistent", False) or not hasattr(
                self.worker, "_reset_persistent_windows"
            ):
                raise RuntimeError("prefill epoch reset requires a persistent PyPTO worker")
            previous_reset = self.worker._reset_persistent_windows
        try:
            if attention_reset:
                self.worker._reset_persistent_windows = True
            try:
                self.worker.run(attention_program.compiled, *a_args, config=self.run_config)
            finally:
                if attention_reset:
                    self.worker._reset_persistent_windows = previous_reset
            if moe_reset:
                self.worker._reset_persistent_windows = True
            try:
                self.worker.run(self.programs[1].compiled, *m_args, config=self.run_config)
            finally:
                if moe_reset:
                    self.worker._reset_persistent_windows = previous_reset
        except BaseException:
            self._failed = True
            raise
        return result

    @staticmethod
    def _check_weights(moe):
        import torch
        from pypto.runtime import StackedDeviceTensor

        for name in ("routed_w1", "routed_w2", "routed_w3", "mxfp4_pair_lut"):
            value = moe[name]
            expected = torch.int16 if name == "mxfp4_pair_lut" else torch.uint8
            if not isinstance(value, StackedDeviceTensor) or value.dtype != expected:
                raise ValueError(f"{name} must be a resident packed-FP4 ABI tensor ({expected})")


# compressed_metadata
@dataclass(frozen=True)
class CompressedMetadata:
    position_ids: torch.Tensor
    request_ids: torch.Tensor
    query_start_loc: torch.Tensor
    compressed_lens: torch.Tensor
    compressed_slots: torch.Tensor
    compressed_rope_positions: torch.Tensor
    index_block_table: torch.Tensor
    state_block_table: torch.Tensor
    group_counts: tuple[int, ...]


def prepare_compressed_metadata(
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    ratio: int,
    compressed_group: str,
    cache_pages: int,
    max_requests: int,
    state_blocks: int,
    max_prepare_bytes: int = 16 << 20,
) -> CompressedMetadata:
    """Lower full-history pages with joint KV/index physical page numbering.

    The resource allocator must use the same page IDs for compressed KV and
    index-key pools. Each pool stores 128 compressed rows per page. This does
    not select candidates/Top-K or materialize compressed attention indices.
    Those remain device outputs from the appropriate lib producer composite.

    Request IDs are local packed batch rows; state slots are stable ledger IDs.
    Ratio-2 publishes only completed pairs, rotating them at the pair's first
    position. An odd-length chunk leaves its tail in the request's state block.
    The formulas match lib metadata.paged_slots/compressor_metadata at fbe92bfc.
    """
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("compressed metadata requires a ForwardStep and SegmentTopology")
    if type(ratio) is not int or ratio not in (1, 2):
        raise ValueError("compression ratio must be 1 or 2")
    if (type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128
            or type(max_requests) is not int or max_requests <= 0
            or type(state_blocks) is not int or not 0 < state_blocks <= 2**31 - 1
            or type(max_prepare_bytes) is not int or max_prepare_bytes <= 0):
        raise ValueError("cache, request, state and allocation capacities must be positive and fit the ABI")
    if not isinstance(compressed_group, str) or not compressed_group:
        raise ValueError("compressed_group must identify a jointly numbered KV/index pool")
    if step.phase not in ("prefill", "decode"):
        raise ValueError("unsupported forward phase")
    groups = [[] for _ in range(topology.dp)]
    counts, owners, slots, names = [0] * topology.dp, set(), set(), set()
    columns = 1
    for request in step.requests:
        group, count = request.partition, len(request.token_ids)
        if (type(group) is not int or not 0 <= group < topology.dp or count <= 0
                or type(request.start) is not int or not 0 <= request.start < request.end <= 2**31 - 1):
            raise ValueError("invalid compressed request partition or extent")
        if request.request_id in names:
            raise ValueError("duplicate request in forward step")
        names.add(request.request_id)
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one token per request")
        if counts[group] + count > topology.capacity or len(groups[group]) >= max_requests:
            raise ValueError("compressed requests exceed token or request capacity")
        if type(request.state_slot) is not int or not 0 <= request.state_slot < state_blocks:
            raise ValueError("request state slot is outside the compressor pool")
        key = (group, request.state_slot)
        if ratio == 2 and key in slots:
            raise ValueError("requests must own distinct compressor state slots")
        slots.add(key)
        pages = request.pages.get(compressed_group)
        needed = (request.end // ratio + 127) // 128
        if not isinstance(pages, (tuple, list)) or len(pages) < needed:
            raise ValueError("compressed pages must cover all completed history")
        for page in pages:
            if type(page) is not int or not 0 <= page < cache_pages:
                raise ValueError("compressed page is outside the joint KV/index pool")
            if (group, page) in owners:
                raise ValueError("compressed pages must be private within each DP group")
            owners.add((group, page))
        columns = max(columns, len(pages))
        groups[group].append((request, counts[group], tuple(pages)))
        counts[group] += count
    # Include DP scratch plus replicated TP outputs and conservative small work arrays.
    estimate = topology.dp * (topology.capacity * 40 + max_requests * (columns + 4) * 4) * (topology.tp + 1)
    if estimate > max_prepare_bytes:
        raise ValueError("compressed metadata exceeds its allocation budget")
    shape = (topology.dp, topology.capacity)
    positions = torch.zeros(shape, dtype=torch.int32)
    ids = torch.full(shape, -1, dtype=torch.int32)
    lens = torch.zeros(shape, dtype=torch.int32)
    writes = torch.full(shape, -1, dtype=torch.int64)
    rope = torch.full(shape, -1, dtype=torch.int32)
    starts = torch.zeros(topology.dp, max_requests + 1, dtype=torch.int32)
    tables = torch.full((topology.dp, max_requests, columns), -1, dtype=torch.int32)
    states = torch.full((topology.dp, max_requests, 1), -1, dtype=torch.int32)
    for group, requests in enumerate(groups):
        for batch_row, (request, offset, pages) in enumerate(requests):
            count = len(request.token_ids)
            section = slice(offset, offset + count)
            p = torch.arange(request.start, request.end, dtype=torch.int32)
            positions[group, section] = p
            ids[group, section] = batch_row
            lens[group, section] = (p + 1) // ratio
            if ratio == 2:
                states[group, batch_row, 0] = request.state_slot
            if pages:
                tables[group, batch_row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            starts[group, batch_row + 1:] = offset + count
            for local, position in enumerate(range(request.start, request.end)):
                if (position + 1) % ratio == 0:
                    logical = position // ratio
                    writes[group, offset + local] = pages[logical // 128] * 128 + logical % 128
                    rope[group, offset + local] = position + 1 - ratio
    def replicate(tensor):
        return tensor.repeat_interleave(topology.tp, dim=0)
    return CompressedMetadata(*(replicate(t) for t in (positions, ids, starts, lens, writes, rope, tables, states)),
                              tuple(counts))


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


# prefill_segment
_WEIGHTS = (
    "x_hc pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale "
)
_WINDOW = "window_slots window_indices window_cache window_cache_scale "
_OUTPUTS = "attn_input attn_output next_pre_mix x_hc_out num_tokens attention_epoch"
C2A_FULL_ARGS = (_WEIGHTS + "freqs_cos freqs_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale token_to_req_indices compressed_lens "
    "index_cache index_cache_scale index_block_table position_ids compressed_freqs_cos compressed_freqs_sin "
    "compressed_rope_positions compressor_wkv compressor_wgate query_start_loc state_block_table state_cache "
    "compressor_norm_weight compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale "
    "index_weights_proj topk_indices " + _OUTPUTS).split()
C2A_REUSE_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale compressed_indices " + _OUTPUTS).split()
C1A_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale request_ids compressed_lens index_cache index_cache_scale "
    "index_block_table compressed_rope_cos compressed_rope_sin compressor_wkv compressor_norm_weight "
    "compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale index_weights_proj "
    "topk_indices candidate_mask compressed_indices " + _OUTPUTS).split()
PREFILL_ARGUMENTS = {
    "swa": ATTENTION_ARGS, "c2a_full": C2A_FULL_ARGS, "c2a_reuse": C2A_REUSE_ARGS,
    "c1a_full": C1A_ARGS, "c1a_reindex": C1A_ARGS, "c1a_reuse": C1A_ARGS,
}


def bind_prefill_producers(plans, arguments):
    """Bind one ordered forward chain to its actual producer allocations.

    Like V4's worker-owned cache arguments, handles stay resident for the whole
    dispatch. V4.1 additionally shares compressed KV/index pools and physical
    Top-K rows according to LayerPlan. This performs no tensor computation.
    Every producer must execute earlier in this chain, so a previous request's
    transient Top-K/candidate rows cannot satisfy a missing dependency.

    C1A's common ABI retains unused Full weights in Reindex/Reuse. Bind those
    slots to their real producer weights; never allocate placeholder weights.
    Keep unused index slots in separate allocations. The common C1A ABI marks
    them writable even when a selected mode does not use them; aliasing those
    arguments makes the runtime reject overlapping write ranges (lib fbe92bfc).
    """
    plans = tuple(plans)
    if not plans or any(not isinstance(p, LayerPlan) for p in plans):
        raise ValueError("prefill chain requires LayerPlan entries")
    if any(b.layer_id != a.layer_id + 1 for a, b in zip(plans, plans[1:])):
        raise ValueError("prefill chain must execute consecutive layers in order")
    result, seen = {}, {}
    for plan in plans:
        if plan.mode not in PREFILL_ARGUMENTS:
            raise ValueError("unsupported prefill mode")
        current = dict(arguments[plan.layer_id])
        if plan.mode != "swa":
            family, mode = plan.mode.split("_")

            def producer(source, kind):
                if source == plan.layer_id:
                    if mode != "full" and not (kind == "index" and mode == "reindex"):
                        raise ValueError(f"{plan.mode} cannot produce its own {kind}")
                    return current
                if source not in seen or not seen[source].mode.startswith(family + "_"):
                    raise ValueError(f"missing preceding {kind} producer for layer {plan.layer_id}")
                source_mode = seen[source].mode.split("_")[1]
                if source_mode != "full" and not (kind == "index" and source_mode == "reindex"):
                    raise ValueError(f"invalid {kind} producer mode")
                return result[source]

            if mode == "full" and (plan.kv_source != plan.layer_id or plan.index_source != plan.layer_id):
                raise ValueError("Full must own its KV and index outputs")
            if mode == "reindex" and (family != "c1a" or plan.index_source != plan.layer_id):
                raise ValueError("Reindex must own its C1A index output")
            kv = producer(plan.kv_source, "KV")
            index = producer(plan.index_source, "index")
            if plan.index_source != plan.layer_id and seen[plan.index_source].kv_source != plan.kv_source:
                raise ValueError("index selection must address the same KV producer")
            for name in ("compressed_cache", "compressed_cache_scale"):
                current[name] = kv[name]
            if family == "c1a":
                candidate = producer(plan.candidate_source, "candidate")
                if plan.candidate_source != plan.kv_source:
                    raise ValueError("C1A candidates must use the same KV producer")
                for name in ("index_cache", "index_cache_scale", "compressor_wkv", "compressor_norm_weight",
                             "index_wk", "index_norm_weight"):
                    current[name] = kv[name]
                current["candidate_mask"] = candidate["candidate_mask"]
                if mode == "reuse":
                    for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                        current[name] = index[name]
                    current["compressed_indices"] = index["topk_indices"]
                if current["compressed_indices"] is current["topk_indices"]:
                    raise ValueError("C1A index ABI arguments require separate allocations")
            elif mode == "reuse":
                current["compressed_indices"] = index["topk_indices"]
        elif any(source is not None for source in (plan.kv_source, plan.index_source, plan.candidate_source)):
            raise ValueError("SWA must not declare compressed producers")
        result[plan.layer_id], seen[plan.layer_id] = current, plan
    return result


def compile_prefill_segments(compiler, lib_root, topology, modes):
    """Compile existing composite entries, sharing one packed-FP4 MoE program.

    C1A must use prefill_c1a_sp, not the older replicated-residual wrappers.
    No model operators are composed by this serving module.
    """
    import pypto.language as pl

    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in PREFILL_ARGUMENTS for mode in modes):
        raise ValueError("unsupported or empty prefill mode selection")
    swa, moe = load_segment_modules(lib_root, topology)
    attention = {}
    for mode in modes:
        if mode == "swa":
            entry = swa.make_hc_program(topology.capacity, topology.world, epochs=1)
        elif mode.startswith("c2a_"):
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_" + mode)
            entry = module.make_hc_program(topology.capacity, topology.world, epochs=1)
        else:
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_c1a_sp")
            entry = module.make_program(mode.removeprefix("c1a_"), topology.world, epochs=1)
        if tuple(entry.param_names) != tuple(PREFILL_ARGUMENTS[mode]):
            raise ValueError(f"{mode} prefill parameter order disagrees with the selected lib")
        attention[mode] = compiler.compile("v41_prefill_" + mode, entry, attention_epoch=pl.RUNTIME)
    if tuple(moe.l3_moe.param_names) != tuple(MOE_ARGS):
        raise ValueError("prefill MoE parameter order disagrees with the selected lib")
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return attention, ffn


class PrefillSegment(SwaSegment):
    """Carry device residual/pre_mix through a selected prefill mode and MoE.

    The worker must own all supplied programs via make_segment_worker(). Each
    Attention program advances its own epoch; the shared MoE advances on every
    layer. Cache ownership, cross-layer producer bindings and padded metadata
    remain caller responsibilities. Any runtime failure poisons the whole worker.
    """

    def __init__(self, worker, attention_programs, moe_program, topology,
                 attention_counts, moe_counts, run_config):
        if not attention_programs or any(mode not in PREFILL_ARGUMENTS for mode in attention_programs):
            raise ValueError("unsupported or empty prefill program set")
        self.attention_programs = dict(attention_programs)
        super().__init__(worker, (next(iter(attention_programs.values())), moe_program),
                         topology, attention_counts, moe_counts, run_config)

    def run_layer(self, state, attention, moe, *, group_counts, mode):
        if mode not in self.attention_programs:
            raise ValueError(f"prefill mode was not compiled: {mode}")
        return self._run_layer(
            state, attention, moe, group_counts=group_counts,
            attention_program=self.attention_programs[mode], argument_names=PREFILL_ARGUMENTS[mode],
            input_mix="incoming_pre_mix" if mode == "swa" else "pre_mix",
            output_residual="output" if mode == "swa" else "x_hc_out",
        )

    def run_chain(self, state, plans, attention, moe, *, group_counts):
        """Execute one request step through its producers and consumers in order.

        The resource owner must prepare every layer for the SAME packed step
        and retain all allocations until this synchronous call returns. This
        is still a bounded prefill segment, not a complete serving backend.
        """
        plans = tuple(plans)
        bound = bind_prefill_producers(plans, attention)
        self.topology.counts(group_counts)
        # Resolve the whole chain before any cache mutation. Runtime state,
        # active counts and per-program epochs are provided by run_layer.
        dynamic = {"x_hc", "pre_mix", "incoming_pre_mix", "num_tokens", "attention_epoch", "moe_epoch"}
        for plan in plans:
            if plan.mode not in self.attention_programs:
                raise ValueError(f"prefill mode was not compiled: {plan.mode}")
            for name in PREFILL_ARGUMENTS[plan.mode]:
                if name not in dynamic:
                    bound[plan.layer_id][name]
            for name in MOE_ARGS:
                if name not in dynamic:
                    moe[plan.layer_id][name]
            self._check_weights(moe[plan.layer_id])
        for plan in plans:
            state = self.run_layer(state, bound[plan.layer_id], moe[plan.layer_id],
                                   group_counts=group_counts, mode=plan.mode)
        return state


# segment_inputs
@dataclass(frozen=True)
class SegmentInputs:
    residual: torch.Tensor
    pre_mix: torch.Tensor
    # Source indices into the original packed request order; -1 means padding.
    row_indices: torch.Tensor
    # (world rank, local row) in the original request order, for output selection.
    request_last_rows: tuple[tuple[int, int], ...]
    group_counts: tuple[int, ...]


def prepare_segment_inputs(
    embeddings: torch.Tensor,
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    max_prepare_bytes: int = 1 << 30,
) -> SegmentInputs:
    """Pack fresh embedding inputs without depending on request batch order.

    Each DP group contains contiguous local-capacity TP slabs, including empty
    slabs. Padding is zero in both residual and pre_mix. row_indices supplies the
    same mapping for positions and other token metadata. This does not lower
    cache pages or initialize a production all-mode model adapter.
    """
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0:
        raise ValueError("max_prepare_bytes must be a positive integer")
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("segment input preparation requires a ForwardStep and SegmentTopology")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.layout != torch.strided
            or embeddings.device.type != "cpu" or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2 or embeddings.shape[1] <= 0):
        raise ValueError("embeddings must be packed CPU BF16 [active_tokens, hidden_size]")
    counts = [0] * topology.dp
    source_ranges, last_rows = [], []
    offset = 0
    request_ids = set()
    for request in step.requests:
        partition, count = request.partition, len(request.token_ids)
        if type(partition) is not int or not 0 <= partition < topology.dp or count <= 0:
            raise ValueError("requests require valid DP partitions and nonempty token slices")
        if request.request_id in request_ids:
            raise ValueError("duplicate request in forward step")
        request_ids.add(request.request_id)
        start = counts[partition]
        if start + count > topology.capacity:
            raise ValueError("DP token count exceeds segment capacity; split the step first")
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one fresh token per request")
        source_ranges.append((partition, start, offset, count))
        owner, row = divmod(start + count - 1, topology.local_capacity)
        last_rows.append((partition * topology.tp + owner, row))
        counts[partition] += count
        offset += count
    if step.phase not in ("prefill", "decode") or embeddings.shape[0] != offset:
        raise ValueError("embedding rows must match a prefill/decode forward step")
    hidden = embeddings.shape[1]
    # Include output storage and the largest temporary expanded FP32 row block.
    allocated_rows = topology.world * topology.local_capacity
    estimate = allocated_rows * (4 * hidden * 4 + 4 * 4 + 8) + offset * hidden * 4
    if estimate > max_prepare_bytes:
        raise ValueError(f"segment preparation requires estimated {estimate} bytes; budget={max_prepare_bytes}")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite")
    residual = torch.zeros(topology.dp, topology.capacity, 4, hidden, dtype=torch.float32)
    pre_mix = torch.zeros(topology.dp, topology.capacity, 4, dtype=torch.float32)
    row_indices = torch.full((topology.dp, topology.capacity), -1, dtype=torch.int64)
    for partition, start, source, count in source_ranges:
        residual[partition, start:start + count].copy_(embeddings[source:source + count, None])
        pre_mix[partition, start:start + count, 0] = 1
        row_indices[partition, start:start + count] = torch.arange(source, source + count)
    return SegmentInputs(
        residual.reshape(topology.world, topology.local_capacity, 4, hidden),
        pre_mix.reshape(topology.world, topology.local_capacity, 4),
        row_indices.reshape(topology.world, topology.local_capacity),
        tuple(last_rows), tuple(counts),
    )


# swa_metadata
@dataclass(frozen=True)
class SwaWindowMetadata:
    positions: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    group_counts: tuple[int, ...]


def gather_swa_rope_rows(
    metadata: SwaWindowMetadata,
    tables: tuple[torch.Tensor, torch.Tensor],
    *,
    max_prepare_bytes: int = 16 << 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather lib-built FP32 SWA tables in the same order as cache metadata.

    Like V4, the executor builds tables with the selected lib's host helper.
    This boundary only selects rows; it does not implement model-side RoPE.
    Padding receives identity rotation and does not index the table.
    """
    if not isinstance(metadata, SwaWindowMetadata) or len(tables) != 2:
        raise ValueError("SWA RoPE requires metadata and a cosine/sine table pair")
    positions, slots = metadata.positions, metadata.window_slots
    if (positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.ndim != 2
            or slots.device.type != "cpu" or slots.dtype != torch.int64 or slots.shape != positions.shape):
        raise ValueError("SWA positions and slots must be matching CPU INT64 matrices")
    cos, sin = tables
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.dtype != torch.float32
           or t.layout != torch.strided or t.ndim != 2 for t in tables):
        raise ValueError("SWA RoPE tables must be CPU FP32 matrices")
    if cos.shape != sin.shape or min(cos.shape) <= 0:
        raise ValueError("SWA cosine/sine tables must have matching nonempty shapes")
    estimate = positions.numel() * (cos.shape[1] * 4 * 4 + 32)
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0 or estimate > max_prepare_bytes:
        raise ValueError("SWA RoPE preparation exceeds its allocation budget")
    active = slots >= 0
    selected = positions[active]
    if bool(((selected < 0) | (selected >= cos.shape[0])).any()):
        raise ValueError("active SWA position is outside the RoPE table")
    selected_cos, selected_sin = cos[selected], sin[selected]
    if not bool(torch.isfinite(selected_cos).all() and torch.isfinite(selected_sin).all()):
        raise ValueError("selected SWA RoPE rows must be finite")
    shape = (*positions.shape, cos.shape[1])
    rows_cos, rows_sin = torch.ones(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    rows_cos[active], rows_sin[active] = selected_cos, selected_sin
    return rows_cos, rows_sin


def prepare_swa_window_metadata(
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    cache_pages: int,
    window_group: str = "window",
    max_prepare_bytes: int = 32 << 20,
) -> SwaWindowMetadata:
    """Map each active row to a unique cache write and at most 128 causal reads.

    Page IDs address a partition-private cache of [pages, 128, 1, 512] payloads.
    Full-history tables preserve old rows while the whole new chunk is published;
    a 128-slot modulo ring would overwrite keys needed by early chunk queries.
    TP peers receive the same group-global metadata. No cache contents, RoPE,
    request ownership or completion state are changed by this host preparation.
    """
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("SWA metadata requires a ForwardStep and SegmentTopology")
    if type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError("cache_pages must fit the INT32 window-index ABI")
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0:
        raise ValueError("max_prepare_bytes must be a positive integer")
    if step.phase not in ("prefill", "decode"):
        raise ValueError("unsupported forward phase")
    counts, ranges, owners, requests = [0] * topology.dp, [], set(), set()
    for request in step.requests:
        group, count = request.partition, len(request.token_ids)
        if (type(group) is not int or not 0 <= group < topology.dp or count <= 0
                or type(request.start) is not int or request.start < 0):
            raise ValueError("request partition, start and token count must be valid")
        if request.request_id in requests:
            raise ValueError("duplicate request in forward step")
        requests.add(request.request_id)
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one token per request")
        if counts[group] + count > topology.capacity:
            raise ValueError("DP token count exceeds SWA capacity")
        pages = request.pages.get(window_group)
        if not isinstance(pages, (tuple, list)) or len(pages) < (request.end + 127) // 128:
            raise ValueError("SWA requires a full-history physical page table through the chunk end")
        for page in pages:
            if type(page) is not int or not 0 <= page < cache_pages:
                raise ValueError("SWA physical page is outside its partition cache pool")
            key = (group, page)
            if key in owners:
                raise ValueError("SWA pages must be private; shared or repeated physical pages are unsupported")
            owners.add(key)
        ranges.append((group, counts[group], request.start, count, tuple(pages)))
        counts[group] += count
    rows = topology.dp * topology.capacity
    estimate = rows * (128 * 4 + 2 * 8) * (topology.tp + 1)
    if estimate > max_prepare_bytes:
        raise ValueError(f"SWA metadata requires estimated {estimate} bytes; budget={max_prepare_bytes}")
    positions = torch.zeros(topology.dp, topology.capacity, dtype=torch.int64)
    slots = torch.full_like(positions, -1)
    indices = torch.full((topology.dp, topology.capacity, 128), -1, dtype=torch.int32)
    for group, row, start, count, pages in ranges:
        for local in range(count):
            position = start + local
            positions[group, row + local] = position
            slots[group, row + local] = pages[position // 128] * 128 + position % 128
            visible = range(max(0, position - 127), position + 1)
            addresses = [pages[p // 128] * 128 + p % 128 for p in visible]
            indices[group, row + local, :len(addresses)] = torch.tensor(addresses, dtype=torch.int32)
    return SwaWindowMetadata(
        positions.repeat_interleave(topology.tp, dim=0),
        slots.repeat_interleave(topology.tp, dim=0),
        indices.repeat_interleave(topology.tp, dim=0), tuple(counts),
    )


# input_resources
@dataclass(frozen=True)
class InputHostResources:
    topology: SegmentTopology
    residual: torch.Tensor
    pre_mix: torch.Tensor

    @property
    def inherited_host_tensors(self):
        return self.residual, self.pre_mix


def make_input_host_resources(topology: SegmentTopology, hidden_size: int) -> InputHostResources:
    """Allocate shared staging before the persistent DistributedWorker forks."""
    if not isinstance(topology, SegmentTopology) or type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("input staging requires a topology and positive hidden size")
    residual = torch.empty((topology.world, topology.local_capacity, 4, hidden_size),
                           dtype=torch.float32).share_memory_()
    pre_mix = torch.empty((topology.world, topology.local_capacity, 4),
                          dtype=torch.float32).share_memory_()
    return InputHostResources(topology, residual, pre_mix)


class InputStateRuntime:
    """Upload each packed step into fixed device handles for its phase capacity."""

    def __init__(self, worker, host: InputHostResources):
        if not isinstance(host, InputHostResources):
            raise ValueError("input state requires pre-fork host resources")
        for tensor in host.inherited_host_tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or not tensor.is_shared() or not tensor.is_contiguous()):
                raise ValueError("input staging must be contiguous shared CPU FP32")
        topology = host.topology
        if (host.residual.shape[:3] != (topology.world, topology.local_capacity, 4)
                or host.pre_mix.shape != host.residual.shape[:-1]):
            raise ValueError("input staging shapes disagree with the TP-local layout")
        self.worker, self.host = worker, host
        self.failed = False
        self._resident = []
        try:
            self.residual = alloc_device_buffer(worker, host.residual.shape, torch.float32, stacked=True)
            self._resident.append(self.residual)
            self.pre_mix = alloc_device_buffer(worker, host.pre_mix.shape, torch.float32, stacked=True)
            self._resident.append(self.pre_mix)
        except BaseException:
            self.close()
            raise

    def stage(self, embeddings: torch.Tensor, step: ForwardStep) -> tuple[LayerState, SegmentInputs]:
        if self.failed:
            raise RuntimeError("input upload failed; recreate the worker before reuse")
        packed = prepare_segment_inputs(embeddings, step, self.host.topology)
        self.host.residual.copy_(packed.residual)
        self.host.pre_mix.copy_(packed.pre_mix)
        try:
            for source, destination in ((self.host.residual, self.residual),
                                        (self.host.pre_mix, self.pre_mix)):
                for rank, shard in enumerate(destination.shards):
                    row = source[rank]
                    self.worker.copy_to(shard.data_ptr, row.data_ptr(),
                                        row.numel() * row.element_size(), worker_id=rank)
        except BaseException:
            self.failed = True
            raise
        inputs = SegmentInputs(self.host.residual, self.host.pre_mix,
                               packed.row_indices, packed.request_last_rows, packed.group_counts)
        return LayerState(self.residual, self.pre_mix, "tp_local_token"), inputs

    def close(self):
        for tensor in reversed(self._resident):
            self.worker.free_stacked_tensor(tensor)
        self._resident.clear()


# prefill_layer_resources
class PrefillLayerResources:
    """Own only transient device buffers; weight and cache handles are borrowed."""

    def __init__(self, worker, topology, plans, weights, cache, metadata, *,
                 kv_sources, index_sources, c2a_sources, max_seq_len, index_topk=512):
        self.worker = worker
        self.topology = topology
        self.plans = tuple(plans)
        self.metadata = metadata
        if (not self.plans or any(plan.mode not in PREFILL_ARGUMENTS for plan in self.plans)
                or any(type(n) is not int or n <= 0 for n in (max_seq_len, index_topk))):
            raise ValueError("prefill binding requires ordered modes and positive model capacities")
        self._owned = []
        world, local, tokens = topology.world, topology.local_capacity, topology.capacity

        def allocate(shape, dtype):
            value = alloc_device_buffer(worker, shape, dtype, stacked=True)
            self._owned.append(value)
            return value

        try:
            # A layer's Attention and MoE outputs must differ from each other
            # and from the preceding MoE output. Three slots suffice for all layers.
            states = [(allocate((world, local, 4, 5120), torch.float32),
                       allocate((world, local, 4), torch.float32)) for _ in range(3)]
            attention_hidden = allocate((world, tokens, 5120), torch.bfloat16)
            attention_out = allocate((world, local, 5120), torch.bfloat16)
            moe_mixed = allocate((world, local, 5120), torch.bfloat16)
            topk = {plan.layer_id: allocate((world, tokens, index_topk), torch.int32)
                    for plan in self.plans if plan.index_source == plan.layer_id}
            unused_topk = allocate((world, tokens, index_topk), torch.int32)
            unused_indices = allocate((world, tokens, index_topk), torch.int32)
            candidate = {plan.layer_id: allocate((world, tokens, max_seq_len), torch.uint8)
                         for plan in self.plans if plan.mode == "c1a_full"}
            attention, moe = {}, {}
            for plan in self.plans:
                layer = plan.layer_id
                a, m = weights.prefill_weights(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources)
                a = dict(a, **cache.prefill_views(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources))
                a.update(metadata.arguments(plan.mode))
                attn_residual, attn_mix = states[(2 * layer) % 3]
                moe_residual, moe_mix = states[(2 * layer + 1) % 3]
                a.update(attn_input=attention_hidden, attn_output=attention_out,
                         hidden=attention_hidden, attn_out=attention_out,
                         output=attn_residual, x_hc_out=attn_residual,
                         next_pre_mix=attn_mix)
                if plan.mode.startswith("c1a_"):
                    a.update(topk_indices=topk.get(layer, unused_topk),
                             compressed_indices=unused_indices,
                             candidate_mask=candidate[plan.candidate_source])
                elif plan.mode == "c2a_full":
                    a["topk_indices"] = topk[layer]
                elif plan.mode == "c2a_reuse":
                    a["compressed_indices"] = unused_indices
                m = dict(m, next_pre_mix=moe_mix, x_next=moe_residual, x_mixed=moe_mixed)
                attention[layer], moe[layer] = a, m
            bound = bind_prefill_producers(self.plans, attention)
            dynamic = {"x_hc", "pre_mix", "incoming_pre_mix", "num_tokens",
                       "attention_epoch", "moe_epoch"}
            for plan in self.plans:
                missing_a = set(PREFILL_ARGUMENTS[plan.mode]) - dynamic - bound[plan.layer_id].keys()
                missing_m = set(MOE_ARGS) - dynamic - moe[plan.layer_id].keys()
                if missing_a or missing_m:
                    raise ValueError(f"layer {plan.layer_id} has unbound prefill ABI fields: "
                                     f"{sorted(missing_a | missing_m)}")
            self.attention, self.moe = bound, moe
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in reversed(self._owned):
            self.worker.free_stacked_tensor(value)
        self._owned.clear()


# prefill_metadata_resources
class PrefillMetadataResources:
    """Keep C2A and C1A page/length contracts separate across one layer chain."""

    def __init__(self, topology, *, max_seq_len, max_requests, window_pages,
                 c2a_pages, c1a_pages, rope_tables, compressed_rope_tables):
        if any(type(value) is not int or value <= 0 for value in
               (max_seq_len, max_requests, window_pages, c2a_pages, c1a_pages)):
            raise ValueError("prefill metadata capacities must be positive integers")
        for pair in (rope_tables, compressed_rope_tables):
            if (len(pair) != 2 or any(not isinstance(value, torch.Tensor) or
                value.device.type != "cpu" or value.dtype != torch.float32 or
                value.ndim != 2 or value.shape[0] < max_seq_len for value in pair)
                    or pair[0].shape != pair[1].shape):
                raise ValueError("prefill requires lib FP32 RoPE tables covering the model context")
        self.topology = topology
        self.max_requests = max_requests
        self.window_pages = window_pages
        self.compressed_pages = {"c2a": c2a_pages, "c1a": c1a_pages}
        self.fields = {}

        def shared(name, shape, dtype, fill=0):
            value = torch.empty(shape, dtype=dtype).share_memory_()
            value.fill_(fill)
            self.fields[name] = value
            return value

        world, tokens, width = topology.world, topology.capacity, rope_tables[0].shape[1]
        shared("attention_counts", (world, 1), torch.int32)
        shared("moe_counts", (world,), torch.int32)
        shared("window_slots", (world, tokens), torch.int64, -1)
        shared("window_indices", (world, tokens, 128), torch.int32, -1)
        for name in ("rope_cos", "rope_sin", "compressed_rope_cos", "compressed_rope_sin"):
            shared(name, (world, tokens, width), torch.float32)
        for name in ("c2a_compressed_rope_cos", "c2a_compressed_rope_sin"):
            shared(name, (world, tokens, width), torch.float32)
        for prefix, pages in self.compressed_pages.items():
            for name, shape, dtype, fill in (
                ("position_ids", (world, tokens), torch.int32, 0),
                ("request_ids", (world, tokens), torch.int32, -1),
                ("query_start_loc", (world, max_requests + 1), torch.int32, 0),
                ("compressed_lens", (world, tokens), torch.int32, 0),
                ("compressed_slots", (world, tokens), torch.int64, -1),
                ("compressed_rope_positions", (world, tokens), torch.int32, -1),
                ("index_block_table", (world, max_requests, pages), torch.int32, -1),
                ("state_block_table", (world, max_requests, 1), torch.int32, -1),
            ):
                shared(prefix + "_" + name, shape, dtype, fill)
        self.tables = {}
        for prefix, pair in (("", rope_tables), ("compressed_", compressed_rope_tables)):
            for suffix, value in zip(("cos", "sin"), pair):
                target = shared(prefix + "freqs_" + suffix,
                                (world, max_seq_len, width), torch.float32)
                target.copy_(value[:max_seq_len].unsqueeze(0).expand_as(target))
            self.tables[prefix] = pair

    @property
    def inherited_host_tensors(self):
        return tuple(self.fields.values())

    def _compressed_rope_rows(self, positions):
        cos, sin = self.tables["compressed_"]
        active = positions >= 0
        if bool((positions[active] >= cos.shape[0]).any()):
            raise ValueError("compressed RoPE position exceeds the lib table")
        result_cos = torch.ones((*positions.shape, cos.shape[1]), dtype=torch.float32)
        result_sin = torch.zeros_like(result_cos)
        result_cos[active] = cos[positions[active].long()]
        result_sin[active] = sin[positions[active].long()]
        return result_cos, result_sin

    def stage(self, step):
        swa = prepare_swa_window_metadata(
            step, self.topology, cache_pages=self.window_pages)
        self.fields["window_slots"].copy_(swa.window_slots)
        self.fields["window_indices"].copy_(swa.window_indices)
        cos, sin = gather_swa_rope_rows(
            swa, self.tables[""], max_prepare_bytes=64 << 20)
        self.fields["rope_cos"].copy_(cos)
        self.fields["rope_sin"].copy_(sin)
        attention, moe = self.topology.counts(swa.group_counts)
        self.fields["attention_counts"][:, 0].copy_(torch.tensor(attention, dtype=torch.int32))
        self.fields["moe_counts"].copy_(torch.tensor(moe, dtype=torch.int32))
        for prefix, ratio in (("c2a", 2), ("c1a", 1)):
            metadata = prepare_compressed_metadata(
                step, self.topology, ratio=ratio, compressed_group=prefix,
                cache_pages=self.compressed_pages[prefix], max_requests=self.max_requests,
                state_blocks=self.max_requests,
            )
            if metadata.group_counts != swa.group_counts:
                raise ValueError("SWA and compressed metadata token layouts disagree")
            for name in ("position_ids", "request_ids", "query_start_loc", "compressed_lens",
                         "compressed_slots", "compressed_rope_positions", "state_block_table"):
                self.fields[prefix + "_" + name].copy_(getattr(metadata, name))
            table = self.fields[prefix + "_index_block_table"]
            source = metadata.index_block_table
            if source.shape[2] > table.shape[2]:
                raise ValueError("compressed page table exceeds its pre-fork capacity")
            table.fill_(-1)
            table[:, :, :source.shape[2]].copy_(source)
            rows = self._compressed_rope_rows(metadata.compressed_rope_positions)
            if prefix == "c2a":
                self.fields["c2a_compressed_rope_cos"].copy_(rows[0])
                self.fields["c2a_compressed_rope_sin"].copy_(rows[1])
            else:
                self.fields["compressed_rope_cos"].copy_(rows[0])
                self.fields["compressed_rope_sin"].copy_(rows[1])
        return swa.group_counts

    def arguments(self, mode):
        common = {name: self.fields[name] for name in
                  ("window_slots", "window_indices", "rope_cos", "rope_sin")}
        if mode == "swa":
            return common
        if mode not in ("c2a_full", "c2a_reuse", "c1a_full", "c1a_reindex", "c1a_reuse"):
            raise ValueError("unknown V4.1 prefill metadata mode")
        prefix = mode[:3]
        names = ("request_ids", "compressed_lens", "compressed_slots", "index_block_table")
        result = dict(common, **{name: self.fields[prefix + "_" + name] for name in names})
        if mode == "c2a_full":
            result.update({
                "freqs_cos": self.fields["freqs_cos"],
                "freqs_sin": self.fields["freqs_sin"],
                "compressed_freqs_cos": self.fields["compressed_freqs_cos"],
                "compressed_freqs_sin": self.fields["compressed_freqs_sin"],
                "token_to_req_indices": result["request_ids"],
                "position_ids": self.fields["c2a_position_ids"],
                "compressed_rope_positions": self.fields["c2a_compressed_rope_positions"],
                "compressed_rope_cos": self.fields["c2a_compressed_rope_cos"],
                "compressed_rope_sin": self.fields["c2a_compressed_rope_sin"],
                "query_start_loc": self.fields["c2a_query_start_loc"],
                "state_block_table": self.fields["c2a_state_block_table"],
            })
        elif mode.startswith("c1a_"):
            result["compressed_rope_cos"] = self.fields["compressed_rope_cos"]
            result["compressed_rope_sin"] = self.fields["compressed_rope_sin"]
        return result
