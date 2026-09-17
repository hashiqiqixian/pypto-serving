# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Opt-in A5 TP1 native attention with scheduler-owned page addressing.

The L2 worker owns separate, persistent payload/scale buffers. Host tensors are
used only for current activations, metadata, returned FP32 partials and bounded
rollback journals. All page allocation, visibility and generation decisions
remain with CacheTransaction; index storage uses its source main-KV page order
so a native physical Top-K index addresses both independently allocated pools.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch

from .attention import rotary_frequencies


def _load_library():
    roots = [Path(os.environ["PYPTO_LIB_ROOT"])] if os.environ.get("PYPTO_LIB_ROOT") else []
    roots.extend(parent / "pypto-lib" for parent in Path(__file__).resolve().parents)
    for root in roots:
        if (root / "models/deepseek_v4_1_flash/local_attention.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from models.deepseek_v4_1_flash import local_attention
            if root.resolve() not in Path(local_attention.__file__).resolve().parents:
                raise RuntimeError("a different pypto-lib models package is already imported")
            return local_attention
    raise RuntimeError("native V4.1 attention requires pypto-lib; set PYPTO_LIB_ROOT")


def _validate_native_config(config, runtime, cache_layouts, library_config, *, platform, device_ids):
    if platform != "a5" or len(device_ids) != 1:
        raise ValueError("native V4.1 attention currently requires A5 with exactly one TP device")
    reference = library_config.FLASH
    fields = ("hidden_size", "num_attention_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank",
              "o_lora_rank", "o_groups", "sliding_window", "index_n_heads", "index_head_dim",
              "index_topk", "candidate_topk_blocks", "candidate_block_size")
    for field in fields:
        if getattr(config, field) != getattr(reference, field):
            raise ValueError(f"native V4.1 attention requires Flash {field}={getattr(reference, field)}")
    if config.text_config["rms_norm_eps"] != reference.rms_norm_eps:
        raise ValueError("native V4.1 attention requires the Flash RMS epsilon")
    if runtime.page_size % 256 or any(layout.rows_per_page % 128 for layout in cache_layouts):
        raise ValueError("native V4.1 attention requires page_size divisible by 256 source tokens")
    chunk = runtime.max_prefill_tokens_per_request or runtime.max_num_batched_tokens
    if chunk > library_config.PREFILL_MAX_TOKENS:
        raise ValueError("native V4.1 attention chunk exceeds the library prefill bound")
    if any(plan.compress_ratio == 2 and plan.attention_kind == "reindex" for plan in config.layer_plan):
        raise ValueError("native ratio-2 attention does not support Reindex layers")


class NativeCacheIO:
    """Byte transfers through L2 kernels, retaining each DeviceTensor's owner Buffer."""

    def __init__(self, worker, kernels, run_config, artifact_dir):
        self.worker, self.kernels = worker, kernels
        self.run_config, self.artifact_dir = run_config, artifact_dir
        self.handles = {}

    @staticmethod
    def _view(tensor):
        # dataclass replacement preserves the Buffer on runtimes that retain
        # it, including the ABA/liveness identity checked by ChipWorker.run.
        return replace(tensor, shape=(1, tensor.nbytes), dtype=torch.uint8)

    def _run(self, mode, *args):
        if mode not in self.handles:
            import pypto.language as pl
            config = replace(self.run_config, save_kernels_dir=str(Path(self.artifact_dir.name) / mode))
            scalars = {"size": pl.RUNTIME}
            if mode != "zero":
                scalars["offset"] = pl.RUNTIME
            compiled = self.kernels[mode].compile(config=config, **scalars)
            self.handles[mode] = self.worker.register(compiled)
        self.handles[mode](*args)

    def zero(self, tensor):
        self._run("zero", self._view(tensor), tensor.nbytes)

    def read(self, tensor, offset, size):
        if not 0 <= offset <= tensor.nbytes - size or size <= 0:
            raise ValueError("native cache read exceeds the resident allocation")
        host = torch.empty((1, size), dtype=torch.uint8)
        self._run("read", self._view(tensor), host, offset, size)
        return host

    def write(self, tensor, offset, host):
        size = host.numel()
        if host.device.type != "cpu" or host.dtype != torch.uint8 or not host.is_contiguous():
            raise ValueError("native cache restore requires contiguous CPU bytes")
        if not 0 <= offset <= tensor.nbytes - size or size <= 0:
            raise ValueError("native cache restore exceeds the resident allocation")
        self._run("write", host.reshape(1, size), self._view(tensor), offset, size)

    def clear(self):
        self.handles.clear()
        self.kernels.clear()


class NativeAttention:
    """One worker's packed attention state, with in-place transactional recovery."""

    def __init__(self, backend, *, worker, kernels, arguments, run_config, pack_weights, cache_io,
                 weight_budget_bytes=256 << 20, artifact_dir=None):
        self.backend, self.worker, self.kernels = backend, worker, kernels
        self.arguments = arguments
        self.run_config, self.pack_weights = run_config, pack_weights
        self.cache_io = cache_io
        self.layouts, self.limits = backend.pool.layouts, backend.pool.limits
        self.weight_budget_bytes = weight_budget_bytes
        self.artifact_dir = artifact_dir
        self.buffers = {}
        self.states = {}
        self.state_leases = {}
        self.leases = {}
        self.weights = OrderedDict()
        self.weight_bytes = 0
        self.handles = {}
        self.journal = None
        self.outer_journal = None
        self._batch_leases = None
        self._closed = False
        self._closing = False

    def _zeros(self, shape, dtype):
        tensor = self.worker.alloc_tensor(shape, dtype)
        try:
            self.cache_io.zero(tensor)
            return tensor
        except BaseException:
            self.worker.free_tensor(tensor)
            raise

    def _pool(self, name):
        if name not in self.buffers:
            layout = self.layouts[name]
            dim = self.backend.config.index_head_dim if layout.kind == "index_k" else self.backend.config.head_dim
            payload = dim if layout.kind == "swa" else dim // 2
            scale = dim // (16 if layout.kind == "main_kv" else 32)
            dtype = torch.float8_e4m3fn if layout.kind == "swa" else torch.uint8
            scale_dtype = torch.float8_e4m3fn if layout.kind == "main_kv" else torch.float8_e8m0fnu
            blocks = self.limits[name] * layout.rows_per_page // 128
            data = self._zeros((blocks, 128, 1, payload), dtype)
            try:
                scales = self._zeros((blocks, 128, 1, scale), scale_dtype)
            except BaseException:
                self.worker.free_tensor(data)
                raise
            self.buffers[name] = data, scales
        return self.buffers[name]

    def _row(self, transaction, name, position, *, write=False):
        slot = transaction.map_position(name, position, write=write)
        layout = self.layouts[name]
        if layout.kind == "index_k":
            # Validate the real index lease first, then use a deterministic
            # backing permutation. This does not require equal scheduler IDs.
            slot = transaction.map_position(f"main_kv.{layout.layer_id}", position, write=write)
        if not 0 <= slot.page_id < self.limits[name]:
            raise ValueError("native cache address exceeds the scheduler pool")
        return slot.page_id * layout.rows_per_page + slot.row_offset

    def _snapshot(self, key, segments):
        if self.journal is None:
            raise RuntimeError("native cache mutation requires an active batch")
        if key not in self.journal:
            saved = []
            for tensor, offset, size in segments:
                host = self.cache_io.read(tensor, offset, size)
                saved.append((tensor, offset, host))
            self.journal[key] = saved
            if self.outer_journal is not None and key not in self.outer_journal:
                # Snapshots are immutable; restore copies bytes into the same
                # device allocation and never installs a journal as live data.
                self.outer_journal[key] = saved

    def _dirty(self, transaction, name):
        layout = self.layouts[name]
        pools = self._pool(name)
        pages = {
            self._row(transaction, name, slot.logical_row * layout.compress_ratio, write=True)
            // layout.rows_per_page
            for slot in transaction.write_slots(name)
        }
        for page in pages:
            segments = []
            for tensor in pools:
                size = layout.rows_per_page * tensor.shape[-1]
                segments.append((tensor, page * size, size))
            self._snapshot((name, page), segments)

    def begin(self, contexts):
        if self._closed or self._closing:
            raise RuntimeError("native attention provider is closed")
        self.journal = {}
        self._batch_leases = dict(self.leases)
        self._collect_states()
        for context in contexts:
            lease = context.cache.lease
            key = context.work.request_id, context.generation
            if (lease.request_id, lease.generation) != key:
                raise ValueError("native context and cache generation disagree")
            if key in self.leases and self.leases[key] is not lease:
                raise ValueError("native request ID/generation refers to a different cache lease")
            self.leases[key] = lease

    def _restore(self, journal):
        for segments in journal.values():
            for tensor, offset, host in segments:
                self.cache_io.write(tensor, offset, host)

    def abort(self):
        if self.journal is not None:
            self._restore(self.journal)
        self.journal = None
        if self._batch_leases is not None:
            self.leases = self._batch_leases
            self._batch_leases = None
        self._collect_states()

    def checkpoint(self):
        if self.outer_journal is not None:
            raise RuntimeError("nested native attention checkpoints are unsupported")
        self.outer_journal = {}
        return self.outer_journal, dict(self.leases)

    def finish_checkpoint(self, checkpoint, *, restore=False):
        journal, leases = checkpoint
        if journal is not self.outer_journal:
            raise ValueError("native attention checkpoint identity mismatch")
        if restore:
            self._restore(journal)
            self.leases = leases
        self.outer_journal = None
        self.journal = None
        self._collect_states()

    def release(self, request_id, generation):
        self.leases.pop((request_id, generation), None)
        self._collect_states()

    def _collect_states(self):
        for key in tuple(self.states):
            live = any(self.state_leases[key] is lease for lease in self.leases.values())
            journal_key = "compressor", key
            protected = journal_key in (self.journal or {}) or journal_key in (self.outer_journal or {})
            if not live and not protected:
                self.worker.free_tensor(self.states.pop(key))
                del self.state_leases[key]

    def _state(self, transaction, source):
        key = id(transaction.lease), source
        if key not in self.states:
            self.states[key] = self._zeros((32, 2, self.backend.config.head_dim), torch.float32)
            self.state_leases[key] = transaction.lease
        state = self.states[key]
        # The backend dispatches one request at a time; native state row zero
        # is private to this lease. No second batch-slot allocator is needed.
        self._snapshot(("compressor", key), [(state, 0, state.nbytes)])
        return state

    def _metadata(self, layer_id, transaction):
        plan = self.backend.config.layer_plan[layer_id]
        ratio = plan.compress_ratio
        count = transaction.end_pos - transaction.start_pos
        positions = torch.arange(transaction.start_pos, transaction.end_pos, dtype=torch.int32)
        frequency = rotary_frequencies(self.backend.config, bool(ratio), torch.device("cpu"))
        angles = positions.float()[:, None] * frequency
        window = f"swa.{layer_id}"
        indices = torch.full((count, 128), -1, dtype=torch.int32)
        for token, position in enumerate(positions.tolist()):
            rows = [self._row(transaction, window, row) for row in transaction.visible_rows(window, position)]
            indices[token, :len(rows)] = torch.tensor(rows, dtype=torch.int32)
        values = {
            "rope_cos": angles.cos(), "rope_sin": angles.sin(),
            "window_slots": torch.tensor([
                self._row(transaction, window, position, write=True) for position in positions.tolist()
            ], dtype=torch.int64),
            "window_indices": indices,
        }
        values["window_cache"], values["window_cache_scale"] = self._pool(window)
        self._dirty(transaction, window)
        if not ratio:
            return values
        source = plan.kv_source_layer_id
        main, index = f"main_kv.{source}", f"index_k.{source}"
        values["compressed_cache"], values["compressed_cache_scale"] = self._pool(main)
        if plan.attention_kind == "reuse":
            values["compressed_indices"] = transaction.index_for(layer_id)
            return values
        lengths = (positions + 1) // ratio
        completed = transaction.end_pos // ratio
        table = torch.zeros((1, max(1, (completed + 127) // 128)), dtype=torch.int32)
        for block in range((completed + 127) // 128):
            table[0, block] = self._row(transaction, index, block * 128 * ratio) // 128
        values.update(request_ids=torch.zeros(count, dtype=torch.int32), compressed_lens=lengths,
                      index_block_table=table)
        values["index_cache"], values["index_cache_scale"] = self._pool(index)
        values["topk_indices"] = torch.full((count, self.backend.config.index_topk), -1, dtype=torch.int32)
        if plan.attention_kind == "reindex":
            values["candidate_mask"] = transaction.candidates_for(layer_id)
            return values
        self._dirty(transaction, main)
        self._dirty(transaction, index)
        compressed_positions = positions // ratio * ratio
        compressed_angles = compressed_positions.float()[:, None] * frequency
        slots = [self._row(transaction, main, position // ratio * ratio, write=True)
                 if (position + 1) % ratio == 0 else -1 for position in positions.tolist()]
        values.update(compressed_rope_cos=compressed_angles.cos(), compressed_rope_sin=compressed_angles.sin(),
                      compressed_slots=torch.tensor(slots, dtype=torch.int64))
        if ratio == 2:
            values.update(position_ids=positions, compressor_state_rows=torch.zeros(count, dtype=torch.int64),
                          compressor_state=self._state(transaction, source))
        else:
            values["candidate_mask"] = torch.zeros((count, max(128, (completed + 127) // 128 * 128)),
                                                     dtype=torch.uint8)
        return values

    def _layer_weights(self, layer, ratio, names):
        if layer in self.weights:
            self.weights.move_to_end(layer)
            return self.weights[layer]
        # Keep a bounded resident layer bundle; checkpoint tiles remain owned
        # by the existing TensorStore rather than dequantizing the full model.
        packed = self.pack_weights(self.backend.ops.weights, layer, ratio, names)
        size = sum(value.numel() * value.element_size() for value in packed.values())
        if size > self.weight_budget_bytes:
            raise ValueError("native attention weight bundle exceeds the resident weight budget")
        while self.weights and self.weight_bytes + size > self.weight_budget_bytes:
            _, old = self.weights.popitem(last=False)
            for tensor in old.values():
                self.worker.free_tensor(tensor)
                self.weight_bytes -= tensor.nbytes
        resident = {}
        try:
            for name, value in packed.items():
                resident[name] = self.worker.alloc_tensor(value.shape, value.dtype, init=value)
        except BaseException:
            for tensor in resident.values():
                self.worker.free_tensor(tensor)
            raise
        self.weights[layer] = resident
        self.weight_bytes += size
        return resident

    def forward(self, layer_id, x, context, states):
        transaction = context.cache
        key = context.work.request_id, context.generation
        if self.leases.get(key) is not transaction.lease or self.journal is None:
            raise RuntimeError("native attention requires this request's active cache lease")
        state = states[layer_id]
        if state.position != transaction.start_pos or len(x) != transaction.end_pos - transaction.start_pos:
            raise ValueError("native attention input and cache endpoints disagree")
        plan = self.backend.config.layer_plan[layer_id]
        mode = "swa" if not plan.compress_ratio else f"c{plan.compress_ratio}a_{plan.attention_kind}"
        if plan.compress_ratio and not plan.owns_main_kv:
            if states[plan.kv_source_layer_id].position != transaction.end_pos:
                raise ValueError("native compressed KV source has not executed")
        kernel = self.kernels[mode]
        parameters = self.arguments[mode]
        values = self._metadata(layer_id, transaction)
        values["x"] = x.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        output_name = "partial" if mode.startswith("c2a") else "output"
        values[output_name] = torch.empty((len(x), self.backend.config.hidden_size), dtype=torch.float32)
        values["num_tokens"] = len(x)
        weight_names = tuple(name for name in parameters if name not in values)
        values.update(self._layer_weights(layer_id, plan.compress_ratio, weight_names))
        if mode not in self.handles:
            import pypto.language as pl
            config = self.run_config
            if self.artifact_dir is not None:
                config = replace(config, save_kernels_dir=str(Path(self.artifact_dir.name) / mode))
            compiled = kernel.compile(num_tokens=pl.RUNTIME, config=config)
            self.handles[mode] = self.worker.register(compiled)
        self.handles[mode](*(values[name] for name in parameters))
        if plan.owns_index_results:
            transaction.publish_index(layer_id, values["topk_indices"])
        if layer_id == self.backend.config.candidate_source_layer_id:
            transaction.publish_candidates(layer_id, values["candidate_mask"])
        state.position = transaction.end_pos
        for rows in (state.main, state.index, state.swa):
            if rows is not None:
                rows.length = transaction.end_pos // rows.layout.compress_ratio
        partial = values[output_name].to(self.backend.ops.device)
        return self.backend.ops.all_reduce(partial).to(torch.bfloat16)

    def diagnostics(self):
        return {"provider": "pypto-lib-a5-local-attention", "world_size": 1,
                "cache_bytes": sum(t.nbytes for pair in self.buffers.values() for t in pair),
                "compressor_state_bytes": sum(t.nbytes for t in self.states.values()),
                "weight_bytes": self.weight_bytes, "compiled_modes": sorted(self.handles),
                "journal_bytes": sum(host.numel() for segments in (self.journal or {}).values()
                                     for _, _, host in segments)}

    def close(self):
        if self._closed:
            return
        self._closing = True
        # ChipWorker cleanup is terminal but retryable. Retain its ownership
        # graph and artifacts if close raises so a second close can finish it.
        self.worker.close()
        self.cache_io.clear()
        self.buffers.clear()
        self.states.clear()
        self.state_leases.clear()
        self.weights.clear()
        self.leases.clear()
        self.handles.clear()
        self.kernels.clear()
        self.journal = self.outer_journal = None
        self._batch_leases = None
        if self.artifact_dir is not None:
            self.artifact_dir.cleanup()
        self._closed = True


def create_backend(*, config, runtime, cache_layouts, weight_loader, device_ids, platform="a5",
                   pypto_build_dir=None, use_compile_cache=False):
    """kernel_factory entry; retain the default bridge for the rest of the model."""
    # Reject unsupported hardware before library/worker initialization or HBM allocation.
    if platform != "a5" or len(device_ids) != 1:
        raise ValueError("native V4.1 attention currently requires A5 TP1")
    library = _load_library()
    library_config = library.load_kernel_configuration(tp_size=1, ep_size=2)
    _validate_native_config(config, runtime, cache_layouts, library_config, platform=platform, device_ids=device_ids)
    kernels = library.load_local_attention_kernels(tp_size=1, ep_size=2)
    from pypto.runtime import ChipWorker, RunConfig

    from .backend import create_backend as create_bridge
    from .native_weights import pack_native_attention_weights

    backend = create_bridge(config=config, runtime=runtime, cache_layouts=cache_layouts,
                            weight_loader=weight_loader, device_ids=device_ids, platform=platform,
                            pypto_build_dir=pypto_build_dir, use_compile_cache=use_compile_cache)
    artifacts = None
    worker = None
    try:
        artifacts = tempfile.TemporaryDirectory(prefix="v41-native-attention-", dir=pypto_build_dir)
        run_config = RunConfig(platform=platform, device_id=device_ids[0], save_kernels=True)
        worker = ChipWorker(config=run_config)
        cache_io = NativeCacheIO(worker, library.load_local_cache_kernels(), run_config, artifacts)
        backend.attention_provider = NativeAttention(
            backend, worker=worker, kernels=kernels, arguments=library.ARGUMENT_NAMES, run_config=run_config,
            pack_weights=pack_native_attention_weights, cache_io=cache_io, artifact_dir=artifacts)
        return backend
    except BaseException:
        if worker is not None:
            worker.close()
        if artifacts is not None:
            artifacts.cleanup()
        backend.close()
        raise
