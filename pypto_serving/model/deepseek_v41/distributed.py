# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded rank RPC for the real V4.1 tensor backend and HCCL collectives.

Each rank reuses DeepSeekV41Backend and PyptoMatmulOps, whose compiled launches
already use PyPTO's dispatcher. The existing L3 worker transports compiled
callables, not a persistent Torch/HCCL Python rank, so this layer only adds rank
lifetime, transaction ordering, and plain-data IPC. No second scheduler exists.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import wait
from pathlib import Path

import torch

from .cache import V41CacheState
from .config import DeepSeekV41Config
from .npu_runner import V41ExecutionContext, V41WorkItem


_MAX_TENSOR_BYTES = 64 << 20
_DTYPES = {str(dtype): dtype for dtype in (
    torch.bfloat16, torch.float16, torch.float32, torch.float64,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool,
)}


def _timeout(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds")
    return value


def _pack_tensor(value: torch.Tensor) -> dict:
    if value.dtype not in _DTYPES.values() or value.numel() * value.element_size() > _MAX_TENSOR_BYTES:
        raise ValueError("rank IPC tensor dtype or 64 MiB tensor budget exceeded")
    owned = value.detach().cpu().contiguous()
    return {"dtype": str(owned.dtype), "shape": tuple(owned.shape),
            "data": owned.reshape(-1).view(torch.uint8).numpy().tobytes()}


def _unpack_tensor(payload: dict) -> torch.Tensor:
    dtype, shape, data = _DTYPES[payload["dtype"]], tuple(payload["shape"]), payload["data"]
    if any(type(size) is not int or size < 0 for size in shape) or not isinstance(data, bytes):
        raise ValueError("invalid rank IPC tensor shape or bytes")
    count = math.prod(shape)
    if len(data) != count * torch.empty((), dtype=dtype).element_size() or len(data) > _MAX_TENSOR_BYTES:
        raise ValueError("invalid rank IPC tensor byte count")
    if not count:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).clone()


def _pack_vision(value):
    if value is None:
        return None
    token_types = value.token_types.tolist() if isinstance(value.token_types, torch.Tensor) else value.token_types
    return {"tokens": tuple(value.tokens), "token_types": tuple(token_types), "images": tuple(
        {"start": image.start, "patches": _pack_tensor(image.patches),
         "n_vit_h": image.n_vit_h, "n_vit_w": image.n_vit_w,
         "types": _pack_tensor(image.types)} for image in value.images
    )}


def _unpack_vision(value):
    if value is None:
        return None
    from .vision import ImageInput, VisionInputs

    images = tuple(ImageInput(item["start"], _unpack_tensor(item["patches"]), item["n_vit_h"],
                              item["n_vit_w"], _unpack_tensor(item["types"])) for item in value["images"])
    return VisionInputs(tuple(value["tokens"]), tuple(value["token_types"]), images)


def serialize_context(context: V41ExecutionContext) -> dict:
    """Copy scheduler-owned tables and input values; never pickle leases or MappingProxy."""
    work = context.work
    return {"request_id": work.request_id, "generation": context.generation,
            "token_ids": tuple(work.token_ids), "start_pos": work.start_pos,
            "end_pos": context.cache.end_pos, "partition": work.partition, "mode": work.mode,
            "pages": {name: tuple(pages) for name, pages in work.block_ids_by_group.items()},
            "engram_hashes": context.engram_hashes, "multimodal": _pack_vision(work.multimodal)}


class TorchCollective:
    """The same rank group owns tensor-parallel and expert-parallel arithmetic."""

    def all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        result = value.contiguous()
        dist.all_reduce(result)
        return result

    def all_gather(self, value: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        result = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(result, value.contiguous())
        return torch.cat(result, dim=-1)


def _ascend_backend(settings: dict, rank: int, world_size: int):
    import torch_npu  # noqa: F401 - registers Ascend devices

    from .backend import DeepSeekV41Backend
    from .numerics import TensorOps
    from .pypto_ops import PyptoMatmulOps
    from .tensor_store import DeepSeekV41TensorStore
    from .vision import VisionConfig

    device = settings["device_ids"][rank]
    torch.npu.set_device(device)
    raw = json.loads((Path(settings["model_dir"]) / "config.json").read_text())
    config = DeepSeekV41Config.from_dict(raw)
    store = DeepSeekV41TensorStore(settings["model_dir"], raw, rank=rank, world_size=world_size,
                                  max_load_bytes=settings["max_load_bytes"])
    provider = None
    try:
        provider = PyptoMatmulOps(device_id=device, build_dir=settings["build_dir"],
                                  platform=settings["platform"])
        ops = TensorOps(store, device=f"npu:{device}", rank=rank, world_size=world_size,
                        matmul_provider=provider, collective=TorchCollective())
        return DeepSeekV41Backend(config, settings["runtime"], settings["layouts"], ops,
                                  vision_config=VisionConfig.from_config(raw), platform=settings["platform"])
    except BaseException:
        if provider is not None:
            provider.close()
        store.close()
        raise


class RankSession:
    """Mirror host page ownership and dispatch one already initialized tensor rank."""

    def __init__(self, backend, cache: V41CacheState, rank: int):
        self.backend, self.cache, self.rank = backend, cache, rank
        self.leases = {}
        self.pending = None
        self.checkpoints = {}

    def begin(self, records: tuple[dict, ...]):
        if self.pending is not None:
            raise RuntimeError("rank already has a pending batch")
        pending = {"transactions": [], "bindings": [], "created": [], "ticket": None}
        self.pending = pending
        contexts = []
        try:
            for item in records:
                key = item["request_id"], item["generation"]
                lease = self.leases.get(key)
                if lease is None:
                    lease = self.cache.create(key[0], generation=key[1], partition=item["partition"])
                    self.leases[key] = lease
                    pending["created"].append(key)
                else:
                    pending["bindings"].append(self.cache.snapshot_bindings(lease))
                self.cache.bind(lease, item["pages"])
                transaction = self.cache.prepare(lease, item["start_pos"], item["end_pos"])
                pending["transactions"].append(transaction)
                work = V41WorkItem(key[0], item["token_ids"], item["start_pos"], item["pages"],
                                   item["partition"], item["mode"], _unpack_vision(item["multimodal"]))
                contexts.append(V41ExecutionContext(work, key[1], transaction, item["engram_hashes"]))
            ticket = self.backend.begin_batch(tuple(contexts))
            pending["ticket"] = ticket
            outputs = []
            for context in contexts:
                hidden = self.backend.embed(ticket, context)
                for layer in self.cache.layer_plan:
                    if layer.requires_engram:
                        hidden = self.backend.engram(ticket, layer, hidden, context)
                    hidden = self.backend.layer(ticket, layer, hidden, context)
                logits = self.backend.head(ticket, hidden, context)
                if self.rank == 0:
                    outputs.append(_pack_tensor(logits.float()))
            self.cache.validate_commit_many(pending["transactions"])
            return outputs if self.rank == 0 else None
        except BaseException:
            self.abort()
            raise

    def commit(self) -> None:
        if self.pending is None:
            raise RuntimeError("rank has no pending batch to commit")
        pending = self.pending
        self.cache.validate_commit_many(pending["transactions"])
        self.backend.commit_batch(pending["ticket"])
        self.cache.commit_many(pending["transactions"])
        self.pending = None

    def abort(self) -> None:
        if self.pending is None:
            return
        pending = self.pending
        if pending["ticket"] is not None:
            self.backend.abort_batch(pending["ticket"])
        for transaction in reversed(pending["transactions"]):
            self.cache.abort(transaction)
        for key in pending["created"]:
            self.cache.release(self.leases.pop(key))
            self.backend.release_request(*key)
        self.cache.restore_bindings_many(pending["bindings"])
        self.pending = None

    def checkpoint(self, tag: int) -> None:
        if self.pending is not None or self.checkpoints:
            raise RuntimeError("rank checkpoint requires an inactive, uncheckpointed session")
        metadata = {key: (lease, self.cache.valid_length(lease), self.cache.snapshot_bindings(lease))
                    for key, lease in self.leases.items()}
        self.checkpoints[tag] = self.backend.checkpoint(), metadata

    def finish_checkpoint(self, tag: int, restore: bool) -> None:
        backend_checkpoint, metadata = self.checkpoints[tag]
        if self.pending is not None:
            raise RuntimeError("finish_checkpoint requires no pending rank batch")
        self.backend.finish_checkpoint(backend_checkpoint, restore=restore)
        if restore:
            for key in tuple(self.leases):
                if key not in metadata:
                    self.cache.release(self.leases.pop(key))
            for key, (lease, _, _) in metadata.items():
                if self.leases.get(key) is not lease:
                    raise RuntimeError("a checkpointed request was released before rollback")
            self.cache.restore_bindings_many([entry[2] for entry in metadata.values()], restore_visibility=True)
        del self.checkpoints[tag]

    def release(self, key: tuple[str, int]) -> None:
        if self.pending is not None or self.checkpoints:
            raise RuntimeError("cannot release a request during a rank transaction/checkpoint")
        self.backend.release_request(*key)
        lease = self.leases.pop(key, None)
        if lease is not None:
            self.cache.release(lease)

    def dispatch(self, command: str, payload):
        if command == "diagnostics":
            return self.backend.diagnostics()
        if command == "begin":
            return self.begin(payload)
        if command == "commit":
            return self.commit()
        if command == "abort":
            return self.abort()
        if command == "release":
            return self.release(payload)
        if command == "checkpoint":
            return self.checkpoint(payload)
        if command == "finish_checkpoint":
            return self.finish_checkpoint(*payload)
        if command == "propose":
            if self.pending is not None:
                raise RuntimeError("proposal requires a committed target batch")
            proposal = self.backend.propose(*payload)
            if self.rank == 0:
                return {"output_ids": _pack_tensor(proposal.output_ids),
                        "logits": _pack_tensor(proposal.logits), "confidence": _pack_tensor(proposal.confidence)}
            return None
        raise ValueError(f"unknown rank command: {command}")


def _rank_main(connection, settings: dict, rank: int, init_method: str, factory, collective_backend: str) -> None:
    backend = None
    sequence = 0
    try:
        if os.name == "posix":
            os.setsid()  # All subsequently launched PyPTO children belong to this task-owned process group.
        torch.set_num_threads(1)
        import torch.distributed as dist

        if collective_backend == "hccl":
            import torch_npu  # noqa: F401

            torch.npu.set_device(settings["device_ids"][rank])
        dist.init_process_group(backend=collective_backend, init_method=init_method, rank=rank,
                                world_size=len(settings["device_ids"]),
                                timeout=timedelta(seconds=settings["rpc_timeout"]))
        backend = factory(settings, rank, len(settings["device_ids"]))
        cache = V41CacheState(backend.config, page_size=settings["runtime"].page_size,
                              max_seq_len=settings["runtime"].max_seq_len,
                              max_chunk_tokens=settings["runtime"].max_prefill_tokens_per_request
                              or settings["runtime"].max_num_batched_tokens)
        session = RankSession(backend, cache, rank)
        connection.send((0, True, (backend.capabilities, backend.num_pages)))
        while True:
            sequence, command, payload = connection.recv()
            if command == "close":
                break
            result = session.dispatch(command, payload)
            connection.send((sequence, True, result))
        backend.close()
        backend = None
        dist.destroy_process_group()
        connection.send((sequence, True, None))
    except BaseException as error:
        try:
            connection.send((sequence, False, f"rank {rank}: {type(error).__name__}: {error}\n"
                             + traceback.format_exc(limit=8)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                pass
        connection.close()


@dataclass
class _Ticket:
    contexts: tuple
    logits: tuple[torch.Tensor, ...]
    active: bool = True


@dataclass(frozen=True)
class _Checkpoint:
    owner: object
    tag: int


class DistributedV41Backend:
    """Run one complete forward concurrently on each rank, committing after host validation.

    Parent layer callbacks access the completed forward's result; computation is
    performed once in begin_batch by the same numerical backend on every rank.
    PYPTO_V41_INIT_TIMEOUT (300 s) and PYPTO_V41_RPC_TIMEOUT (600 s) must be positive.
    The private factory/backend overrides exist for explicit CPU collective tests;
    the production factory selects HCCL, Torch-NPU and real PyPTO kernels for the
    requested Ascend platform. A2/A3 is the default; A5 remains explicit.
    """

    def __init__(self, *, config, runtime, cache_layouts, weight_loader, device_ids,
                 platform: str = "a2a3", pypto_build_dir=None, use_compile_cache=False, _rank_factory=None,
                 _collective_backend="hccl") -> None:
        if platform not in ("a2a3", "a5"):
            raise ValueError("distributed V4.1 platform must be a2a3 or a5")
        ids = tuple(device_ids)
        if not 2 <= len(ids) <= 16 or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("distributed V4.1 requires 2..16 distinct nonnegative device IDs")
        config.expert_ownership(len(ids), 0)
        if _rank_factory is None and _collective_backend != "hccl":
            raise ValueError("the production distributed backend requires HCCL")
        self._lock, self._owner = threading.RLock(), object()
        self._closed, self._ticket, self._sequence = False, None, 0
        self._terminated, self._checkpoint = False, None
        self._processes, self._connections = [], []
        self._rpc_timeout = _timeout("PYPTO_V41_RPC_TIMEOUT", 600)
        init_timeout = _timeout("PYPTO_V41_INIT_TIMEOUT", 300)
        self._rendezvous = tempfile.TemporaryDirectory(prefix="pypto-v41-ranks-")
        settings = {"model_dir": str(weight_loader.model_dir), "max_load_bytes": weight_loader.max_load_bytes,
                    "runtime": runtime, "layouts": tuple(cache_layouts), "device_ids": ids,
                    "build_dir": pypto_build_dir, "rpc_timeout": self._rpc_timeout, "platform": platform}
        context = mp.get_context("spawn")
        try:
            init_method = (Path(self._rendezvous.name) / "store").as_uri()
            for rank in range(len(ids)):
                parent, child = context.Pipe()
                process = context.Process(target=_rank_main,
                                          args=(child, settings, rank, init_method,
                                                _rank_factory or _ascend_backend, _collective_backend),
                                          daemon=False)
                self._connections.append(parent)
                self._processes.append(process)
                process.start()
                child.close()
            records = self._receive(0, time.monotonic() + init_timeout)
            self.capabilities, self.num_pages = records[0]
            if any(record != records[0] for record in records[1:]) or self.capabilities.world_size != len(ids):
                raise RuntimeError("V4.1 ranks disagree on capabilities or physical page capacity")
            if self.capabilities.platform != platform:
                raise RuntimeError(f"V4.1 rank platform {self.capabilities.platform!r} differs from {platform!r}")
        except BaseException:
            self._terminate()
            raise

    def _receive(self, sequence: int, deadline: float, send_errors=None) -> list:
        pending, result = set(range(len(self._connections))), [None] * len(self._connections)
        indexes = {connection: rank for rank, connection in enumerate(self._connections)}
        while pending:
            if send_errors:
                raise RuntimeError("V4.1 rank RPC send failed") from send_errors[0]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"V4.1 rank RPC {sequence} timed out waiting for ranks {sorted(pending)}")
            ready = wait([self._connections[rank] for rank in pending], timeout=min(remaining, 0.1))
            for connection in ready:
                rank = indexes[connection]
                reply, ok, value = connection.recv()
                if reply != sequence:
                    raise RuntimeError(f"V4.1 rank {rank} returned stale RPC {reply}, expected {sequence}")
                if not ok:
                    raise RuntimeError(f"V4.1 worker failure: {value}")
                result[rank] = value
                pending.remove(rank)
            for rank in pending:
                if not self._processes[rank].is_alive():
                    raise RuntimeError(f"V4.1 rank {rank} exited with code {self._processes[rank].exitcode}")
        return result

    def _rpc(self, command: str, payload=None, *, timeout: float | None = None) -> list:
        if self._closed:
            raise RuntimeError("V4.1 distributed backend is closed or a rank failed")
        self._sequence += 1
        sequence = self._sequence
        errors = []

        def send(connection):
            try:
                connection.send((sequence, command, payload))
            except BaseException as error:
                errors.append(error)

        # A blocked pipe write must not bypass the RPC deadline. These task-owned
        # daemon senders are released when failed ranks and their pipe endpoints close.
        threads = [threading.Thread(target=send, args=(connection,), daemon=True)
                   for connection in self._connections]
        deadline = time.monotonic() + (self._rpc_timeout if timeout is None else timeout)
        try:
            for thread in threads:
                thread.start()
            result = self._receive(sequence, deadline, errors)
            if errors:
                raise RuntimeError("V4.1 rank RPC send failed") from errors[0]
            return result
        except BaseException:
            self._terminate()
            raise

    def _terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        self._closed = True
        for process in self._processes:
            if process.pid is None:
                continue
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    if process.is_alive():
                        with suppress(OSError):
                            process.terminate()
            elif process.is_alive():
                with suppress(OSError):
                    process.terminate()
        deadline = time.monotonic() + 3
        for process in self._processes:
            if process.pid is not None:
                process.join(max(0, deadline - time.monotonic()))
                if process.is_alive():
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except OSError:
                            with suppress(OSError):
                                process.kill()
                    else:
                        with suppress(OSError):
                            process.kill()
                    process.join(1)
        for connection in self._connections:
            with suppress(OSError):
                connection.close()
        # Preserve the original worker exception even if a Windows file-store
        # handle has not finished closing during bounded process cleanup.
        with suppress(OSError):
            self._rendezvous.cleanup()

    def _check(self, ticket: _Ticket, context=None) -> int:
        if self._closed or ticket is not self._ticket or not ticket.active:
            raise RuntimeError("inactive distributed V4.1 batch")
        if context is None:
            return -1
        for index, entry in enumerate(ticket.contexts):
            if entry is context:
                return index
        raise ValueError("context does not belong to this distributed batch")

    def begin_batch(self, contexts: tuple) -> _Ticket:
        with self._lock:
            if self._ticket is not None and self._ticket.active:
                raise RuntimeError("distributed V4.1 already has a pending batch")
            replies = self._rpc("begin", tuple(serialize_context(context) for context in contexts))
            try:
                self._ticket = _Ticket(contexts, tuple(_unpack_tensor(row) for row in replies[0]))
            except BaseException:
                self._rpc("abort")
                raise
            if len(self._ticket.logits) != len(contexts):
                self.abort_batch(self._ticket)
                raise ValueError("rank zero returned the wrong number of vocabulary rows")
            return self._ticket

    def embed(self, ticket, context):
        return self._check(ticket, context)

    def engram(self, ticket, layer, state, context):
        if state != self._check(ticket, context):
            raise ValueError("distributed result handle belongs to a different request")
        return state

    def layer(self, ticket, layer, state, context):
        return self.engram(ticket, layer, state, context)

    def head(self, ticket, state, context):
        if state != self._check(ticket, context):
            raise ValueError("distributed result handle belongs to a different request")
        return ticket.logits[state]

    def commit_batch(self, ticket) -> None:
        with self._lock:
            self._check(ticket)
            self._rpc("commit")
            ticket.active = False

    def abort_batch(self, ticket) -> None:
        with self._lock:
            self._check(ticket)
            self._rpc("abort")
            ticket.active = False

    def release_request(self, request_id: str, generation: int) -> None:
        with self._lock:
            if self._checkpoint is not None or (self._ticket is not None and self._ticket.active):
                raise RuntimeError("cannot release during a distributed transaction/checkpoint")
            self._rpc("release", (request_id, generation))

    def checkpoint(self) -> _Checkpoint:
        with self._lock:
            if self._checkpoint is not None or (self._ticket is not None and self._ticket.active):
                raise RuntimeError("checkpoint requires an inactive, uncheckpointed distributed backend")
            tag = self._sequence + 1
            self._rpc("checkpoint", tag)
            self._checkpoint = _Checkpoint(self._owner, tag)
            return self._checkpoint

    def finish_checkpoint(self, checkpoint: _Checkpoint, *, restore: bool = False) -> None:
        with self._lock:
            if (not isinstance(checkpoint, _Checkpoint) or checkpoint is not self._checkpoint
                    or checkpoint.owner is not self._owner):
                raise ValueError("checkpoint belongs to a different distributed backend")
            self._rpc("finish_checkpoint", (checkpoint.tag, restore))
            self._checkpoint = None
            if self._ticket is not None:
                self._ticket.active = False

    def propose(self, request_id: str, generation: int, anchor: int):
        from .draft import DraftOutput

        with self._lock:
            if self._ticket is not None and self._ticket.active:
                raise RuntimeError("proposal requires a committed distributed target batch")
            result = self._rpc("propose", (request_id, generation, anchor))[0]
            return DraftOutput(*(_unpack_tensor(result[name]) for name in ("output_ids", "logits", "confidence")))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    self._rpc("close", timeout=min(self._rpc_timeout, 5))
                finally:
                    self._terminate()

    def diagnostics(self) -> dict:
        """Return actual per-rank counters without summing replicated logical capacity."""
        with self._lock:
            return {"ranks": self._rpc("diagnostics")}
