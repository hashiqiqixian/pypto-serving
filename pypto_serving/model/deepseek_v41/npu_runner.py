# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transactional host execution for V4.1; arithmetic is supplied by an A5 kernel bundle.

The runner never substitutes another model or synthetic logits. A backend's state
includes Hyper-Connections/pre_mix and its caches, compressors and Engram state.
begin_batch/abort_batch must cover *all* of those tensors. head applies the final
hc_pre, norm and output projection and returns the last token's vocabulary logits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Protocol

from .cache import CacheLease, CacheTransaction, V41CacheState
from .config import DeepSeekV41Config, LayerPlan
from .engram import EngramHashState
from .dspark import choose_verification_count


@dataclass(frozen=True)
class V41WorkItem:
    request_id: str
    token_ids: tuple[int, ...]
    start_pos: int
    block_ids_by_group: Mapping[str, Sequence[int]]
    partition: int = 0
    mode: str = "prefill"
    multimodal: Any = None


@dataclass(frozen=True)
class V41ExecutionContext:
    work: V41WorkItem
    generation: int
    cache: CacheTransaction
    engram_hashes: tuple


class V41KernelBackend(Protocol):
    """A batch commit is atomic; abort remains valid after a failed commit.

    begin_batch must clean up internally if it raises before returning a ticket.
    Outputs must own their storage through the caller's sampling step.
    Backend methods must not mutate host cache metadata supplied in a context.
    """

    def begin_batch(self, contexts: tuple[V41ExecutionContext, ...]) -> object: ...
    def embed(self, ticket: object, context: V41ExecutionContext) -> object: ...
    def engram(
        self, ticket: object, layer: LayerPlan, state: object, context: V41ExecutionContext
    ) -> object: ...
    def layer(
        self, ticket: object, layer: LayerPlan, state: object, context: V41ExecutionContext
    ) -> object: ...
    def head(self, ticket: object, state: object, context: V41ExecutionContext) -> object: ...
    def commit_batch(self, ticket: object) -> None: ...
    def abort_batch(self, ticket: object) -> None: ...
    def release_request(self, request_id: str, generation: int) -> None: ...
    def close(self) -> None: ...


@dataclass
class _Request:
    lease: CacheLease
    history: EngramHashState
    generation: int
    decode_ready: bool = False


class DeepSeekV41Runner:
    """Serialize host state transitions while allowing backend-internal parallelism."""

    def __init__(
        self,
        config: DeepSeekV41Config,
        cache: V41CacheState,
        backend: V41KernelBackend,
        history_factory: Callable[[], EngramHashState],
        *,
        max_requests: int,
    ) -> None:
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be positive")
        self.config, self.cache, self.backend = config, cache, backend
        self.history_factory, self.max_requests = history_factory, max_requests
        self._requests: dict[str, _Request] = {}
        self._next_generation = 0
        self._lock = RLock()
        self._closed = False
        self.speculation_stats = {name: 0 for name in ("rounds", "proposed", "verified", "accepted", "rejected")}

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("V4.1 runner is closed or backend recovery failed; recreate the executor")

    def execute(self, items: Sequence[V41WorkItem], *, validate_outputs: Callable[[list], Any]) -> Any:
        """Execute all backbone layers, then commit only after output validation."""
        with self._lock:
            self._check_open()
            if not items or len({item.request_id for item in items}) != len(items):
                raise ValueError("a batch must contain distinct, nonempty requests")
            contexts, pending, created, snapshots, bindings = [], [], [], [], []
            ticket = None
            began = False
            try:
                for item in items:
                    if not isinstance(item.request_id, str) or not item.request_id:
                        raise ValueError("request_id must be nonempty")
                    if item.mode not in ("prefill", "decode"):
                        raise ValueError("only prefill and autoregressive decode are implemented")
                    if not item.token_ids or any(
                        type(t) is not int or not 0 <= t < self.config.vocab_size for t in item.token_ids
                    ):
                        raise ValueError("token IDs must be in the checkpoint vocabulary")
                    if item.mode == "decode" and len(item.token_ids) != 1:
                        raise ValueError("autoregressive decode consumes exactly one token")
                    request = self._requests.get(item.request_id)
                    is_new = request is None
                    if request is None:
                        if item.mode != "prefill" or item.start_pos != 0:
                            raise ValueError("unknown request must start with prefill at position zero")
                        if len(self._requests) >= self.max_requests:
                            raise ValueError("V4.1 request capacity exceeded")
                        history = self.history_factory()
                        self._next_generation += 1
                        lease = self.cache.create(
                            item.request_id, generation=self._next_generation, partition=item.partition
                        )
                        request = _Request(lease, history, self._next_generation)
                        self._requests[item.request_id] = request
                        created.append(item.request_id)
                    if request.decode_ready != (item.mode == "decode"):
                        raise ValueError("prefill/decode phase mismatch; finalize terminal prefill first")
                    if request.lease.partition != item.partition:
                        raise ValueError("a request cannot change its cache partition")
                    if not is_new:
                        bindings.append(self.cache.snapshot_bindings(request.lease))
                    self.cache.bind(request.lease, item.block_ids_by_group)
                    tx = self.cache.prepare(
                        request.lease, item.start_pos, item.start_pos + len(item.token_ids)
                    )
                    pending.append(tx)
                    snapshots.append((request, request.history.snapshot()))
                    mask = None
                    if item.multimodal is not None:
                        mask = tuple(kind < 0 for kind in item.multimodal.token_types[
                            item.start_pos: item.start_pos + len(item.token_ids)])
                        if len(mask) != len(item.token_ids):
                            raise ValueError("multimodal token types do not cover the prefill chunk")
                    hashes = request.history.advance(item.token_ids, start_pos=item.start_pos, token_mask=mask)
                    contexts.append(V41ExecutionContext(item, request.generation, tx, hashes))
                ticket = self.backend.begin_batch(tuple(contexts))
                began = True
                outputs = []
                for context in contexts:
                    state = self.backend.embed(ticket, context)
                    for layer in self.config.layer_plan:
                        if layer.requires_engram:
                            state = self.backend.engram(ticket, layer, state, context)
                        state = self.backend.layer(ticket, layer, state, context)
                    outputs.append(self.backend.head(ticket, state, context))
                result = validate_outputs(outputs)
                self.cache.validate_commit_many(pending)
                self.backend.commit_batch(ticket)
                self.cache.commit_many(pending)
                return result
            except BaseException:
                try:
                    if began:
                        self.backend.abort_batch(ticket)
                    for tx in reversed(pending):
                        self.cache.abort(tx)
                    for request, snapshot in snapshots:
                        request.history.restore(snapshot)
                    for request_id in created:
                        request = self._requests.pop(request_id)
                        self.cache.release(request.lease)
                        self.backend.release_request(request_id, request.generation)
                    self.cache.restore_bindings_many(bindings)
                except BaseException as recovery_error:
                    self._closed = True
                    raise RuntimeError(
                        "V4.1 backend/state rollback failed; executor is unusable"
                    ) from recovery_error
                raise

    def speculate(self, items: Sequence[V41WorkItem], *, extra_tokens: int,
                  validate_outputs: Callable[[list], Any],
                  minimum_score: float | None = None) -> tuple[Any, list[list[int]]]:
        """Greedy DSpark with sequential target verification and atomic batch recovery.

        Only accepted prefix tokens are consumed by the target; the final
        correction remains pending for the next dispatch. The initial path pays
        for sequential verification and does not claim speculative acceleration.
        """
        with self._lock:
            self._check_open()
            if type(extra_tokens) is not int or not 1 <= extra_tokens <= self.config.text_config["dspark_block_size"]:
                raise ValueError("DSpark extra tokens exceed the configured draft block")
            if not items or any(item.mode != "decode" for item in items):
                raise ValueError("DSpark requires existing decode requests")
            if len({item.request_id for item in items}) != len(items):
                raise ValueError("duplicate DSpark request IDs")
            choose_verification_count((), 0, minimum_score)
            stats = {name: 0 for name in self.speculation_stats}
            snapshots = []
            for item in items:
                request = self._requests[item.request_id]
                snapshots.append((request, request.history.snapshot(), self.cache.snapshot_bindings(request.lease)))
            checkpoint = self.backend.checkpoint()
            try:
                logits = self.execute(items, validate_outputs=validate_outputs)
                emitted = [[int(row.argmax().item())] for row in logits]
                for index, item in enumerate(items):
                    count = min(extra_tokens, self.cache.max_seq_len - item.start_pos - 1)
                    # The proposal graph evaluates its entire bidirectional block.
                    # At the sequence boundary keep the normal target prediction.
                    if self.cache.max_seq_len - item.start_pos - 1 < self.config.text_config["dspark_block_size"]:
                        continue
                    request = self._requests[item.request_id]
                    proposal = self.backend.propose(item.request_id, request.generation, emitted[index][0])
                    ids = proposal.output_ids.reshape(-1).cpu().tolist()
                    scores = proposal.confidence.reshape(-1).cpu().tolist()
                    if (len(ids) != self.config.text_config["dspark_block_size"] + 1
                            or len(scores) != len(ids) - 1 or ids[0] != emitted[index][0]
                            or any(type(token) is not int or not 0 <= token < self.config.vocab_size for token in ids)):
                        raise ValueError("DSpark proposal has invalid anchor, length or token IDs")
                    count = choose_verification_count(scores, count, minimum_score)
                    stats["rounds"] += 1
                    stats["proposed"] += len(ids) - 1
                    for offset in range(count):
                        work = replace(item, token_ids=(emitted[index][-1],), start_pos=item.start_pos + offset + 1)
                        target = self.execute([work], validate_outputs=validate_outputs)
                        predicted = int(target[0].argmax().item())
                        emitted[index].append(predicted)
                        stats["verified"] += 1
                        if predicted != ids[offset + 1]:
                            stats["rejected"] += 1
                            break
                        stats["accepted"] += 1
                self.backend.finish_checkpoint(checkpoint)
                for name, value in stats.items():
                    self.speculation_stats[name] += value
                return logits, emitted
            except BaseException:
                try:
                    self.backend.finish_checkpoint(checkpoint, restore=True)
                    for request, history, bindings in snapshots:
                        request.history.restore(history)
                    self.cache.restore_bindings_many([entry[2] for entry in snapshots], restore_visibility=True)
                except BaseException as recovery_error:
                    self._closed = True
                    raise RuntimeError("DSpark batch recovery failed; recreate executor") from recovery_error
                raise

    def finalize_prefill(self, request_ids: Sequence[str]) -> None:
        with self._lock:
            self._check_open()
            if len(set(request_ids)) != len(request_ids):
                raise ValueError("duplicate request IDs")
            requests = [self._requests[rid] for rid in request_ids]
            if any(request.decode_ready or request.history.position == 0 for request in requests):
                raise ValueError("terminal prefill must be finalized exactly once")
            for request in requests:
                request.decode_ready = True

    def position(self, request_id: str) -> int:
        with self._lock:
            return self._requests[request_id].history.position

    def release(self, request_ids: Sequence[str]) -> None:
        with self._lock:
            self._check_open()
            for request_id in dict.fromkeys(request_ids):
                request = self._requests.get(request_id)
                if request is not None:
                    try:
                        self.backend.release_request(request_id, request.generation)
                        self.cache.release(request.lease)
                    except BaseException:
                        self._closed = True
                        raise
                    del self._requests[request_id]

    def close(self) -> None:
        with self._lock:
            # Always let the backend reclaim tensors even after an abort failure.
            try:
                self.backend.close()
            finally:
                self._closed = True
                self._requests.clear()
