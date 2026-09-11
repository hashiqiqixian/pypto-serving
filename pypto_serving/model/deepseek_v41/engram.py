# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded, per-request CPU Engram hashes and sparse row lookup.

Semantics follow DeepSeek-V4.1-Flash inference/engram.py and model.py at
dba1be0a40aa45a94ad051997016db3960a90277. This is an independent host implementation;
it does not claim NPU hashing, allocate an embedding table, or implement its gate.
The exact tokenizer normalizers and PCG64 multiplier stream require tokenizers and
NumPy only when constructing their respective inputs. State and lookup use stdlib.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    if value % 3 == 0:
        return value == 3
    divisor = 5
    while divisor * divisor <= value:
        if value % divisor == 0 or value % (divisor + 2) == 0:
            return False
        divisor += 6
    return True


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @property
    def hash_columns(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> EngramLayout | None:
        """Accept the HF text_config mapping (or its containing config)."""
        text = config.get("text_config", config)
        layer_ids = tuple(text.get("engram_layer_ids", ()))
        if not layer_ids:
            return None
        if any(type(i) is not int or i < 0 for i in layer_ids) or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Engram layer IDs must be distinct nonnegative integers")
        count = _integer(text["engram_max_ngram_size"], "engram_max_ngram_size", 2)
        heads = _integer(text["engram_n_heads"], "engram_n_heads", 1)
        dim = _integer(text["engram_head_dim"], "engram_head_dim", 1)
        base = _integer(text["engram_vocab_size"], "engram_vocab_size", 2)
        rows = tuple(text["engram_num_embeddings"])
        if len(rows) != len(layer_ids):
            raise ValueError("Engram table row counts must match the number of layers")
        used, primes = set(), []
        for table_rows in rows:
            _integer(table_rows, "engram_num_embeddings", 1)
            per_layer = []
            for _ in range(count - 1):
                current, per_ngram = base - 1, []
                for _ in range(heads):
                    current += 1
                    while current in used or not _prime(current):
                        current += 1
                    used.add(current)
                    per_ngram.append(current)
                per_layer.append(tuple(per_ngram))
            if sum(sum(group) for group in per_layer) != table_rows:
                raise ValueError("Engram table row count does not match the configured prime buckets")
            primes.append(tuple(per_layer))
        return cls(count, layer_ids, rows, tuple(primes), heads, dim)


def build_compressed_token_map(tokenizer) -> tuple[tuple[int, ...], int]:
    """Use raw tokenizer decode and the pinned normalization order, including byte tokens."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys, result = {}, []
    for token_id in range(len(tokenizer)):
        decoded = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in decoded
            else normalize.normalize_str(decoded) or decoded
        )
        if not isinstance(key, str):
            raise ValueError(f"tokenizer did not supply a string for token {token_id}")
        if key not in keys:
            keys[key] = len(keys)
        result.append(keys[key])
    return tuple(result), len(keys)


def token_map_sha256(token_map: Sequence[int]) -> str:
    """Stable digest of compressed IDs encoded as unsigned 64-bit little-endian integers."""
    digest = hashlib.sha256()
    for value in token_map:
        _integer(value, "compressed token ID")
        if value >= 1 << 63:
            raise ValueError("compressed token IDs must fit signed int64")
        digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


def compute_hash_multipliers(layout: EngramLayout, compressed_vocab_size: int) -> tuple[tuple[int, ...], ...]:
    """Reproduce the official per-layer NumPy PCG64 stream exactly."""
    import numpy as np

    size = _integer(compressed_vocab_size, "compressed_vocab_size", 1)
    bound = max(1, (((1 << 63) - 1) // size) // 2)
    return tuple(
        tuple(
            int(value) * 2 + 1
            for value in np.random.default_rng(10007 * layer).integers(
                0, bound, size=layout.max_ngram_size, dtype=np.int64
            )
        )
        for layer in layout.layer_ids
    )


@dataclass(frozen=True)
class EngramSnapshot:
    owner: object
    position: int
    retained_start: int
    history: tuple[int, ...]


class EngramHashState:
    """One request's compressed-token history; image positions are stored as DEAD.

    History retains at most rollback_window + max_ngram_size - 1 entries. A
    snapshot captures this bounded state and can restore an entire failed chunk.
    Ordinary rollback supports only positions with enough retained lookback.
    """

    DEAD = -1

    def __init__(
        self,
        layout: EngramLayout,
        token_map: Sequence[int],
        *,
        compressed_vocab_size: int,
        pad_token_id: int,
        rollback_window: int = 256,
        multipliers: Sequence[Sequence[int]] | None = None,
        expected_token_map_sha256: str | None = None,
    ):
        self.layout = layout
        self.token_map = tuple(token_map)
        size = _integer(compressed_vocab_size, "compressed_vocab_size", 1)
        self.token_map_sha256 = token_map_sha256(self.token_map)
        unique_ids = set(self.token_map)
        if (
            len(unique_ids) != size
            or min(unique_ids, default=-1) != 0
            or max(unique_ids, default=-1) != size - 1
        ):
            raise ValueError("token map must exactly cover the configured compressed vocabulary")
        if expected_token_map_sha256 is not None and self.token_map_sha256 != expected_token_map_sha256:
            raise ValueError("compressed token map digest differs from the pinned mapping")
        pad = _integer(pad_token_id, "pad_token_id")
        if pad >= len(self.token_map):
            raise ValueError("pad_token_id is outside the tokenizer vocabulary")
        self.pad_id = self.token_map[pad]
        self.rollback_window = _integer(rollback_window, "rollback_window")
        values = compute_hash_multipliers(layout, size) if multipliers is None else multipliers
        self.multipliers = tuple(tuple(row) for row in values)
        if len(self.multipliers) != len(layout.layer_ids) or any(
            len(row) != layout.max_ngram_size
            or any(type(value) is not int or not 0 < value < 1 << 63 or value % 2 != 1 for value in row)
            for row in self.multipliers
        ):
            raise ValueError("hash multipliers must be odd positive int64 values with [layers,ngram] shape")
        self.position = 0
        self._retained_start = 0
        self._history: tuple[int, ...] = ()
        self._owner = object()

    @property
    def retained_tokens(self) -> int:
        return len(self._history)

    @property
    def minimum_rollback_position(self) -> int:
        first = self._retained_start + self.layout.max_ngram_size - 1 if self._retained_start else 0
        return max(first, self.position - self.rollback_window)

    def snapshot(self) -> EngramSnapshot:
        return EngramSnapshot(self._owner, self.position, self._retained_start, self._history)

    def restore(self, snapshot: EngramSnapshot) -> None:
        if not isinstance(snapshot, EngramSnapshot) or snapshot.owner is not self._owner:
            raise ValueError("Engram snapshot belongs to another request")
        self.position, self._retained_start, self._history = (
            snapshot.position,
            snapshot.retained_start,
            snapshot.history,
        )

    def rollback(self, position: int) -> None:
        position = _integer(position, "rollback position")
        if not self.minimum_rollback_position <= position <= self.position:
            raise ValueError("rollback position is outside the retained Engram window")
        self._history = self._history[: position - self._retained_start]
        self.position = position

    def advance(
        self,
        token_ids: Sequence[int],
        *,
        start_pos: int,
        token_mask: Sequence[bool] | None = None,
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Return [position][Engram layer][hash column]; validate before mutating state."""
        if _integer(start_pos, "start_pos") != self.position:
            raise ValueError(f"Engram position mismatch: expected {self.position}, got {start_pos}")
        ids = tuple(token_ids)
        if any(type(token) is not int or not 0 <= token < len(self.token_map) for token in ids):
            raise ValueError("token ID is outside the tokenizer vocabulary")
        mask = (True,) * len(ids) if token_mask is None else tuple(token_mask)
        if len(mask) != len(ids) or any(type(value) is not bool for value in mask):
            raise ValueError("token_mask must contain one bool per token")
        capacity = self.rollback_window + self.layout.max_ngram_size - 1
        history, result = deque(self._history), []
        for token, is_text in zip(ids, mask):
            history.append(self.token_map[token] if is_text else self.DEAD)
            blocked, context = False, []
            for shift in range(self.layout.max_ngram_size):
                source = history[-1 - shift] if shift < len(history) else self.DEAD
                blocked = blocked or source == self.DEAD
                context.append(self.pad_id if blocked else source)
            per_position = []
            for layer, multipliers in enumerate(self.multipliers):
                products = [
                    ((token * factor + (1 << 63)) % (1 << 64)) - (1 << 63)
                    for token, factor in zip(context, multipliers)
                ]
                rolling, offset, columns = products[0], 0, []
                for shift in range(1, self.layout.max_ngram_size):
                    rolling ^= products[shift]
                    for prime in self.layout.primes[layer][shift - 1]:
                        columns.append(rolling % prime + offset)
                        offset += prime
                per_position.append(tuple(columns))
            result.append(tuple(per_position))
            if len(history) > capacity:
                history.popleft()
        self.position += len(ids)
        self._history = tuple(history)
        self._retained_start = self.position - len(self._history)
        return tuple(result)


@dataclass(frozen=True)
class EngramShard:
    rank: int
    global_start: int
    valid_rows: int
    capacity_rows: int


def shard_plan(layout: EngramLayout, layer_id: int, world_size: int) -> tuple[EngramShard, ...]:
    """Contiguous ceil-divided table shards; padding is never a valid lookup row."""
    size = _integer(world_size, "world_size", 1)
    rows = layout.num_embeddings[layout.layer_ids.index(layer_id)]
    capacity = (rows + size - 1) // size
    return tuple(
        EngramShard(rank, rank * capacity, max(0, min(capacity, rows - rank * capacity)), capacity)
        for rank in range(size)
    )


RowReader = Callable[[int, int, tuple[int, ...]], Mapping[int, Sequence[float]]]


class SparseEngramLookup:
    """Read only requested rows, deduplicating and bounding each reader call.

    reader(layer_id, rank, local_row_ids) returns a mapping from every requested
    local ID to a dequantized head_dim vector. The reader owns storage placement
    and FP8/UE8M0 row decoding; this class owns sharding, bounds, and output order.
    """

    def __init__(
        self, layout: EngramLayout, reader: RowReader, *, world_size: int, max_rows_per_read: int = 256
    ):
        self.layout, self.reader = layout, reader
        self.world_size = _integer(world_size, "world_size", 1)
        self.max_rows_per_read = _integer(max_rows_per_read, "max_rows_per_read", 1)

    def lookup(self, layer_id: int, row_ids: Sequence[int]) -> tuple[tuple[float, ...], ...]:
        shards = shard_plan(self.layout, layer_id, self.world_size)
        total = self.layout.num_embeddings[self.layout.layer_ids.index(layer_id)]
        ids = tuple(row_ids)
        if any(type(row) is not int or not 0 <= row < total for row in ids):
            raise ValueError("Engram row ID is outside the unpadded table")
        grouped: dict[int, list[int]] = {}
        for row in sorted(set(ids)):
            rank, local = divmod(row, shards[0].capacity_rows)
            grouped.setdefault(rank, []).append(local)
        values = {}
        for rank, local_ids in grouped.items():
            for begin in range(0, len(local_ids), self.max_rows_per_read):
                requested = tuple(local_ids[begin : begin + self.max_rows_per_read])
                found = self.reader(layer_id, rank, requested)
                if not isinstance(found, Mapping) or set(found) != set(requested):
                    raise ValueError("row reader must return exactly the requested local IDs")
                for local in requested:
                    vector = tuple(found[local])
                    if len(vector) != self.layout.head_dim or any(
                        not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
                        for value in vector
                    ):
                        raise ValueError("row reader returned a nonfinite or incorrectly shaped embedding")
                    values[shards[rank].global_start + local] = tuple(float(value) for value in vector)
        return tuple(values[row] for row in ids)
