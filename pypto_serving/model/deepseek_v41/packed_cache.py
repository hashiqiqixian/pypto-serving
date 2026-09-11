# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed tensor pages addressed exclusively through scheduler cache transactions."""

from __future__ import annotations

import torch

from .cache import CacheGroupLayout, CacheTransaction
from .numerics import QuantizedRows, quantize_rows


class PackedPagePool:
    """Lazy physical storage, with a bounded capacity and reversible page writes.

    The scheduler owns allocation and request page tables. This object only owns
    byte tensors for those physical IDs. A journal preserves the previous page
    on first write, including ring overwrites and failure after backend commit.
    """

    def __init__(self, layouts: tuple[CacheGroupLayout, ...], num_pages: int, *, device: str = "cpu"):
        stride = layouts[0].max_blocks_per_seq
        if type(num_pages) is not int or num_pages < stride or num_pages % stride:
            raise ValueError("cache capacity must contain integral scheduler request slots")
        self.layouts = {layout.name: layout for layout in layouts}
        self.limits = {name: num_pages // stride * layout.max_blocks_per_seq
                       for name, layout in self.layouts.items()}
        self.device = torch.device(device)
        self.pages: dict[tuple[str, int], torch.Tensor] = {}
        self.journal: dict[tuple[str, int], torch.Tensor | None] | None = None
        self.outer_journal: dict[tuple[str, int], torch.Tensor | None] | None = None

    def begin(self) -> None:
        # The previous commit's journal remains valid until the next batch.
        self.journal = {}

    def page(self, name: str, page_id: int, *, write: bool = False) -> torch.Tensor:
        layout = self.layouts[name]
        if not 0 <= page_id < self.limits[name]:
            raise ValueError("physical cache page exceeds the registered pool")
        key = name, page_id
        if write:
            if self.journal is None:
                raise RuntimeError("cache write requires a backend transaction")
            if key not in self.journal:
                self.journal[key] = self.pages[key].clone() if key in self.pages else None
            if self.outer_journal is not None and key not in self.outer_journal:
                old = self.journal[key]
                # An inner abort installs its snapshot as the live page. Keep
                # the outer snapshot independent so a retry cannot mutate it.
                self.outer_journal[key] = old.clone() if old is not None else None
            if key not in self.pages:
                self.pages[key] = torch.zeros((layout.rows_per_page, layout.row_bytes),
                                              dtype=torch.uint8, device=self.device)
        if key not in self.pages:
            raise ValueError("read from an unwritten cache page")
        return self.pages[key]

    def abort(self) -> None:
        if self.journal is None:
            raise RuntimeError("no cache transaction to abort")
        for key, old in self.journal.items():
            if old is None:
                self.pages.pop(key, None)
            else:
                self.pages[key] = old
        self.journal = None

    def close(self) -> None:
        self.journal = None
        self.outer_journal = None
        self.pages.clear()

    def checkpoint(self) -> object:
        if self.outer_journal is not None:
            raise RuntimeError("nested cache checkpoints are unsupported")
        self.outer_journal = {}
        return self.outer_journal

    def finish_checkpoint(self, checkpoint: object, *, restore: bool = False) -> None:
        if checkpoint is not self.outer_journal:
            raise ValueError("cache checkpoint identity mismatch")
        if restore:
            for key, old in self.outer_journal.items():
                if old is None:
                    self.pages.pop(key, None)
                else:
                    self.pages[key] = old
        self.outer_journal = None
        self.journal = None


class PackedRows:
    """Attention RowStore over actual FP8/FP4 payload and scale byte pages.

    A binding is valid only for the current host transaction. Rows are logical
    append endpoints; ring reuse and compressed source positions are resolved by
    CacheTransaction.map_position, never by an independent allocator.
    """

    def __init__(self, pool: PackedPagePool, name: str, dim: int, fmt: str):
        self.pool, self.name, self.dim, self.fmt = pool, name, dim, fmt
        self.layout = pool.layouts[name]
        self.block = 16 if fmt == "fp4_e2m1_e4m3" else 32
        self.payload_bytes = dim if fmt.startswith("fp8") else dim // 2
        if self.payload_bytes + dim // self.block != self.layout.row_bytes:
            raise ValueError("packed row format differs from scheduler group layout")
        self.length = 0
        self.transaction: CacheTransaction | None = None

    def bind(self, transaction: CacheTransaction) -> None:
        if self.length != transaction.start_pos // self.layout.compress_ratio:
            raise ValueError("packed row endpoint differs from request cache endpoint")
        self.transaction = transaction

    def _slot(self, row: int, *, write: bool = False):
        if self.transaction is None:
            raise RuntimeError("packed rows require a live cache transaction")
        return self.transaction.map_position(self.name, row * self.layout.compress_ratio, write=write)

    def append(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dim or not len(values):
            raise ValueError("packed append requires nonempty [rows,dim] values")
        packed = getattr(values, "_pypto_quantized_payload", None)
        if packed is None:
            packed = quantize_rows(values, self.fmt, self.block)
        if packed.fmt != self.fmt or packed.block_size != self.block:
            raise ValueError("cache producer used the wrong quantizer")
        data = torch.cat((packed.values, packed.scales), -1).to(self.pool.device)
        slots = [self._slot(row, write=True) for row in range(self.length, self.length + len(values))]
        for offset, slot in enumerate(slots):
            self.pool.page(self.name, slot.page_id, write=True)[slot.row_offset].copy_(data[offset])
        self.length += len(values)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache gather requires an integer row vector")
        if bool(((indices < 0) | (indices >= self.length)).any()):
            raise ValueError("cache gather is outside the retained endpoint")
        slots = [self._slot(row) for row in indices.cpu().tolist()]
        data = torch.empty((len(slots), self.layout.row_bytes), dtype=torch.uint8, device=self.pool.device)
        for offset, slot in enumerate(slots):
            data[offset].copy_(self.pool.page(self.name, slot.page_id)[slot.row_offset])
        packed = QuantizedRows(data[:, :self.payload_bytes], data[:, self.payload_bytes:], self.fmt, self.block)
        return packed.dequantize().to(indices.device)

    def read(self, start: int, end: int) -> torch.Tensor:
        if not 0 <= start < end <= self.length:
            raise ValueError("packed read requires a retained nonempty interval")
        return self.gather(torch.arange(start, end, device=self.pool.device))

    def truncate(self, length: int) -> None:
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("cache truncation cannot extend the endpoint")
        self.length = length
