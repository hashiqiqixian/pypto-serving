# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded checkpoint slices for V4.1 text, vision, DSpark and sparse Engram.

Quantized matrix tiles expose unscaled BF16 E4M3/E2M1 values plus FP32 scales
per output row and K32 block. The caller applies scales after its FP32 dot
product. This uses the shared safetensors reader's real slicing API, never a
whole-tensor read followed by a slice. Checkpoints are immutable during a store
lifetime. Headers are validated once per shard; payloads on every slice read.

Load budgets count returned tensors and conservative temporary tensor buffers;
the separately bounded row cache, Python metadata, mmap address space, and caller
retained prior outputs are excluded. The consumer must not retain every yielded
matrix tile. CPU placement is explicit; no device-resident embedding is implied.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore, SafeOpenFn

from .checkpoint import read_header, read_index
from .config import DeepSeekV41Config
from .draft_spec import draft_weight_specs
from .numerics import decode_e2m1, decode_e4m3, decode_ue8m0
from .vision import VisionConfig, vision_weight_shapes
from .weight_loader import DeepSeekV41WeightLoader, ParallelMode, WeightLoadBudgetError
from .weight_spec import TensorSpec, backbone_weight_specs


_EXPERT = re.compile(r"^(layers|mtp)\.\d+\.ffn\.experts\.(\d+)\.")
_SHARED = re.compile(r"^mtp\.(\d+)\.(embed|head)(?:\.weight)?$")


class DeepSeekV41TensorStore:
    """Provide owned CPU weights, streamed matrix tiles and selected table rows.

    Args:
        model_dir: Immutable checkpoint directory with a safetensors index.
        raw_config: Complete published-format config, including vision and draft metadata.
        rank: Rank in the reference TP/EP group.
        world_size: Number of ranks in that group.
        parallel_mode: Explicit reference TP+EP or EP-only ownership.
        max_load_bytes: Positive per-operation tensor-buffer budget.
        row_cache_bytes: Maximum retained decoded embedding bytes, zero disables caching.
        out_tile_rows: Output rows per matrix tile, a positive multiple of 32.
        dense_k_tile: Reduction width for unquantized matrices.
        prefetch_rows: Maximum consecutive selected embedding rows per read.
        placement: ``cpu`` uses the row LRU; ``uncached`` reads without retaining rows.
        safe_open_fn: Optional shared safetensors opener for instrumentation.
    """

    def __init__(
        self,
        model_dir: str | Path,
        raw_config: Mapping[str, Any],
        *,
        rank: int = 0,
        world_size: int = 1,
        parallel_mode: ParallelMode = "reference_tp",
        max_load_bytes: int = 256 << 20,
        row_cache_bytes: int = 64 << 20,
        out_tile_rows: int = 128,
        dense_k_tile: int = 1024,
        prefetch_rows: int = 64,
        placement: str = "cpu",
        safe_open_fn: SafeOpenFn | None = None,
    ) -> None:
        self.config = DeepSeekV41Config.from_dict(raw_config)
        self.config.expert_ownership(world_size, rank)
        if parallel_mode not in ("ep_only", "reference_tp"):
            raise ValueError("parallel_mode must be ep_only or reference_tp")
        if placement not in ("cpu", "uncached"):
            raise ValueError("embedding placement must be cpu or uncached")
        for name, value in (
            ("max_load_bytes", max_load_bytes),
            ("out_tile_rows", out_tile_rows),
            ("dense_k_tile", dense_k_tile),
            ("prefetch_rows", prefetch_rows),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if out_tile_rows % 32 or type(row_cache_bytes) is not int or row_cache_bytes < 0:
            raise ValueError("out_tile_rows must align to 32; row_cache_bytes must be nonnegative")
        if parallel_mode == "reference_tp" and any(
            size % world_size
            for size in (
                self.config.o_groups,
                self.config.num_attention_heads,
                self.config.index_n_heads,
                self.config.vocab_size,
            )
        ):
            raise ValueError("reference_tp requires complete groups, heads and vocabulary shards")
        self.rank, self.world_size, self.parallel_mode = rank, world_size, parallel_mode
        self.max_load_bytes, self.out_tile_rows = max_load_bytes, out_tile_rows
        self.dense_k_tile, self.prefetch_rows = dense_k_tile, prefetch_rows
        self.row_cache_bytes = row_cache_bytes if placement == "cpu" else 0
        self.placement = placement
        self._draft_count = raw_config["text_config"]["num_nextn_predict_layers"]
        self._draft_experts = raw_config["text_config"].get("dspark_n_routed_experts", 0)
        if self._draft_count and self._draft_experts % world_size:
            raise ValueError("DSpark routed experts must divide evenly across ranks")
        self.specs = backbone_weight_specs(raw_config)
        if self._draft_count:
            self.specs.update(draft_weight_specs(raw_config))
        if "vision_config" in raw_config:
            self.specs.update(
                {
                    name: TensorSpec(shape, "BF16", name, "identity")
                    for name, shape in vision_weight_shapes(VisionConfig.from_config(raw_config)).items()
                }
            )
        self.model_dir = Path(model_dir).resolve()
        index = read_index(self.model_dir / "model.safetensors.index.json")
        unknown = index["weight_map"].keys() - self.specs.keys()
        if unknown:
            raise ValueError(f"unknown V4.1 checkpoint tensors: {sorted(unknown)[:8]}")
        self.store = LazySafetensorsStore(
            model_dir=self.model_dir, weight_map=index["weight_map"], device="cpu", safe_open_fn=safe_open_fn
        )
        self._shard_names: dict[str, list[str]] = {}
        for name, shard in index["weight_map"].items():
            self._shard_names.setdefault(shard, []).append(name)
        self._validated_shards: set[str] = set()
        self._rows: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        self._cached_bytes = 0
        self._closed = False

    def _name(self, name: str) -> str:
        if self._closed:
            raise RuntimeError("tensor store is closed")
        if not isinstance(name, str):
            raise ValueError("tensor name must be a string")
        shared = _SHARED.fullmatch(name)
        if shared:
            if int(shared[1]) >= self._draft_count:
                raise ValueError("shared DSpark alias refers to an absent stage")
            name = shared[2] + ".weight"
        elif name not in self.specs:
            name += ".weight"
        if name not in self.specs:
            raise KeyError(f"unknown V4.1 tensor: {name}")
        expert = _EXPERT.match(name)
        if expert:
            count = self._draft_experts if expert[1] == "mtp" else self.config.n_routed_experts
            per_rank = count // self.world_size
            if not self.rank * per_rank <= int(expert[2]) < (self.rank + 1) * per_rank:
                raise ValueError(f"expert tensor is not owned by this rank: {name}")
        self.store.require([name])
        return name

    def _budget(self, size: int) -> None:
        if size > self.max_load_bytes:
            raise WeightLoadBudgetError(
                f"tensor operation requires {size} bytes; budget is {self.max_load_bytes}"
            )

    def _layout(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        spec = self.specs[name]
        shape, offsets = list(spec.shape), [0] * len(spec.shape)
        if spec.dtype == "I8":
            shape[-1] *= 2
        axis = DeepSeekV41WeightLoader._axis(spec, self.parallel_mode)
        if axis is not None:
            if "row_shard_ceil" in spec.conversion:
                shape[axis] = (shape[axis] + self.world_size - 1) // self.world_size
            else:
                if shape[axis] % self.world_size:
                    raise ValueError(f"tensor dimension does not divide across reference TP ranks: {name}")
                shape[axis] //= self.world_size
                if spec.dtype == "F8_E4M3" and shape[axis] % 32:
                    raise ValueError(f"reference TP split crosses a 32-value quantization block: {name}")
            offsets[axis] = self.rank * shape[axis]
        return tuple(shape), tuple(offsets)

    def _read(self, name: str, ranges: tuple[slice, ...]) -> torch.Tensor:
        spec = self.specs[name]
        shape = tuple(part.stop - part.start for part in ranges)
        self._budget(math.prod(shape) * 32)
        filename = self.store.filename_for(name)
        path = self.store.path_for(name).resolve()
        if path.parent != self.model_dir:
            raise ValueError("checkpoint shard resolves outside model_dir")
        if filename not in self._validated_shards:
            header, _ = read_header(path)
            for source in self._shard_names[filename]:
                item, expected = header.get(source), self.specs[source]
                if item is None or tuple(item["shape"]) != expected.shape or item["dtype"] != expected.dtype:
                    raise ValueError(f"checkpoint header shape/dtype mismatch: {source}")
            self._validated_shards.add(filename)
        value = self.store.load_slice(name, ranges)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"source tensor must be a CPU tensor: {name}")
        if tuple(value.shape) != shape:
            raise ValueError(f"source tensor contract mismatch: {name}, expected {shape}/{spec.dtype}")
        # A safetensors column slice can retain the full source row stride.
        # Compact only this metadata-budgeted selection before applying the
        # whole-tensor validator and making the independently owned byte copy.
        value = value.contiguous()
        DeepSeekV41WeightLoader._validate_tensor(name, value, replace(spec, shape=shape))
        # Own only the requested bytes: no mmap backing or larger row batch can
        # accidentally remain alive through a returned tensor/cache view.
        return value.view(torch.uint8).clone().view(value.dtype)

    def matrix_shape(self, name: str) -> tuple[int, int]:
        """Return local logical [N,K], counting both FP4 values per packed byte."""
        shape, _ = self._layout(self._name(name))
        if len(shape) != 2:
            raise ValueError("matrix access requires a two-dimensional tensor")
        return shape

    def matrix_format(self, name: str) -> str:
        """Return fp8/fp4/bf16/float32 according to the reference execution contract."""
        spec = self.specs[self._name(name)]
        if "dequantize_fp8" in spec.conversion:
            return "bf16"
        if "cast_bf16_to_fp32" in spec.conversion or "cast_to_fp32_for_routing" in spec.conversion:
            return "float32"
        formats = {"F8_E4M3": "fp8", "I8": "fp4", "BF16": "bf16", "F32": "float32"}
        if spec.dtype not in formats:
            raise ValueError("a quantization scale is not a matrix weight")
        return formats[spec.dtype]

    def packed_fp4(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one owned expert matrix without expanding its checkpoint nibbles.

        This TP1 bring-up API bounds the complete returned bundle and read scratch.
        It deliberately rejects tensor sharding; expert ownership is still checked.
        """
        if self.world_size != 1:
            raise ValueError("native packed expert loading currently requires TP1")
        name = self._name(name)
        if self.matrix_format(name) != "fp4":
            raise ValueError("packed_fp4 requires a checkpoint FP4 expert matrix")
        rows, width = self.matrix_shape(name)
        scale_name = self._name(name.removesuffix(".weight") + ".scale")
        row_bytes = width // 2 + width // 32
        self._budget(rows * row_bytes + min(rows, self.out_tile_rows) * row_bytes * 32)
        payload = torch.empty((rows, width // 2), dtype=torch.uint8)
        scales = torch.empty((rows, width // 32), dtype=torch.float8_e8m0fnu)
        for start in range(0, rows, self.out_tile_rows):
            end = min(rows, start + self.out_tile_rows)
            payload[start:end].copy_(self._read(name, (slice(start, end), slice(0, width // 2))).view(torch.uint8))
            scales[start:end].copy_(self._read(scale_name, (slice(start, end), slice(0, width // 32))))
        return payload, scales

    def matrix_tiles(self, name: str) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor | None]]:
        """Yield local offsets, normalized [Ntile,Ktile] values and per-row scales.

        Tiles are ordered by output block, then by ascending reduction offset;
        every output block is complete before the next block begins.

        Quantized weights always use complete K32 tiles. wo_a is the explicit
        exception: it is converted to BF16 and has no separate returned scale.
        """
        name = self._name(name)
        spec = self.specs[name]
        shape, offsets = self._layout(name)
        if len(shape) != 2 or ".engram.embed." in name or spec.dtype == "F8_E8M0":
            raise ValueError("matrix_tiles requires a linear weight, not a table or scale")
        rows, inner = shape
        quantized = spec.dtype in ("F8_E4M3", "I8")
        if quantized and inner % 32:
            raise ValueError("quantized matrix reduction must contain complete K32 blocks")
        step = 32 if quantized else self.dense_k_tile
        for row in range(0, rows, self.out_tile_rows):
            count = min(self.out_tile_rows, rows - row)
            for column in range(0, inner, step):
                width = min(step, inner - column)
                self._budget(count * width * 64 + count * 32)
                start_row, start_col = row + offsets[0], column + offsets[1]
                divisor = 2 if spec.dtype == "I8" else 1
                value = self._read(
                    name,
                    (
                        slice(start_row, start_row + count),
                        slice(start_col // divisor, (start_col + width) // divisor),
                    ),
                )
                if quantized:
                    scale_name = name.removesuffix(".weight") + ".scale"
                    self.store.require([scale_name])
                    block_rows = 1 if spec.dtype == "I8" else 32
                    first, last = start_row // block_rows, (start_row + count + block_rows - 1) // block_rows
                    raw_scale = self._read(
                        scale_name, (slice(first, last), slice(start_col // 32, start_col // 32 + 1))
                    )
                    scale = decode_ue8m0(raw_scale.view(torch.uint8))[:, 0]
                    scale = scale[torch.arange(start_row, start_row + count) // block_rows - first]
                    decode = decode_e2m1 if spec.dtype == "I8" else decode_e4m3
                    values = decode(value.view(torch.uint8)).to(torch.bfloat16)
                    if "dequantize_fp8" in spec.conversion:
                        values = (values.float() * scale[:, None]).to(torch.bfloat16)
                        if not bool(torch.isfinite(values).all()):
                            raise ValueError(f"dequantized wo_a overflows BF16: {name}")
                        scale = None
                    yield row, column, values, scale
                else:
                    dtype = torch.float32 if self.matrix_format(name) == "float32" else torch.bfloat16
                    yield row, column, value.to(dtype), None

    def weight(self, name: str) -> torch.Tensor:
        """Load a budgeted small tensor, or stream a grouped BF16 wo_a projection."""
        name = self._name(name)
        spec = self.specs[name]
        shape, offsets = self._layout(name)
        if "row_shard_ceil" in spec.conversion or name.endswith("embed.weight"):
            raise ValueError("embedding tables must use selected-row embedding access")
        if len(shape) == 2 and spec.dtype != "F8_E8M0":
            dtype = torch.float32 if self.matrix_format(name) == "float32" else torch.bfloat16
            scratch = min(shape[0], self.out_tile_rows) * min(shape[1], self.dense_k_tile) * 64
            self._budget(math.prod(shape) * torch.empty((), dtype=dtype).element_size() + scratch)
            result = torch.empty(shape, dtype=dtype)
            for row, column, value, scale in self.matrix_tiles(name):
                if scale is not None:
                    value = (value.float() * scale[:, None]).to(dtype)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"dequantized weight overflows: {name}")
                result[row : row + value.shape[0], column : column + value.shape[1]].copy_(value)
            if "view_grouped_output_projection" in spec.conversion:
                groups = self.config.o_groups // (
                    self.world_size if self.parallel_mode == "reference_tp" else 1
                )
                result = result.reshape(groups, self.config.o_lora_rank, shape[1])
            return result
        ranges = tuple(slice(offset, offset + size) for offset, size in zip(offsets, shape))
        return self._read(name, ranges)

    def _embedding_ids(self, name: str, ids: torch.Tensor) -> tuple[str, dict[int, list[int]], int]:
        name = self._name(name)
        spec = self.specs[name]
        if not name.endswith("embed.weight") or len(spec.shape) != 2:
            raise ValueError("embedding requires a token or Engram table")
        if (
            not isinstance(ids, torch.Tensor)
            or ids.device.type != "cpu"
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("embedding IDs must be an integer CPU tensor")
        self._budget(ids.numel() * (spec.shape[1] * 2 + 16) + self.prefetch_rows * spec.shape[1] * 64)
        if bool(((ids < 0) | (ids >= spec.shape[0])).any()):
            raise ValueError("embedding ID is outside the unpadded table")
        shape, offsets = self._layout(name)
        start, end = offsets[0], min(spec.shape[0], offsets[0] + shape[0])
        positions: dict[int, list[int]] = {}
        for index, row in enumerate(ids.reshape(-1).tolist()):
            if start <= row < end:
                positions.setdefault(row, []).append(index)
        return name, positions, spec.shape[1]

    def _cache_row(self, key: tuple[str, int], value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        if size > self.row_cache_bytes:
            return
        while self._cached_bytes + size > self.row_cache_bytes:
            _, old = self._rows.popitem(last=False)
            self._cached_bytes -= old.numel() * old.element_size()
        self._rows[key] = value
        self._cached_bytes += size

    def _selected_rows(self, name: str, requested: list[int]) -> Iterator[tuple[int, torch.Tensor]]:
        cursor, dimension = 0, self.specs[name].shape[1]
        while cursor < len(requested):
            row = requested[cursor]
            key = (name, row)
            if key in self._rows:
                self._rows.move_to_end(key)
                yield row, self._rows[key]
                cursor += 1
                continue
            stop = cursor + 1
            while (
                stop < len(requested)
                and stop - cursor < self.prefetch_rows
                and requested[stop] == requested[stop - 1] + 1
                and (name, requested[stop]) not in self._rows
            ):
                stop += 1
            end = requested[stop - 1] + 1
            values = self._read(name, (slice(row, end), slice(0, dimension)))
            if self.specs[name].dtype == "F8_E4M3":
                scale_name = name.removesuffix(".weight") + ".scale"
                self.store.require([scale_name])
                scale = self._read(scale_name, (slice(row, end), slice(0, dimension // 32)))
                values = (
                    (
                        decode_e4m3(values.view(torch.uint8)).unflatten(-1, (-1, 32))
                        * decode_ue8m0(scale.view(torch.uint8)).unsqueeze(-1)
                    )
                    .flatten(-2)
                    .to(torch.bfloat16)
                )
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("Engram dequantization overflows BF16")
            for offset in range(end - row):
                value = values[offset].clone()
                self._cache_row((name, row + offset), value)
                yield row + offset, value
            cursor = stop

    def embedding(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Return selected CPU BF16 rows, with zeros for rows owned by another rank."""
        name, positions, dimension = self._embedding_ids(name, ids)
        output = torch.zeros((ids.numel(), dimension), dtype=torch.bfloat16)
        for row, value in self._selected_rows(name, sorted(positions)):
            output[positions[row]] = value
        return output.reshape(*ids.shape, dimension)

    def prefetch(self, name: str, ids: torch.Tensor) -> None:
        """Populate the bounded CPU row cache for explicitly selected upcoming IDs."""
        name, positions, _ = self._embedding_ids(name, ids)
        for _ in self._selected_rows(name, sorted(positions)):
            pass

    def close(self) -> None:
        """Release cached rows and reject later accesses; no open reader is retained."""
        self._rows.clear()
        self._cached_bytes = 0
        self._closed = True
