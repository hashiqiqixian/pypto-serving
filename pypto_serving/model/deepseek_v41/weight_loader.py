# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded V4.1 CPU weight loading into an explicit reference layout.

This reuses LazySafetensorsStore; it neither defines an Ascend ABI nor uploads data.
``ep_only`` partitions routed experts and replicates every other tensor. Explicit
``reference_tp`` additionally applies inference/convert.py's tensor/Engram sharding.
Both preserve global expert IDs. FP4 remains packed I8 with E2M1 metadata; it is not
renamed to INT4, unpacked to BF16, or changed to a device-specific float4 dtype.

Returned wo_a weights are contiguous BF16 [local_groups, o_rank, heads_per_group*D].
Other linear weights stay [N,K], or [N,K/2] for packed FP4. UE8M0 stays native E8M0.
The shared store reads whole tensors: budget estimates therefore include full source
tensors even when only a TP slice is retained. Huge Engram tables are rejected by the
default budget; this is not a streaming table loader. Budgets cover tensor buffers
for one call, not the process, mmap virtual address space, or previously returned calls.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from pypto_serving.model.common.weights.store import LazySafetensorsStore, SafeOpenFn
from pypto_serving.model.deepseek_v41.checkpoint import DTYPE_BYTES, read_header, read_index
from pypto_serving.model.deepseek_v41.config import DeepSeekV41Config
from pypto_serving.model.deepseek_v41.weight_spec import TensorSpec, backbone_weight_specs


ParallelMode = Literal["ep_only", "reference_tp"]
DEFAULT_MAX_LOAD_BYTES = 256 * 1024 * 1024
_EXPERT = re.compile(r"\.ffn\.experts\.(\d+)\.")


@dataclass(frozen=True)
class ReferenceWeights:
    """One owned CPU tensor bundle, with source provenance and layout declarations."""

    tensors: dict[str, torch.Tensor]
    formats: dict[str, str]
    source_names: tuple[str, ...]
    parallel_mode: ParallelMode
    rank: int
    world_size: int
    estimated_peak_bytes: int


class WeightLoadBudgetError(ValueError):
    """A requested load exceeds its explicit conservative tensor-buffer budget."""


class DeepSeekV41WeightLoader:
    """Select, validate, shard, and transform V4.1 weights on CPU.

    Args:
        model_dir: Directory containing model.safetensors.index.json and its shards.
        raw_config: Complete HF config, validated independently from V4.
        max_load_bytes: Positive per-call tensor-buffer limit; never unlimited.
        safe_open_fn: Optional shared-store opener for instrumentation or tests.
    """

    def __init__(
        self,
        model_dir: str | Path,
        raw_config: Mapping[str, Any],
        *,
        max_load_bytes: int = DEFAULT_MAX_LOAD_BYTES,
        safe_open_fn: SafeOpenFn | None = None,
    ) -> None:
        if type(max_load_bytes) is not int or max_load_bytes <= 0:
            raise ValueError("max_load_bytes must be a positive integer")
        self.config = DeepSeekV41Config.from_dict(raw_config)
        self.specs = backbone_weight_specs(raw_config)
        self.max_load_bytes = max_load_bytes
        self.model_dir = Path(model_dir).resolve()
        index = read_index(self.model_dir / "model.safetensors.index.json")
        unknown = [
            name
            for name in index["weight_map"]
            if name not in self.specs and not name.startswith(("vision.", "aligner.", "image_", "mtp."))
        ]
        if unknown:
            raise ValueError(f"unrecognized V4.1 text tensors in index: {unknown[:8]}")
        self.store = LazySafetensorsStore(
            model_dir=self.model_dir, weight_map=index["weight_map"], device="cpu", safe_open_fn=safe_open_fn
        )
        self._layer_names: dict[int, list[str]] = {}
        self._global_names: list[str] = []
        for name in self.specs:
            if name.startswith("layers."):
                self._layer_names.setdefault(int(name.split(".", 2)[1]), []).append(name)
            else:
                self._global_names.append(name)

    def load_layer(
        self,
        layer_id: int,
        rank: int = 0,
        world_size: int = 1,
        *,
        parallel_mode: ParallelMode = "ep_only",
        tensor_names: Sequence[str] | None = None,
    ) -> ReferenceWeights:
        """Load a layer's owned tensors, or an explicit subset with quantization companions.

        ``tensor_names`` uses complete source names. Missing tensors, foreign-rank
        experts and names belonging to other layers are errors. Selecting a weight
        or scale automatically includes its counterpart. No other layer is loaded.
        """
        if type(layer_id) is not int or not 0 <= layer_id < self.config.num_hidden_layers:
            raise ValueError("layer_id must identify a backbone layer")
        return self._load(self._layer_names[layer_id], rank, world_size, parallel_mode, tensor_names)

    def load_globals(
        self,
        rank: int = 0,
        world_size: int = 1,
        *,
        parallel_mode: ParallelMode = "ep_only",
        tensor_names: Sequence[str] | None = None,
    ) -> ReferenceWeights:
        """Load embedding, norm, and FP32 output head, optionally selecting a bounded subset."""
        return self._load(self._global_names, rank, world_size, parallel_mode, tensor_names)

    def _select(
        self,
        candidates: Sequence[str],
        rank: int,
        world_size: int,
        parallel_mode: ParallelMode,
        requested: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if parallel_mode not in ("ep_only", "reference_tp"):
            raise ValueError("parallel_mode must be ep_only or reference_tp")
        ownership = self.config.expert_ownership(world_size, rank)
        if parallel_mode == "reference_tp":
            for value in (self.config.o_groups, self.config.num_attention_heads, self.config.index_n_heads):
                if value % world_size:
                    raise ValueError("reference_tp requires complete attention groups and heads per rank")
        allowed = set()
        for name in candidates:
            expert = _EXPERT.search(name)
            if expert is None or ownership.start <= int(expert[1]) < ownership.stop:
                allowed.add(name)
        if requested is None:
            return tuple(name for name in candidates if name in allowed)
        if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence) or not requested:
            raise ValueError("tensor_names must be a nonempty sequence of source names")
        if any(not isinstance(name, str) or name not in allowed for name in requested):
            raise ValueError("tensor_names must belong to the requested layer/global scope and rank")
        selected = set(requested)
        for name in requested:
            companion = (
                name.removesuffix(".scale") + ".weight"
                if name.endswith(".scale")
                else (name.removesuffix(".weight") + ".scale" if name.endswith(".weight") else "")
            )
            if companion in allowed:
                selected.add(companion)
        return tuple(name for name in candidates if name in selected)

    @staticmethod
    def _axis(spec: TensorSpec, mode: ParallelMode) -> int | None:
        if mode == "reference_tp":
            if "tp_shard_axis0" in spec.conversion or "row_shard_ceil" in spec.conversion:
                return 0
            if "tp_shard_axis1" in spec.conversion:
                return 1
        return None

    def _output_shape(self, spec: TensorSpec, mode: ParallelMode, world_size: int) -> tuple[int, ...]:
        shape = list(spec.shape)
        axis = self._axis(spec, mode)
        if axis is not None:
            if "row_shard_ceil" in spec.conversion:
                shape[axis] = (shape[axis] + world_size - 1) // world_size
            else:
                if shape[axis] % world_size:
                    raise ValueError(f"TP split is not divisible for {spec.runtime_name}: {spec.shape}")
                shape[axis] //= world_size
                if spec.dtype == "F8_E4M3" and shape[axis] % 32:
                    raise ValueError(
                        f"TP quantized split must align to 32-element scale blocks: {spec.runtime_name}"
                    )
        return tuple(shape)

    def _estimate(self, names: Sequence[str], mode: ParallelMode, world_size: int) -> int:
        source = output = scratch = 0
        for name in names:
            spec = self.specs[name]
            count = math.prod(spec.shape)
            source += count * DTYPE_BYTES[spec.dtype]
            # Includes finite checks and FP32 wo_a/scale temporaries, conservatively
            # using the full source shape even when a TP slice will be smaller.
            scratch = max(scratch, count * 16)
            shape = self._output_shape(spec, mode, world_size)
            if "consume_scale_for_dequantization" in spec.conversion:
                continue
            width = DTYPE_BYTES[spec.dtype]
            if "dequantize_fp8_32x32_ue8m0_to_bf16" in spec.conversion:
                if any(dim % 32 for dim in shape):
                    raise ValueError("wo_a dequantization requires complete local 32x32 tiles")
                width = 2
            elif "cast_bf16_to_fp32" in spec.conversion or "cast_to_fp32_for_routing" in spec.conversion:
                width = 4
            output += math.prod(shape) * width
        return source + output + scratch

    def _validate_headers(self, names: Sequence[str]) -> None:
        by_shard: dict[Path, list[str]] = {}
        for name in names:
            path = self.store.path_for(name).resolve()
            if path.parent != self.model_dir:
                raise ValueError(f"shard resolves outside checkpoint directory: {path.name}")
            by_shard.setdefault(path, []).append(name)
        for path, selected in by_shard.items():
            header, _ = read_header(path)
            for name in selected:
                item, spec = header.get(name), self.specs[name]
                if item is None or tuple(item["shape"]) != spec.shape or item["dtype"] != spec.dtype:
                    raise ValueError(
                        f"source header contract mismatch: {name}, expected {spec.shape}/{spec.dtype}"
                    )

    @staticmethod
    def _validate_tensor(name: str, value: torch.Tensor, spec: TensorSpec) -> None:
        expected = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "I8": torch.int8,
            "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
            "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
        }[spec.dtype]
        if expected is None:
            raise RuntimeError(f"PyTorch does not expose source dtype {spec.dtype}")
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"source tensor must be a CPU tensor: {name}")
        if tuple(value.shape) != spec.shape or value.dtype != expected or not value.is_contiguous():
            raise ValueError(f"source tensor contract mismatch: {name}, expected {spec.shape}/{spec.dtype}")
        if spec.dtype == "F8_E8M0":
            finite = not bool((value.view(torch.uint8) == 255).any())
        elif spec.dtype == "F8_E4M3":
            finite = not bool(((value.view(torch.uint8) & 127) == 127).any())
        else:
            finite = spec.dtype == "I8" or bool(torch.isfinite(value).all())
        if not finite:
            raise ValueError(f"non-finite source tensor or scale: {name}")

    def _slice(self, value: torch.Tensor, spec: TensorSpec, mode: ParallelMode, rank: int, world_size: int):
        axis = self._axis(spec, mode)
        if axis is None:
            return value
        shape = self._output_shape(spec, mode, world_size)
        width = shape[axis]
        begin = rank * width
        count = min(width, max(0, value.shape[axis] - begin))
        return value.narrow(axis, min(begin, value.shape[axis]), count)

    def _convert(
        self,
        name: str,
        loaded: Mapping[str, torch.Tensor],
        mode: ParallelMode,
        rank: int,
        world_size: int,
    ) -> tuple[torch.Tensor, str]:
        spec = self.specs[name]
        value = self._slice(loaded[name], spec, mode, rank, world_size)
        operation = spec.conversion
        if "dequantize_fp8_32x32_ue8m0_to_bf16" in operation:
            scale_name = name.removesuffix(".weight") + ".scale"
            scale = self._slice(loaded[scale_name], self.specs[scale_name], mode, rank, world_size).float()
            rows, columns = value.shape
            result = value.float().reshape(rows // 32, 32, columns // 32, 32)
            result.mul_(scale[:, None, :, None])
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"wo_a dequantization overflows FP32: {name}")
            result = result.to(torch.bfloat16)
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"wo_a dequantization overflows BF16: {name}")
            groups = self.config.o_groups // world_size if mode == "reference_tp" else self.config.o_groups
            return result.reshape(groups, self.config.o_lora_rank, columns), "bf16_grouped_wo_a"
        if "cast_bf16_to_fp32" in operation or "cast_to_fp32_for_routing" in operation:
            return value.float().contiguous(), "float32_row_major"
        if "row_shard_ceil" in operation and mode == "reference_tp":
            shape = self._output_shape(spec, mode, world_size)
            # Raw bytes avoid requiring arithmetic kernels for E8M0 on CPU.
            pad = 127 if spec.dtype == "F8_E8M0" else 0
            raw = torch.full(shape, pad, dtype=torch.uint8)
            raw[: value.shape[0]].copy_(value.view(torch.uint8))
            return raw.view(value.dtype), "ue8m0_row_block32" if pad else "fp8_engram_row_major"
        if spec.dtype in ("F8_E4M3", "F8_E8M0", "I8"):
            result = value.view(torch.uint8).clone(memory_format=torch.contiguous_format).view(value.dtype)
            format_name = {
                "F8_E4M3": "fp8_e4m3fn_row_major",
                "F8_E8M0": "ue8m0_scale",
                "I8": "packed_e2m1_i8_low_nibble_first",
            }[spec.dtype]
            return result, format_name
        return value.clone(
            memory_format=torch.contiguous_format
        ), "bf16_row_major" if spec.dtype == "BF16" else "float32_row_major"

    def _load(
        self,
        candidates: Sequence[str],
        rank: int,
        world_size: int,
        mode: ParallelMode,
        requested: Sequence[str] | None,
    ) -> ReferenceWeights:
        names = self._select(candidates, rank, world_size, mode, requested)
        self.store.require(names)
        peak = self._estimate(names, mode, world_size)
        if peak > self.max_load_bytes:
            raise WeightLoadBudgetError(
                f"load requires an estimated {peak} tensor-buffer bytes; budget is {self.max_load_bytes}. "
                "Select fewer tensor_names or provide an explicit resource-checked budget. "
                "TP/Engram slicing still requires the full source tensor in the shared store."
            )
        self._validate_headers(names)
        loaded = self.store.load_many(names)
        for name in names:
            self._validate_tensor(name, loaded[name], self.specs[name])
        tensors, formats = {}, {}
        for name in names:
            spec = self.specs[name]
            if "consume_scale_for_dequantization" not in spec.conversion:
                tensors[spec.runtime_name], formats[spec.runtime_name] = self._convert(
                    name, loaded, mode, rank, world_size
                )
        return ReferenceWeights(tensors, formats, names, mode, rank, world_size, peak)
