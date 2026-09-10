# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent V4.1 checkpoint configuration and sequential CSA2 ownership.

This describes the published checkpoint; it does not register an executable model.
The reference executes every backbone layer in order. Its config does not identify
an encoder/decoder split, so a ratio transition must not become a CED boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class ConfigError(ValueError):
    """A checkpoint field violates the independent V4.1 contract."""


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _integers(value: Any, name: str, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be an integer array, got {value!r}")
    return tuple(_integer(item, f"{name}[{i}]", minimum) for i, item in enumerate(value))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _unique_ids(values: tuple[int, ...], name: str, limit: int) -> None:
    if tuple(sorted(set(values))) != values:
        raise ConfigError(f"{name} must be strictly increasing without duplicates")
    if any(value >= limit for value in values):
        raise ConfigError(f"{name} must reference backbone layers in [0, {limit})")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate config key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class QuantizationSpec:
    """Storage formats, distinguished from the main KV and index cache formats."""

    quant_method: str
    activation_scheme: str
    weight_block_size: tuple[int, int]
    scale_fmt: str
    expert_dtype: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantizationSpec:
        data = _mapping(data, "quantization_config")
        expected = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
        unknown = set(data) - set(expected) - {"weight_block_size"}
        if unknown:
            raise ConfigError(f"unsupported quantization metadata: {sorted(unknown)}")
        for field, value in expected.items():
            if data.get(field) != value:
                raise ConfigError(f"quantization_config.{field} must be {value!r}, got {data.get(field)!r}")
        block = _integers(data.get("weight_block_size"), "quantization_config.weight_block_size", 1)
        if block != (32, 32):
            raise ConfigError(f"dense FP8 weight_block_size must be (32, 32), got {block}")
        return cls(weight_block_size=block, **expected)

    @property
    def expert_block_size(self) -> int:
        """Reference fp4_block_size: one UE8M0 scale per 32 elements along K."""
        return 32


@dataclass(frozen=True)
class LayerPlan:
    """One backbone layer's owners; SWA is always private to this layer."""

    layer_id: int
    compress_ratio: int
    attention_kind: str
    kv_source_layer_id: int | None
    index_source_layer_id: int | None
    candidate_source_layer_id: int | None
    owns_main_kv: bool
    owns_index_keys: bool
    owns_index_results: bool
    requires_engram: bool

    @property
    def swa_owner_layer_id(self) -> int:
        return self.layer_id


@dataclass(frozen=True)
class ExpertOwnership:
    """Contiguous routed-expert partition; shared experts are evaluated on every rank."""

    ep_size: int
    rank: int
    start: int
    stop: int
    shared_experts: int

    @property
    def local_expert_count(self) -> int:
        return self.stop - self.start

    def local_index(self, expert_id: int) -> int:
        expert_id = _integer(expert_id, "expert_id", 0)
        if not self.start <= expert_id < self.stop:
            raise ConfigError(
                f"expert {expert_id} is not owned by rank {self.rank}: [{self.start}, {self.stop})"
            )
        return expert_id - self.start


@dataclass(frozen=True)
class DeepSeekV41Config:
    """Validated HF config, preserving metadata without inheriting V4 defaults."""

    text_config: Mapping[str, Any]
    quantization: QuantizationSpec
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    max_position_embeddings: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    dspark_target_layer_ids: tuple[int, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> DeepSeekV41Config:
        """Read a HF config without importing PyTorch or allocating model weights."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle, object_pairs_hook=_json_object))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeepSeekV41Config:
        data = _mapping(data, "config")
        if data.get("model_type") != "deepseek_v41":
            raise ConfigError(f"model_type must be 'deepseek_v41', got {data.get('model_type')!r}")
        if data.get("architectures") != ["DeepseekV41ForCausalLM"]:
            raise ConfigError("architectures must be ['DeepseekV41ForCausalLM']")
        text = _mapping(data.get("text_config"), "text_config")
        if text.get("model_type") != "deepseek_v41_text":
            raise ConfigError("text_config.model_type must be 'deepseek_v41_text'")
        fields = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "sliding_window",
            "max_position_embeddings",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
        )
        values = {name: _integer(text.get(name), f"text_config.{name}") for name in fields}
        for name in ("num_nextn_predict_layers", "candidate_topk_blocks", "candidate_block_size"):
            values[name] = _integer(text.get(name), f"text_config.{name}", 0)
        values["candidate_source_layer_id"] = _integer(
            text.get("candidate_source_layer_id"), "text_config.candidate_source_layer_id", -1
        )
        for name in (
            "compress_ratios",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "engram_layer_ids",
            "engram_num_embeddings",
            "dspark_target_layer_ids",
        ):
            values[name] = _integers(text.get(name), f"text_config.{name}")
        config = cls(
            text_config=_freeze(text),
            quantization=QuantizationSpec.from_dict(data.get("quantization_config")),
            **values,
        )
        config._validate()
        return config

    def _validate(self) -> None:
        count = self.num_hidden_layers
        if len(self.compress_ratios) != count + self.num_nextn_predict_layers:
            raise ConfigError("compress_ratios must contain every backbone and draft layer")
        if any(ratio not in (0, 1, 2) for ratio in self.compress_ratios):
            raise ConfigError(
                "V4.1 reference supports compress_ratios 0, 1, or 2; V4 ratios are not compatible"
            )
        if any(self.compress_ratios[count:]):
            raise ConfigError("V4.1 draft layers must use ratio 0 in this checkpoint contract")
        for field in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
            _unique_ids(getattr(self, field), field, count)
        _unique_ids(self.dspark_target_layer_ids, "dspark_target_layer_ids", count)
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ConfigError("num_experts_per_tok exceeds n_routed_experts")
        if self.n_shared_experts != 1:
            raise ConfigError("V4.1 reference requires exactly one shared expert")
        if self.qk_rope_head_dim > min(self.head_dim, self.index_head_dim) or self.qk_rope_head_dim % 2:
            raise ConfigError("qk_rope_head_dim must be even and fit both attention and index heads")
        if self.num_key_value_heads != 1:
            raise ConfigError("V4.1 latent attention requires num_key_value_heads=1")
        if self.num_attention_heads % self.o_groups:
            raise ConfigError("num_attention_heads must be divisible by o_groups")
        if not set(self.kv_source_layer_ids) <= set(self.index_source_layer_ids):
            raise ConfigError("each KV source must also produce index keys and index results")
        for source in set(self.kv_source_layer_ids) | set(self.index_source_layer_ids):
            if self.compress_ratios[source] == 0:
                raise ConfigError(f"source layer {source} has no main KV path (compress_ratio=0)")
        candidate = self.candidate_source_layer_id
        if candidate >= 0:
            if candidate not in self.index_source_layer_ids:
                raise ConfigError("candidate_source_layer_id must be an index source")
            if not self.candidate_topk_blocks or not self.candidate_block_size:
                raise ConfigError("candidate selection requires positive topk_blocks and block_size")
            # Candidates name blocks of the shared index-key cache, which must stay the same.
            if any(source > candidate for source in self.kv_source_layer_ids):
                raise ConfigError(
                    "a KV source after the candidate source would invalidate candidate ownership"
                )
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ConfigError("engram_num_embeddings must have one entry per engram_layer_id")
        if self.engram_layer_ids:
            for name in (
                "engram_max_ngram_size",
                "engram_vocab_size",
                "engram_n_heads",
                "engram_head_dim",
                "engram_compressed_vocab_size",
            ):
                _integer(self.text_config.get(name), f"text_config.{name}")
            _integer(self.text_config.get("engram_pad_token_id"), "text_config.engram_pad_token_id", 0)
            if any(rows <= 0 for rows in self.engram_num_embeddings):
                raise ConfigError("engram_num_embeddings entries must be positive")
        if self.num_nextn_predict_layers:
            if not self.dspark_target_layer_ids:
                raise ConfigError("draft layers require dspark_target_layer_ids")
            for name in ("dspark_markov_rank", "dspark_n_routed_experts", "dspark_num_experts_per_tok"):
                _integer(self.text_config.get(name), f"text_config.{name}")
            if self.text_config["dspark_num_experts_per_tok"] > self.text_config["dspark_n_routed_experts"]:
                raise ConfigError("DSpark activated experts exceed routed experts")
        # Constructing the plan validates read-before-write and sharing across ratio changes.
        self.layer_plan

    @property
    def layer_plan(self) -> tuple[LayerPlan, ...]:
        """Resolve the most recent producers, matching reference shared_attn writes."""
        plans = []
        kv_owner = index_owner = None
        for layer_id, ratio in enumerate(self.compress_ratios[: self.num_hidden_layers]):
            owns_kv = layer_id in self.kv_source_layer_ids
            owns_index = layer_id in self.index_source_layer_ids
            if owns_kv:
                kv_owner = layer_id
            if owns_index:
                index_owner = layer_id
            if ratio:
                if kv_owner is None or index_owner is None:
                    raise ConfigError(f"layer {layer_id} reads main KV/index before a source writes them")
                if self.compress_ratios[kv_owner] != ratio or self.compress_ratios[index_owner] != ratio:
                    raise ConfigError(f"layer {layer_id} shares KV/index with an incompatible compress_ratio")
                if index_owner < kv_owner:
                    raise ConfigError(f"layer {layer_id} reuses index results from before its KV source")
            candidate = self.candidate_source_layer_id
            attention_kind = (
                "swa" if not ratio else "full" if owns_kv else "reindex" if owns_index else "reuse"
            )
            plans.append(
                LayerPlan(
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    attention_kind=attention_kind,
                    kv_source_layer_id=kv_owner if ratio else None,
                    index_source_layer_id=index_owner if ratio else None,
                    candidate_source_layer_id=candidate if ratio and 0 <= candidate <= layer_id else None,
                    owns_main_kv=owns_kv,
                    owns_index_keys=owns_kv,
                    owns_index_results=owns_index,
                    requires_engram=layer_id in self.engram_layer_ids,
                )
            )
        return tuple(plans)

    @property
    def ced_split(self) -> None:
        """No explicit split is present in the published HF/inference configuration."""
        return None

    @property
    def text_requires_engram(self) -> bool:
        """Engram is part of the text forward, independent of speculative decoding or images."""
        return bool(self.engram_layer_ids)

    def expert_ownership(self, ep_size: int = 8, rank: int = 0) -> ExpertOwnership:
        ep_size = _integer(ep_size, "ep_size")
        rank = _integer(rank, "rank", 0)
        if rank >= ep_size:
            raise ConfigError(f"rank {rank} must be less than ep_size {ep_size}")
        if self.n_routed_experts % ep_size:
            raise ConfigError(
                f"n_routed_experts {self.n_routed_experts} is not divisible by ep_size {ep_size}"
            )
        local_count = self.n_routed_experts // ep_size
        return ExpertOwnership(
            ep_size, rank, rank * local_count, (rank + 1) * local_count, self.n_shared_experts
        )


@dataclass(frozen=True)
class DeepSeekV41Profile:
    """A V4.1 execution shape request; this is not a claim of an available NPU executor."""

    config: DeepSeekV41Config
    expert_parallel_size: int = 8

    def __post_init__(self) -> None:
        self.config.expert_ownership(self.expert_parallel_size)

    @property
    def local_expert_count(self) -> int:
        return self.config.n_routed_experts // self.expert_parallel_size

    @property
    def model_type(self) -> str:
        return "deepseek_v41"
