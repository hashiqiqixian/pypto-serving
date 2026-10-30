# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint contracts for the DeepSeek V4.1 DSpark drafter.

The V4.1 backbone loader intentionally excludes deferred draft modules.  This
module keeps their checkpoint contract separate from backbone loading and only
loads the small replicated heads needed by the Markov stage.  Layer weights are
exposed through per-layer source names so a future packer can process one
layer at a time without constructing the full three-layer bank in Host memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


DSPARK_DRAFT_LAYERS = (0, 1, 2)

# Raw checkpoint names and the names used by l3_dspark_drafter /
# l3_markov_sample_from_hidden.  The projections are still in checkpoint
# orientation; the eventual packer owns transpose, quantization and EP/TP
# placement.  Keeping this map explicit prevents accidental use of backbone
# layers as draft weights.
DSPARK_HEAD_SOURCE_TO_ABI = {
    "mtp.0.main_proj.weight": "main_proj_weight",
    "mtp.0.main_norm.weight": "main_norm_weight",
    "mtp.2.norm.weight": "final_norm_weight",
    "mtp.2.markov_head.markov_w1.weight": "markov_w1",
    "mtp.2.markov_head.markov_w2.weight": "markov_w2",
    "mtp.2.confidence_head.proj.weight": "confidence_weight",
}

_LAYER_SUFFIXES = (
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn.wq_a.weight",
    "attn.wq_a.scale",
    "attn.wq_b.weight",
    "attn.wq_b.scale",
    "attn.wkv.weight",
    "attn.wkv.scale",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn.attn_sink",
    "attn.wo_a.weight",
    "attn.wo_a.scale",
    "attn.wo_b.weight",
    "attn.wo_b.scale",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w3.scale",
)


def dspark_v41_required_weight_names(n_routed_experts: int) -> tuple[str, ...]:
    """Return the deferred checkpoint names required by the K=5 DSpark path."""
    if type(n_routed_experts) is not int or n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be a positive integer")
    names = list(DSPARK_HEAD_SOURCE_TO_ABI)
    for layer in DSPARK_DRAFT_LAYERS:
        names.extend(f"mtp.{layer}.{suffix}" for suffix in _LAYER_SUFFIXES)
        names.extend(
            f"mtp.{layer}.ffn.experts.{expert}.{suffix}"
            for expert in range(n_routed_experts)
            for suffix in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
        )
    return tuple(names)


def dspark_v41_layer_weight_names(layer: int, n_routed_experts: int) -> tuple[str, ...]:
    """Return one draft layer's raw names for bounded, per-layer staging."""
    if type(layer) is not int or layer not in DSPARK_DRAFT_LAYERS:
        raise ValueError("DSpark draft layer must be one of 0, 1 or 2")
    all_names = dspark_v41_required_weight_names(n_routed_experts)
    prefix = f"mtp.{layer}."
    return tuple(name for name in all_names if name.startswith(prefix))


@dataclass(frozen=True)
class V41DSparkHeadWeights:
    """Small replicated heads in the ABI names expected by the lib stages."""

    tensors: Mapping[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class V41DSparkWeightContract:
    """Validate and lazily read deferred DSpark tensors from a V41 loader.

    ``loader`` is deliberately duck-typed to keep this contract usable by the
    existing V41WeightLoader and by host tests.  It must expose ``store`` and
    ``text``.  No device allocation or full-bank materialization occurs here.
    """

    def __init__(self, loader):
        if not hasattr(loader, "store") or not hasattr(loader, "text"):
            raise TypeError("V41 DSpark contract requires a V41WeightLoader-like object")
        self.loader = loader
        self.n_routed_experts = loader.text.get("n_routed_experts")
        if type(self.n_routed_experts) is not int or self.n_routed_experts <= 0:
            raise ValueError("text.n_routed_experts must be a positive integer")

    @property
    def required_names(self) -> tuple[str, ...]:
        return dspark_v41_required_weight_names(self.n_routed_experts)

    def validate_index(self) -> None:
        """Fail before payload I/O when a speculative checkpoint is incomplete."""
        self.loader.store.require(self.required_names)

    def load_heads(self) -> V41DSparkHeadWeights:
        """Load only the replicated head tensors and normalize main-proj layout.

        The checkpoint stores ``main_proj`` as ``[D, 3D]`` while the lib ABI
        consumes ``[3D, D]``.  Other head tensors already use their ABI logical
        orientation; quantized layer transforms remain the packer's concern.
        """
        values = self.loader.store.load_many(tuple(DSPARK_HEAD_SOURCE_TO_ABI))
        main_proj = values["mtp.0.main_proj.weight"]
        main_norm = values["mtp.0.main_norm.weight"]
        markov_w1 = values["mtp.2.markov_head.markov_w1.weight"]
        markov_w2 = values["mtp.2.markov_head.markov_w2.weight"]
        confidence = values["mtp.2.confidence_head.proj.weight"]
        if main_proj.ndim != 2 or main_norm.ndim != 1:
            raise ValueError("DSpark main projection must be rank-2 with a rank-1 norm")
        hidden = int(main_norm.shape[0])
        if tuple(main_proj.shape) != (hidden, 3 * hidden):
            raise ValueError(
                f"DSpark main_proj must be [D, 3*D] with D={hidden}, got {tuple(main_proj.shape)}"
            )
        if markov_w1.ndim != 2 or tuple(markov_w1.shape) != tuple(markov_w2.shape):
            raise ValueError("DSpark Markov tables must have matching rank-2 shapes")
        markov_rank = int(markov_w1.shape[1])
        if tuple(confidence.shape) != (1, hidden + markov_rank):
            raise ValueError(
                f"DSpark confidence weight must be [1, {hidden + markov_rank}], "
                f"got {tuple(confidence.shape)}"
            )
        result = {
            DSPARK_HEAD_SOURCE_TO_ABI[name]: value
            for name, value in values.items()
        }
        result["main_proj_weight"] = main_proj.transpose(0, 1).contiguous()
        return V41DSparkHeadWeights(result)

    def load_layer(self, layer: int) -> Mapping[str, torch.Tensor]:
        """Load one raw draft layer after validating its source names.

        The returned tensors retain checkpoint orientation.  This is the
        bounded input to the future V4.1 FP8/FP4 packer and is intentionally
        not uploaded to an NPU by this module.
        """
        names = dspark_v41_layer_weight_names(layer, self.n_routed_experts)
        self.loader.store.require(names)
        return self.loader.store.load_many(names)
