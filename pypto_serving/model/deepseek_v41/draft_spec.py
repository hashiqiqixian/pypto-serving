# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pinned V4.1 DSpark checkpoint contracts; shared embed/head remain backbone-owned."""

from __future__ import annotations

from collections.abc import Mapping

from .weight_spec import TensorSpec


def draft_weight_specs(config: Mapping) -> dict[str, TensorSpec]:
    text, quant = config["text_config"], config["quantization_config"]
    if (
        quant.get("quant_method"),
        tuple(quant.get("weight_block_size", ())),
        quant.get("scale_fmt"),
        quant.get("expert_dtype"),
    ) != ("fp8", (32, 32), "ue8m0", "fp4"):
        raise ValueError("DSpark requires FP8 32x32 UE8M0 and E2M1 expert block-32 weights")

    def positive(name):
        value = text[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    stages, dim, inter = (
        positive("num_nextn_predict_layers"),
        positive("hidden_size"),
        positive("moe_intermediate_size"),
    )
    heads, head_dim = positive("num_attention_heads"), positive("head_dim")
    groups, q_rank, o_rank = positive("o_groups"), positive("q_lora_rank"), positive("o_lora_rank")
    hc, experts = positive("hc_mult"), positive("dspark_n_routed_experts")
    vocab, markov_rank = positive("vocab_size"), positive("dspark_markov_rank")
    backbone = positive("num_hidden_layers")
    targets = text["dspark_target_layer_ids"]
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(type(i) is not int or not 0 <= i < backbone for i in targets)
    ):
        raise ValueError("DSpark target layers must be unique backbone layer IDs")
    if heads % groups or dim % 32 or inter % 32 or text["n_shared_experts"] != 1:
        raise ValueError("DSpark requires divisible groups, block-32 experts, and one shared expert")
    if tuple(text["compress_ratios"][backbone:]) != (0,) * stages:
        raise ValueError("DSpark layers require zero compression ratios")
    specs = {}

    def add(name, shape, dtype, conversion="identity", target=None):
        specs[name] = TensorSpec(shape, dtype, target or name, conversion)

    def dense(prefix, out_dim, in_dim, shard="replicate", dequant=False):
        target = prefix + ".weight"
        operation = "dequantize_fp8_32x32_ue8m0_to_bf16" if dequant else "preserve_fp8_32x32_ue8m0"
        add(target, (out_dim, in_dim), "F8_E4M3", f"{operation};{shard}")
        add(
            prefix + ".scale",
            ((out_dim + 31) // 32, (in_dim + 31) // 32),
            "F8_E8M0",
            f"consume_scale_for_dequantization;{shard}" if dequant else f"preserve_ue8m0;{shard}",
            target=target if dequant else None,
        )

    for stage in range(stages):
        prefix = f"mtp.{stage}"
        attn = prefix + ".attn"
        add(attn + ".attn_sink", (heads,), "F32", "tp_shard_axis0")
        dense(attn + ".wq_a", q_rank, dim)
        dense(attn + ".wq_b", heads * head_dim, q_rank, "tp_shard_axis0")
        dense(attn + ".wkv", head_dim, dim)
        dense(
            attn + ".wo_a",
            groups * o_rank,
            heads * head_dim // groups,
            "tp_shard_axis0;view_grouped_output_projection",
            dequant=True,
        )
        dense(attn + ".wo_b", dim, groups * o_rank, "tp_shard_axis1")
        add(attn + ".q_norm.weight", (q_rank,), "BF16")
        add(attn + ".kv_norm.weight", (head_dim,), "BF16")
        for sublayer in ("attn", "ffn"):
            mix = hc * (2 + hc)
            add(f"{prefix}.{sublayer}_norm.weight", (dim,), "BF16")
            add(f"{prefix}.hc_{sublayer}_fn", (mix, hc * dim), "F32")
            add(f"{prefix}.hc_{sublayer}_base", (mix,), "F32")
            add(f"{prefix}.hc_{sublayer}_scale", (3,), "F32")
        add(prefix + ".ffn.gate.weight", (experts, dim), "BF16", "cast_to_fp32_for_routing")
        add(prefix + ".ffn.gate.bias", (experts,), "F32")
        add(prefix + ".ffn.gate.bias_vl", (experts,), "F32", "retain_unused_draft_bias")
        for projection, out_dim, in_dim in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
            dense(f"{prefix}.ffn.shared_experts.{projection}", out_dim, in_dim)
            for expert in range(experts):
                name = f"{prefix}.ffn.experts.{expert}.{projection}"
                add(
                    name + ".weight",
                    (out_dim, in_dim // 2),
                    "I8",
                    "reinterpret_packed_e2m1_low_nibble_first;ep_select_expert",
                )
                add(
                    name + ".scale",
                    (out_dim, in_dim // 32),
                    "F8_E8M0",
                    "preserve_ue8m0_per_row_block32;ep_select_expert",
                )
    dense("mtp.0.main_proj", dim, dim * len(targets))
    add("mtp.0.main_norm.weight", (dim,), "BF16")
    prefix = f"mtp.{stages - 1}"
    add(prefix + ".norm.weight", (dim,), "BF16")
    add(prefix + ".markov_head.embed.weight", (vocab, markov_rank), "BF16", "tp_shard_axis0")
    add(prefix + ".markov_head.head.weight", (vocab, markov_rank), "BF16", "tp_shard_axis0;cast_bf16_to_fp32")
    add(prefix + ".confidence_head.proj.weight", (1, dim + markov_rank), "BF16", "cast_bf16_to_fp32")
    return specs
