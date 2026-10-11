# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind resident prefill weights, cache views, metadata and layer scratch."""

import torch

from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

from .prefill_segment import PREFILL_ARGUMENTS, bind_prefill_producers
from .swa_segment import MOE_ARGS


class PrefillLayerResources:
    """Own only transient device buffers; weight and cache handles are borrowed."""

    def __init__(self, worker, topology, plans, weights, cache, metadata, *,
                 kv_sources, index_sources, c2a_sources, max_seq_len, index_topk=512):
        self.worker = worker
        self.topology = topology
        self.plans = tuple(plans)
        self.metadata = metadata
        if (not self.plans or any(plan.mode not in PREFILL_ARGUMENTS for plan in self.plans)
                or any(type(n) is not int or n <= 0 for n in (max_seq_len, index_topk))):
            raise ValueError("prefill binding requires ordered modes and positive model capacities")
        self._owned = []
        world, local, tokens = topology.world, topology.local_capacity, topology.capacity

        def allocate(shape, dtype):
            value = alloc_device_buffer(worker, shape, dtype, stacked=True)
            self._owned.append(value)
            return value

        try:
            # A layer's Attention and MoE outputs must differ from each other
            # and from the preceding MoE output. Three slots suffice for all layers.
            states = [(allocate((world, local, 4, 5120), torch.float32),
                       allocate((world, local, 4), torch.float32)) for _ in range(3)]
            attention_hidden = allocate((world, tokens, 5120), torch.bfloat16)
            attention_out = allocate((world, local, 5120), torch.bfloat16)
            moe_mixed = allocate((world, local, 5120), torch.bfloat16)
            topk = {plan.layer_id: allocate((world, tokens, index_topk), torch.int32)
                    for plan in self.plans if plan.index_source == plan.layer_id}
            unused_topk = allocate((world, tokens, index_topk), torch.int32)
            unused_indices = allocate((world, tokens, index_topk), torch.int32)
            candidate = {plan.layer_id: allocate((world, tokens, max_seq_len), torch.uint8)
                         for plan in self.plans if plan.mode == "c1a_full"}
            attention, moe = {}, {}
            for plan in self.plans:
                layer = plan.layer_id
                a, m = weights.prefill_weights(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources)
                a = dict(a, **cache.prefill_views(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources))
                a.update(metadata.arguments(plan.mode))
                attn_residual, attn_mix = states[(2 * layer) % 3]
                moe_residual, moe_mix = states[(2 * layer + 1) % 3]
                a.update(attn_input=attention_hidden, attn_output=attention_out,
                         hidden=attention_hidden, attn_out=attention_out,
                         output=attn_residual, x_hc_out=attn_residual,
                         next_pre_mix=attn_mix)
                if plan.mode.startswith("c1a_"):
                    a.update(topk_indices=topk.get(layer, unused_topk),
                             compressed_indices=unused_indices,
                             candidate_mask=candidate[plan.candidate_source])
                elif plan.mode == "c2a_full":
                    a["topk_indices"] = topk[layer]
                elif plan.mode == "c2a_reuse":
                    a["compressed_indices"] = unused_indices
                m = dict(m, next_pre_mix=moe_mix, x_next=moe_residual, x_mixed=moe_mixed)
                attention[layer], moe[layer] = a, m
            bound = bind_prefill_producers(self.plans, attention)
            dynamic = {"x_hc", "pre_mix", "incoming_pre_mix", "num_tokens",
                       "attention_epoch", "moe_epoch"}
            for plan in self.plans:
                missing_a = set(PREFILL_ARGUMENTS[plan.mode]) - dynamic - bound[plan.layer_id].keys()
                missing_m = set(MOE_ARGS) - dynamic - moe[plan.layer_id].keys()
                if missing_a or missing_m:
                    raise ValueError(f"layer {plan.layer_id} has unbound prefill ABI fields: "
                                     f"{sorted(missing_a | missing_m)}")
            self.attention, self.moe = bound, moe
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in reversed(self._owned):
            self.worker.free_stacked_tensor(value)
        self._owned.clear()
