# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident per-layer decode Block arguments with distinct C2A/C1A metadata."""

import torch

from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer


class DecodeLayerResources:
    """Own decode scratch and selections; borrow uploaded weights and caches."""

    def __init__(self, worker, topology, plans, weights, cache, metadata, param_names, *,
                 kv_sources, index_sources, c2a_sources, index_topk=512):
        self.worker, self.topology, self.plans = worker, topology, tuple(plans)
        if len(self.plans) != 40 or tuple(p.layer_id for p in self.plans) != tuple(range(40)):
            raise ValueError("decode layer binding requires the complete ordered backbone")
        if type(index_topk) is not int or index_topk <= 0:
            raise ValueError("decode index top-K must be positive")
        self._owned = []
        world, local, tokens = topology.world, topology.local_capacity, topology.capacity

        def allocate(shape, dtype):
            value = alloc_device_buffer(worker, shape, dtype, stacked=True)
            self._owned.append(value)
            return value

        try:
            states = [(allocate((world, local, 4, 5120), torch.float32),
                       allocate((world, local, 4), torch.float32)) for _ in range(2)]
            scratch = {
                "attention_output": allocate((world, local, 5120), torch.bfloat16),
                "attention_hidden": allocate((world, local, 4, 5120), torch.float32),
                "attention_pre_mix": allocate((world, local, 4), torch.float32),
                "gathered": allocate((world, tokens, 5120), torch.bfloat16),
                "x_mixed": allocate((world, local, 5120), torch.bfloat16),
            }
            topk = {source: allocate((world, tokens, index_topk), torch.int32)
                    for source in index_sources}
            unused_topk = allocate((world, tokens, index_topk), torch.int32)
            unused_indices = allocate((world, tokens, index_topk), torch.int32)
            compressed_pages = max(cache.plan.physical_group_blocks[group] for group, families in
                                   cache.plan.group_sources.items() if "compressed" in families)
            candidate = allocate((world, tokens, compressed_pages * 128), torch.uint8)
            c2a_metadata = metadata.arguments("c2a_full")
            c1a_metadata = metadata.arguments("c1a_full")
            result = {}
            for plan in self.plans:
                layer, mode = plan.layer_id, plan.mode
                a, m = weights.prefill_weights(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources)
                arguments = dict(a)
                arguments.update(m)
                arguments["ffn_norm_weight"] = arguments.pop("norm_weight")
                family_metadata = c1a_metadata if mode.startswith("c1a_") else c2a_metadata
                arguments.update(family_metadata)
                arguments.update(cache.prefill_views(
                    plan, kv_sources=kv_sources, index_sources=index_sources,
                    c2a_sources=c2a_sources))
                if mode == "swa":
                    kv_source, index_source = kv_sources[0], index_sources[0]
                else:
                    kv_source, index_source = plan.kv_source, plan.index_source
                kv_slot = kv_sources.index(kv_source)
                index_slot = index_sources.index(index_source)
                for name in ("compressor_norm_weight",):
                    arguments[name] = weights.view(name, kv_slot)
                for name in ("index_wk", "index_norm_weight"):
                    arguments[name] = weights.view(name, index_sources.index(kv_source))
                for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                    arguments[name] = weights.view(name, index_slot)
                arguments["compressor_wkv"] = weights.view("c2a_compressor_wkv", 0)
                arguments["compressor_wgate"] = weights.view("c2a_compressor_wgate", 0)
                arguments["c1a_compressor_wkv"] = weights.view("c1a_compressor_wkv", 0)
                if mode == "c2a_full":
                    c2a_slot = c2a_sources.index(layer)
                    arguments["compressor_wkv"] = weights.view("c2a_compressor_wkv", c2a_slot)
                    arguments["compressor_wgate"] = weights.view("c2a_compressor_wgate", c2a_slot)
                if mode == "swa":
                    for name in ("compressed_cache", "compressed_cache_scale",
                                 "index_cache", "index_cache_scale"):
                        slot = kv_slot if name.startswith("compressed_") else index_slot
                        arguments[name] = cache.view(name + "_pool", slot)
                if mode.startswith("c1a_"):
                    for name in ("index_cache", "index_cache_scale"):
                        arguments[name] = cache.view(name + "_pool", index_sources.index(kv_source))
                arguments["state_cache"] = cache.view(
                    "state_cache_pool", c2a_sources.index(layer) if mode == "c2a_full" else 0)
                arguments["token_to_req_indices"] = c2a_metadata["token_to_req_indices"]
                arguments["position_ids"] = c2a_metadata["position_ids"]
                arguments["query_start_loc"] = c2a_metadata["query_start_loc"]
                arguments["state_block_table"] = c2a_metadata["state_block_table"]
                arguments["topk_indices"] = topk.get(layer, unused_topk)
                arguments["candidate_mask"] = candidate
                arguments["compressed_indices"] = (
                    topk[index_source] if mode.endswith("reuse") else unused_indices)
                arguments.update(scratch)
                arguments["x_next"], arguments["next_pre_mix"] = states[layer % 2]
                reserved = {"x_hc", "incoming_pre_mix", "layer_id", "attention_num_tokens",
                            "moe_num_tokens", "attention_epoch", "moe_epoch"}
                missing = set(param_names) - reserved - arguments.keys()
                if missing:
                    raise ValueError(f"decode layer {layer} has unbound ABI fields: {sorted(missing)}")
                result[layer] = arguments
            self.arguments = result
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in reversed(self._owned):
            self.worker.free_stacked_tensor(value)
        self._owned.clear()
