# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep the V4.1 input, backbone and final-output resources on one worker."""

from dataclasses import replace

from .cache_resources import DecodeCachePools, allocate_cache_with_retry
from .composite import MissingCompositeInterface
from .decode_padding import pad_single_decode
from .decode_weight_resources import ResidentDecodeWeights
from .decode_layer_resources import DecodeLayerResources
from .final_output_resources import FinalOutputRuntime
from .input_resources import InputStateRuntime
from .prefill_layer_resources import PrefillLayerResources
from .swa_segment import make_segment_worker


class V41ResidentResources:
    """Own resident buffers and their shared host mappings until worker close.

    This is the resource lifetime for a production binding, not an executable
    binding by itself: the caller still has to lower every prefill/decode ABI
    argument and dispatch the compiled layer programs.
    """

    def __init__(self, programs, run_config, *, weight_bank, weight_param_names,
                 cache_plan, state_reset, page_resets, input_host, final_host,
                 prefill_host, final_program, final_param_names, topology,
                 decode_input_host=None, decode_host=None, worker_factory=make_segment_worker):
        weight_bank.finish(weight_param_names)
        if decode_host is not None and (decode_input_host is None or
                decode_host.topology != decode_input_host.topology):
            raise ValueError("decode metadata and input must share one TP-local topology")
        inherited = (input_host.inherited_host_tensors
                     + (() if decode_input_host is None else decode_input_host.inherited_host_tensors)
                     + prefill_host.inherited_host_tensors
                     + (() if decode_host is None else decode_host.inherited_host_tensors)
                     + final_host.all_host_tensors
                     + (state_reset, *page_resets.values()))
        if any(not tensor.is_shared() for tensor in inherited):
            raise ValueError("all inherited V4.1 host buffers must be shared before worker creation")
        if not getattr(weight_bank, "prepacked", False) and any(
                not tensor.is_shared() for tensor in weight_bank.tensors.values()):
            raise ValueError("all inherited V4.1 host buffers must be shared before worker creation")
        if final_program not in programs:
            raise ValueError("final composite must be registered on the shared worker")

        self.worker = worker_factory(programs, run_config,
                                     tuple(weight_bank.tensors.values()) + inherited)
        self.host_resources = (weight_bank, input_host, decode_input_host, prefill_host, decode_host,
                               final_host, state_reset, page_resets)
        self.prefill_host = prefill_host
        self.decode_host = decode_host
        self._owned = []
        self._step = None
        self._inputs = None
        self._group_counts = None
        self._prefill = None
        self._decode_layers = None
        self.closed = False
        try:
            self.weights = self._own(ResidentDecodeWeights(
                self.worker, weight_bank, weight_param_names))
            self.input = self._own(InputStateRuntime(self.worker, input_host))
            self.decode_input = (None if decode_input_host is None else
                                 self._own(InputStateRuntime(self.worker, decode_input_host)))
            self.final = self._own(FinalOutputRuntime(
                self.worker, final_program, final_param_names, run_config, topology,
                final_host, prefill_local_capacity=prefill_host.topology.local_capacity))
            if callable(cache_plan):
                slots, plan_for_slots = cache_plan(self.worker)
                cache = allocate_cache_with_retry(
                    self.worker, slots, plan_for_slots, state_reset, page_resets)
            else:
                cache = DecodeCachePools(self.worker, cache_plan, state_reset, page_resets)
            self.cache = self._own(cache)
            if decode_host is not None:
                plan = cache.plan
                if decode_host.max_requests < plan.physical_request_slots:
                    raise ValueError("decode metadata must reserve the filler state slot")
                if decode_host.window_pages < plan.physical_group_blocks[plan.family_groups.window]:
                    raise ValueError("decode window metadata must cover the filler page")
                for family in ("c2a", "c1a"):
                    group = getattr(plan.family_groups, family)
                    if decode_host.compressed_pages[family] < plan.physical_group_blocks[group]:
                        raise ValueError(f"decode {family} metadata must cover the filler page")
        except BaseException:
            self.close()
            raise

    def _own(self, resource):
        self._owned.append(resource)
        return resource

    def begin_step(self, embeddings, step):
        if self.closed or self._step is not None:
            raise RuntimeError("V4.1 worker is closed or a step is already active")
        dispatch_step = step
        if step.phase == "decode":
            dispatch_step, embeddings, padding_partition = pad_single_decode(
                step, embeddings, self.cache.plan)
            if padding_partition is not None:
                self.cache.reset_padding(padding_partition)
        host = self.prefill_host if step.phase == "prefill" else self.decode_host
        counts = host.stage(dispatch_step) if host is not None else None
        input_runtime = self.decode_input if step.phase == "decode" else self.input
        if input_runtime is None:
            raise ValueError("decode requires its own TP-local input capacity")
        state, inputs = input_runtime.stage(embeddings, dispatch_step)
        if dispatch_step is not step:
            inputs = replace(inputs, request_last_rows=inputs.request_last_rows[:len(step.requests)])
        self._step, self._inputs = step, inputs
        self._group_counts = counts
        return state

    def prefill_arguments(self, mode, step):
        if self.closed or step is not self._step or step.phase != "prefill":
            raise ValueError("prefill arguments require the active prefill step")
        return self.prefill_host.arguments(mode), self._group_counts

    def bind_prefill(self, plans, *, kv_sources, index_sources, c2a_sources,
                     max_seq_len, index_topk=512):
        """Complete per-layer arguments using this worker's resident resources."""
        if self.closed or self._prefill is not None or self._step is not None:
            raise RuntimeError("prefill must be bound once before a request step")
        self._prefill = self._own(PrefillLayerResources(
            self.worker, self.prefill_host.topology, plans, self.weights, self.cache,
            self.prefill_host, kv_sources=kv_sources, index_sources=index_sources,
            c2a_sources=c2a_sources, max_seq_len=max_seq_len, index_topk=index_topk))
        return self._prefill

    def run_prefill(self, segment, state, step):
        if (self.closed or self._prefill is None or step is not self._step
                or step.phase != "prefill" or segment.worker is not self.worker):
            raise ValueError("prefill dispatch requires the active step and shared worker")
        return segment.run_chain(
            state, self._prefill.plans, self._prefill.attention, self._prefill.moe,
            group_counts=self._group_counts)

    def decode_backbone_metadata(self, step):
        """Reject the current lib ABI before binding unequal C2A/C1A fields."""
        if (self.closed or step is not self._step or step.phase != "decode"
                or self.decode_host is None):
            raise ValueError("decode metadata requires the active decode step")
        raise MissingCompositeInterface(
            "decode_fwd accepts one compressed_lens, compressed_slots, "
            "index_block_table and compressed_rope pair for both C2A and C1A; "
            "the two families have different values and page namespaces")

    def bind_decode_layers(self, plans, param_names, *, kv_sources, index_sources,
                           c2a_sources, index_topk=512):
        """Prepare the lib's complete per-layer Block on this worker."""
        if (self.closed or self._decode_layers is not None or self._step is not None
                or self.decode_host is None or self.decode_input is None):
            raise RuntimeError("decode Blocks require pre-fork metadata and one idle worker")
        self._decode_layers = self._own(DecodeLayerResources(
            self.worker, self.decode_host.topology, plans, self.weights, self.cache,
            self.decode_host, param_names, kv_sources=kv_sources,
            index_sources=index_sources, c2a_sources=c2a_sources,
            index_topk=index_topk))
        return self._decode_layers

    def run_decode_layers(self, segment, state, step):
        if (self.closed or self._decode_layers is None or step is not self._step
                or step.phase != "decode" or segment.worker is not self.worker):
            raise ValueError("decode Blocks require the active step and shared worker")
        for plan in self._decode_layers.plans:
            state = segment.run_layer(
                state, plan, self._decode_layers.arguments[plan.layer_id],
                group_counts=self._group_counts)
        return state

    def finish_step(self, state, step):
        if self.closed or step is not self._step:
            raise ValueError("final output must use the active request step")
        try:
            return self.final.run(state, self._inputs)
        finally:
            self._step = self._inputs = self._group_counts = None

    def reset_request(self, owner):
        self._step = self._inputs = self._group_counts = None
        self.cache.reset_request(owner)

    def close(self):
        if self.closed:
            return
        self._step = self._inputs = self._group_counts = None
        try:
            for resource in reversed(self._owned):
                resource.close()
        finally:
            self.worker.close()
            self.closed = True
            self._owned.clear()
