# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pre-fork output weights and buffers for the V4.1 final lib composite."""

from dataclasses import dataclass

import torch

from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

from .final_output import FinalOutput, collect_final_logits, prepare_final_output_rows
from .swa_segment import SegmentTopology
from .weight_loader import V41WeightLoader


@dataclass(frozen=True)
class FinalOutputHostResources:
    norm_weight: torch.Tensor
    head_weight: torch.Tensor
    logit_row_indices: torch.Tensor
    logits: torch.Tensor
    sampled_ids: torch.Tensor

    @property
    def inherited_host_tensors(self):
        return (self.norm_weight, self.head_weight, self.logit_row_indices,
                self.logits, self.sampled_ids)


def load_final_output_host_resources(model_dir, topology: SegmentTopology, *,
                                     max_logit_rows: int = 16, chunk_rows: int = 64,
                                     max_host_bytes: int = 4 << 30) -> FinalOutputHostResources:
    """Load one TP vocabulary shard at a time before the worker forks."""
    if not isinstance(topology, SegmentTopology) or (topology.tp, topology.dp) != (4, 2):
        raise ValueError("V4.1 final output requires TP4/DP2 placement")
    if any(type(value) is not int or value <= 0 for value in
           (max_logit_rows, chunk_rows, max_host_bytes)):
        raise ValueError("final output capacities and memory budget must be positive")
    loader = V41WeightLoader(model_dir, tp_size=topology.tp, tp_rank=0,
                             ep_size=topology.world, ep_rank=0)
    vocab, hidden = loader.config.vocab_size, loader.config.hidden_size
    if vocab % topology.tp:
        raise ValueError("LM-head vocabulary must divide TP size")
    shard_rows = vocab // topology.tp
    sampled_pad = 8
    required_bytes = (topology.world * shard_rows * hidden * 2
                      + topology.world * hidden * 2
                      + topology.world * max_logit_rows * vocab * 4
                      + topology.world * max_logit_rows * (1 + sampled_pad) * 4)
    if required_bytes > max_host_bytes:
        raise ValueError(f"final output host buffers require {required_bytes} bytes")

    norm = torch.empty((topology.world, hidden), dtype=torch.bfloat16).share_memory_()
    head = torch.empty((topology.world, shard_rows, hidden), dtype=torch.bfloat16).share_memory_()
    indices = torch.full((topology.world, max_logit_rows), -1, dtype=torch.int32).share_memory_()
    logits = torch.zeros((topology.world, max_logit_rows, vocab), dtype=torch.float32).share_memory_()
    sampled = torch.full((topology.world, max_logit_rows, sampled_pad), -1,
                         dtype=torch.int32).share_memory_()

    norm_value = loader.load("norm.weight").weight
    if norm_value.dtype != torch.bfloat16 or tuple(norm_value.shape) != (hidden,):
        raise ValueError("checkpoint final Norm weight disagrees with the lib output ABI")
    norm.copy_(norm_value.expand_as(norm))
    for tp_rank in range(topology.tp):
        rank_loader = (loader if tp_rank == 0 else V41WeightLoader(
            model_dir, tp_size=topology.tp, tp_rank=tp_rank,
            ep_size=topology.world, ep_rank=tp_rank))
        for start in range(0, shard_rows, chunk_rows):
            stop = min(start + chunk_rows, shard_rows)
            value = rank_loader.load_rows("head.weight", start, stop).weight
            if value.dtype != torch.bfloat16 or tuple(value.shape) != (stop - start, hidden):
                raise ValueError("checkpoint LM-head shard disagrees with the lib output ABI")
            head[tp_rank, start:stop].copy_(value)
        head[tp_rank + topology.tp].copy_(head[tp_rank])
    return FinalOutputHostResources(norm, head, indices, logits, sampled)


class FinalOutputRuntime:
    """Keep output weights resident and stage only request rows per dispatch.

    The host resources must exist before the shared DistributedWorker forks.
    The caller owns and closes the worker after this runtime releases its tensors.
    """

    def __init__(self, worker, program, param_names, run_config, topology: SegmentTopology,
                 host: FinalOutputHostResources):
        if not isinstance(host, FinalOutputHostResources) or not isinstance(topology, SegmentTopology):
            raise ValueError("final output requires host resources and a segment topology")
        world, local = topology.world, topology.local_capacity
        if any(not tensor.is_shared() for tensor in host.inherited_host_tensors):
            raise ValueError("final output host buffers must be shared before worker creation")
        if (host.norm_weight.dtype != torch.bfloat16 or host.norm_weight.shape != (world, host.head_weight.shape[-1])
                or host.head_weight.dtype != torch.bfloat16 or host.head_weight.shape[0] != world
                or host.logit_row_indices.dtype != torch.int32 or host.logit_row_indices.shape != (world, 16)
                or host.logits.dtype != torch.float32
                or host.logits.shape != (world, 16, host.head_weight.shape[1] * topology.tp)
                or host.sampled_ids.dtype != torch.int32 or host.sampled_ids.shape != (world, 16, 8)):
            raise ValueError("final output host buffers disagree with the lib TP4/DP2 ABI")
        self.worker, self.host, self.topology = worker, host, topology
        self.dispatcher = FinalOutput(worker, program, param_names, run_config)
        self._resident = []
        try:
            self.norm_weight = worker.alloc_stacked_tensor(host.norm_weight)
            self._resident.append(self.norm_weight)
            self.head_weight = worker.alloc_stacked_tensor(host.head_weight)
            self._resident.append(self.head_weight)
            self.normed = alloc_device_buffer(
                worker, (world, local, host.head_weight.shape[-1]), torch.bfloat16, stacked=True,
            )
            self._resident.append(self.normed)
        except BaseException:
            self.close()
            raise

    def run(self, state, inputs):
        rows = prepare_final_output_rows(
            inputs, self.host.logit_row_indices, local_capacity=self.topology.local_capacity,
        )
        self.host.logits.zero_()
        self.host.sampled_ids.fill_(-1)
        self.dispatcher.run(state, {
            "norm_weight": self.norm_weight,
            "head_weight": self.head_weight,
            "logit_row_indices": self.host.logit_row_indices,
            "normed": self.normed,
            "logits": self.host.logits,
            "sampled_ids": self.host.sampled_ids,
        })
        return collect_final_logits(self.host.logits, rows)

    def close(self):
        for tensor in reversed(self._resident):
            self.worker.free_stacked_tensor(tensor)
        self._resident.clear()
