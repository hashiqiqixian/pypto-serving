# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Worker-owned entry state for the V4.1 TP-local composite boundary."""

from dataclasses import dataclass

import torch

from pypto_serving.model.common.runner.buffer_set import alloc_device_buffer

from .composite import LayerState
from .request_state import ForwardStep
from .segment_inputs import SegmentInputs, prepare_segment_inputs
from .swa_segment import SegmentTopology


@dataclass(frozen=True)
class InputHostResources:
    topology: SegmentTopology
    residual: torch.Tensor
    pre_mix: torch.Tensor

    @property
    def inherited_host_tensors(self):
        return self.residual, self.pre_mix


def make_input_host_resources(topology: SegmentTopology, hidden_size: int) -> InputHostResources:
    """Allocate shared staging before the persistent DistributedWorker forks."""
    if not isinstance(topology, SegmentTopology) or type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("input staging requires a topology and positive hidden size")
    residual = torch.empty((topology.world, topology.local_capacity, 4, hidden_size),
                           dtype=torch.float32).share_memory_()
    pre_mix = torch.empty((topology.world, topology.local_capacity, 4),
                          dtype=torch.float32).share_memory_()
    return InputHostResources(topology, residual, pre_mix)


class InputStateRuntime:
    """Upload each packed step into fixed device handles for its phase capacity."""

    def __init__(self, worker, host: InputHostResources):
        if not isinstance(host, InputHostResources):
            raise ValueError("input state requires pre-fork host resources")
        for tensor in host.inherited_host_tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or not tensor.is_shared() or not tensor.is_contiguous()):
                raise ValueError("input staging must be contiguous shared CPU FP32")
        topology = host.topology
        if (host.residual.shape[:3] != (topology.world, topology.local_capacity, 4)
                or host.pre_mix.shape != host.residual.shape[:-1]):
            raise ValueError("input staging shapes disagree with the TP-local layout")
        self.worker, self.host = worker, host
        self.failed = False
        self._resident = []
        try:
            self.residual = alloc_device_buffer(worker, host.residual.shape, torch.float32, stacked=True)
            self._resident.append(self.residual)
            self.pre_mix = alloc_device_buffer(worker, host.pre_mix.shape, torch.float32, stacked=True)
            self._resident.append(self.pre_mix)
        except BaseException:
            self.close()
            raise

    def stage(self, embeddings: torch.Tensor, step: ForwardStep) -> tuple[LayerState, SegmentInputs]:
        if self.failed:
            raise RuntimeError("input upload failed; recreate the worker before reuse")
        packed = prepare_segment_inputs(embeddings, step, self.host.topology)
        self.host.residual.copy_(packed.residual)
        self.host.pre_mix.copy_(packed.pre_mix)
        try:
            for source, destination in ((self.host.residual, self.residual),
                                        (self.host.pre_mix, self.pre_mix)):
                for rank, shard in enumerate(destination.shards):
                    row = source[rank]
                    self.worker.copy_to(shard.data_ptr, row.data_ptr(),
                                        row.numel() * row.element_size(), worker_id=rank)
        except BaseException:
            self.failed = True
            raise
        inputs = SegmentInputs(self.host.residual, self.host.pre_mix,
                               packed.row_indices, packed.request_last_rows, packed.group_counts)
        return LayerState(self.residual, self.pre_mix, "tp_local_token"), inputs

    def close(self):
        for tensor in reversed(self._resident):
            self.worker.free_stacked_tensor(tensor)
        self._resident.clear()
