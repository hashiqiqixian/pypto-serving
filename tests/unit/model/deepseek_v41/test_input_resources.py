# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-to-device state handoff; recording worker does not execute a kernel."""

import torch
import pytest
from pypto.runtime import DeviceTensor

from pypto_serving.model.deepseek_v41.input_resources import InputStateRuntime, make_input_host_resources
from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


class RecordingWorker:
    def __init__(self):
        self.next_ptr = 0
        self.copies = []
        self.freed = []

    def alloc_tensor(self, shape, dtype, *, worker_id):
        self.next_ptr += 1
        return DeviceTensor(self.next_ptr, shape, dtype)

    def copy_to(self, destination, source, count, *, worker_id):
        self.copies.append((worker_id, destination, count))

    def free_stacked_tensor(self, tensor):
        self.freed.append(tensor)


def test_input_runtime_uploads_tp_local_state_and_keeps_request_rows():
    topology = SegmentTopology(tp=4, dp=2, local_capacity=2)
    host = make_input_host_resources(topology, 4)
    worker = RecordingWorker()
    runtime = InputStateRuntime(worker, host)
    step = ForwardStep("prefill", (
        RequestSlice("a", 0, 0, 0, (3, 4), 2, {}),
    ), 1)
    state, inputs = runtime.stage(torch.ones(2, 4, dtype=torch.bfloat16), step)
    assert state.layout == "tp_local_token"
    assert state.residual.shape == (8, 2, 4, 4)
    assert inputs.group_counts == (2, 0)
    assert inputs.request_last_rows == ((0, 1),)
    assert host.residual[0, 0].sum().item() == 16
    assert host.residual[4].count_nonzero().item() == 0
    assert len(worker.copies) == 16
    assert all(tensor.is_shared() for tensor in host.inherited_host_tensors)
    runtime.close()
    runtime.close()
    assert len(worker.freed) == 2


def test_partial_input_upload_poisons_runtime():
    class FailingWorker(RecordingWorker):
        def copy_to(self, destination, source, count, *, worker_id):
            raise RuntimeError("copy failed")

    worker = FailingWorker()
    runtime = InputStateRuntime(worker, make_input_host_resources(SegmentTopology(4, 2, 1), 4))
    step = ForwardStep("decode", (RequestSlice("a", 0, 0, 1, (3,), 1, {}),), 1)
    embeddings = torch.ones(1, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="copy failed"):
        runtime.stage(embeddings, step)
    with pytest.raises(RuntimeError, match="recreate the worker"):
        runtime.stage(embeddings, step)
    runtime.close()
