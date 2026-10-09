# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.model.deepseek_v41 import final_output_resources
from pypto_serving.model.deepseek_v41.composite import LayerState
from pypto_serving.model.deepseek_v41.segment_inputs import SegmentInputs
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


def test_final_weights_are_tp_sharded_and_dp_replicated_before_worker(monkeypatch):
    reads = []

    class Loader:
        def __init__(self, model_dir, *, tp_rank, **kwargs):
            self.tp_rank = tp_rank
            self.config = SimpleNamespace(vocab_size=16, hidden_size=4)

        def load(self, name):
            assert name == "norm.weight"
            return SimpleNamespace(weight=torch.arange(4, dtype=torch.bfloat16))

        def load_rows(self, name, start, stop):
            assert name == "head.weight"
            reads.append((self.tp_rank, start, stop))
            return SimpleNamespace(weight=torch.full((stop - start, 4), self.tp_rank + 1,
                                                     dtype=torch.bfloat16))

    monkeypatch.setattr(final_output_resources, "V41WeightLoader", Loader)
    resources = final_output_resources.load_final_output_host_resources(
        "unused", SegmentTopology.for_decode(), max_logit_rows=2, chunk_rows=3,
    )
    assert len(reads) == 8
    assert reads[:2] == [(0, 0, 3), (0, 3, 4)]
    assert resources.head_weight.shape == (8, 4, 4)
    for rank in range(8):
        assert torch.all(resources.head_weight[rank] == rank % 4 + 1)
    assert resources.norm_weight[7].tolist() == [0, 1, 2, 3]
    assert all(tensor.is_shared() for tensor in resources.inherited_host_tensors)
    assert resources.logit_row_indices.shape == (8, 2)
    assert resources.logits.shape == (8, 2, 16)


def test_final_output_rejects_host_budget_before_allocation(monkeypatch):
    class Loader:
        def __init__(self, model_dir, **kwargs):
            self.config = SimpleNamespace(vocab_size=16, hidden_size=4)

    monkeypatch.setattr(final_output_resources, "V41WeightLoader", Loader)
    with pytest.raises(ValueError, match="host buffers require"):
        final_output_resources.load_final_output_host_resources(
            "unused", SegmentTopology.for_decode(), max_host_bytes=1,
        )


class RecordingWorker:
    def __init__(self):
        self.next_ptr = 1
        self.freed = []
        self.calls = []

    def alloc_tensor(self, shape, dtype, *, worker_id):
        ptr = self.next_ptr
        self.next_ptr += 1
        return DeviceTensor(ptr, shape, dtype)

    def alloc_stacked_tensor(self, host):
        ids = tuple(range(host.shape[0]))
        shards = [self.alloc_tensor(host.shape[1:], host.dtype, worker_id=i) for i in ids]
        return StackedDeviceTensor(shards, host.shape, ids)

    def free_stacked_tensor(self, tensor):
        self.freed.append(tensor)

    def run(self, program, *args, config):
        self.calls.append((program, config))
        rows, logits = args[4], args[6]
        for rank in range(rows.shape[0]):
            for slot in range(rows.shape[1]):
                if rows[rank, slot] >= 0:
                    logits[rank, slot].fill_(rank * 10 + slot)


def test_final_runtime_dispatches_owned_request_logits():
    world, rows, vocab, hidden = 8, 16, 16, 4
    host = final_output_resources.FinalOutputHostResources(
        torch.ones(world, hidden, dtype=torch.bfloat16).share_memory_(),
        torch.ones(world, vocab // 4, hidden, dtype=torch.bfloat16).share_memory_(),
        torch.full((world, rows), -1, dtype=torch.int32).share_memory_(),
        torch.zeros(world, rows, vocab, dtype=torch.float32).share_memory_(),
        torch.zeros(world, rows, 8, dtype=torch.int32).share_memory_(),
    )
    worker = RecordingWorker()
    params = ("x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
              "normed", "logits", "sampled_ids", "done_epoch")
    runtime = final_output_resources.FinalOutputRuntime(
        worker, SimpleNamespace(compiled="output"), params, None, SegmentTopology.for_decode(), host,
    )
    source_rows = torch.full((world, 48), -1, dtype=torch.int64)
    source_rows[5, 3], source_rows[1, 2] = 0, 1
    inputs = SegmentInputs(None, None, source_rows, ((5, 3), (1, 2)), (0, 2))
    result = runtime.run(LayerState("residual", "mix", "tp_local_token"), inputs)
    assert result.shape == (2, vocab)
    assert result[:, 0].tolist() == [50, 10]
    assert worker.calls == [("output", None)]
    runtime.close()
    runtime.close()
    assert len(worker.freed) == 3


def test_final_runtime_rejects_unshared_output_buffer():
    world, rows, vocab, hidden = 8, 16, 16, 4
    host = final_output_resources.FinalOutputHostResources(
        torch.ones(world, hidden, dtype=torch.bfloat16).share_memory_(),
        torch.ones(world, vocab // 4, hidden, dtype=torch.bfloat16).share_memory_(),
        torch.full((world, rows), -1, dtype=torch.int32).share_memory_(),
        torch.zeros(world, rows, vocab, dtype=torch.float32),
        torch.zeros(world, rows, 8, dtype=torch.int32).share_memory_(),
    )
    worker = RecordingWorker()
    with pytest.raises(ValueError, match="must be shared"):
        final_output_resources.FinalOutputRuntime(
            worker, SimpleNamespace(compiled="output"),
            ("x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
             "normed", "logits", "sampled_ids", "done_epoch"),
            None, SegmentTopology.for_decode(), host,
        )
    assert worker.next_ptr == 1
