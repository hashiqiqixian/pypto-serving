# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The prefill owner must bind producers and recycle only completed scratch."""

from types import SimpleNamespace
import json
from pathlib import Path

from pypto_serving.model.deepseek_v41.execution_plan import LayerPlan, plan_layers
from pypto_serving.model.deepseek_v41.prefill_layer_resources import PrefillLayerResources
from pypto_serving.model.deepseek_v41.prefill_segment import PREFILL_ARGUMENTS
from pypto_serving.model.deepseek_v41.swa_segment import MOE_ARGS, SegmentTopology


class RecordingWorker:
    def __init__(self):
        self.freed = []

    def free_stacked_tensor(self, tensor):
        self.freed.append(tensor)


def test_layer_binding_borrows_producers_and_rotates_state(monkeypatch):
    allocated = []

    def allocate(worker, shape, dtype, *, stacked):
        assert stacked
        value = SimpleNamespace(shape=shape, dtype=dtype)
        allocated.append(value)
        return value

    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.prefill_layer_resources.alloc_device_buffer", allocate)
    plans = (
        LayerPlan(0, "swa", None, None, None),
        LayerPlan(1, "c1a_full", 1, 1, 1),
        LayerPlan(2, "c1a_reuse", 1, 1, 1),
    )

    class Weights:
        def prefill_weights(self, plan, **sources):
            return ({name: object() for name in PREFILL_ARGUMENTS[plan.mode]},
                    {name: object() for name in MOE_ARGS})

    class Cache:
        def prefill_views(self, plan, **sources):
            return {"window_cache": object()}

    class Metadata:
        def arguments(self, mode):
            return {"window_slots": object()}

    worker = RecordingWorker()
    resource = PrefillLayerResources(
        worker, SegmentTopology(4, 2, 2), plans, Weights(), Cache(), Metadata(),
        kv_sources=(1,), index_sources=(1,), c2a_sources=(), max_seq_len=16,
    )
    first, producer, consumer = (resource.attention[plan.layer_id] for plan in plans)
    assert consumer["compressed_cache"] is producer["compressed_cache"]
    assert consumer["compressed_indices"] is producer["topk_indices"]
    assert consumer["candidate_mask"] is producer["candidate_mask"]
    assert consumer["topk_indices"] is not consumer["compressed_indices"]
    assert resource.moe[0]["x_next"] is not resource.attention[1]["x_hc_out"]
    assert resource.moe[0]["x_next"] is not resource.attention[1]["next_pre_mix"]
    assert resource.moe[0]["x_next"] is not first["output"]
    assert producer["attn_input"] is consumer["attn_input"]
    resource.close()
    assert worker.freed == list(reversed(allocated))


def test_all_checkpoint_layer_modes_bind_their_real_producers(monkeypatch):
    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.prefill_layer_resources.alloc_device_buffer",
        lambda worker, shape, dtype, *, stacked: SimpleNamespace(shape=shape, dtype=dtype))
    fixture = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"
    plans = plan_layers(json.loads(fixture.read_text()))

    class Weights:
        def prefill_weights(self, plan, **sources):
            return ({name: object() for name in PREFILL_ARGUMENTS[plan.mode]},
                    {name: object() for name in MOE_ARGS})

    class Cache:
        def prefill_views(self, plan, **sources):
            return {}

    class Metadata:
        def arguments(self, mode):
            return {}

    worker = RecordingWorker()
    resource = PrefillLayerResources(
        worker, SegmentTopology(4, 2, 2), plans, Weights(), Cache(), Metadata(),
        kv_sources=(2, 8, 14, 20), index_sources=(2, 8, 14, 20, 24, 28, 32, 36),
        c2a_sources=(2, 8, 14), max_seq_len=16)
    assert len(resource.attention) == len(resource.moe) == 40
    assert resource.attention[39]["compressed_indices"] is resource.attention[36]["topk_indices"]
    assert resource.attention[39]["candidate_mask"] is resource.attention[20]["candidate_mask"]
    resource.close()
