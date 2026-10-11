# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-layer decode keeps producer selections and compression families distinct."""

import json
from pathlib import Path
from types import SimpleNamespace

from pypto_serving.model.deepseek_v41.decode_layer_resources import DecodeLayerResources
from pypto_serving.model.deepseek_v41.decode_weights import _ATTENTION, _MOE
from pypto_serving.model.deepseek_v41.execution_plan import plan_layers
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


def test_decode_layer_binds_real_producers_and_recycles_completed_state(monkeypatch):
    allocated, freed = [], []

    def allocate(worker, shape, dtype, *, stacked):
        assert stacked
        value = SimpleNamespace(shape=shape, dtype=dtype)
        allocated.append(value)
        return value

    monkeypatch.setattr(
        "pypto_serving.model.deepseek_v41.decode_layer_resources.alloc_device_buffer", allocate)
    fixture = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"
    plans = plan_layers(json.loads(fixture.read_text()))

    class Weights:
        def prefill_weights(self, plan, **sources):
            return ({name: object() for name in _ATTENTION.values()},
                    {**{name: object() for name in _MOE.values()}, "mxfp4_pair_lut": object()})

        def view(self, name, slot):
            return object()

    class Cache:
        plan = SimpleNamespace(
            group_blocks={"window": 2, "c2a": 2, "c1a": 2},
            physical_group_blocks={"window": 3, "c2a": 3, "c1a": 3},
            group_sources={"window": {"window": (0,)},
                           "c2a": {"compressed": (0,), "index": (0,)},
                           "c1a": {"compressed": (1,), "index": (1,)}})

        def prefill_views(self, plan, **sources):
            return {}

        def view(self, name, slot):
            return object()

    class Metadata:
        c2a = {"compressed_lens": object(), "index_block_table": object(),
               "token_to_req_indices": object(), "position_ids": object(),
               "query_start_loc": object(), "state_block_table": object()}
        c1a = {"compressed_lens": object(), "index_block_table": object()}

        def arguments(self, mode):
            return self.c2a if mode == "c2a_full" else self.c1a

    worker = SimpleNamespace(free_stacked_tensor=freed.append)
    resource = DecodeLayerResources(
        worker, SegmentTopology.for_decode(), plans, Weights(), Cache(), Metadata(),
        ("x_hc", "incoming_pre_mix", "x_next", "next_pre_mix", "compressed_lens"),
        kv_sources=(2, 8, 14, 20), index_sources=(2, 8, 14, 20, 24, 28, 32, 36),
        c2a_sources=(2, 8, 14))
    bound = resource.arguments
    assert bound[3]["compressed_indices"] is bound[2]["topk_indices"]
    assert bound[25]["compressed_indices"] is bound[24]["topk_indices"]
    assert bound[39]["candidate_mask"] is bound[20]["candidate_mask"]
    assert bound[3]["compressed_lens"] is not bound[21]["compressed_lens"]
    assert bound[0]["x_next"] is bound[2]["x_next"]
    assert bound[0]["x_next"] is not bound[1]["x_next"]
    resource.close()
    assert freed == list(reversed(allocated))
