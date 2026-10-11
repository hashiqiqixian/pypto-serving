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

from pypto_serving.model.deepseek_v41.composite import LayerState
from pypto_serving.model.deepseek_v41.final_output import (
    FinalOutput, collect_final_logits, prepare_final_output_rows,
)
from pypto_serving.model.deepseek_v41.segment_inputs import SegmentInputs


PARAMS = ("x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
          "normed", "logits", "sampled_ids", "done_epoch")


class RecordingWorker:
    def __init__(self):
        self.calls = []

    def run(self, program, *args, config):
        self.calls.append((program, args, config))


def test_output_uses_final_state_and_advances_epoch():
    worker = RecordingWorker()
    output = FinalOutput(worker, SimpleNamespace(compiled="program"), PARAMS, "config")
    state = LayerState("final-residual", "final-mix", "tp_local_token")
    arguments = {name: object() for name in PARAMS if name not in ("x_hc", "pre_mix", "done_epoch")}

    for epoch in (1, 2):
        assert output.run(state, arguments) == (arguments["logits"], arguments["sampled_ids"])
        _, values, _ = worker.calls[-1]
        assert values[:2] == ("final-residual", "final-mix")
        assert values[-1].value == epoch


def test_output_rejects_missing_weights_before_launch():
    worker = RecordingWorker()
    output = FinalOutput(worker, SimpleNamespace(compiled="program"), PARAMS, "config")
    state = LayerState(object(), object(), "tp_local_token")
    with pytest.raises(ValueError, match="head_weight"):
        output.run(state, {"norm_weight": object()}, )
    assert worker.calls == []


def test_final_rows_preserve_request_order_across_ranks():
    source_rows = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int64)
    inputs = SegmentInputs(None, None, source_rows, ((1, 0), (0, 1)), (2,))
    shared = torch.empty(2, 2, dtype=torch.int32).share_memory_()
    rows = prepare_final_output_rows(inputs, shared, local_capacity=3)
    assert rows.indices.tolist() == [[1, -1], [0, -1]]
    assert rows.indices is shared

    logits = torch.zeros(2, 2, 4, dtype=torch.float32)
    logits[1, 0] = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    logits[0, 0] = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    result = collect_final_logits(logits, rows)
    assert result.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    logits.zero_()
    assert result[0, 0] == 1


def test_final_rows_reject_per_rank_lm_head_overflow():
    source_rows = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int64)
    inputs = SegmentInputs(None, None, source_rows, ((0, 0), (0, 1)), (2,))
    shared = torch.full((2, 1), 7, dtype=torch.int32).share_memory_()
    with pytest.raises(ValueError, match="capacity"):
        prepare_final_output_rows(inputs, shared, local_capacity=2)
    assert shared.tolist() == [[7], [7]]
