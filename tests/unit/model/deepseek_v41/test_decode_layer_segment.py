# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete decode Block dispatch must preserve state and reject unsafe DP counts."""

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.composite import LayerState, MissingCompositeInterface
from pypto_serving.model.deepseek_v41.decode_backbone import DecodeLayerSegment
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


PARAMS = ("x_hc", "incoming_pre_mix", "weight", "x_next", "next_pre_mix",
          "moe_num_tokens", "layer_id", "attention_num_tokens", "attention_epoch", "moe_epoch")


class Worker:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.reset_states = []

    def run(self, program, *args, config):
        self.reset_states.append(getattr(self, "_reset_persistent_windows", None))
        if self.fail:
            raise RuntimeError("device dispatch failed")
        self.calls.append((program, args, config))


def test_decode_block_passes_full_state_and_runtime_scalars_in_lib_order():
    worker = Worker()
    counts = torch.zeros(8, dtype=torch.int32).share_memory_()
    segment = DecodeLayerSegment(
        worker, SimpleNamespace(compiled="block"), PARAMS, "config",
        SegmentTopology.for_decode(), counts)
    first = LayerState(object(), object(), "tp_local_token")
    output, mix = object(), object()
    args = {"weight": object(), "x_next": output, "next_pre_mix": mix}
    result = segment.run_layer(first, SimpleNamespace(layer_id=20), args, group_counts=(1, 1))
    assert result == LayerState(output, mix, "tp_local_token")
    assert counts.tolist() == [1, 0, 0, 0, 1, 0, 0, 0]
    program, actual, config = worker.calls[0]
    assert program == "block" and config == "config"
    assert actual[0] is first.residual and actual[1] is first.pre_mix
    assert actual[3] is output and actual[4] is mix and actual[5] is counts
    assert [int(value.value) for value in actual[-4:]] == [20, 1, 1, 1]


def test_decode_block_rejects_asymmetric_dp_before_cache_mutation():
    worker = Worker()
    segment = DecodeLayerSegment(
        worker, SimpleNamespace(compiled="block"), PARAMS, None,
        SegmentTopology.for_decode(), torch.zeros(8, dtype=torch.int32).share_memory_())
    with pytest.raises(MissingCompositeInterface, match="isolated filler rows and cache pages"):
        segment.run_layer(LayerState(object(), object(), "tp_local_token"),
                          SimpleNamespace(layer_id=0),
                          {"weight": object(), "x_next": object(), "next_pre_mix": object()},
                          group_counts=(1, 0))
    assert worker.calls == []


def test_decode_block_failure_poisons_dispatcher():
    worker = Worker()
    worker.fail = True
    segment = DecodeLayerSegment(
        worker, SimpleNamespace(compiled="block"), PARAMS, None,
        SegmentTopology.for_decode(), torch.zeros(8, dtype=torch.int32).share_memory_())
    state = LayerState(object(), object(), "tp_local_token")
    arguments = {"weight": object(), "x_next": object(), "next_pre_mix": object()}
    with pytest.raises(RuntimeError, match="device dispatch failed"):
        segment.run_layer(state, SimpleNamespace(layer_id=0), arguments, group_counts=(1, 1))
    with pytest.raises(RuntimeError, match="recreate the worker"):
        segment.run_layer(state, SimpleNamespace(layer_id=0), arguments, group_counts=(1, 1))


def test_decode_block_resets_persistent_windows_before_epoch_reuse():
    worker = Worker()
    worker._persistent = True
    worker._reset_persistent_windows = False
    segment = DecodeLayerSegment(
        worker, SimpleNamespace(compiled="block"), PARAMS, None,
        SegmentTopology.for_decode(), torch.zeros(8, dtype=torch.int32).share_memory_())
    segment.epoch = 512
    state = LayerState(object(), object(), "tp_local_token")
    arguments = {"weight": object(), "x_next": object(), "next_pre_mix": object()}

    segment.run_layer(state, SimpleNamespace(layer_id=0), arguments, group_counts=(1, 1))
    assert worker.reset_states == [True]
    assert worker._reset_persistent_windows is False
    assert segment.epoch == 1
    assert worker.calls[0][1][-2].value == 1

    segment.run_layer(state, SimpleNamespace(layer_id=1), arguments, group_counts=(1, 1))
    assert worker.reset_states == [True, False]
    assert segment.epoch == 2


def test_decode_block_restores_window_policy_after_reset_failure():
    worker = Worker()
    worker._persistent = True
    worker._reset_persistent_windows = False
    worker.fail = True
    segment = DecodeLayerSegment(
        worker, SimpleNamespace(compiled="block"), PARAMS, None,
        SegmentTopology.for_decode(), torch.zeros(8, dtype=torch.int32).share_memory_())
    segment.epoch = 512
    with pytest.raises(RuntimeError, match="device dispatch failed"):
        segment.run_layer(
            LayerState(object(), object(), "tp_local_token"), SimpleNamespace(layer_id=0),
            {"weight": object(), "x_next": object(), "next_pre_mix": object()},
            group_counts=(1, 1),
        )
    assert worker.reset_states == [True]
    assert worker._reset_persistent_windows is False
    assert segment.failed
