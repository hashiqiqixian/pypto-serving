# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident scratch binding preserves CPU outputs and dispatch epoch ownership."""
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41 import npu_runner


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("epoch", [1, 512])
@pytest.mark.parametrize("fail", [False, True])
def test_decode_binds_device_scratch_and_preserves_epochs(monkeypatch, epoch, fail, resident):
    names = npu_runner._DSPARK_RESIDENT_DECODE_SCRATCH
    arguments = {name: torch.zeros((4, 12, 5, 1)) for name in names}
    for name in ('accepted_ids', 'accepted_counts', 'last_sampled', 'draft_ids', 'confidences',
                 'slot_ids', 'generations', 'state_tokens', 'state_meta'):
        arguments[name] = torch.ones((4, 12))
    arguments['moe_epoch'] = 999
    original = dict(arguments)
    device = {name: object() for name in names}
    compiled = object()
    calls = []
    worker = SimpleNamespace(_reset_persistent_windows=False)

    def run(program, *values, config):
        calls.append((program, dict(zip(arguments, values)), worker._reset_persistent_windows))
        if fail:
            raise RuntimeError('device failure')

    worker.run = run
    resources = object.__new__(npu_runner.V41ResidentResources)
    resources.closed = resources.failed = False
    resources.worker = worker
    resources.run_config = object()
    resources.dspark_programs = {'decode': SimpleNamespace(compiled=compiled)}
    resources.dspark_param_names = {'decode': tuple(arguments)}
    resources.dspark_buffers = SimpleNamespace(decode_workspace=device) if resident else None
    resources._dspark_drafter_epochs = {id(compiled): epoch}
    module = SimpleNamespace(DSPARK_CHUNK_REQUESTS=8, DSPARK_DRAFT_LAYERS=3)
    monkeypatch.setattr(npu_runner.importlib, 'import_module', lambda _: module)
    if fail:
        with pytest.raises(RuntimeError, match='device failure'):
            resources.dispatch_dspark('decode', arguments)
    else:
        resources.dispatch_dspark('decode', arguments)
    bound = calls[0][1]
    assert all(bound[name] is (device[name] if resident else arguments[name]) for name in names)
    assert all(bound[name] is arguments[name] for name in arguments if name not in names and name!='moe_epoch')
    assert all(arguments[name] is value for name,value in original.items())
    expected_epoch = 1 if epoch==512 else epoch
    assert bound['moe_epoch']==expected_epoch
    assert calls[0][2] == (epoch==512)
    assert not worker._reset_persistent_windows
    assert resources._dspark_drafter_epochs[id(compiled)] == (epoch if fail else expected_epoch+6)


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_scratch_readback_respects_phase_and_explicit_outputs(prefill, override):
    resources = object.__new__(npu_runner.V41ResidentResources)
    host = {'target_hidden': torch.zeros((4, 2))}
    device = {'target_hidden': torch.full((4, 2), 7.0)}
    resources.dspark_host_resources = SimpleNamespace(decode_workspace=host)
    resources.dspark_buffers = SimpleNamespace(decode_workspace=device)
    outputs = {'target_hidden': lambda: torch.full((4, 2), 9.0)} if override else {}
    lowered = SimpleNamespace(workspace=dict(host) if prefill else host, outputs=outputs)
    actual = resources._read_dspark_output(lowered, 'target_hidden')
    expected = 9 if override else 0 if prefill else 7
    assert (actual == expected).all()
    actual.zero_()
    assert (device['target_hidden'] == 7).all()
