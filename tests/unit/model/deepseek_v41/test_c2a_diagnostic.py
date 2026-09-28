# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ragged diagnostics retain real causal prefixes and explicit inactive padding."""
import pytest
import torch

from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology
from tools.validate_v41_c2a_chain import select_active_state


def test_select_ragged_prefix_without_mutating_saved_state():
    topology = SegmentTopology(tp=2, dp=2)
    residual = torch.arange(4 * 16 * 4 * 5120, dtype=torch.float32).reshape(4, 16, 4, 5120)
    mix = torch.ones(4, 16, 4)
    saved = dict(actual_residual=residual, actual_pre_mix=mix)
    selected, selected_mix = select_active_state(saved, topology, [31, 0])
    assert torch.equal(selected[0], residual[0])
    assert torch.equal(selected[1, :15], residual[1, :15])
    assert not selected[1, 15:].count_nonzero() and not selected[2:].count_nonzero()
    assert not selected_mix[1, 15:].count_nonzero() and not selected_mix[2:].count_nonzero()
    assert residual[2:].count_nonzero() and mix[2:].eq(1).all()


@pytest.mark.parametrize("counts", [[33, 0], [1], [-1, 32]])
def test_select_prefix_rejects_invalid_counts_before_reading_state(counts):
    with pytest.raises(ValueError, match="active-token count"):
        select_active_state({}, SegmentTopology(tp=2, dp=2), counts)
