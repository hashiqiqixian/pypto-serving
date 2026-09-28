# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reference diagnostics must reject local or accumulated numerical failures."""
from types import SimpleNamespace

import pytest
import torch

from tools.validate_v41_swa_segment import compare_saved


@pytest.mark.parametrize("failure", [None, "residual", "pre_mix", "stage"])
def test_saved_comparison_keeps_all_acceptance_gates(failure):
    residual = torch.ones(1, 2, 4, 3)
    mix = torch.ones(1, 2, 4)
    data = {"actual_residual": residual.clone(), "expected_residual": residual,
            "actual_pre_mix": mix.clone(), "expected_pre_mix": mix}
    if failure in ("residual", "pre_mix"):
        data["actual_" + failure].add_(1)
    if failure == "stage":
        data["stages"] = [{"results": {"output": (False, "stage mismatch")}}]

    def comparator(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        assert inputs["num_tokens"].tolist() == [2]
        assert actual_outputs["x_next"] is actual
        assert expected_outputs["x_next"] is expected
        return torch.equal(actual, expected), "residual mismatch"

    moe = SimpleNamespace(_local_mhc_compare=lambda counts: comparator)
    topology = SimpleNamespace(local_capacity=2, world=1)
    if failure is None:
        compare_saved(data, moe, topology)
    else:
        with pytest.raises(AssertionError):
            compare_saved(data, moe, topology)
