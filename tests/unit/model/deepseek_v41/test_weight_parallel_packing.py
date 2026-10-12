# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parallel expert staging preserves the complete layer's bytes and rank order."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.weight_loader import _load_layer_weights


class _Loader:
    def __init__(self, model_dir, rank, world):
        self.model_dir = model_dir.resolve()
        self.tp_size = 1
        self.tp_rank = 0
        self.ep_size = world
        self.ep_rank = rank
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.text = {
            "compress_ratios": [0],
            "kv_source_layer_ids": [],
            "index_source_layer_ids": [],
            "n_routed_experts": world,
        }
        self.store = SimpleNamespace(slice_session=nullcontext)

    def load(self, name, *, session):
        value = (sum(name.encode()) + self.ep_rank) % 256
        weight = torch.full((2, 16), value, dtype=torch.uint8)
        scale = torch.full((2, 16), (value + 1) % 256, dtype=torch.uint8)
        return SimpleNamespace(weight=weight, scale=scale)


def test_four_rank_parallel_experts_match_serial_layer(tmp_path):
    topology = SimpleNamespace(tp=1, world=4)
    loaders = tuple(_Loader(tmp_path, rank, 4) for rank in range(4))
    serial = _load_layer_weights(tmp_path, 0, topology, 1 << 20,
                                 swa_only=False, loaders=loaders)
    parallel = _load_layer_weights(tmp_path, 0, topology, 1 << 20,
                                   swa_only=False, loaders=loaders,
                                   parallel_experts=True)
    for serial_fields, parallel_fields in zip(serial, parallel):
        assert tuple(serial_fields) == tuple(parallel_fields)
        for name, value in serial_fields.items():
            assert torch.equal(value.view(torch.uint8), parallel_fields[name].view(torch.uint8))


def test_parallel_experts_rejects_eight_rank_path(tmp_path):
    topology = SimpleNamespace(tp=1, world=8)
    loaders = tuple(_Loader(tmp_path, rank, 8) for rank in range(8))
    with pytest.raises(ValueError, match="four-rank"):
        _load_layer_weights(tmp_path, 0, topology, 1 << 20,
                            swa_only=False, loaders=loaders, parallel_experts=True)
