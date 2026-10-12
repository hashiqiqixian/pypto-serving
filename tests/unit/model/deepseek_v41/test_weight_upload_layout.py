# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small-field upload staging preserves every rank and layer byte."""

import pytest
import torch
from pypto.runtime import DeviceTensor

from pypto_serving.model.deepseek_v41.weight_loader import (
    DecodeWeightBank,
    DecodeWeightLayout,
    ResidentFullDecodeWeights,
)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_rank_major_upload_preserves_slot_major_payload(dtype):
    layout = DecodeWeightLayout(2, 1, 1, 1, world=2, tp_size=1)
    source = torch.arange(24, dtype=torch.uint8).view(2, 2, 6)
    if dtype is torch.bfloat16:
        source = source.view(dtype)
    bank = DecodeWeightBank(layout)
    bank.tensors = {"wq_a": source}
    bank.prepare_rank_major_uploads(max_field_bytes=source.numel() * source.element_size())
    rank_major = bank.rank_major_uploads["wq_a"]
    assert rank_major.is_shared() and rank_major.is_contiguous()
    for slot in range(2):
        for rank in range(2):
            assert torch.equal(rank_major[rank, slot].view(torch.uint8),
                               source[slot, rank].view(torch.uint8))


def test_large_upload_field_keeps_original_layout():
    layout = DecodeWeightLayout(2, 1, 1, 1, world=2, tp_size=1)
    bank = DecodeWeightBank(layout)
    bank.tensors = {"wq_a": torch.zeros((2, 2, 6), dtype=torch.uint8)}
    bank.prepare_rank_major_uploads(max_field_bytes=1)
    assert not bank.rank_major_uploads


def test_rank_major_field_uploads_once_per_rank():
    source = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3).share_memory_()
    layout = DecodeWeightLayout(2, 1, 1, 1, world=2, tp_size=1)
    bank = DecodeWeightBank(layout)
    bank.tensors = {"hc_attn_fn": source}
    bank.finish = lambda names: None
    bank.prepare_rank_major_uploads()

    class Worker:
        def __init__(self):
            self.uploads = []

        def alloc_tensor(self, shape, dtype, *, init=None, worker_id):
            assert init is not None and init.is_contiguous()
            self.uploads.append((worker_id, init.clone()))
            return DeviceTensor(1000 + worker_id, shape, dtype)

        def free_stacked_tensor(self, tensor):
            pass

    worker = Worker()
    resident = ResidentFullDecodeWeights(worker, bank, ("hc_attn_fn",))
    assert len(worker.uploads) == 2
    for rank, uploaded in worker.uploads:
        assert torch.equal(uploaded.view(2, 2, 3), source[:, rank])
    resident.close()
