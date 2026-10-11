# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Isolate the empty DP partition of a single-request decode dispatch."""

from types import MappingProxyType

import torch

from .request_state import ForwardStep, RequestSlice


def pad_single_decode(step: ForwardStep, embeddings: torch.Tensor, cache_plan):
    """Return a device-only filler; the request ledger retains the original step."""
    if not isinstance(step, ForwardStep) or step.phase != "decode":
        raise ValueError("decode padding requires a decode ForwardStep")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16 or embeddings.ndim != 2
            or embeddings.shape[0] != len(step.token_ids)):
        raise ValueError("decode padding requires one packed BF16 embedding per token")
    counts = tuple(sum(len(request.token_ids) for request in step.requests
                       if request.partition == partition) for partition in range(2))
    if counts == (1, 1):
        return step, embeddings, None
    if len(step.requests) != 1 or counts not in ((1, 0), (0, 1)):
        raise ValueError("only one-request asymmetric decode has isolated filler resources")
    if (len(cache_plan.group_blocks) != 3 or
            set(cache_plan.group_blocks) != set(cache_plan.group_sources) or
            any(cache_plan.physical_group_blocks.get(group, 0) <= blocks
                for group, blocks in cache_plan.group_blocks.items()) or
            cache_plan.physical_request_slots <= cache_plan.request_slots):
        raise ValueError("decode filler requires all three reserved cache families")
    partition = counts.index(0)
    name = "__v41_decode_padding__"
    while any(request.request_id == name for request in step.requests):
        name += "_"
    filler = RequestSlice(
        request_id=name, partition=partition, state_slot=cache_plan.request_slots,
        start=0, token_ids=(step.requests[0].token_ids[0],), prompt_length=1,
        pages=MappingProxyType({group: (blocks,) for group, blocks in cache_plan.group_blocks.items()}),
    )
    padded = ForwardStep(step.phase, (*step.requests, filler), step.epoch)
    return padded, torch.cat((embeddings, embeddings[:1]), dim=0), partition
