# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private device regression for twelve offset-three C2A verification slabs."""
import argparse
from pathlib import Path
import sys

import pypto.language as pl
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--lib-root', type=Path, required=True)
parser.add_argument('-d', '--device', type=int, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.lib_root / 'models/deepseek_v4_1_flash'))
from compressor import COMPRESSED_CACHE_GROUP, HEAD_DIM, ROPE_DIM, c2a_compressor_publish, golden_c2a_compressor
from config import MOE_TOKENS, STATE_CAPACITY, STATE_WIDTH
from golden import TensorSpec, ratio_allclose, run
from quant import pack_bf16_nz


REQUESTS = 12
ACTIVE = REQUESTS * 6
TOKENS = MOE_TOKENS
BLOCKS = REQUESTS + 2
needed_groups = sum((6 + 3 + 3) // 4 for _ in range(REQUESTS))
old_capacity = (TOKENS + 3) // 4 + REQUESTS
print(f'Capacity counterexample: tokens={TOKENS}, requests={REQUESTS}, needed={needed_groups}, old={old_capacity}')
assert needed_groups == 36 and old_capacity == 32


@pl.jit
def check(
    kv: pl.Tensor[[TOKENS, HEAD_DIM], pl.FP32],
    score: pl.Tensor[[TOKENS, HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[TOKENS], pl.INT32],
    request_ids: pl.Tensor[[TOKENS], pl.INT32],
    query_starts: pl.Tensor[[REQUESTS + 1], pl.INT32],
    state_blocks: pl.Tensor[[REQUESTS, 1], pl.INT32],
    slots: pl.Tensor[[TOKENS], pl.INT64],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[TOKENS, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[TOKENS, ROPE_DIM // 2], pl.FP32],
    state_cache: pl.InOut[pl.Tensor[[BLOCKS, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    latent: pl.InOut[pl.Tensor[[TOKENS, HEAD_DIM], pl.BF16]],
):
    cache = pl.create_tensor([1, 128, 1, HEAD_DIM // 2], dtype=pl.UINT8)
    scales = pl.create_tensor([1, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], dtype=pl.BF16)
    ready = pl.system.task_dummy(deps=[])
    c2a_compressor_publish(
        kv, score, query_starts, positions, request_ids, state_blocks,
        state_cache, norm_weight, latent, pl.cast(ACTIVE, pl.INT32), ready,
        cosine, sine, slots, cache, scales,
    )
    return state_cache, latent


generator = torch.Generator().manual_seed(42)
kv = (torch.randint(0, 2, (TOKENS, HEAD_DIM), generator=generator) * 2 - 1).float()
positions = torch.full((TOKENS,), -1, dtype=torch.int32)
positions[:ACTIVE] = torch.arange(3, 9, dtype=torch.int32).repeat(REQUESTS)
request_ids = torch.full((TOKENS,), -1, dtype=torch.int32)
request_ids[:ACTIVE] = torch.arange(REQUESTS, dtype=torch.int32).repeat_interleave(6)
blocks = torch.arange(1, REQUESTS + 1, dtype=torch.int32).reshape(REQUESTS, 1)
state = torch.zeros(BLOCKS, STATE_CAPACITY, STATE_WIDTH)
state[0].fill_(0.25)
state[-1].fill_(0.75)
state[1:-1, 2, :HEAD_DIM] = 1.0
weight = pack_bf16_nz(torch.eye(HEAD_DIM, dtype=torch.bfloat16))


def golden(values):
    values['latent'].copy_(golden_c2a_compressor(
        values['kv'].bfloat16(), values['query_starts'], values['positions'],
        values['request_ids'], values['state_blocks'], values['state_cache'],
        weight, torch.zeros_like(weight), values['norm_weight'],
    )[0])


specs = [
    TensorSpec('kv', [TOKENS, HEAD_DIM], torch.float32, init_value=kv),
    TensorSpec('score', [TOKENS, HEAD_DIM], torch.float32, init_value=0.0),
    TensorSpec('positions', [TOKENS], torch.int32, init_value=positions),
    TensorSpec('request_ids', [TOKENS], torch.int32, init_value=request_ids),
    TensorSpec('query_starts', [REQUESTS + 1], torch.int32,
               init_value=torch.arange(REQUESTS + 1, dtype=torch.int32) * 6),
    TensorSpec('state_blocks', [REQUESTS, 1], torch.int32, init_value=blocks),
    TensorSpec('slots', [TOKENS], torch.int64, init_value=-1),
    TensorSpec('norm_weight', [HEAD_DIM], torch.bfloat16, init_value=1.0),
    TensorSpec('cosine', [TOKENS, ROPE_DIM // 2], torch.float32, init_value=1.0),
    TensorSpec('sine', [TOKENS, ROPE_DIM // 2], torch.float32, init_value=0.0),
    TensorSpec('state_cache', [BLOCKS, STATE_CAPACITY, STATE_WIDTH], torch.float32, init_value=state),
    TensorSpec('latent', [TOKENS, HEAD_DIM], torch.bfloat16, init_value=0.0),
]
result = run(
    fn=check, specs=specs, golden_fn=golden,
    compare_fn={
        'state_cache': ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        'latent': ratio_allclose(atol=1e-3, rtol=1e-3, valid_rows=ACTIVE, zero_tail=True),
    },
    config={'backend_type': 'a5', 'device_id': args.device},
)
print(result)
if not result.passed:
    raise SystemExit(result.error or 1)
