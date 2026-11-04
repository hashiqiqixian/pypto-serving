# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private int32 notification counter probe without any model or blocking wait."""

import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lib-root", type=Path, required=True)
parser.add_argument("--devices", required=True)
parser.add_argument("--rounds", type=int, default=600)
parser.add_argument("--compile-only", action="store_true")
args = parser.parse_args()
devices = [int(value) for value in args.devices.split(",")]
sys.path.insert(0, str(args.lib_root))

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
import torch

from golden import TensorSpec, normalize_config, run

WORLD = len(devices)
INCREMENTS = 6
torch.set_num_threads(1)


@pl.jit
def probe_rank(
    signal: pld.DistributedTensor[[1, 16], pl.INT32],
    observed: pl.Out[pl.Tensor[[1, 16], pl.INT32]],
    rank: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="increment_signal") as updated:
        for step in pl.range(INCREMENTS):
            pld.system.notify(signal, peer=rank, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.spmd(1, name_hint="read_signal", deps=[updated]):
        block = pl.tile.get_block_idx()
        value = pld.tile.remote_load(signal, peer=rank, offsets=[block, 0], shape=[1, 16])
        observed = pl.tile.store(value, [block, 0], observed)


@pl.jit.host
def probe(observed: pl.Out[pl.Tensor[[WORLD, 1, 16], pl.INT32]]):
    buffer = pld.alloc_window_buffer([1, 16], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        signal = pld.window(buffer, [1, 16], dtype=pl.INT32)
        probe_rank(signal, observed[rank], rank, device=rank)


observed = torch.zeros((WORLD, 1, 16), dtype=torch.int32).share_memory_()
config = {
    "backend_type": "a5", "execution_mode": False,
    "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
}
result = run(fn=probe, specs=[TensorSpec("observed", tuple(observed.shape), torch.int32)],
             config=config, compile_only=True)
if not result.passed:
    raise RuntimeError(result.error)
if args.compile_only:
    print(f"COMPILE PASS: {result.work_dir}", flush=True)
    raise SystemExit(0)
cfg = normalize_config(config)
compiled = DistributedCompiledProgram.from_dir(
    result.work_dir, backend_type=cfg.backend_type, execution_mode=cfg.execution_mode,
    distributed_config=cfg.distributed_config,
)
with compiled.prepare(cfg, persistent=True, reset_persistent_windows=False,
                      inherited_host_tensors=(observed,)) as runtime:
    for iteration in range(args.rounds):
        runtime(observed, config=cfg)
        values = observed[:, 0, 0].tolist()
        expected = (iteration + 1) * INCREMENTS
        if iteration % 64 == 0 or expected >= 2040:
            print(f"round={iteration + 1} expected={expected} observed={values}", flush=True)
        if values != [expected] * WORLD:
            raise AssertionError(f"round={iteration + 1} expected={expected} observed={values}")
print(f"PASS: {args.rounds} rounds, counter={args.rounds * INCREMENTS}", flush=True)
