# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private repeated-device test of the existing V4.1 attention signal protocol."""

import argparse
import ctypes
from pathlib import Path
import sys
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lib-root", type=Path, required=True)
parser.add_argument("--devices", required=True)
parser.add_argument("--rounds", type=int, default=600)
parser.add_argument("--compile-only", action="store_true")
parser.add_argument("--serving-dispatch", action="store_true")
args = parser.parse_args()
devices = [int(value) for value in args.devices.split(",")]
sys.path.insert(0, str(args.lib_root))
sys.path.insert(0, str(args.lib_root / "models/deepseek_v4_1_flash"))

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
import torch

from golden import ScalarSpec, TensorSpec, normalize_config, run
from o_proj import D, PREFILL_MAX_TOKENS, TP_SIZE, prefill_tp_output_all_reduce

WORLD = len(devices)
ROWS = 6
LAYERS = 3
assert TP_SIZE == 1 and args.rounds > 0
torch.set_num_threads(1)


@pl.jit
def repeat_rank(
    source: pl.Tensor[[ROWS, D], pl.BF16],
    output: pl.Out[pl.Tensor[[LAYERS, ROWS, D], pl.BF16]],
    window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    for layer in pl.range(LAYERS):
        prefill_tp_output_all_reduce(
            source, window, signal, output[layer], rank, pl.cast(0, pl.INT32),
            pl.cast(ROWS, pl.INT32), epoch + layer,
        )


@pl.jit.host
def repeat(
    head_hidden: pl.Tensor[[WORLD, ROWS, D], pl.BF16],
    output: pl.Out[pl.Tensor[[WORLD, LAYERS, ROWS, D], pl.BF16]],
    attention_epoch: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        window = pld.window(window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        repeat_rank(head_hidden[rank], output[rank], window, signal, rank, attention_epoch, device=rank)


source = (torch.arange(WORLD * ROWS * D).reshape(WORLD, ROWS, D) % 31).to(torch.bfloat16)
source.share_memory_()
output = torch.zeros((WORLD, LAYERS, ROWS, D), dtype=torch.bfloat16).share_memory_()
specs = [
    TensorSpec("head_hidden", tuple(source.shape), torch.bfloat16, init_value=source),
    TensorSpec("output", tuple(output.shape), torch.bfloat16),
    ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True),
    ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True),
]
config = {
    "backend_type": "a5", "execution_mode": False,
    "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
}
result = run(fn=repeat, specs=specs, config=config, compile_only=True)
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
start = time.monotonic()
with compiled.prepare(cfg, persistent=True, reset_persistent_windows=False,
                      inherited_host_tensors=(source, output)) as runtime:
    resident = None
    rollovers = 0
    previous_end = 0
    if args.serving_dispatch:
        from pypto_serving.model.deepseek_v41.npu_runner import V41ResidentResources

        resident = object.__new__(V41ResidentResources)
        resident.closed = resident.failed = False
        resident.worker, resident.run_config = runtime, cfg
        resident._dspark_drafter_epochs = {}
        resident.dspark_programs = {"dspark_drafter": SimpleNamespace(compiled=compiled)}
        resident.dspark_param_names = {"dspark_drafter": tuple(repeat.param_names)}
    for iteration in range(args.rounds):
        end_epoch = (iteration + 1) * LAYERS
        if resident is None:
            epoch = ctypes.c_int32(iteration * LAYERS + 1)
            runtime(source, output, epoch, epoch, config=cfg)
        else:
            resident.dispatch_dspark("dspark_drafter", {
                "head_hidden": source, "output": output, "attention_epoch": 1, "moe_epoch": 1,
            })
            end_epoch = resident._dspark_drafter_epochs[id(compiled)] - 1
            rollovers += end_epoch < previous_end
            previous_end = end_epoch
        if iteration % 64 == 0 or iteration + 1 == args.rounds:
            expected = source[:, None].expand_as(output)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            print(f"round={iteration + 1} epoch_end={end_epoch} rollovers={rollovers} "
                  f"seconds={time.monotonic() - start:.3f}", flush=True)
print(f"PASS: {args.rounds} rounds, {args.rounds * LAYERS} attention epochs", flush=True)
