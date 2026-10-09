# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dispatch the lib-owned final-state output composite."""

import ctypes
import importlib
from dataclasses import dataclass

import torch

from .composite import LayerState
from .segment_inputs import SegmentInputs
from .swa_segment import load_segment_modules


@dataclass(frozen=True)
class FinalOutputRows:
    indices: torch.Tensor
    locations: tuple[tuple[int, int], ...]


def prepare_final_output_rows(inputs: SegmentInputs, indices: torch.Tensor, *,
                              local_capacity: int) -> FinalOutputRows:
    """Stage final-token rows into a pre-fork shared buffer in request order."""
    if not isinstance(inputs, SegmentInputs) or type(local_capacity) is not int or local_capacity <= 0:
        raise ValueError("final output requires valid packed inputs and rank capacities")
    if (not isinstance(indices, torch.Tensor) or indices.device.type != "cpu"
            or indices.dtype != torch.int32 or indices.ndim != 2
            or min(indices.shape) <= 0 or not indices.is_contiguous() or not indices.is_shared()):
        raise ValueError("LM-head row indices must be a pre-fork shared CPU INT32 matrix")
    world, max_logit_rows = indices.shape
    if tuple(inputs.row_indices.shape) != (world, local_capacity):
        raise ValueError("final output row map differs from the layer-state layout")
    counts = [0] * world
    locations = []
    selected = set()
    for rank, row in inputs.request_last_rows:
        if (type(rank) is not int or not 0 <= rank < world or type(row) is not int
                or not 0 <= row < local_capacity or inputs.row_indices[rank, row] < 0):
            raise ValueError("request final row is absent from the packed layer state")
        if (rank, row) in selected:
            raise ValueError("two requests cannot select the same final token row")
        selected.add((rank, row))
        slot = counts[rank]
        if slot >= max_logit_rows:
            raise ValueError("request final rows exceed the lib LM-head capacity on one rank")
        counts[rank] += 1
        locations.append((rank, slot))
    indices.fill_(-1)
    for (rank, row), (_, slot) in zip(inputs.request_last_rows, locations):
        indices[rank, slot] = row
    return FinalOutputRows(indices, tuple(locations))


def collect_final_logits(logits: torch.Tensor, rows: FinalOutputRows) -> torch.Tensor:
    """Return owned CPU logits in request order after output completion."""
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
            or logits.ndim != 3 or logits.shape[:2] != rows.indices.shape
            or logits.dtype != torch.float32):
        raise ValueError("lib output must be CPU FP32 [rank, logit_row, vocabulary]")
    if not rows.locations:
        raise ValueError("final output requires at least one request")
    result = torch.stack([logits[rank, slot] for rank, slot in rows.locations]).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("lib output contains non-finite request logits")
    return result


def make_final_output_program(lib_root, topology):
    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.final_output")
    return module.make_final_output_program(topology.tp, topology.dp)


def compile_final_output(compiler, lib_root, topology):
    import pypto.language as pl

    entry = make_final_output_program(lib_root, topology)
    return compiler.compile("v41_final_output", entry, done_epoch=pl.RUNTIME), tuple(entry.param_names)


class FinalOutput:
    """Dispatch final logits while retaining one completion epoch per worker."""

    def __init__(self, worker, program, param_names, run_config):
        self.worker = worker
        self.program = program
        self.param_names = tuple(param_names)
        self.run_config = run_config
        self.epoch = 0
        self.failed = False
        required = {"x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
                    "normed", "logits", "sampled_ids", "done_epoch"}
        if set(self.param_names) != required or len(self.param_names) != len(required):
            raise ValueError("final output ABI differs from the lib composite")

    def run(self, state: LayerState, arguments):
        if self.failed:
            raise RuntimeError("output dispatch failed; close the worker before retrying")
        if state.layout != "tp_local_token":
            raise ValueError("final output requires TP-local residual and pre_mix")
        reserved = {"x_hc", "pre_mix", "done_epoch"}
        if reserved & arguments.keys():
            raise ValueError("final state and epoch must come from this dispatch")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing output ABI arguments: " + ", ".join(sorted(missing)))
        if self.epoch >= (2**31 - 1) // 16:
            raise OverflowError("output completion epoch exhausted; recreate worker")
        next_epoch = self.epoch + 1
        bound = dict(arguments, x_hc=state.residual, pre_mix=state.pre_mix,
                     done_epoch=ctypes.c_int32(next_epoch))
        try:
            self.worker.run(self.program.compiled, *(bound[name] for name in self.param_names),
                            config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        self.epoch = next_epoch
        return arguments["logits"], arguments["sampled_ids"]
