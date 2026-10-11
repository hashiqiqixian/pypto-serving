# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving dispatch boundary for lib's complete V4.1 decode backbone."""

import ctypes
import importlib
import logging
import os

import torch

from .composite import LayerState, MissingCompositeInterface
from .swa_segment import load_segment_modules


_LOG = logging.getLogger(__name__)


def compile_decode_backbone(compiler, lib_root, topology):
    """Compile the existing 40-layer lib entry without rebuilding model operators."""
    import pypto.language as pl

    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_fwd")
    config = importlib.import_module("models.deepseek_v4_1_flash.config")
    if module.N_LAYERS != config.FLASH.num_hidden_layers or module.N_LAYERS != 40:
        raise ValueError("decode backbone and checkpoint layer schedules disagree")
    program = compiler.compile(
        "v41_decode_fwd", module.l3_decode_fwd, attention_num_tokens=pl.RUNTIME,
    )
    return program, tuple(module.l3_decode_fwd.param_names)


def compile_decode_layer(compiler, lib_root, topology):
    """Compile lib's complete per-layer Block with family-specific metadata."""
    import pypto.language as pl

    load_segment_modules(lib_root, topology)
    module = importlib.import_module("models.deepseek_v4_1_flash.decode_layer")
    entry = module.l3_decode_layer
    program = compiler.compile(
        "v41_decode_layer", entry, layer_id=pl.RUNTIME,
        attention_num_tokens=pl.RUNTIME, attention_epoch=pl.RUNTIME,
        moe_epoch=pl.RUNTIME,
    )
    return program, tuple(entry.param_names)


class DecodeBackbone:
    """Run one device-resident decode step with explicit lib-owned state/cache ABI."""

    def __init__(self, worker, program, param_names, run_config):
        self.worker = worker
        self.program = program
        self.param_names = tuple(param_names)
        self.run_config = run_config
        self.failed = False
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError("duplicate decode ABI parameter")
        if not {"x_hc", "pre_mix", "attention_num_tokens"} <= set(self.param_names):
            raise ValueError("decode ABI must expose residual, delayed mix and active count")

    def run(self, state: LayerState, arguments, *, active_tokens: int) -> LayerState:
        if self.failed:
            raise RuntimeError("decode dispatch failed; close the worker before retrying")
        if state.layout != "tp_local_token":
            raise ValueError("decode backbone requires TP-local residual and pre_mix")
        if type(active_tokens) is not int or not 0 < active_tokens <= 192:
            raise ValueError("decode requires a compact active batch of 1..192 tokens")
        reserved = {"x_hc", "pre_mix", "attention_num_tokens"}
        if reserved & arguments.keys():
            raise ValueError("decode state/count must come from this dispatch, not argument buffers")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing decode ABI arguments: " + ", ".join(sorted(missing)))
        bound = dict(arguments, x_hc=state.residual, pre_mix=state.pre_mix,
                     attention_num_tokens=ctypes.c_int32(active_tokens))
        try:
            self.worker.run(self.program.compiled, *(bound[name] for name in self.param_names),
                            config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        return state


class DecodeLayerSegment:
    """Dispatch lib's complete Block once per layer on one persistent worker."""

    def __init__(self, worker, program, param_names, run_config, topology, moe_counts):
        expected = {"x_hc", "incoming_pre_mix", "x_next", "next_pre_mix",
                    "layer_id", "attention_num_tokens", "moe_num_tokens",
                    "attention_epoch", "moe_epoch"}
        self.param_names = tuple(param_names)
        if len(set(self.param_names)) != len(self.param_names) or not expected <= set(self.param_names):
            raise ValueError("decode layer ABI is missing complete Block state or counts")
        if (not isinstance(moe_counts, torch.Tensor) or moe_counts.device.type != "cpu"
                or moe_counts.dtype != torch.int32 or moe_counts.shape != (topology.world,)
                or not moe_counts.is_shared()):
            raise ValueError("decode MoE counts must be shared before worker startup")
        self.worker, self.program, self.run_config = worker, program, run_config
        self.topology, self.moe_counts = topology, moe_counts
        self.epoch = 0
        self.failed = False

    def run_layer(self, state: LayerState, plan, arguments, *, group_counts):
        if self.failed:
            raise RuntimeError("decode Block failed; recreate the worker")
        if state.layout != "tp_local_token":
            raise ValueError("decode Block requires TP-local state")
        if type(plan.layer_id) is not int or not 0 <= plan.layer_id < 40:
            raise ValueError("decode Block requires a checkpoint layer ID")
        if (len(group_counts) != self.topology.dp or
                any(type(n) is not int or not 0 <= n <= self.topology.capacity for n in group_counts)):
            raise ValueError("decode Block requires one bounded count per DP group")
        if len(set(group_counts)) != 1 or not group_counts[0]:
            raise MissingCompositeInterface(
                "decode_layer has one scalar attention_num_tokens for both DP groups; "
                "asymmetric or empty groups require isolated filler rows and cache pages "
                "before serving can safely use that ABI")
        reserved = {"x_hc", "incoming_pre_mix", "layer_id", "attention_num_tokens",
                    "moe_num_tokens", "attention_epoch", "moe_epoch"}
        if reserved & arguments.keys():
            raise ValueError("decode Block state and runtime scalars must come from dispatch")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing decode Block arguments: " + ", ".join(sorted(missing)))
        if (arguments["x_next"] is state.residual or arguments["next_pre_mix"] is state.pre_mix
                or arguments["x_next"] is arguments["next_pre_mix"]):
            raise ValueError("decode Block input and output state must not alias")
        if self.epoch >= (2**31 - 1) // 2:
            raise OverflowError("decode Block communication epoch exhausted")
        _, local = self.topology.counts(group_counts)
        self.moe_counts.copy_(torch.tensor(local, dtype=torch.int32))
        reset_windows = self.epoch == 512
        epoch = 1 if reset_windows else self.epoch + 1
        bound = dict(arguments, x_hc=state.residual, incoming_pre_mix=state.pre_mix,
                     layer_id=ctypes.c_int32(plan.layer_id),
                     attention_num_tokens=ctypes.c_int32(group_counts[0]),
                     moe_num_tokens=self.moe_counts,
                     attention_epoch=ctypes.c_int32(epoch), moe_epoch=ctypes.c_int32(epoch))
        trace = os.environ.get("PYPTO_V41_TRACE_DECODE") == "1"
        if trace:
            positions = arguments["position_ids"]
            lengths = arguments["compressed_lens"]
            slots = arguments["compressed_slots"]
            _LOG.info(
                "V4.1 decode enter layer=%d mode=%s epoch=%d groups=%s pos=%s lens=%s slots=%s",
                plan.layer_id, plan.mode, epoch, group_counts,
                [int(positions[rank, 0]) for rank in (0, 4)],
                [int(lengths[rank, 0]) for rank in (0, 4)],
                [int(slots[rank, 0]) for rank in (0, 4)],
            )
        if reset_windows:
            if not getattr(self.worker, "_persistent", False) or not hasattr(
                self.worker, "_reset_persistent_windows"
            ):
                raise RuntimeError("decode epoch reset requires a persistent PyPTO worker")
            previous_reset = self.worker._reset_persistent_windows
            self.worker._reset_persistent_windows = True
        try:
            self.worker.run(self.program.compiled,
                            *(bound[name] for name in self.param_names), config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        finally:
            if reset_windows:
                self.worker._reset_persistent_windows = previous_reset
        if trace:
            _LOG.info("V4.1 decode done layer=%d epoch=%d", plan.layer_id, epoch)
        self.epoch = epoch
        return LayerState(arguments["x_next"], arguments["next_pre_mix"], "tp_local_token")
