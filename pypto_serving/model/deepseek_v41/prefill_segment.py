# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit prefill half-layer dispatch for the lib's sequence-parallel entries.

Signatures were audited at lib fbe92bfc. This module does not allocate caches,
invent metadata, or enable the full CLI backend. The caller must supply the
entire audited ABI and retain every program's communication windows.
"""
import importlib

from .swa_segment import ATTENTION_ARGS, SwaSegment, load_segment_modules


_WEIGHTS = (
    "x_hc pre_mix hc_attn_fn hc_attn_scale hc_attn_base attn_norm_weight "
    "wq_a wq_a_scale q_norm_weight wq_b wq_b_scale wkv wkv_scale kv_norm_weight "
    "attn_sink wo_a wo_b wo_b_scale "
)
_WINDOW = "window_slots window_indices window_cache window_cache_scale "
_OUTPUTS = "attn_input attn_output next_pre_mix x_hc_out num_tokens attention_epoch"
C2A_FULL_ARGS = (_WEIGHTS + "freqs_cos freqs_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale token_to_req_indices compressed_lens "
    "index_cache index_cache_scale index_block_table position_ids compressed_freqs_cos compressed_freqs_sin "
    "compressed_rope_positions compressor_wkv compressor_wgate query_start_loc state_block_table state_cache "
    "compressor_norm_weight compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale "
    "index_weights_proj topk_indices " + _OUTPUTS).split()
C2A_REUSE_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale compressed_indices " + _OUTPUTS).split()
C1A_ARGS = (_WEIGHTS + "rope_cos rope_sin " + _WINDOW +
    "compressed_cache compressed_cache_scale request_ids compressed_lens index_cache index_cache_scale "
    "index_block_table compressed_rope_cos compressed_rope_sin compressor_wkv compressor_norm_weight "
    "compressed_slots index_wk index_norm_weight index_wq_b index_wq_b_scale index_weights_proj "
    "topk_indices candidate_mask compressed_indices " + _OUTPUTS).split()
PREFILL_ARGUMENTS = {
    "swa": ATTENTION_ARGS, "c2a_full": C2A_FULL_ARGS, "c2a_reuse": C2A_REUSE_ARGS,
    "c1a_full": C1A_ARGS, "c1a_reindex": C1A_ARGS, "c1a_reuse": C1A_ARGS,
}


def compile_prefill_segments(compiler, lib_root, topology, modes):
    """Compile existing composite entries, sharing one packed-FP4 MoE program.

    C1A must use prefill_c1a_sp, not the older replicated-residual wrappers.
    No model operators are composed by this serving module.
    """
    import pypto.language as pl

    modes = tuple(dict.fromkeys(modes))
    if not modes or any(mode not in PREFILL_ARGUMENTS for mode in modes):
        raise ValueError("unsupported or empty prefill mode selection")
    swa, moe = load_segment_modules(lib_root, topology)
    attention = {}
    for mode in modes:
        if mode == "swa":
            entry = swa.make_hc_program(topology.capacity, topology.world, epochs=1)
        elif mode.startswith("c2a_"):
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_" + mode)
            entry = module.make_hc_program(topology.capacity, topology.world, epochs=1)
        else:
            module = importlib.import_module("models.deepseek_v4_1_flash.prefill_c1a_sp")
            entry = module.make_program(mode.removeprefix("c1a_"), topology.world, epochs=1)
        attention[mode] = compiler.compile("v41_prefill_" + mode, entry, attention_epoch=pl.RUNTIME)
    ffn = compiler.compile("v41_moe_segment", moe.l3_moe, moe_epoch=pl.RUNTIME)
    return attention, ffn


class PrefillSegment(SwaSegment):
    """Carry device residual/pre_mix through a selected prefill mode and MoE.

    The worker must own all supplied programs via make_segment_worker(). Each
    Attention program advances its own epoch; the shared MoE advances on every
    layer. Cache ownership, cross-layer producer bindings and padded metadata
    remain caller responsibilities. Any runtime failure poisons the whole worker.
    """

    def __init__(self, worker, attention_programs, moe_program, topology,
                 attention_counts, moe_counts, run_config):
        if not attention_programs or any(mode not in PREFILL_ARGUMENTS for mode in attention_programs):
            raise ValueError("unsupported or empty prefill program set")
        self.attention_programs = dict(attention_programs)
        super().__init__(worker, (next(iter(attention_programs.values())), moe_program),
                         topology, attention_counts, moe_counts, run_config)

    def run_layer(self, state, attention, moe, *, group_counts, mode):
        if mode not in self.attention_programs:
            raise ValueError(f"prefill mode was not compiled: {mode}")
        return self._run_layer(
            state, attention, moe, group_counts=group_counts,
            attention_program=self.attention_programs[mode], argument_names=PREFILL_ARGUMENTS[mode],
            input_mix="incoming_pre_mix" if mode == "swa" else "pre_mix",
            output_residual="output" if mode == "swa" else "x_hc_out",
        )
