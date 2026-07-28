# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    KVCacheGroupSpec,
    KVCacheSpec,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.deepseek.weight_loader import (
    DeepSeekV4GlobalWeights,
    DeepSeekV4MtpWeights,
    DeepSeekV4StackedLayerWeights,
    DeepSeekV4WeightStore,
)
from pypto_serving.tools.profile import profile_span


logger = logging.getLogger(__name__)


def _kernel_trace_name(kernel_name: str) -> str:
    """Map model-specific L3 callable names to stable profiling lanes."""
    if "prefill" in kernel_name:
        return "kernel.prefill_fwd"
    if "decode" in kernel_name:
        return "kernel.decode_fwd"
    return f"kernel.{kernel_name}"


def _add_run_timing_args(args: dict[str, Any], timing: Any) -> None:
    """Attach runtime host/device timings to a profiling event when available."""
    if timing is None:
        return
    host_wall_us = getattr(timing, "host_wall_us", None)
    device_wall_us = getattr(timing, "device_wall_us", None)
    if host_wall_us is not None:
        args["host_wall_us"] = float(host_wall_us)
        args["host_wall_ms"] = float(host_wall_us) / 1000.0
    if device_wall_us is not None:
        args["device_wall_us"] = float(device_wall_us)
        args["device_wall_ms"] = float(device_wall_us) / 1000.0


DEEPSEEK_V4_RANKS = 8
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_VOCAB_SIZE = 129280
DEEPSEEK_V4_BLOCK_SIZE = 128
DEEPSEEK_V4_DECODE_BATCH = 4
DEEPSEEK_V4_DECODE_SEQ = 2
DEEPSEEK_V4_DECODE_TOKENS = DEEPSEEK_V4_DECODE_BATCH * DEEPSEEK_V4_DECODE_SEQ
DEEPSEEK_V4_PREFILL_BATCH = 1
DEEPSEEK_V4_PREFILL_SEQ = 128
# Prefill and decode share scheduler-owned rank-local physical pools. Group
# block IDs are local to each DP rank and address worker-resident cache shards.
DEEPSEEK_V4_PREFILL_ORI_MAX_BLOCKS = 128
DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS = 128
DEEPSEEK_V4_ORI_TABLE_MAX_BLOCKS = 128
DEEPSEEK_V4_SLIDING_WINDOW = 128
DEEPSEEK_V4_CMP_MAX_BLOCKS = 32
DEEPSEEK_V4_IDX_MAX_BLOCKS = 64
DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS = 64
DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_C128_STATE_BLOCK_SIZE = 8
DEEPSEEK_V4_C4_STATE_BLOCK_SIZE = 4
DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS = DEEPSEEK_V4_CMP_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS = DEEPSEEK_V4_IDX_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS = 2048
DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_IDX_HEAD_DIM = 128
DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
DEEPSEEK_V4_HCA_STATE_DIM = 2 * DEEPSEEK_V4_HCA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_STATE_DIM = 2 * DEEPSEEK_V4_CSA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_INNER_STATE_DIM = 2 * DEEPSEEK_V4_CSA_INNER_OUT_DIM
DEEPSEEK_V4_RMS_NORM_EPS = 1e-6
DEEPSEEK_V4_HC_EPS = 1e-6
# Layer-stacking counts for the packed all-layer decode_fwd kernel.
DEEPSEEK_V4_FWD_NUM_LAYERS = 43
DEEPSEEK_V4_CSA_NUM_LAYERS = 21
DEEPSEEK_V4_HCA_NUM_LAYERS = 20
DEEPSEEK_V4_LM_HEAD_TP_SIZE = 4
DEEPSEEK_V4_MAX_LOGIT_ROWS = 8

# Policy values indicate whether a resident argument contains mutable request
# cache state and therefore must be invalidated before its Host backing is
# reused for a later request.
_MAIN_STATIC_RESIDENT_POLICY = {
    "freqs_cos": False,
    "freqs_sin": False,
    "hc_head_fn": False,
    "hc_head_scale": False,
    "hc_head_base": False,
    "final_norm_w": False,
    "lm_head_weight": False,
}
_MAIN_CACHE_RESIDENT_POLICY = {
    "kv_cache": True,
    "cmp_kv": True,
    "idx_kv_cache": True,
    "idx_kv_scale": True,
    "hca_compress_state": True,
    "csa_compress_state": True,
    "csa_inner_compress_state": True,
}
_PREFILL_RESIDENT_POLICY = {
    **_MAIN_STATIC_RESIDENT_POLICY,
    **_MAIN_CACHE_RESIDENT_POLICY,
}
_DECODE_RESIDENT_POLICY = {
    **_MAIN_STATIC_RESIDENT_POLICY,
    **_MAIN_CACHE_RESIDENT_POLICY,
}
_MTP_RESIDENT_POLICY = {
    "freqs_cos": False,
    "freqs_sin": False,
    "lm_head_weight": False,
    "kv_cache": True,
}


def build_deepseek_v4_cache_group_specs(
    num_hidden_layers: int,
    compress_ratios: Sequence[int] | None = None,
    *,
    decode_batch: int = DEEPSEEK_V4_DECODE_TOKENS,
) -> tuple[KVCacheGroupSpec, ...]:
    """Describe cache namespaces and reserve padding pages for the decode specialization."""
    decode_batch = int(decode_batch)
    if decode_batch <= 0:
        raise ValueError("decode_batch must be positive")
    all_layers = tuple(range(int(num_hidden_layers)))
    ratios = tuple(int(ratio) for ratio in (compress_ratios or ()))[:num_hidden_layers]
    csa_layers = tuple(index for index, ratio in enumerate(ratios) if ratio == 4) or all_layers
    hca_layers = tuple(index for index, ratio in enumerate(ratios) if ratio == 128) or all_layers

    def group(
        name: str,
        layers: tuple[int, ...],
        *,
        block_size: int,
        element_bytes: int,
        row_width: int,
        max_blocks: int,
        compress_ratio: int = 1,
    ) -> KVCacheGroupSpec:
        if max_blocks <= decode_batch:
            raise ValueError(
                f"DeepSeekV4 {name} cache needs more than {decode_batch} physical blocks"
            )
        blocks_per_request = max(1, max_blocks // DEEPSEEK_V4_DECODE_BATCH)
        return KVCacheGroupSpec(
            name=name,
            layer_indices=layers,
            spec=KVCacheSpec(
                block_size=block_size,
                page_size_bytes=block_size * row_width * element_bytes,
                compress_ratio=compress_ratio,
            ),
            # One rank-local decode row owns one disjoint ring slice. This is
            # the same fixed-pool partitioning used by pypto-lib's B4 host
            # metadata, while allocation/release remains scheduler-owned.
            max_blocks_per_seq=blocks_per_request,
            # pypto-lib exposes one fixed physical pool per DP rank. The
            # scheduler shares that pool across the rank-local decode rows and
            # applies backpressure when their combined working sets do not fit.
            # Attention/compressor kernels execute every fixed B row even when
            # MoE receives a smaller num_tokens prefix. Reserve one physical
            # scratch page per kernel row so padding can never alias a live
            # scheduler-owned page.
            num_blocks=max_blocks - decode_batch,
            num_partitions=DEEPSEEK_V4_RANKS,
        )

    return (
        group(
            "ori",
            all_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=2,
            row_width=DEEPSEEK_V4_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS,
        ),
        group(
            "cmp",
            all_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=2,
            row_width=DEEPSEEK_V4_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_CMP_MAX_BLOCKS,
            # CSA is the least-compressed consumer of this shared family.
            compress_ratio=4,
        ),
        group(
            "idx",
            csa_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=2,
            row_width=DEEPSEEK_V4_IDX_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_IDX_MAX_BLOCKS,
            compress_ratio=4,
        ),
        group(
            "hca_state",
            hca_layers,
            block_size=DEEPSEEK_V4_C128_STATE_BLOCK_SIZE,
            element_bytes=4,
            row_width=DEEPSEEK_V4_HCA_STATE_DIM,
            max_blocks=DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS,
        ),
        group(
            "csa_state",
            csa_layers,
            block_size=DEEPSEEK_V4_C4_STATE_BLOCK_SIZE,
            element_bytes=4,
            row_width=DEEPSEEK_V4_CSA_STATE_DIM,
            max_blocks=DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS,
        ),
        group(
            "csa_inner_state",
            csa_layers,
            block_size=DEEPSEEK_V4_C4_STATE_BLOCK_SIZE,
            element_bytes=4,
            row_width=DEEPSEEK_V4_CSA_INNER_STATE_DIM,
            max_blocks=DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS,
        ),
    )


# Argument order for the packed all-43-layer ``l3_prefill_fwd`` kernel. This
# mirrors pypto-lib prefill_fwd.py ``l3_prefill_fwd`` host signature: every
# layer-stacked weight/state tensor in core-parameter order, followed by the
# ``hc_head`` collapse weights, final RMSNorm input, device LM-head weights, and
# hidden/logit outputs and owner-major execution metadata.
# The cache pools are ``pl.InOut`` tensors shared by prefill and decode; mutable
# block tables, slot mappings and token metadata remain shared host inputs.
_PREFILL_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "cmp_kv",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "idx_kv_cache",
    "idx_kv_scale",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "freqs_cos",
    "freqs_sin",
    "ori_block_table",
    "cmp_block_table",
    "idx_block_table",
    "ori_slot_mapping",
    "position_ids",
    "input_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "lm_head_weight",
    "hidden_out",
    "logits",
    "num_tokens_per_owner",
    "logit_row_indices",
)

# Argument order for the packed all-43-layer ``l3_decode_fwd`` kernel. This
# mirrors pypto-lib decode_fwd.py ``l3_decode_fwd`` host signature: after the
# ``hc_head`` collapse weights the kernel performs final RMSNorm and device
# LM-head projection.
_DECODE_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "freqs_cos",
    "freqs_sin",
    "block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "lm_head_weight",
    "hidden_out",
    "logits",
    "num_tokens_per_owner",
    "logit_row_indices",
)

_MTP_PREFILL_TENSOR_ORDER = (
    "hidden_states", "prev_hidden_states",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "ori_block_table", "ori_slot_mapping",
    "position_ids", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "lm_head_weight", "hidden_out", "pre_hc_hidden_out", "logits", "logit_row_indices",
)

_MTP_DECODE_TENSOR_ORDER = (
    "hidden_states", "prev_pre_hc_hidden", "position_ids",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "swa_slot_mapping", "swa_indices", "swa_lens",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "lm_head_weight", "hidden_out", "next_pre_hc_hidden", "logits", "logit_row_indices",
)

_DECODE_INPUT_TENSOR_FIELDS = (
    "input_ids",
    "position_ids",
    "kv_seq_lens",
    "block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "num_tokens_per_owner",
    "logit_row_indices",
)


@dataclass(frozen=True)
class DeepSeekV4CacheLayout:
    """Static cache layout baked into the current DeepSeekV4 kernels."""

    ranks: int = DEEPSEEK_V4_RANKS
    hc_mult: int = DEEPSEEK_V4_HC_MULT
    block_size: int = DEEPSEEK_V4_BLOCK_SIZE
    decode_batch: int = DEEPSEEK_V4_DECODE_BATCH
    decode_seq: int = DEEPSEEK_V4_DECODE_SEQ
    decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS
    prefill_batch: int = DEEPSEEK_V4_PREFILL_BATCH
    prefill_seq: int = DEEPSEEK_V4_PREFILL_SEQ
    prefill_ori_max_blocks: int = DEEPSEEK_V4_PREFILL_ORI_MAX_BLOCKS
    decode_ori_max_blocks: int = DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS
    ori_table_max_blocks: int = DEEPSEEK_V4_ORI_TABLE_MAX_BLOCKS
    sliding_window: int = DEEPSEEK_V4_SLIDING_WINDOW
    cmp_max_blocks: int = DEEPSEEK_V4_CMP_MAX_BLOCKS
    idx_max_blocks: int = DEEPSEEK_V4_IDX_MAX_BLOCKS
    hca_state_max_blocks: int = DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS
    csa_state_max_blocks: int = DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS
    csa_inner_state_max_blocks: int = DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS
    c128_state_block_size: int = DEEPSEEK_V4_C128_STATE_BLOCK_SIZE
    c4_state_block_size: int = DEEPSEEK_V4_C4_STATE_BLOCK_SIZE
    prefill_cmp_max_blocks: int = DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS
    prefill_idx_max_blocks: int = DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS
    prefill_hca_state_max_blocks: int = DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS
    prefill_csa_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS
    prefill_csa_inner_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS

    @property
    def prefill_cmp_block_num(self) -> int:
        """Physical cmp_kv blocks per layer in the packed prefill kernel."""
        return self.prefill_cmp_max_blocks

    @property
    def prefill_idx_block_num(self) -> int:
        """Physical idx_kv_cache blocks per CSA layer in the packed prefill kernel."""
        return self.prefill_idx_max_blocks

    def validate_runtime(self, config: ModelConfig, runtime: RuntimeConfig, device_ids: Sequence[int]) -> None:
        """Validate serving/runtime options against kernel-fixed dimensions."""
        if len(device_ids) != self.ranks:
            raise ValueError(f"DeepSeekV4 requires exactly {self.ranks} devices, got {len(device_ids)}")
        if runtime.page_size != self.block_size:
            raise ValueError(f"DeepSeekV4 kernels require page_size={self.block_size}, got {runtime.page_size}")
        global_decode_capacity = self.ranks * self.decode_batch
        if runtime.max_batch_size > global_decode_capacity:
            raise ValueError(
                f"DeepSeekV4 decode kernels support at most {global_decode_capacity} global active rows "
                f"({self.decode_batch} per rank x {self.ranks} ranks), "
                f"got max_batch_size={runtime.max_batch_size}"
            )
        decode_state_capacity = self.prefill_csa_state_max_blocks * self.c4_state_block_size
        if runtime.max_seq_len > decode_state_capacity:
            raise ValueError(
                "DeepSeekV4 pypto-lib decode CSA state tables currently support at most "
                f"max_seq_len={decode_state_capacity}, got {runtime.max_seq_len}. "
                "Increase the decode CSA state table depth in pypto-lib before serving longer contexts."
            )
        if self.decode_tokens != self.decode_batch * self.decode_seq:
            raise ValueError("DeepSeekV4 layout decode_tokens must equal decode_batch * decode_seq")
        expected = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "vocab_size": 129280,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
        }
        if actual != expected:
            mismatch = ", ".join(f"{name}={actual[name]} expected {value}" for name, value in expected.items())
            raise ValueError("DeepSeekV4 W8A8 kernels require Flash shape: " + mismatch)


def deepseek_v4_decode_layout(num_speculative_tokens: int) -> DeepSeekV4CacheLayout:
    """Select the smallest fixed eight-row decode tile for one MTP chunk.

    The pypto-lib decode programs always process eight rows per DP rank. MTP
    groups those rows into power-of-two request-local sequences so the target
    can verify as many drafts as possible without sacrificing more batch
    capacity than necessary. Draft depths larger than seven use repeated S=8
    target chunks.
    """
    num_speculative_tokens = int(num_speculative_tokens)
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")
    if num_speculative_tokens == 0:
        decode_seq = 1
    elif num_speculative_tokens == 1:
        decode_seq = 2
    elif num_speculative_tokens <= 3:
        decode_seq = 4
    else:
        decode_seq = 8
    return DeepSeekV4CacheLayout(
        decode_batch=DEEPSEEK_V4_DECODE_TOKENS // decode_seq,
        decode_seq=decode_seq,
        decode_tokens=DEEPSEEK_V4_DECODE_TOKENS,
    )


@dataclass(frozen=True)
class DeepSeekV4CacheMetadataBuilder:
    """Build kernel metadata from scheduler-owned rank-local cache block IDs."""

    layout: DeepSeekV4CacheLayout = field(default_factory=DeepSeekV4CacheLayout)

    @staticmethod
    def block_table_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        *,
        max_blocks: int,
    ) -> torch.Tensor:
        """Build a padded block table from scheduler-owned physical IDs."""
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        table = torch.zeros((len(per_request_block_ids), max_blocks), dtype=torch.int32)
        for row, block_ids in enumerate(per_request_block_ids):
            if len(block_ids) > max_blocks:
                raise ValueError(f"row {row} has {len(block_ids)} blocks, maximum is {max_blocks}")
            if any(int(block_id) < 0 for block_id in block_ids):
                raise ValueError("block IDs must not be negative")
            if block_ids:
                table[row, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32)
        return table

    @staticmethod
    def ring_block_table_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        *,
        max_blocks: int,
    ) -> torch.Tensor:
        """Expand scheduler-owned physical IDs across a logical ring table."""
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        table = torch.empty((len(per_request_block_ids), max_blocks), dtype=torch.int32)
        for row, block_ids in enumerate(per_request_block_ids):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ring block-table row {row} has no allocated blocks")
            if any(block_id < 0 for block_id in ids):
                raise ValueError("block IDs must not be negative")
            repeated = torch.tensor(ids, dtype=torch.int32).repeat(math.ceil(max_blocks / len(ids)))
            table[row].copy_(repeated[:max_blocks])
        return table

    @staticmethod
    def slot_mapping_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        block_size: int,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        """Map logical positions through scheduler-owned physical blocks."""
        if block_size <= 0 or compress_ratio <= 0:
            raise ValueError("block_size and compress_ratio must be positive")
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            for col, position in enumerate(row_positions):
                logical = int(position) // compress_ratio
                block_index, offset = divmod(logical, block_size)
                if not block_ids:
                    raise ValueError(f"slot-mapping row {row} has no allocated blocks")
                mapping[row, col] = int(block_ids[block_index % len(block_ids)]) * block_size + offset
        return mapping

    def sliding_window_slot_mapping_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Map absolute positions into one scheduler-owned sliding-window block."""
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            if not block_ids:
                raise ValueError(f"sliding-window row {row} has no allocated block")
            block_id = int(block_ids[0])
            for col, position in enumerate(row_positions):
                mapping[row, col] = block_id * self.layout.block_size + int(position) % self.layout.block_size
        return mapping

    def paged_ori_block_table_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Expand scheduler-owned ori ring blocks into the absolute logical table."""
        for row, block_ids in enumerate(per_request_block_ids):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ori ring row {row} has no allocated blocks")
            if len(ids) > self.layout.decode_ori_max_blocks:
                raise ValueError(
                    f"ori ring row {row} has {len(ids)} blocks, maximum is "
                    f"{self.layout.decode_ori_max_blocks}"
                )
            if any(block_id < 0 for block_id in ids):
                raise ValueError("ori ring block IDs must not be negative")
        return self.ring_block_table_from_ids(
            per_request_block_ids,
            max_blocks=self.layout.ori_table_max_blocks,
        )

    def paged_decode_slot_mapping_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Map absolute decode writes through scheduler-owned ori ring blocks."""
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        block_size = int(self.layout.block_size)
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ori ring row {row} has no allocated blocks")
            for col, position in enumerate(row_positions):
                logical_block, offset = divmod(int(position), block_size)
                mapping[row, col] = ids[logical_block % len(ids)] * block_size + offset
        return mapping

    def swa_window_indices_and_lens_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        exclude_current: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower SWA windows through scheduler-owned ori ring blocks."""
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        per_row = max((len(row) for row in positions), default=0)
        total = len(positions) * per_row
        window = int(self.layout.sliding_window)
        block_size = int(self.layout.block_size)
        indices = torch.full((total, window), -1, dtype=torch.int32)
        lens = torch.zeros((total,), dtype=torch.int32)
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ori ring row {row} has no allocated blocks")
            overlay = {int(position) for position in row_positions} if exclude_current else set()
            for seq_index, position in enumerate(row_positions):
                token = row * per_row + seq_index
                absolute_position = int(position)
                start = max(0, absolute_position - window + 1)
                out_index = 0
                for visible_position in range(start, absolute_position + 1):
                    if visible_position in overlay:
                        continue
                    logical_block, offset = divmod(visible_position, block_size)
                    physical_block = ids[logical_block % len(ids)]
                    indices[token, out_index] = physical_block * block_size + offset
                    out_index += 1
                lens[token] = out_index
        return indices, lens

    @staticmethod
    def compressed_slot_mapping_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        block_size: int,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Map compression-boundary positions through physical cache blocks."""
        if block_size <= 0 or compress_ratio <= 0:
            raise ValueError("block_size and compress_ratio must be positive")
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            for col, position in enumerate(row_positions):
                position = int(position)
                if (position + 1) % compress_ratio != 0:
                    continue
                logical = position // compress_ratio
                block_index, offset = divmod(logical, block_size)
                if not block_ids:
                    raise ValueError(f"compressed slot-mapping row {row} has no allocated blocks")
                mapping[row, col] = int(block_ids[block_index % len(block_ids)]) * block_size + offset
        return mapping

    @staticmethod
    def state_slot_mapping_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        state_block_size: int,
    ) -> torch.Tensor:
        """Map absolute positions through physical compressor-state blocks."""
        return DeepSeekV4CacheMetadataBuilder.slot_mapping_from_ids(
            per_request_block_ids,
            positions,
            block_size=state_block_size,
        )

    @staticmethod
    def replicate_first_row(tensor: torch.Tensor, *, actual_rows: int, kernel_rows: int) -> torch.Tensor:
        """Pad kernel inputs by replicating row 0 into inactive rows."""
        if actual_rows <= 0:
            raise ValueError("actual_rows must be positive")
        if kernel_rows < actual_rows:
            raise ValueError("kernel_rows must be >= actual_rows")
        if tensor.shape[0] < actual_rows:
            raise ValueError("tensor has fewer rows than actual_rows")
        out = torch.empty((kernel_rows, *tensor.shape[1:]), dtype=tensor.dtype)
        out[:actual_rows].copy_(tensor[:actual_rows])
        if actual_rows < kernel_rows:
            out[actual_rows:].copy_(tensor[0:1].expand(kernel_rows - actual_rows, *tensor.shape[1:]))
        return out


class DeepSeekV4InputBuilder:
    """Build fixed-shape host inputs for DeepSeekV4 HC-stack kernels."""

    def __init__(self, *, layout: DeepSeekV4CacheLayout, hidden_size: int) -> None:
        self.layout = layout
        self.hidden_size = int(hidden_size)

    def prefill_x_hc(
        self,
        embeddings: Sequence[torch.Tensor],
        *,
        ranks: Sequence[int],
        token_rows: int,
    ) -> torch.Tensor:
        """Build distinct rank-local prefill token streams for one EP dispatch."""
        if token_rows <= 0:
            raise ValueError("prefill token rows must be positive")
        if not embeddings or len(embeddings) != len(ranks):
            raise ValueError("prefill embeddings and ranks must be non-empty and aligned")
        if len(set(int(rank) for rank in ranks)) != len(ranks):
            raise ValueError("one prefill dispatch can contain at most one request per rank")

        padded_rows = []
        for rows in embeddings:
            if rows.ndim != 2 or rows.shape[0] <= 0 or int(rows.shape[1]) != self.hidden_size:
                raise ValueError("rank-local prefill embeddings must have shape [tokens, hidden]")
            if rows.shape[0] > token_rows:
                raise ValueError("rank-local prefill embeddings exceed the kernel token rows")
            rows = rows.to(torch.float32)
            padded = torch.zeros((token_rows, self.hidden_size), dtype=rows.dtype, device=rows.device)
            padded[: rows.shape[0]].copy_(rows)
            if rows.shape[0] < token_rows:
                pad_indices = torch.arange(token_rows - rows.shape[0], device=rows.device) % rows.shape[0]
                padded[rows.shape[0] :].copy_(rows.index_select(0, pad_indices))
            padded_rows.append(padded)

        # The scalar num_tokens contract is common to every rank. Inactive ranks
        # therefore run a harmless filler stream whose outputs and cache writes
        # are discarded; active ranks are overwritten below with their own data.
        rank_rows = padded_rows[0].unsqueeze(0).expand(self.layout.ranks, -1, -1).clone()
        for rank, rows in zip(ranks, padded_rows, strict=True):
            rank = int(rank)
            if not 0 <= rank < self.layout.ranks:
                raise ValueError(f"prefill rank {rank} is out of range")
            rank_rows[rank].copy_(rows)
        return self._expand_hc(rank_rows)

    def decode_x_hc(
        self,
        embeddings: torch.Tensor,
        *,
        ranks: Sequence[int],
        local_rows: Sequence[int],
    ) -> torch.Tensor:
        """Pack one autoregressive token per request into rank-local rows."""
        token_rows = embeddings.unsqueeze(1).expand(-1, self.layout.decode_seq, -1)
        return self._pack_decode_x_hc(token_rows, ranks=ranks, local_rows=local_rows)

    def decode_token_rows_x_hc(
        self,
        embeddings: torch.Tensor,
        *,
        ranks: Sequence[int],
        local_rows: Sequence[int],
    ) -> torch.Tensor:
        """Pack explicit request-local token rows for target verification."""
        return self._pack_decode_x_hc(embeddings, ranks=ranks, local_rows=local_rows)

    def _pack_decode_x_hc(
        self,
        token_rows: torch.Tensor,
        *,
        ranks: Sequence[int],
        local_rows: Sequence[int],
    ) -> torch.Tensor:
        """Pack explicit per-request token rows into the fixed decode tile."""
        if (
            token_rows.ndim != 3
            or token_rows.shape[0] <= 0
            or int(token_rows.shape[1]) != self.layout.decode_seq
            or int(token_rows.shape[2]) != self.hidden_size
        ):
            raise ValueError("decode token rows must have shape [requests, decode_seq, hidden]")
        if len(ranks) != token_rows.shape[0] or len(local_rows) != token_rows.shape[0]:
            raise ValueError("decode token rows, ranks, and local rows must align")
        token_rows = token_rows.to(torch.float32)
        rows = torch.zeros(
            (self.layout.ranks, self.layout.decode_tokens, self.hidden_size),
            dtype=token_rows.dtype,
            device=token_rows.device,
        )
        fallback = token_rows[0, -1]
        for rank in range(self.layout.ranks):
            rows[rank].copy_(fallback.reshape(1, -1).expand(self.layout.decode_tokens, -1))
        for index, (rank, local_row) in enumerate(zip(ranks, local_rows, strict=True)):
            rank = int(rank)
            local_row = int(local_row)
            if not 0 <= rank < self.layout.ranks:
                raise ValueError(f"decode rank {rank} is out of range")
            if not 0 <= local_row < self.layout.decode_batch:
                raise ValueError(f"decode local row {local_row} is out of range")
            start = local_row * self.layout.decode_seq
            rows[rank, start : start + self.layout.decode_seq].copy_(token_rows[index])
        return self._expand_hc(rows)

    def _expand_hc(self, rank_rows: torch.Tensor) -> torch.Tensor:
        if (
            rank_rows.ndim != 3
            or rank_rows.shape[0] != self.layout.ranks
            or rank_rows.shape[2] != self.hidden_size
        ):
            raise ValueError(
                "rank rows must have shape "
                f"[{self.layout.ranks}, tokens, {self.hidden_size}], got {tuple(rank_rows.shape)}"
            )
        return (
            rank_rows.unsqueeze(2)
            .expand(self.layout.ranks, rank_rows.shape[1], self.layout.hc_mult, self.hidden_size)
            .contiguous()
        )


@dataclass
class DeepSeekV4L3Callable:
    """Compiled HOST-dispatched DeepSeekV4 program."""

    compiled: object
    name: str
    block_dim: int | None = None
    aicpu_thread_num: int = 4


@dataclass
class _StaticDeviceTensor:
    """Rank-stacked CPU tensor marker uploaded to every chip worker once."""

    tensor: torch.Tensor
    cache_state: bool = False


@dataclass
class _TransientDeviceTensor:
    """CPU tensor marker uploaded for one layer dispatch and then freed."""

    tensor: torch.Tensor


@dataclass
class DeepSeekV4LayerCache:
    """Shared host backing for the rank-sharded DeepSeekV4 cache pools."""

    kv_cache: torch.Tensor
    cmp_kv: torch.Tensor
    idx_kv_cache: torch.Tensor
    idx_kv_scale: torch.Tensor
    hca_compress_state: torch.Tensor
    csa_compress_state: torch.Tensor
    csa_inner_compress_state: torch.Tensor


@dataclass
class DeepSeekV4DeviceCache:
    """Worker-resident rank shards shared by packed prefill and decode."""

    kv_cache: StackedDeviceTensor
    cmp_kv: StackedDeviceTensor
    idx_kv_cache: StackedDeviceTensor
    idx_kv_scale: StackedDeviceTensor
    hca_compress_state: StackedDeviceTensor
    csa_compress_state: StackedDeviceTensor
    csa_inner_compress_state: StackedDeviceTensor


@dataclass
class DeepSeekV4CompiledKernels:
    """Compiled-kernel placeholder and immutable DeepSeekV4 runtime metadata."""

    layout: DeepSeekV4CacheLayout
    model_dir: str
    weight_map: dict[str, str]
    weight_store: DeepSeekV4WeightStore
    compress_ratios: tuple[int, ...]
    layer_plan: tuple["DeepSeekV4LayerPlan", ...]
    kernel_dir: str
    prefill: DeepSeekV4L3Callable | None = None
    decode: DeepSeekV4L3Callable | None = None
    mtp_prefill: DeepSeekV4L3Callable | None = None
    mtp_decode: DeepSeekV4L3Callable | None = None
    freqs_cos: torch.Tensor | None = None
    freqs_sin: torch.Tensor | None = None
    platform: str = "a2a3"
    device_id: int = 0
    n_routed_experts: int = 256
    num_hash_layers: int = 3
    embedding_weight: torch.Tensor | None = None
    num_speculative_tokens: int = 0

    def l3_callables(self) -> tuple[DeepSeekV4L3Callable, ...]:
        """Return every compiled L3 program that the shared worker may run."""
        callables: list[DeepSeekV4L3Callable] = []
        if self.prefill is not None:
            callables.append(self.prefill)
        if self.decode is not None:
            callables.append(self.decode)
        if self.mtp_prefill is not None:
            callables.append(self.mtp_prefill)
        if self.mtp_decode is not None:
            callables.append(self.mtp_decode)
        return tuple(callables)


@dataclass(frozen=True)
class DeepSeekV4PreparedPrefillInputs:
    """Fixed-shape host tensors derived from one serving prefill chunk."""

    request_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    actual_tokens: tuple[int, ...]
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    ori_block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    hca_cmp_slot_mapping: torch.Tensor
    hca_state_slot_mapping: torch.Tensor
    csa_cmp_slot_mapping: torch.Tensor
    csa_idx_slot_mapping: torch.Tensor
    csa_state_slot_mapping: torch.Tensor
    csa_inner_state_slot_mapping: torch.Tensor
    num_tokens_per_owner: torch.Tensor
    logit_row_indices: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4PreparedDecodeInputs:
    """Fixed-shape host tensors derived from one decode scheduler batch."""

    request_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    local_rows: tuple[int, ...]
    per_rank_counts: tuple[int, ...]
    actual_batch: int
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    window_swa_indices: torch.Tensor
    window_swa_lens: torch.Tensor
    swa_slot_mapping: torch.Tensor
    swa_indices: torch.Tensor
    swa_lens: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    hca_cmp_slot_mapping: torch.Tensor
    hca_state_slot_mapping: torch.Tensor
    csa_cmp_slot_mapping: torch.Tensor
    csa_idx_slot_mapping: torch.Tensor
    csa_state_slot_mapping: torch.Tensor
    csa_inner_state_slot_mapping: torch.Tensor
    block_ids_by_group: tuple[dict[str, tuple[int, ...]], ...]
    num_tokens_per_owner: torch.Tensor
    logit_row_indices: torch.Tensor


@dataclass(frozen=True)
class _DeepSeekV4DecodeAssignment:
    """Mapping from scheduler order to rank-local kernel rows."""

    ranks: tuple[int, ...]
    local_rows: tuple[int, ...]
    per_rank_counts: tuple[int, ...]
    indices_by_rank: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _DeepSeekV4MainDecodeOutput:
    """Main-model tensors produced by one packed decode dispatch."""

    inputs: DeepSeekV4PreparedDecodeInputs
    hidden: torch.Tensor
    pre_hc_hidden: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class _DeepSeekV4MtpVerification:
    """Accepted tokens and recurrent inputs produced by target verification."""

    accepted_token_ids: list[list[int]]
    tail_token_ids: torch.Tensor
    tail_pre_hc_hidden: torch.Tensor
    tail_positions: torch.Tensor
    first_logits: torch.Tensor


@dataclass
class _DeepSeekV4DecodeSharedBuffers:
    """Reusable decode shared-memory buffers inherited by the L3 chip workers."""

    x_hc_a: torch.Tensor
    x_hc_b: torch.Tensor
    pre_hc_hidden_out: torch.Tensor
    x_out: torch.Tensor
    tensors: dict[str, torch.Tensor]


@dataclass
class _DeepSeekV4PrefillFwdSharedBuffers:
    """Reusable packed-prefill shared buffers inherited by the L3 chip workers.

    For the single ``l3_prefill_fwd`` dispatch the work caches are flattened 5-D
    (kv_cache/cmp_kv stack across all 43 hidden layers, idx_kv_cache across the 21
    compress_ratio==4 layers) and the compress-state kv/score caches stack across
    the CSA (x21) and HCA (x20) groups. The per-step metadata, RoPE tables and
    compress-state block tables are shared single per-rank copies (the kernel
    slices them per layer). ``tensors`` is keyed by ``_PREFILL_FWD_TENSOR_ORDER``
    name (excluding the stacked weights, which live in ``_stacked_host_weights``,
    and ``freqs_*``/``x_hc`` which are tracked explicitly). The final normalized
    hidden output is held separately in ``_prefill_output_buffer``.
    """

    x_hc: torch.Tensor
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor
    tensors: dict[str, torch.Tensor]


@dataclass
class _DeepSeekV4MtpSharedBuffers:
    """MTP weights, unified SWA cache, recurrent state, and outputs."""

    weights: dict[str, torch.Tensor]
    prefill_hidden_in: torch.Tensor
    prefill_prev_hidden_in: torch.Tensor
    prefill_input_ids: torch.Tensor
    prefill_position_ids: torch.Tensor
    prefill_block_table: torch.Tensor
    prefill_slot_mapping: torch.Tensor
    prefill_kv_cache: torch.Tensor
    decode_hidden_in: torch.Tensor
    decode_prev_hidden_in: torch.Tensor
    decode_input_ids: torch.Tensor
    decode_position_ids: torch.Tensor
    decode_slot_mapping: torch.Tensor
    decode_swa_indices: torch.Tensor
    decode_swa_lens: torch.Tensor
    decode_kv_cache: torch.Tensor
    prefill_hidden_out: torch.Tensor
    prefill_pre_hc_out: torch.Tensor
    prefill_logits: torch.Tensor
    prefill_logit_row_indices: torch.Tensor
    decode_hidden_out: torch.Tensor
    decode_pre_hc_out: torch.Tensor
    decode_logits: torch.Tensor
    decode_logit_row_indices: torch.Tensor


@dataclass(frozen=True)
class _DeepSeekV4MtpPrefillContext:
    """Request-local inputs retained until the first sampled token is known."""

    rank: int
    actual_tokens: int
    hidden_states: torch.Tensor
    prev_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DeepSeekV4MtpRequestState:
    """Speculative state owned by one serving request."""

    prefill_context: _DeepSeekV4MtpPrefillContext | None = None
    draft_token_id: int | None = None
    draft_pre_hc_hidden: torch.Tensor | None = None
    draft_position: int | None = None
    proposed_tokens: int = 0
    accepted_tokens: int = 0


@dataclass(frozen=True)
class DeepSeekV4LayerPlan:
    """Per-layer execution metadata for DeepSeekV4 serving."""

    layer_id: int
    compress_ratio: int
    attention_kind: str
    include_tid2eid: bool
    include_gate_bias: bool


def deepseek_v4_attention_kind(compress_ratio: int) -> str:
    """Return the DeepSeekV4 attention family for a compression ratio."""
    if compress_ratio == 0:
        return "swa"
    if compress_ratio == 128:
        return "hca"
    if compress_ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeekV4 attention compress ratio: {compress_ratio}")


def build_deepseek_v4_layer_plan(
    *,
    compress_ratios: Sequence[int],
    num_hidden_layers: int,
    num_hash_layers: int,
) -> tuple[DeepSeekV4LayerPlan, ...]:
    """Build the per-layer serving plan from config metadata."""
    if len(compress_ratios) < num_hidden_layers:
        raise ValueError("compress_ratios must include at least one entry per hidden layer")
    return tuple(
        DeepSeekV4LayerPlan(
            layer_id=layer_id,
            compress_ratio=int(compress_ratios[layer_id]),
            attention_kind=deepseek_v4_attention_kind(int(compress_ratios[layer_id])),
            include_tid2eid=layer_id < num_hash_layers,
            include_gate_bias=layer_id >= num_hash_layers,
        )
        for layer_id in range(num_hidden_layers)
    )


def accept_mtp_tokens(main_token_ids: torch.Tensor, draft_token_ids: torch.Tensor) -> list[list[int]]:
    """Accept the longest matching MTP prefix plus one target-model token."""
    main = main_token_ids.detach().cpu().to(torch.long)
    draft = draft_token_ids.detach().cpu().to(torch.long)
    if main.ndim != 2 or main.shape[1] < 2:
        raise ValueError(f"main_token_ids must have shape [batch, K+1], got {tuple(main.shape)}")
    if draft.ndim != 2 or draft.shape != (main.shape[0], main.shape[1] - 1):
        raise ValueError(
            "draft_token_ids must have shape "
            f"{(main.shape[0], main.shape[1] - 1)}, got {tuple(draft.shape)}"
        )
    accepted: list[list[int]] = []
    for row in range(main.shape[0]):
        matched = 0
        while matched < draft.shape[1] and int(draft[row, matched]) == int(main[row, matched]):
            matched += 1
        accepted.append([int(token) for token in main[row, : matched + 1]])
    return accepted


class DeepSeekV4ModelRunner(ModelRunner):
    """Runner boundary for DeepSeekV4 W8A8 kernels and model-specific caches."""

    def __init__(self, *, compiled: DeepSeekV4CompiledKernels) -> None:
        super().__init__()
        self._compiled = compiled
        self.cache_metadata = DeepSeekV4CacheMetadataBuilder(layout=compiled.layout)
        self.input_builder: DeepSeekV4InputBuilder | None = None
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[
            tuple[int, tuple[int, ...], torch.dtype], StackedDeviceTensor
        ] = {}
        self._l3_cache_tensor_keys: set[tuple[int, tuple[int, ...], torch.dtype]] = set()
        self._decode_work_cache: DeepSeekV4LayerCache | None = None
        self._decode_device_cache: DeepSeekV4DeviceCache | None = None
        self._decode_cache_block_ids: dict[str, dict[str, set[int]]] = {}
        self._global_weights: DeepSeekV4GlobalWeights | None = None
        self._static_final_norm_weight: torch.Tensor | None = None
        self._static_lm_head_weight: torch.Tensor | None = None
        self._static_freqs_cos: torch.Tensor | None = None
        self._static_freqs_sin: torch.Tensor | None = None
        self._prefill_fwd_buffers: _DeepSeekV4PrefillFwdSharedBuffers | None = None
        self._decode_buffers: _DeepSeekV4DecodeSharedBuffers | None = None
        self._stacked_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._mtp_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._prefill_output_buffer: torch.Tensor | None = None
        self._prefill_pre_hc_output_buffer: torch.Tensor | None = None
        self._prefill_logits_buffer: torch.Tensor | None = None
        self._decode_logits_buffer: torch.Tensor | None = None
        self._mtp_buffers: _DeepSeekV4MtpSharedBuffers | None = None
        self._mtp_device_kv_cache: StackedDeviceTensor | None = None
        self._mtp_request_states: dict[str, _DeepSeekV4MtpRequestState] = {}
        self._mtp_proposed_tokens = 0
        self._mtp_accepted_tokens = 0
        if compiled.num_speculative_tokens:
            self._decode_flow = self._run_mtp_decode
            self._prefill_completion = self._capture_mtp_prefill_context
        else:
            self._decode_flow = self._run_autoregressive_decode
            self._prefill_completion = self._ignore_prefill_context

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Initialize runner state and return scheduler-only KV block capacity.

        DeepSeekV4 owns its model-specific cache tensors, while the scheduler's
        grouped ``KvCacheManager`` is the sole owner of request partitions and
        block IDs. No generic KV tensors are allocated here.
        """
        self.input_builder = DeepSeekV4InputBuilder(
            layout=self._compiled.layout,
            hidden_size=config.hidden_size,
        )
        self._decode_cache_block_ids.clear()
        if runtime.total_kv_pages is not None:
            return int(runtime.total_kv_pages)
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        return int(runtime.max_batch_size * max_blocks_per_seq)

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Discard cache ownership metadata for finished or preempted requests."""
        request_ids = tuple(request_ids)
        for request_id in request_ids:
            self._decode_cache_block_ids.pop(request_id, None)
            state = self._mtp_request_states.pop(request_id, None)
            if state is not None and state.proposed_tokens:
                logger.info(
                    "DeepSeekV4 MTP acceptance for %s: accepted=%d proposed=%d rate=%.2f%%",
                    request_id,
                    state.accepted_tokens,
                    state.proposed_tokens,
                    100.0 * state.accepted_tokens / state.proposed_tokens,
                )

    def preflight(self, record: ModelRecord) -> None:
        """Stage host buffers and allocate the resident cache before worker readiness."""
        self._ensure_l3_shared_buffers(record.runtime_model)
        self._materialize_decode_device_cache()
        self._materialize_mtp_device_kv_cache()

    def load_packed_global_weights(self) -> DeepSeekV4GlobalWeights:
        """Load global tensors and shard the device LM head across its TP ranks."""
        if self._global_weights is None:
            loaded = self._compiled.weight_store.load_packed_global_weights(
                ranks=DEEPSEEK_V4_LM_HEAD_TP_SIZE
            )
            exact_weight = loaded.lm_head_weight[:, : loaded.lm_head_layout.vocab_per_rank, :].contiguous()
            self._global_weights = replace(loaded, lm_head_weight=exact_weight)
        return self._global_weights

    def load_stacked_layer_weights(self) -> DeepSeekV4StackedLayerWeights:
        """Load and stack all hidden-layer weights for the packed decode_fwd kernel."""
        compress_ratios = tuple(int(layer.compress_ratio) for layer in self._compiled.layer_plan)
        return self._compiled.weight_store.load_stacked_layer_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=self._compiled.num_hash_layers,
        )

    def load_mtp_weights(self) -> DeepSeekV4MtpWeights:
        """Load the single checkpoint MTP draft layer."""
        return self._compiled.weight_store.load_mtp_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
        )

    def prepare_prefill_inputs(self, model: RuntimeModel, batch: PrefillBatch) -> DeepSeekV4PreparedPrefillInputs:
        """Build DeepSeekV4 prefill host inputs for the current scheduler chunk."""
        builder = self._require_input_builder()
        layout = self._compiled.layout
        request_count = len(batch.request_ids)
        if request_count <= 0 or request_count > layout.ranks * layout.prefill_batch:
            raise ValueError(
                "DeepSeekV4 prefill supports one local request per rank and "
                f"at most {layout.ranks} global requests, got {request_count}"
            )
        if len(batch.cache_partitions) != request_count:
            raise ValueError("DeepSeekV4 prefill requires one cache partition per request")
        ranks = tuple(int(rank) for rank in batch.cache_partitions)
        if len(set(ranks)) != request_count:
            raise ValueError("DeepSeekV4 prefill accepts at most one request per rank per dispatch")
        if min(ranks) < 0 or max(ranks) >= layout.ranks:
            raise ValueError(f"DeepSeekV4 prefill cache partitions must be in [0, {layout.ranks - 1}]")
        group_rows = self._normalize_group_block_ids(
            batch.block_ids_by_group,
            actual_batch=request_count,
        )

        actual_tokens_by_request = []
        kernel_embeddings_by_request = []
        input_ids_by_request = []
        position_ids_by_request = []
        ori_block_tables = []
        cmp_block_tables = []
        idx_block_tables = []
        hca_state_block_tables = []
        csa_state_block_tables = []
        csa_inner_state_block_tables = []
        ori_slot_mappings = []
        hca_cmp_slot_mappings = []
        hca_state_slot_mappings = []
        csa_cmp_slot_mappings = []
        csa_idx_slot_mappings = []
        csa_state_slot_mappings = []
        csa_inner_state_slot_mappings = []

        if batch.input_embeddings is None:
            raise ValueError("DeepSeek V4 prefill requires host input embeddings")

        # Prefill writes directly into the scheduler-owned rank-local physical
        # pools. The same worker-resident shards are passed to decode, so no
        # parent-side cache snapshot or handoff is required.
        for index, (rank, groups) in enumerate(zip(ranks, group_rows, strict=True)):
            actual_tokens = batch.chunk_lens[index]
            chunk_offset = batch.chunk_offsets[index]
            chunk_start = batch.chunk_starts[index]
            positions = list(range(chunk_start, chunk_start + actual_tokens))
            if positions[-1] >= model.runtime.max_seq_len:
                raise ValueError(
                    f"prefill position {positions[-1]} exceeds max_seq_len={model.runtime.max_seq_len}"
                )
            chunk_end = chunk_offset + actual_tokens
            embeddings = batch.input_embeddings[chunk_offset:chunk_end].to(torch.float32).cpu()
            token_ids = batch.token_ids[chunk_offset:chunk_end].detach().cpu().to(torch.long)
            kernel_tokens = self._prefill_kernel_tokens(actual_tokens)
            kernel_positions = self._prefill_kernel_positions(
                positions,
                kernel_tokens=kernel_tokens,
                max_seq_len=model.runtime.max_seq_len,
            )
            kernel_embeddings_by_request.append(self._padded_rows(embeddings, kernel_tokens))
            input_ids_by_request.append(
                self._padded_vector(token_ids, layout.prefill_seq, dtype=torch.long)
            )
            position_ids_by_request.append(
                self._prefill_position_ids(kernel_positions, layout.prefill_seq)
            )
            actual_tokens_by_request.append(actual_tokens)
            ori_block_tables.append(
                self.cache_metadata.ring_block_table_from_ids(
                    (groups["ori"],),
                    max_blocks=layout.prefill_ori_max_blocks,
                )[0]
            )
            cmp_block_tables.append(
                self.cache_metadata.ring_block_table_from_ids(
                    (groups["cmp"],),
                    max_blocks=layout.prefill_cmp_max_blocks,
                )[0]
            )
            idx_block_tables.append(
                self.cache_metadata.ring_block_table_from_ids(
                    (groups["idx"],),
                    max_blocks=layout.prefill_idx_max_blocks,
                )[0]
            )
            hca_state_block_tables.append(
                self.cache_metadata.ring_block_table_from_ids(
                    (groups["hca_state"],),
                    max_blocks=layout.prefill_hca_state_max_blocks,
                )[0]
            )
            csa_state_block_tables.append(
                self.cache_metadata.ring_block_table_from_ids(
                    (groups["csa_state"],),
                    max_blocks=layout.prefill_csa_state_max_blocks,
                )[0]
            )
            csa_inner_state_block_tables.append(
                self.cache_metadata.ring_block_table_from_ids(
                    (groups["csa_inner_state"],),
                    max_blocks=layout.prefill_csa_inner_state_max_blocks,
                )[0]
            )
            ori_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.paged_decode_slot_mapping_from_ids(
                        (groups["ori"],),
                        (positions,),
                    )[0],
                    layout.prefill_seq,
                )
            )
            hca_cmp_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.compressed_slot_mapping_from_ids(
                        (groups["cmp"],),
                        (positions,),
                        block_size=layout.block_size,
                        compress_ratio=128,
                    )[0],
                    layout.prefill_seq,
                )
            )
            hca_state_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.state_slot_mapping_from_ids(
                        (groups["hca_state"],),
                        (positions,),
                        state_block_size=layout.c128_state_block_size,
                    )[0],
                    layout.prefill_seq,
                )
            )
            csa_cmp_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.compressed_slot_mapping_from_ids(
                        (groups["cmp"],),
                        (positions,),
                        block_size=layout.block_size,
                        compress_ratio=4,
                    )[0],
                    layout.prefill_seq,
                )
            )
            csa_idx_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.compressed_slot_mapping_from_ids(
                        (groups["idx"],),
                        (positions,),
                        block_size=layout.block_size,
                        compress_ratio=4,
                    )[0],
                    layout.prefill_seq,
                )
            )
            csa_state_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.state_slot_mapping_from_ids(
                        (groups["csa_state"],),
                        (positions,),
                        state_block_size=layout.c4_state_block_size,
                    )[0],
                    layout.prefill_seq,
                )
            )
            csa_inner_state_slot_mappings.append(
                self._pad_prefill_mapping(
                    self.cache_metadata.state_slot_mapping_from_ids(
                        (groups["csa_inner_state"],),
                        (positions,),
                        state_block_size=layout.c4_state_block_size,
                    )[0],
                    layout.prefill_seq,
                )
            )

        num_tokens_per_owner = torch.zeros(layout.ranks, dtype=torch.int32)
        logit_row_indices = torch.full(
            (layout.ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS),
            -1,
            dtype=torch.int32,
        )
        for rank, actual_tokens in zip(ranks, actual_tokens_by_request, strict=True):
            num_tokens_per_owner[rank] = actual_tokens
            logit_row_indices[rank, 0] = actual_tokens - 1

        return DeepSeekV4PreparedPrefillInputs(
            request_ids=tuple(batch.request_ids),
            ranks=ranks,
            actual_tokens=tuple(actual_tokens_by_request),
            x_hc=builder.prefill_x_hc(
                kernel_embeddings_by_request,
                ranks=ranks,
                token_rows=layout.prefill_seq,
            ),
            input_ids=self._rank_scatter(input_ids_by_request, ranks),
            position_ids=self._rank_scatter(position_ids_by_request, ranks),
            ori_block_table=self._rank_scatter(ori_block_tables, ranks),
            ori_slot_mapping=self._rank_scatter_mappings(ori_slot_mappings, ranks),
            cmp_block_table=self._rank_scatter(cmp_block_tables, ranks),
            idx_block_table=self._rank_scatter(idx_block_tables, ranks),
            hca_compress_state_block_table=self._rank_scatter(hca_state_block_tables, ranks),
            csa_compress_state_block_table=self._rank_scatter(csa_state_block_tables, ranks),
            csa_inner_compress_state_block_table=self._rank_scatter(csa_inner_state_block_tables, ranks),
            hca_cmp_slot_mapping=self._rank_scatter_mappings(hca_cmp_slot_mappings, ranks),
            hca_state_slot_mapping=self._rank_scatter_mappings(hca_state_slot_mappings, ranks),
            csa_cmp_slot_mapping=self._rank_scatter_mappings(csa_cmp_slot_mappings, ranks),
            csa_idx_slot_mapping=self._rank_scatter_mappings(csa_idx_slot_mappings, ranks),
            csa_state_slot_mapping=self._rank_scatter_mappings(csa_state_slot_mappings, ranks),
            csa_inner_state_slot_mapping=self._rank_scatter_mappings(
                csa_inner_state_slot_mappings,
                ranks,
            ),
            num_tokens_per_owner=num_tokens_per_owner,
            logit_row_indices=logit_row_indices,
        )

    @staticmethod
    def _require_decode_hidden_states(batch: DecodeBatch) -> torch.Tensor:
        """Return DeepSeek decode hidden states or reject a device-embedding batch."""
        hidden_states = batch.hidden_states
        if hidden_states is None:
            raise ValueError("DeepSeek V4 decode requires host hidden states")
        return hidden_states

    def prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build inputs for the single-token autoregressive decode flow."""
        hidden_states = self._require_decode_hidden_states(batch)
        assignment = self._decode_assignment(batch)
        if self._compiled.layout.decode_seq != 1 and max(assignment.per_rank_counts) > 1:
            raise ValueError(
                "DeepSeekV4 non-MTP decode supports at most one request per DP rank; "
                "the fixed S=2 kernel can expose only one cache-safe active token per rank"
            )
        builder = self._require_input_builder()
        actual_batch = len(batch.request_ids)
        positions = self._autoregressive_decode_positions(batch, actual_batch)
        embeddings = hidden_states.to(torch.float32).cpu()
        return self._prepare_decode_inputs(
            model,
            batch,
            assignment=assignment,
            active_seq=1,
            positions=positions,
            token_rows=self._autoregressive_decode_token_rows(batch.token_ids, actual_batch),
            x_hc=builder.decode_x_hc(
                embeddings,
                ranks=assignment.ranks,
                local_rows=assignment.local_rows,
            ),
        )

    def prepare_mtp_target_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        token_rows: torch.Tensor,
        positions: Sequence[Sequence[int]],
        active_width: int,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build one fixed-width target-verification chunk."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if token_rows.shape != (actual_batch, layout.decode_seq):
            raise ValueError(
                "MTP target token rows must have shape "
                f"{(actual_batch, layout.decode_seq)}, got {tuple(token_rows.shape)}"
            )
        if not 1 <= int(active_width) <= layout.decode_seq:
            raise ValueError("MTP target active width must fit the fixed decode sequence")
        position_rows = tuple(tuple(int(value) for value in row) for row in positions)
        if len(position_rows) != actual_batch or any(
            len(row) != layout.decode_seq for row in position_rows
        ):
            raise ValueError("MTP target positions must align with the fixed decode sequence")
        assignment = self._decode_assignment(batch)
        builder = self._require_input_builder()
        token_rows = token_rows.detach().cpu().to(torch.long)
        embeddings = self._embedding_rows(token_rows.reshape(-1), torch.float32).reshape(
            actual_batch,
            layout.decode_seq,
            -1,
        )
        prepared = self._prepare_decode_inputs(
            model,
            batch,
            assignment=assignment,
            active_seq=layout.decode_seq,
            positions=position_rows,
            token_rows=token_rows,
            x_hc=builder.decode_token_rows_x_hc(
                embeddings,
                ranks=assignment.ranks,
                local_rows=assignment.local_rows,
            ),
        )
        if active_width < layout.decode_seq:
            for rank, local_row in zip(
                prepared.ranks,
                prepared.local_rows,
                strict=True,
            ):
                padding_start = local_row * layout.decode_seq + active_width
                padding_end = (local_row + 1) * layout.decode_seq
                for name in (
                    "ori_slot_mapping",
                    "swa_slot_mapping",
                    "hca_cmp_slot_mapping",
                    "hca_state_slot_mapping",
                    "csa_cmp_slot_mapping",
                    "csa_idx_slot_mapping",
                    "csa_state_slot_mapping",
                    "csa_inner_state_slot_mapping",
                ):
                    getattr(prepared, name)[rank, padding_start:padding_end].fill_(-1)
        return prepared

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        assignment: _DeepSeekV4DecodeAssignment,
        active_seq: int,
        positions: tuple[tuple[int, ...], ...],
        token_rows: torch.Tensor,
        x_hc: torch.Tensor,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build mode-independent cache metadata around explicit token rows."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        ranks = assignment.ranks
        local_rows = assignment.local_rows
        per_rank_counts = assignment.per_rank_counts
        indices_by_rank = assignment.indices_by_rank
        active_group_ids = self._normalize_group_block_ids(
            batch.block_ids_by_group,
            actual_batch,
        )
        max_position = max(max(row) for row in positions)
        if max_position >= model.runtime.max_seq_len:
            raise ValueError(f"decode position {max_position} exceeds max_seq_len={model.runtime.max_seq_len}")

        field_rows: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "input_ids",
                "position_ids",
                "kv_seq_lens",
                "block_table",
                "ori_slot_mapping",
                "window_swa_indices",
                "window_swa_lens",
                "swa_slot_mapping",
                "swa_indices",
                "swa_lens",
                "cmp_block_table",
                "idx_block_table",
                "hca_compress_state_block_table",
                "csa_compress_state_block_table",
                "csa_inner_compress_state_block_table",
                "hca_cmp_slot_mapping",
                "hca_state_slot_mapping",
                "csa_cmp_slot_mapping",
                "csa_idx_slot_mapping",
                "csa_state_slot_mapping",
                "csa_inner_state_slot_mapping",
            )
        }
        group_limits = {
            "ori": layout.decode_ori_max_blocks,
            "cmp": layout.cmp_max_blocks,
            "idx": layout.idx_max_blocks,
            "hca_state": layout.hca_state_max_blocks,
            "csa_state": layout.csa_state_max_blocks,
            "csa_inner_state": layout.csa_inner_state_max_blocks,
        }

        for rank, request_indices in enumerate(indices_by_rank):
            if request_indices:
                local_positions = [positions[index] for index in request_indices]
                local_groups = [active_group_ids[index] for index in request_indices]
                local_token_rows = token_rows[list(request_indices)]
                local_seq_lens = batch.seq_lens[list(request_indices)]
                local_count = len(request_indices)
            else:
                # All ranks must enter the distributed program with the common
                # scalar num_tokens. This rank contributes filler rows whose
                # cache mappings cover otherwise-unowned scratch blocks.
                local_positions = [positions[0]]
                local_groups = []
                local_token_rows = token_rows[0:1]
                local_seq_lens = batch.seq_lens[0:1]
                local_count = 1

            padded_positions = list(local_positions)
            while len(padded_positions) < layout.decode_batch:
                padded_positions.append(local_positions[0])

            padded_group_ids = {}
            for name, max_blocks in group_limits.items():
                if local_groups:
                    padded_group_ids[name] = self._pad_group_block_ids(
                        [groups[name] for groups in local_groups],
                        max_blocks=max_blocks,
                        kernel_rows=layout.decode_batch,
                    )
                else:
                    padded_group_ids[name] = self._scratch_group_block_ids(
                        max_blocks=max_blocks,
                        kernel_rows=layout.decode_batch,
                    )

            field_rows["input_ids"].append(
                self._pad_decode_token_rows(
                    local_token_rows,
                    local_count,
                    vocab_size=model.config.vocab_size,
                )
            )
            field_rows["position_ids"].append(
                torch.tensor(padded_positions, dtype=torch.int32).reshape(-1)
            )
            field_rows["kv_seq_lens"].append(
                self._decode_kv_seq_lens(local_seq_lens, local_count)
            )
            field_rows["block_table"].append(
                self.cache_metadata.paged_ori_block_table_from_ids(padded_group_ids["ori"])
            )
            field_rows["cmp_block_table"].append(
                self.cache_metadata.block_table_from_ids(
                    padded_group_ids["cmp"], max_blocks=layout.cmp_max_blocks
                )
            )
            field_rows["idx_block_table"].append(
                self.cache_metadata.block_table_from_ids(
                    padded_group_ids["idx"], max_blocks=layout.idx_max_blocks
                )
            )
            field_rows["hca_compress_state_block_table"].append(
                self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["hca_state"],
                    max_blocks=layout.prefill_hca_state_max_blocks,
                )
            )
            field_rows["csa_compress_state_block_table"].append(
                self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["csa_state"],
                    max_blocks=layout.prefill_csa_state_max_blocks,
                )
            )
            field_rows["csa_inner_compress_state_block_table"].append(
                self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["csa_inner_state"],
                    max_blocks=layout.prefill_csa_inner_state_max_blocks,
                )
            )
            field_rows["ori_slot_mapping"].append(
                self.cache_metadata.sliding_window_slot_mapping_from_ids(
                    padded_group_ids["ori"], padded_positions
                ).reshape(-1)
            )
            field_rows["swa_slot_mapping"].append(
                self.cache_metadata.paged_decode_slot_mapping_from_ids(
                    padded_group_ids["ori"], padded_positions
                ).reshape(-1)
            )
            swa_indices, swa_lens = self.cache_metadata.swa_window_indices_and_lens_from_ids(
                padded_group_ids["ori"], padded_positions
            )
            field_rows["swa_indices"].append(swa_indices)
            field_rows["swa_lens"].append(swa_lens)
            # Current pypto-lib uses a cache-first contract for HCA/CSA: the
            # current decode rows are written before sparse attention.
            window_swa_indices, window_swa_lens = swa_indices, swa_lens
            field_rows["window_swa_indices"].append(window_swa_indices)
            field_rows["window_swa_lens"].append(window_swa_lens)
            field_rows["hca_cmp_slot_mapping"].append(
                self.cache_metadata.compressed_slot_mapping_from_ids(
                    padded_group_ids["cmp"],
                    padded_positions,
                    block_size=layout.block_size,
                    compress_ratio=128,
                ).reshape(-1)
            )
            field_rows["hca_state_slot_mapping"].append(
                self.cache_metadata.state_slot_mapping_from_ids(
                    padded_group_ids["hca_state"],
                    padded_positions,
                    state_block_size=layout.c128_state_block_size,
                ).reshape(-1)
            )
            field_rows["csa_cmp_slot_mapping"].append(
                self.cache_metadata.compressed_slot_mapping_from_ids(
                    padded_group_ids["cmp"],
                    padded_positions,
                    block_size=layout.block_size,
                    compress_ratio=4,
                ).reshape(-1)
            )
            field_rows["csa_idx_slot_mapping"].append(
                self.cache_metadata.compressed_slot_mapping_from_ids(
                    padded_group_ids["idx"],
                    padded_positions,
                    block_size=layout.block_size,
                    compress_ratio=4,
                ).reshape(-1)
            )
            field_rows["csa_state_slot_mapping"].append(
                self.cache_metadata.state_slot_mapping_from_ids(
                    padded_group_ids["csa_state"],
                    padded_positions,
                    state_block_size=layout.c4_state_block_size,
                ).reshape(-1)
            )
            field_rows["csa_inner_state_slot_mapping"].append(
                self.cache_metadata.state_slot_mapping_from_ids(
                    padded_group_ids["csa_inner_state"],
                    padded_positions,
                    state_block_size=layout.c4_state_block_size,
                ).reshape(-1)
            )

        num_tokens_per_owner = torch.tensor(
            tuple(count * active_seq for count in per_rank_counts),
            dtype=torch.int32,
        )
        logit_row_indices = torch.full(
            (layout.ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS),
            -1,
            dtype=torch.int32,
        )
        for rank, count in enumerate(per_rank_counts):
            row_count = count * active_seq
            if row_count > DEEPSEEK_V4_MAX_LOGIT_ROWS:
                raise ValueError(
                    f"rank {rank} requires {row_count} logit rows, "
                    f"capacity is {DEEPSEEK_V4_MAX_LOGIT_ROWS}"
                )
            if row_count:
                logit_row_indices[rank, :row_count] = torch.arange(
                    row_count,
                    dtype=torch.int32,
                )

        return DeepSeekV4PreparedDecodeInputs(
            request_ids=tuple(batch.request_ids),
            ranks=ranks,
            local_rows=tuple(local_rows),
            per_rank_counts=per_rank_counts,
            actual_batch=actual_batch,
            x_hc=x_hc,
            input_ids=torch.stack(field_rows["input_ids"]),
            position_ids=torch.stack(field_rows["position_ids"]),
            kv_seq_lens=torch.stack(field_rows["kv_seq_lens"]),
            block_table=torch.stack(field_rows["block_table"]),
            ori_slot_mapping=torch.stack(field_rows["ori_slot_mapping"]),
            window_swa_indices=torch.stack(field_rows["window_swa_indices"]),
            window_swa_lens=torch.stack(field_rows["window_swa_lens"]),
            swa_slot_mapping=torch.stack(field_rows["swa_slot_mapping"]),
            swa_indices=torch.stack(field_rows["swa_indices"]),
            swa_lens=torch.stack(field_rows["swa_lens"]),
            cmp_block_table=torch.stack(field_rows["cmp_block_table"]),
            idx_block_table=torch.stack(field_rows["idx_block_table"]),
            hca_compress_state_block_table=torch.stack(
                field_rows["hca_compress_state_block_table"]
            ),
            csa_compress_state_block_table=torch.stack(
                field_rows["csa_compress_state_block_table"]
            ),
            csa_inner_compress_state_block_table=torch.stack(
                field_rows["csa_inner_compress_state_block_table"]
            ),
            hca_cmp_slot_mapping=torch.stack(field_rows["hca_cmp_slot_mapping"]),
            hca_state_slot_mapping=torch.stack(field_rows["hca_state_slot_mapping"]),
            csa_cmp_slot_mapping=torch.stack(field_rows["csa_cmp_slot_mapping"]),
            csa_idx_slot_mapping=torch.stack(field_rows["csa_idx_slot_mapping"]),
            csa_state_slot_mapping=torch.stack(field_rows["csa_state_slot_mapping"]),
            csa_inner_state_slot_mapping=torch.stack(
                field_rows["csa_inner_state_slot_mapping"]
            ),
            block_ids_by_group=active_group_ids,
            num_tokens_per_owner=num_tokens_per_owner,
            logit_row_indices=logit_row_indices,
        )

    @staticmethod
    def _normalize_group_block_ids(
        rows: Sequence[dict[str, list[int]]],
        actual_batch: int,
    ) -> tuple[dict[str, tuple[int, ...]], ...]:
        """Validate and normalize grouped scheduler metadata for active rows."""
        if not rows:
            raise ValueError("DeepSeekV4 requires grouped KV block IDs")
        if len(rows) != actual_batch:
            raise ValueError(
                f"grouped KV metadata has {len(rows)} rows, expected decode batch {actual_batch}"
            )
        required = ("ori", "cmp", "idx", "hca_state", "csa_state", "csa_inner_state")
        normalized = []
        for row_index, row in enumerate(rows):
            missing = [name for name in required if not row.get(name)]
            if missing:
                raise ValueError(
                    f"decode row {row_index} is missing grouped KV blocks: {', '.join(missing)}"
                )
            normalized.append(
                {name: tuple(int(block_id) for block_id in row[name]) for name in required}
            )
        return tuple(normalized)

    @staticmethod
    def _pad_group_block_ids(
        active_rows: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        kernel_rows: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Pad inactive kernel rows without expanding pypto-lib's fixed pools."""
        if not active_rows or len(active_rows) > kernel_rows:
            raise ValueError("active grouped KV rows must fit the kernel batch")
        normalized = [tuple(int(block_id) for block_id in row) for row in active_rows]
        if any(not row for row in normalized):
            raise ValueError("active grouped KV rows must not be empty")
        used = [block_id for row in normalized for block_id in row]
        if len(used) != len(set(used)):
            raise ValueError("active grouped KV rows must not share physical blocks")
        scratch = DeepSeekV4ModelRunner._scratch_group_block_ids(
            max_blocks=max_blocks,
            kernel_rows=kernel_rows,
        )
        allocator_blocks = max_blocks - kernel_rows
        if any(block_id < 0 or block_id >= allocator_blocks for block_id in used):
            raise ValueError(
                f"grouped KV block IDs must be in [0, {allocator_blocks}); "
                f"[{allocator_blocks}, {max_blocks}) is reserved for kernel padding"
            )
        padded = list(normalized)
        # Attention and compressor cache writes are fixed-B and are not fully
        # gated by num_tokens. Give every inactive row a distinct reserved page
        # instead of mirroring a live request's metadata.
        padded.extend(scratch[: kernel_rows - len(normalized)])
        return tuple(padded)

    @staticmethod
    def _scratch_group_block_ids(
        *,
        max_blocks: int,
        kernel_rows: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return one isolated physical scratch page for every fixed kernel row."""
        if kernel_rows <= 0 or max_blocks <= kernel_rows:
            raise ValueError("cache pool must provide one scratch page per kernel row")
        first = max_blocks - kernel_rows
        return tuple((first + row,) for row in range(kernel_rows))

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        raise NotImplementedError("DeepSeekV4 uses model-specific cache pools, not generic KV tensors")

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        return None

    def run_prefill(self, model, batch: PrefillBatch) -> PrefillResult:
        """Run all DeepSeekV4 hidden layers for one prefill chunk in a single packed call."""
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        with profile_span("DeepSeekV4ModelRunner.prefill.prepare", cat="executor"):
            with profile_span("DeepSeekV4ModelRunner.prefill.ensure_l3_shared_buffers", cat="executor"):
                self._ensure_l3_shared_buffers(model)
            with profile_span("DeepSeekV4ModelRunner.prefill.prepare_inputs", cat="executor"):
                inputs = self.prepare_prefill_inputs(model, batch)
        group_rows = self._normalize_group_block_ids(
            batch.block_ids_by_group,
            actual_batch=len(inputs.request_ids),
        )
        self._initialize_decode_cache_blocks(
            inputs.request_ids,
            inputs.ranks,
            group_rows,
        )
        with profile_span(
            "DeepSeekV4ModelRunner.prefill.prepare_fwd_args",
            cat="executor",
            args={"actual_tokens": max(inputs.actual_tokens)},
        ):
            self._stage_prefill_fwd_inputs(inputs)
            hidden_buffer = self._require_prefill_output_buffer(model.config.hidden_size)
            pre_hc_hidden_buffer = self._require_prefill_pre_hc_output_buffer(model.config.hidden_size)
            logits_buffer = self._require_prefill_logits_buffer(model.config.vocab_size)
            hidden_buffer.zero_()
            pre_hc_hidden_buffer.zero_()
            logits_buffer.zero_()
            args = self._prefill_fwd_args(pre_hc_hidden_buffer, hidden_buffer, logits_buffer)
        self._debug_prefill_dispatch(inputs, args)
        try:
            with profile_span(
                "DeepSeekV4ModelRunner.prefill.l3_dispatch",
                cat="executor",
                args={"actual_tokens": max(inputs.actual_tokens)},
            ):
                self._run_l3(
                    self._require_prefill_callable(),
                    *args,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed prefill dispatch failed "
                f"(tokens={inputs.actual_tokens}, ranks={inputs.ranks})"
            ) from exc
        self._prefill_completion(inputs, pre_hc_hidden_buffer)

        active_hidden = hidden_buffer[:, : max(inputs.actual_tokens), :]
        self._debug_tensor_stats("prefill.output.hidden.active", active_hidden, per_rank=True)
        if self._debug_tensor_stats_enabled() and not self._tensor_is_finite(active_hidden):
            raise RuntimeError("DeepSeekV4 packed prefill produced non-finite active hidden rows")

        logits = torch.stack(
            tuple(logits_buffer[rank, 0] for rank in inputs.ranks),
        ).float()
        return PrefillResult(last_hidden=None, logits=logits)

    def run_decode(self, model, batch: DecodeBatch) -> DecodeResult:
        """Dispatch to the decode flow selected when the model was compiled."""
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        with profile_span("DeepSeekV4ModelRunner.decode.prepare", cat="executor"):
            self._ensure_l3_shared_buffers(model)
        return self._decode_flow(model, batch)

    def _run_autoregressive_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the single-token autoregressive decode flow."""
        output = self._execute_main_decode(
            model,
            self.prepare_decode_inputs(model, batch),
            active_seq=1,
        )
        logits = torch.stack(
            tuple(
                output.logits[rank, local_row]
                for rank, local_row in zip(
                    output.inputs.ranks,
                    output.inputs.local_rows,
                    strict=True,
                )
            )
        ).float()
        return DecodeResult(hidden_states=None, logits=logits)

    def _verify_mtp_drafts(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        drafts: torch.Tensor,
    ) -> _DeepSeekV4MtpVerification:
        """Verify arbitrary draft depth with fixed-width target chunks."""
        layout = self._compiled.layout
        width = layout.decode_seq
        actual_batch = len(batch.request_ids)
        drafts = drafts.detach().cpu().to(torch.long)
        if drafts.ndim != 2 or drafts.shape[0] != actual_batch:
            raise ValueError(
                f"MTP drafts must have shape [batch, K], got {tuple(drafts.shape)}"
            )

        current = batch.token_ids[:actual_batch].detach().cpu().to(torch.long).reshape(actual_batch, -1)[
            :, 0
        ]
        input_sequences = torch.cat((current.reshape(-1, 1), drafts), dim=1)
        base_seq_lens = batch.seq_lens[:actual_batch].detach().cpu().to(torch.int32)
        accepted: list[list[int]] = [[] for _ in range(actual_batch)]
        first_logits: list[torch.Tensor | None] = [None] * actual_batch
        tail_tokens: list[int | None] = [None] * actual_batch
        tail_hidden: list[torch.Tensor | None] = [None] * actual_batch
        tail_positions: list[int | None] = [None] * actual_batch
        pending: dict[int, tuple[int, torch.Tensor, int]] = {}
        active = list(range(actual_batch))
        input_offset = 0

        while active:
            if input_offset:
                continuing = []
                for request_index in active:
                    predicted, previous_hidden, position = pending.pop(request_index)
                    accepted[request_index].append(predicted)
                    if predicted == int(drafts[request_index, input_offset - 1]):
                        continuing.append(request_index)
                    else:
                        tail_tokens[request_index] = predicted
                        tail_hidden[request_index] = previous_hidden
                        tail_positions[request_index] = position
                active = continuing
                if not active:
                    break

            real_width = min(width, input_sequences.shape[1] - input_offset)
            active_rows = input_sequences[active, input_offset : input_offset + real_width]
            if real_width < width:
                active_rows = torch.cat(
                    (
                        active_rows,
                        active_rows[:, -1:].expand(-1, width - real_width),
                    ),
                    dim=1,
                )
            position_rows = []
            for request_index in active:
                first_position = int(base_seq_lens[request_index]) - 1 + input_offset
                real_positions = tuple(first_position + offset for offset in range(real_width))
                position_rows.append(
                    real_positions + (real_positions[-1],) * (width - real_width)
                )

            chunk_batch = self._select_decode_batch_rows(batch, active)
            chunk_batch.seq_lens = torch.tensor(
                [
                    int(base_seq_lens[request_index]) + input_offset + real_width - 1
                    for request_index in active
                ],
                dtype=torch.int32,
            )
            output = self._execute_main_decode(
                model,
                self.prepare_mtp_target_inputs(
                    model,
                    chunk_batch,
                    token_rows=active_rows,
                    positions=position_rows,
                    active_width=real_width,
                ),
                active_seq=width,
            )

            continuing = []
            for chunk_index, request_index in enumerate(active):
                rank = output.inputs.ranks[chunk_index]
                row_start = output.inputs.local_rows[chunk_index] * width
                row_logits = output.logits[rank, row_start : row_start + real_width].float()
                row_predictions = row_logits.argmax(dim=-1)
                if first_logits[request_index] is None:
                    first_logits[request_index] = row_logits[0].clone()

                rejected = False
                for offset in range(real_width - 1):
                    predicted = int(row_predictions[offset])
                    accepted[request_index].append(predicted)
                    draft_index = input_offset + offset
                    if predicted != int(drafts[request_index, draft_index]):
                        tail_tokens[request_index] = predicted
                        tail_hidden[request_index] = (
                            output.pre_hc_hidden[rank, row_start + offset].detach().cpu().clone()
                        )
                        tail_positions[request_index] = position_rows[chunk_index][offset] + 1
                        rejected = True
                        break
                if rejected:
                    continue

                last_offset = real_width - 1
                predicted = int(row_predictions[last_offset])
                previous_hidden = (
                    output.pre_hc_hidden[rank, row_start + last_offset].detach().cpu().clone()
                )
                position = position_rows[chunk_index][last_offset] + 1
                if input_offset + last_offset == drafts.shape[1]:
                    accepted[request_index].append(predicted)
                    tail_tokens[request_index] = predicted
                    tail_hidden[request_index] = previous_hidden
                    tail_positions[request_index] = position
                else:
                    pending[request_index] = (predicted, previous_hidden, position)
                    continuing.append(request_index)

            active = continuing
            input_offset += real_width

        if (
            any(value is None for value in first_logits)
            or any(value is None for value in tail_tokens)
            or any(value is None for value in tail_hidden)
            or any(value is None for value in tail_positions)
        ):
            raise RuntimeError("DeepSeekV4 MTP target verification left incomplete request state")
        return _DeepSeekV4MtpVerification(
            accepted_token_ids=accepted,
            tail_token_ids=torch.tensor(tail_tokens, dtype=torch.long),
            tail_pre_hc_hidden=torch.stack(tail_hidden),
            tail_positions=torch.tensor(tail_positions, dtype=torch.int32),
            first_logits=torch.stack(first_logits).float(),
        )

    def _run_mtp_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Generate and verify the configured number of request-local MTP drafts."""
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError("DeepSeekV4 MTP decode currently requires greedy device sampling")
        num_drafts = self._mtp_draft_count(model, batch)
        drafts = self._propose_mtp_tokens(batch, num_drafts=num_drafts)
        verification = self._verify_mtp_drafts(model, batch, drafts)
        accepted = verification.accepted_token_ids
        proposed = len(batch.request_ids) * drafts.shape[1]
        accepted_drafts = sum(len(tokens) - 1 for tokens in accepted)
        self._mtp_proposed_tokens += proposed
        self._mtp_accepted_tokens += accepted_drafts
        for request_id, tokens in zip(batch.request_ids, accepted, strict=True):
            state = self._require_mtp_request_state(request_id)
            state.proposed_tokens += drafts.shape[1]
            state.accepted_tokens += len(tokens) - 1
        logger.info(
            "DeepSeekV4 MTP acceptance progress: accepted=%d proposed=%d rate=%.2f%%",
            self._mtp_accepted_tokens,
            self._mtp_proposed_tokens,
            (
                100.0 * self._mtp_accepted_tokens / self._mtp_proposed_tokens
                if self._mtp_proposed_tokens
                else 0.0
            ),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DeepSeekV4 MTP step: draft=%s main=%s",
                drafts.tolist(),
                accepted,
            )

        with profile_span(
            "DeepSeekV4ModelRunner.decode.mtp_advance",
            cat="executor",
            args={"accepted_counts": tuple(len(tokens) for tokens in accepted)},
        ):
            next_drafts, next_hidden = self._run_mtp_token_step(
                batch,
                verification.tail_token_ids,
                verification.tail_pre_hc_hidden,
                verification.tail_positions,
            )
            for index, request_id in enumerate(batch.request_ids):
                state = self._require_mtp_request_state(request_id)
                state.draft_token_id = int(next_drafts[index])
                state.draft_pre_hc_hidden = next_hidden[index].clone()
                state.draft_position = int(verification.tail_positions[index]) + 1
        return DecodeResult(
            hidden_states=None,
            logits=verification.first_logits,
            accepted_token_ids=accepted,
        )

    def _mtp_draft_count(self, model: RuntimeModel, batch: DecodeBatch) -> int:
        """Cap this step's draft depth at the shortest remaining context."""
        remaining_context = min(
            model.runtime.max_seq_len - int(seq_len)
            for seq_len in batch.seq_lens[: len(batch.request_ids)]
        )
        return min(self._compiled.num_speculative_tokens, max(0, remaining_context))

    def _execute_main_decode(
        self,
        model: RuntimeModel,
        prepared: DeepSeekV4PreparedDecodeInputs,
        *,
        active_seq: int,
    ) -> _DeepSeekV4MainDecodeOutput:
        """Run the mode-independent packed main-model decode kernel."""
        with profile_span("DeepSeekV4ModelRunner.decode.prepare_inputs", cat="executor"):
            inputs = self._stage_decode_inputs(prepared)
        self._seed_decode_work_cache_from_group_ids(
            inputs.request_ids,
            inputs.ranks,
            inputs.block_ids_by_group,
        )
        decode_buffers = self._require_decode_buffers()
        x_hc = decode_buffers.x_hc_a
        active_decode_tokens = max(inputs.per_rank_counts) * active_seq
        self._debug_tensor_stats("decode.input.initial.active", x_hc[:, :active_decode_tokens, :, :])

        hidden_buffer = self._require_decode_output_buffer(model.config.hidden_size)
        pre_hc_hidden_buffer = decode_buffers.pre_hc_hidden_out
        logits_buffer = self._require_decode_logits_buffer(model.config.vocab_size)
        hidden_buffer.zero_()
        pre_hc_hidden_buffer.zero_()
        num_tokens = active_decode_tokens
        with profile_span(
            "DeepSeekV4ModelRunner.decode.prepare_fwd_args",
            cat="executor",
            args={"actual_tokens": num_tokens},
        ):
            logits_buffer.zero_()
            args = self._decode_fwd_args(
                inputs,
                x_hc,
                pre_hc_hidden_buffer,
                hidden_buffer,
                logits_buffer,
            )
        self._debug_decode_dispatch(inputs, args)
        try:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.l3_dispatch",
                cat="executor",
                args={"actual_tokens": num_tokens},
            ):
                self._run_l3(
                    self._require_decode_callable(),
                    *args,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed decode dispatch failed "
                f"(actual_batch={inputs.actual_batch}, ranks={inputs.ranks})"
            ) from exc
        active_hidden = hidden_buffer[:, :active_decode_tokens, :]
        self._debug_tensor_stats("decode.output.hidden.active", active_hidden, per_rank=True)
        if self._debug_tensor_stats_enabled() and not self._tensor_is_finite(active_hidden):
            raise RuntimeError("DeepSeekV4 packed decode produced non-finite active hidden rows")

        return _DeepSeekV4MainDecodeOutput(
            inputs=inputs,
            hidden=hidden_buffer,
            pre_hc_hidden=pre_hc_hidden_buffer,
            logits=logits_buffer,
        )

    @staticmethod
    def _ignore_prefill_context(
        inputs: DeepSeekV4PreparedPrefillInputs,
        pre_hc_hidden: torch.Tensor,
    ) -> None:
        """Ignore main-prefill intermediates in autoregressive mode."""
        return None

    def _capture_mtp_prefill_context(
        self,
        inputs: DeepSeekV4PreparedPrefillInputs,
        pre_hc_hidden: torch.Tensor,
    ) -> None:
        """Retain request-local main-prefill inputs needed to seed MTP."""
        for request_id, rank, actual_tokens in zip(
            inputs.request_ids,
            inputs.ranks,
            inputs.actual_tokens,
            strict=True,
        ):
            self._mtp_request_states[request_id] = _DeepSeekV4MtpRequestState(
                prefill_context=_DeepSeekV4MtpPrefillContext(
                    rank=rank,
                    actual_tokens=actual_tokens,
                    hidden_states=inputs.x_hc[rank, :, 0].detach().cpu().clone(),
                    prev_hidden_states=pre_hc_hidden[rank].detach().cpu().clone(),
                    input_ids=inputs.input_ids[rank].detach().cpu().clone(),
                    position_ids=inputs.position_ids[rank].detach().cpu().clone(),
                    block_table=inputs.ori_block_table[rank].detach().cpu().clone(),
                    slot_mapping=inputs.ori_slot_mapping[rank].detach().cpu().clone(),
                )
            )

    def _require_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 prefill kernel is not compiled")
        return self._compiled.prefill

    def _require_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 decode kernel is not compiled")
        return self._compiled.decode

    def _ensure_l3_shared_buffers(self, model: RuntimeModel) -> None:
        """Allocate every CPU tensor visible to the L3 worker before it forks.

        ``DistributedWorker`` creates per-chip children on first use. Mutable CPU
        arguments must already live in shared memory at that point; immutable
        weights are registered for fork inheritance. This method prepares both
        groups before the first ``_run_l3`` call.
        """
        with profile_span("DeepSeekV4ModelRunner.prepare.load_global_weights", cat="executor"):
            self.load_packed_global_weights()
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_rope_tables", cat="executor"):
            self._static_freqs_cos_tensor()
            self._static_freqs_sin_tensor()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_decode_buffers", cat="executor"):
            self._ensure_decode_buffers(model.config.hidden_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_mtp_buffers", cat="executor"):
            self._ensure_mtp_buffers(model.config.hidden_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_decode_work_cache", cat="executor"):
            self._ensure_decode_work_cache()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_prefill_outputs", cat="executor"):
            self._require_prefill_output_buffer(model.config.hidden_size)
            self._require_prefill_pre_hc_output_buffer(model.config.hidden_size)
            self._require_prefill_logits_buffer(model.config.vocab_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_final_norm", cat="executor"):
            self._static_final_norm_weight_tensor()
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_lm_head", cat="executor"):
            self._static_lm_head_weight_tensor()
            self._require_decode_logits_buffer(model.config.vocab_size)
        if self._stacked_host_weights is None:
            if self._stacked_device_weights is None:
                with profile_span(
                    "DeepSeekV4ModelRunner.prepare.load_and_pack_layer_weights",
                    cat="executor",
                ):
                    stacked_weights = self.load_stacked_layer_weights()
                with profile_span("DeepSeekV4ModelRunner.prepare.retain_layer_weights", cat="executor"):
                    self._retain_stacked_host_weights(stacked_weights)
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_hc_head", cat="executor"):
            self._hc_head_tensors()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_prefill_fwd_buffers", cat="executor"):
            self._ensure_prefill_fwd_buffers(model.config.hidden_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.validate_shared_buffers", cat="executor"):
            self._assert_l3_shared_buffers_preallocated()
        with profile_span("DeepSeekV4ModelRunner.upload_resident_weights", cat="executor"):
            self._materialize_resident_weights()

    def _assert_l3_shared_buffers_preallocated(self) -> None:
        missing = self._missing_l3_shared_buffers()
        if missing:
            raise RuntimeError(
                "DeepSeekV4 L3 worker cannot start before all shared host buffers are preallocated; "
                "missing: " + ", ".join(missing)
            )

    def _missing_l3_shared_buffers(self) -> list[str]:
        missing: list[str] = []
        expected = {
            "final_norm_w": self._static_final_norm_weight,
            "lm_head_weight": self._static_lm_head_weight,
            "freqs_cos": self._static_freqs_cos,
            "freqs_sin": self._static_freqs_sin,
            "prefill_fwd_buffers": self._prefill_fwd_buffers,
            "decode_buffers": self._decode_buffers,
            "decode_work_cache": self._decode_work_cache,
            "stacked_weights": self._stacked_host_weights or self._stacked_device_weights,
            "hc_head_buffers": self._hc_head_buffers,
            "prefill_output": self._prefill_output_buffer,
            "prefill_pre_hc_output": self._prefill_pre_hc_output_buffer,
            "prefill_logits": self._prefill_logits_buffer,
            "decode_logits": self._decode_logits_buffer,
        }
        if self._compiled.mtp_prefill is not None or self._compiled.mtp_decode is not None:
            expected["mtp_buffers"] = self._mtp_buffers
        for name, value in expected.items():
            if value is None:
                missing.append(name)
        if self._stacked_host_weights is not None and not self._stacked_host_weights:
            missing.append("stacked_weights")
        if self._hc_head_buffers is not None and not self._hc_head_buffers:
            missing.append("hc_head_buffers")
        return missing

    def _prefill_fwd_args(
        self,
        pre_hc_hidden_out: torch.Tensor,
        hidden_out: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Build the single packed ``l3_prefill_fwd`` argument tuple.

        The kernel runs final RMSNorm and the device-side LM-head.
        """
        buffers = self._require_prefill_fwd_buffers()
        stacked = self._require_stacked_weights()
        hc_head = self._hc_head_tensors()
        values = dict(stacked.tensors)
        values.update(
            {
                "x_hc": buffers.x_hc,
                "freqs_cos": buffers.freqs_cos,
                "freqs_sin": buffers.freqs_sin,
                "hc_head_fn": hc_head["hc_head_fn"],
                "hc_head_scale": hc_head["hc_head_scale"],
                "hc_head_base": hc_head["hc_head_base"],
                "final_norm_w": self._static_final_norm_weight_tensor(),
                "pre_hc_hidden_out": pre_hc_hidden_out,
                "lm_head_weight": self._static_lm_head_weight_tensor(),
                "hidden_out": hidden_out,
                "logits": logits,
                "num_tokens_per_owner": buffers.tensors["num_tokens_per_owner"],
                "logit_row_indices": buffers.tensors["logit_row_indices"],
            }
        )
        values.update(buffers.tensors)
        values.update(self._device_cache_values())
        values = self._mark_resident_args(values, _PREFILL_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _PREFILL_FWD_TENSOR_ORDER)

    def _decode_fwd_args(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        x_hc: torch.Tensor,
        pre_hc_hidden_out: torch.Tensor,
        hidden_out: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Build the single packed ``l3_decode_fwd`` argument tuple."""
        cache = self._materialize_decode_device_cache()
        stacked = self._require_stacked_weights()
        hc_head = self._hc_head_tensors()
        values = dict(stacked.tensors)
        values.update(
            {
                "x_hc": x_hc,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": cache.kv_cache,
                "block_table": inputs.block_table,
                "ori_slot_mapping": inputs.ori_slot_mapping,
                "window_swa_indices": inputs.window_swa_indices,
                "window_swa_lens": inputs.window_swa_lens,
                "swa_slot_mapping": inputs.swa_slot_mapping,
                "swa_indices": inputs.swa_indices,
                "swa_lens": inputs.swa_lens,
                "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
                "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
                "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
                "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
                "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
                "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
                "position_ids": inputs.position_ids,
                "kv_seq_lens": inputs.kv_seq_lens,
                "hca_compress_state": cache.hca_compress_state,
                "hca_compress_state_block_table": inputs.hca_compress_state_block_table,
                "csa_compress_state": cache.csa_compress_state,
                "csa_compress_state_block_table": inputs.csa_compress_state_block_table,
                "csa_inner_compress_state": cache.csa_inner_compress_state,
                "csa_inner_compress_state_block_table": inputs.csa_inner_compress_state_block_table,
                "cmp_kv": cache.cmp_kv,
                "cmp_block_table": inputs.cmp_block_table,
                "idx_kv_cache": cache.idx_kv_cache,
                "idx_kv_scale": cache.idx_kv_scale,
                "idx_block_table": inputs.idx_block_table,
                "input_ids": inputs.input_ids,
                "hc_head_fn": hc_head["hc_head_fn"],
                "hc_head_scale": hc_head["hc_head_scale"],
                "hc_head_base": hc_head["hc_head_base"],
                "final_norm_w": self._static_final_norm_weight_tensor(),
                "pre_hc_hidden_out": pre_hc_hidden_out,
                "lm_head_weight": self._static_lm_head_weight_tensor(),
                "hidden_out": hidden_out,
                "logits": logits,
                "num_tokens_per_owner": inputs.num_tokens_per_owner,
                "logit_row_indices": inputs.logit_row_indices,
            }
        )
        values = self._mark_resident_args(values, _DECODE_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _DECODE_FWD_TENSOR_ORDER)

    def _device_cache_values(self) -> dict[str, StackedDeviceTensor]:
        """Return the unified worker-resident cache pools by kernel argument name."""
        cache = self._materialize_decode_device_cache()
        return {
            "kv_cache": cache.kv_cache,
            "cmp_kv": cache.cmp_kv,
            "idx_kv_cache": cache.idx_kv_cache,
            "idx_kv_scale": cache.idx_kv_scale,
            "hca_compress_state": cache.hca_compress_state,
            "csa_compress_state": cache.csa_compress_state,
            "csa_inner_compress_state": cache.csa_inner_compress_state,
        }

    def _mtp_prefill_args(self) -> tuple[Any, ...]:
        buffers = self._require_mtp_buffers()
        kv_cache = self._materialize_mtp_device_kv_cache()
        if kv_cache is None:
            raise RuntimeError("DeepSeekV4 MTP KV cache is unavailable")
        values = dict(self._mtp_device_weights or buffers.weights)
        values.update(
            {
                "hidden_states": buffers.prefill_hidden_in,
                "prev_hidden_states": buffers.prefill_prev_hidden_in,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": kv_cache,
                "ori_block_table": buffers.prefill_block_table,
                "ori_slot_mapping": buffers.prefill_slot_mapping,
                "position_ids": buffers.prefill_position_ids,
                "input_ids": buffers.prefill_input_ids,
                "lm_head_weight": self._static_lm_head_weight_tensor(),
                "hidden_out": buffers.prefill_hidden_out,
                "pre_hc_hidden_out": buffers.prefill_pre_hc_out,
                "logits": buffers.prefill_logits,
                "logit_row_indices": buffers.prefill_logit_row_indices,
            }
        )
        values = self._mark_resident_args(values, _MTP_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _MTP_PREFILL_TENSOR_ORDER)

    def _mtp_decode_args(self) -> tuple[Any, ...]:
        buffers = self._require_mtp_buffers()
        kv_cache = self._materialize_mtp_device_kv_cache()
        if kv_cache is None:
            raise RuntimeError("DeepSeekV4 MTP KV cache is unavailable")
        values = dict(self._mtp_device_weights or buffers.weights)
        values.update(
            {
                "hidden_states": buffers.decode_hidden_in,
                "prev_pre_hc_hidden": buffers.decode_prev_hidden_in,
                "position_ids": buffers.decode_position_ids,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": kv_cache,
                "swa_slot_mapping": buffers.decode_slot_mapping,
                "swa_indices": buffers.decode_swa_indices,
                "swa_lens": buffers.decode_swa_lens,
                "input_ids": buffers.decode_input_ids,
                "lm_head_weight": self._static_lm_head_weight_tensor(),
                "hidden_out": buffers.decode_hidden_out,
                "next_pre_hc_hidden": buffers.decode_pre_hc_out,
                "logits": buffers.decode_logits,
                "logit_row_indices": buffers.decode_logit_row_indices,
            }
        )
        values = self._mark_resident_args(values, _MTP_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _MTP_DECODE_TENSOR_ORDER)

    def _require_mtp_buffers(self) -> _DeepSeekV4MtpSharedBuffers:
        if self._mtp_buffers is None:
            raise RuntimeError("DeepSeekV4 MTP shared buffers are not staged")
        return self._mtp_buffers

    def _require_mtp_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.mtp_prefill is None:
            raise RuntimeError("DeepSeekV4 MTP prefill kernel is not compiled")
        return self._compiled.mtp_prefill

    def _require_mtp_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.mtp_decode is None:
            raise RuntimeError("DeepSeekV4 MTP decode kernel is not compiled")
        return self._compiled.mtp_decode

    def _embedding_rows(self, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        embed = self._compiled.embedding_weight
        if embed is None:
            embed = self._compiled.weight_store.load_tensor("embed.weight").contiguous().cpu()
            self._compiled.embedding_weight = embed
        return embed.index_select(0, token_ids.detach().cpu().to(torch.long).reshape(-1)).to(dtype)

    def _require_mtp_request_state(self, request_id: str) -> _DeepSeekV4MtpRequestState:
        state = self._mtp_request_states.get(request_id)
        if state is None:
            raise RuntimeError(f"DeepSeekV4 MTP state is missing for request {request_id!r}")
        return state

    def _mtp_drafts_for_requests(self, request_ids: Sequence[str]) -> torch.Tensor:
        draft_ids = []
        for request_id in request_ids:
            draft_token_id = self._require_mtp_request_state(request_id).draft_token_id
            if draft_token_id is None:
                raise RuntimeError(f"DeepSeekV4 MTP draft is not initialized for {request_id!r}")
            draft_ids.append(draft_token_id)
        return torch.tensor(draft_ids, dtype=torch.long)

    @staticmethod
    def _select_decode_batch_rows(batch: DecodeBatch, indices: Sequence[int]) -> DecodeBatch:
        """Return a request-aligned subset without changing cache ownership."""
        rows = tuple(int(index) for index in indices)

        def select_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            index = torch.tensor(rows, dtype=torch.long, device=value.device)
            return value.index_select(0, index)

        def select_list(values):
            return [values[index] for index in rows] if values else []

        return replace(
            batch,
            request_ids=select_list(batch.request_ids),
            token_ids=select_tensor(batch.token_ids),
            hidden_states=select_tensor(batch.hidden_states),
            seq_lens=select_tensor(batch.seq_lens),
            kv_allocations=select_list(batch.kv_allocations),
            block_ids=select_list(batch.block_ids),
            block_ids_by_group=select_list(batch.block_ids_by_group),
            cache_partitions=select_list(batch.cache_partitions),
        )

    def _propose_mtp_tokens(
        self,
        batch: DecodeBatch,
        *,
        num_drafts: int | None = None,
    ) -> torch.Tensor:
        """Generate the configured number of drafts by recurrently reusing the MTP layer."""
        if num_drafts is None:
            num_drafts = self._compiled.num_speculative_tokens
        num_drafts = int(num_drafts)
        if not 0 <= num_drafts <= self._compiled.num_speculative_tokens:
            raise ValueError("MTP draft count must fit the compiled speculative depth")
        if num_drafts == 0:
            return torch.empty((len(batch.request_ids), 0), dtype=torch.long)
        self._initialize_mtp_drafts(batch)
        draft_columns = [self._mtp_drafts_for_requests(batch.request_ids)]
        if num_drafts == 1:
            return draft_columns[0].reshape(-1, 1)

        recurrent_hidden = []
        positions = []
        for request_id in batch.request_ids:
            state = self._require_mtp_request_state(request_id)
            if state.draft_pre_hc_hidden is None or state.draft_position is None:
                raise RuntimeError(f"DeepSeekV4 recurrent MTP state is missing for {request_id!r}")
            recurrent_hidden.append(state.draft_pre_hc_hidden)
            positions.append(state.draft_position)
        hidden = torch.stack(recurrent_hidden)
        position_ids = torch.tensor(positions, dtype=torch.int32)

        while len(draft_columns) < num_drafts:
            next_tokens, hidden = self._run_mtp_token_step(
                batch,
                draft_columns[-1],
                hidden,
                position_ids,
            )
            draft_columns.append(next_tokens)
            position_ids = position_ids + 1
        return torch.stack(draft_columns, dim=1)

    def _run_mtp_token_step(
        self,
        batch: DecodeBatch,
        token_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one recurrent MTP token per request inside the fixed decode tile."""
        actual_batch = len(batch.request_ids)
        if token_ids.numel() != actual_batch or positions.numel() != actual_batch:
            raise ValueError("MTP token IDs and positions must align with active requests")
        if previous_hidden.shape[0] != actual_batch:
            raise ValueError("MTP recurrent hidden rows must align with active requests")

        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        assignment = self._decode_assignment(batch)
        token_ids = token_ids.detach().cpu().to(torch.long).reshape(actual_batch)
        positions = positions.detach().cpu().to(torch.int32).reshape(actual_batch)
        previous_hidden = previous_hidden.detach().cpu()
        active_group_ids = self._normalize_group_block_ids(
            batch.block_ids_by_group,
            actual_batch,
        )

        fallback_token = token_ids[0]
        fallback_position = positions[0]
        fallback_hidden = previous_hidden[0]
        kernel_ids = fallback_token.repeat(layout.ranks, layout.decode_tokens)
        kernel_positions = fallback_position.repeat(layout.ranks, layout.decode_tokens)
        kernel_previous_hidden = fallback_hidden.repeat(
            layout.ranks,
            layout.decode_tokens,
            1,
            1,
        )
        slot_mappings = []
        swa_indices_by_rank = []
        swa_lens_by_rank = []
        for rank, request_indices in enumerate(assignment.indices_by_rank):
            for request_index in request_indices:
                row_start = assignment.local_rows[request_index] * layout.decode_seq
                row_slice = slice(row_start, row_start + layout.decode_seq)
                kernel_ids[rank, row_slice].fill_(int(token_ids[request_index]))
                kernel_positions[rank, row_slice].fill_(int(positions[request_index]))
                kernel_previous_hidden[rank, row_slice].copy_(
                    previous_hidden[request_index].expand(layout.decode_seq, -1, -1)
                )

            if request_indices:
                padded_blocks = self._pad_group_block_ids(
                    [active_group_ids[index]["ori"] for index in request_indices],
                    max_blocks=layout.prefill_ori_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
                padded_positions = [
                    (int(positions[index]),) * layout.decode_seq for index in request_indices
                ]
                while len(padded_positions) < layout.decode_batch:
                    padded_positions.append((int(fallback_position),) * layout.decode_seq)
            else:
                padded_blocks = self._scratch_group_block_ids(
                    max_blocks=layout.prefill_ori_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
                padded_positions = [
                    (int(fallback_position),) * layout.decode_seq
                    for _ in range(layout.decode_batch)
                ]
            slot_mappings.append(
                self.cache_metadata.paged_decode_slot_mapping_from_ids(
                    padded_blocks,
                    padded_positions,
                ).reshape(-1)
            )
            rank_swa_indices, rank_swa_lens = (
                self.cache_metadata.swa_window_indices_and_lens_from_ids(
                    padded_blocks,
                    padded_positions,
                )
            )
            swa_indices_by_rank.append(rank_swa_indices)
            swa_lens_by_rank.append(rank_swa_lens)

        buffers.decode_input_ids.copy_(kernel_ids)
        buffers.decode_hidden_in.copy_(
            self._embedding_rows(kernel_ids.reshape(-1), torch.bfloat16).reshape(
                layout.ranks,
                layout.decode_tokens,
                -1,
            )
        )
        buffers.decode_prev_hidden_in.copy_(kernel_previous_hidden)
        buffers.decode_position_ids.copy_(kernel_positions)
        decode_slot_mapping = torch.stack(slot_mappings)
        for rank, local_row in zip(assignment.ranks, assignment.local_rows, strict=True):
            row_start = local_row * layout.decode_seq
            decode_slot_mapping[rank, row_start + 1 : row_start + layout.decode_seq].fill_(-1)
        buffers.decode_slot_mapping.copy_(decode_slot_mapping)
        buffers.decode_swa_indices.copy_(torch.stack(swa_indices_by_rank))
        buffers.decode_swa_lens.copy_(torch.stack(swa_lens_by_rank))
        buffers.decode_hidden_out.zero_()
        buffers.decode_pre_hc_out.zero_()
        buffers.decode_logits.zero_()
        buffers.decode_logit_row_indices.fill_(-1)
        for rank, local_row in zip(assignment.ranks, assignment.local_rows, strict=True):
            buffers.decode_logit_row_indices[rank, local_row] = local_row * layout.decode_seq

        active_tokens = max(assignment.per_rank_counts) * layout.decode_seq
        with profile_span(
            "DeepSeekV4ModelRunner.mtp.decode.l3_dispatch",
            cat="executor",
            args={"actual_tokens": active_tokens},
        ):
            self._run_l3(
                self._require_mtp_decode_callable(),
                *self._mtp_decode_args(),
                self._int32_scalar(active_tokens),
            )

        next_tokens = []
        next_hidden = []
        for rank, local_row in zip(assignment.ranks, assignment.local_rows, strict=True):
            row_start = local_row * layout.decode_seq
            next_tokens.append(int(buffers.decode_logits[rank, local_row].argmax().item()))
            next_hidden.append(buffers.decode_pre_hc_out[rank, row_start].detach().cpu().clone())
        return torch.tensor(next_tokens, dtype=torch.long), torch.stack(next_hidden)

    def _initialize_mtp_drafts(self, batch: DecodeBatch) -> None:
        """Initialize every request's first draft without sharing mutable state."""
        for batch_index, request_id in enumerate(batch.request_ids):
            state = self._require_mtp_request_state(request_id)
            if state.draft_token_id is None:
                self._initialize_mtp_draft(request_id, state, batch, batch_index)

    def _initialize_mtp_draft(
        self,
        request_id: str,
        state: _DeepSeekV4MtpRequestState,
        batch: DecodeBatch,
        batch_index: int,
    ) -> None:
        context = state.prefill_context
        if context is None:
            raise RuntimeError(f"DeepSeekV4 MTP prefill context is missing for {request_id!r}")
        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        n = context.actual_tokens
        owner_rank = context.rank
        first_token = batch.token_ids[batch_index].detach().cpu().to(torch.long).reshape(1)
        first_hidden = self._embedding_rows(first_token, torch.bfloat16)[0]

        shifted_hidden = torch.zeros_like(context.hidden_states, dtype=torch.bfloat16)
        shifted_hidden[: n - 1].copy_(context.hidden_states[1:n].to(torch.bfloat16))
        shifted_hidden[n - 1].copy_(first_hidden)
        buffers.prefill_hidden_in.copy_(
            shifted_hidden.unsqueeze(0).expand(layout.ranks, -1, -1)
        )
        buffers.prefill_prev_hidden_in.copy_(
            context.prev_hidden_states.unsqueeze(0).expand(layout.ranks, -1, -1, -1)
        )

        shifted_ids = context.input_ids.clone()
        shifted_ids[: n - 1].copy_(context.input_ids[1:n])
        shifted_ids[n - 1].copy_(first_token[0])
        buffers.prefill_input_ids.copy_(shifted_ids.unsqueeze(0).expand(layout.ranks, -1))
        buffers.prefill_position_ids.copy_(
            context.position_ids.unsqueeze(0).expand(layout.ranks, -1)
        )
        buffers.prefill_block_table.copy_(
            context.block_table.unsqueeze(0).expand(layout.ranks, -1)
        )
        buffers.prefill_slot_mapping.fill_(-1)
        buffers.prefill_slot_mapping[owner_rank].copy_(context.slot_mapping)
        buffers.prefill_hidden_out.zero_()
        buffers.prefill_pre_hc_out.zero_()
        buffers.prefill_logits.zero_()
        buffers.prefill_logit_row_indices.fill_(-1)
        buffers.prefill_logit_row_indices[owner_rank, 0] = n - 1
        with profile_span(
            "DeepSeekV4ModelRunner.mtp.prefill.l3_dispatch",
            cat="executor",
            args={"actual_tokens": n},
        ):
            self._run_l3(
                self._require_mtp_prefill_callable(),
                *self._mtp_prefill_args(),
                self._int32_scalar(n),
            )
        state.draft_token_id = int(buffers.prefill_logits[owner_rank, 0].argmax().item())
        state.draft_pre_hc_hidden = buffers.prefill_pre_hc_out[owner_rank, n - 1].detach().cpu().clone()
        state.draft_position = int(context.position_ids[n - 1].item()) + 1
        state.prefill_context = None

    def _require_stacked_weights(self) -> DeepSeekV4StackedLayerWeights:
        tensors = self._stacked_device_weights or self._stacked_host_weights
        if tensors is None:
            raise RuntimeError("DeepSeekV4 stacked decode weights are not available")
        return DeepSeekV4StackedLayerWeights(tensors=tensors)

    def _ordered_layer_args(self, values: dict[str, Any], names: Sequence[str]) -> tuple[Any, ...]:
        missing = [name for name in names if name not in values]
        if missing:
            raise KeyError(f"DeepSeekV4 layer dispatch is missing tensors: {', '.join(missing)}")
        return tuple(values[name] for name in names)

    def _debug_prefill_dispatch(
        self,
        inputs: DeepSeekV4PreparedPrefillInputs,
        args: Sequence[Any],
    ) -> None:
        if os.getenv("PYPTO_DSV4_DEBUG") != "1":
            return
        named_args = dict(zip(_PREFILL_FWD_TENSOR_ORDER, args, strict=True))
        interesting = (
            "x_hc",
            "kv_cache",
            "cmp_kv",
            "idx_kv_cache",
            "ori_block_table",
            "cmp_block_table",
            "idx_block_table",
            "input_ids",
            "hidden_out",
            "logits",
        )
        tensor_names = [
            name
            for name, tensor in named_args.items()
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        ]
        non_shared = [name for name in tensor_names if not named_args[name].is_shared()]
        parts = []
        for name in interesting:
            tensor = named_args[name]
            if isinstance(tensor, torch.Tensor):
                parts.append(f"{name}={tuple(tensor.shape)}/{tensor.dtype}/shared={tensor.is_shared()}")
            elif isinstance(tensor, DeviceTensor):
                parts.append(f"{name}=DeviceTensor")
            else:
                parts.append(f"{name}={type(tensor).__name__}")
        print(
            "DeepSeekV4 packed prefill dispatch "
            f"tokens={inputs.actual_tokens} ranks={inputs.ranks} "
            f"worker_started={self._l3_worker is not None} "
            f"cpu_tensor_args={len(tensor_names)} non_shared={non_shared} "
            + " ".join(parts),
            flush=True,
        )
        if os.getenv("PYPTO_DSV4_DEBUG_ARGS") == "1":
            for name in _PREFILL_FWD_TENSOR_ORDER:
                tensor = named_args[name]
                if isinstance(tensor, torch.Tensor):
                    print(
                        "DeepSeekV4 prefill arg "
                        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                        f"device={tensor.device} shared={tensor.is_shared()}",
                        flush=True,
                    )

    def _debug_decode_dispatch(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        args: Sequence[Any],
    ) -> None:
        if os.getenv("PYPTO_DSV4_DEBUG") != "1":
            return
        named_args = dict(zip(_DECODE_FWD_TENSOR_ORDER, args, strict=True))
        interesting = (
            "x_hc",
            "kv_cache",
            "block_table",
            "ori_slot_mapping",
            "cmp_kv",
            "cmp_block_table",
            "idx_kv_cache",
            "idx_block_table",
            "hca_compress_state",
            "hca_state_slot_mapping",
            "csa_compress_state",
            "csa_state_slot_mapping",
            "csa_inner_compress_state",
            "csa_inner_state_slot_mapping",
            "position_ids",
            "kv_seq_lens",
            "input_ids",
            "hidden_out",
            "logits",
        )
        tensor_names = [
            name
            for name, tensor in named_args.items()
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        ]
        non_shared = [name for name in tensor_names if not named_args[name].is_shared()]
        parts = []
        for name in interesting:
            tensor = named_args[name]
            if isinstance(tensor, torch.Tensor):
                parts.append(f"{name}={tuple(tensor.shape)}/{tensor.dtype}/shared={tensor.is_shared()}")
            elif isinstance(tensor, DeviceTensor):
                parts.append(f"{name}=DeviceTensor")
            else:
                parts.append(f"{name}={type(tensor).__name__}")
        print(
            "DeepSeekV4 packed decode dispatch "
            f"actual_batch={inputs.actual_batch} "
            f"active_tokens={max(inputs.per_rank_counts) * self._compiled.layout.decode_seq} "
            f"ranks={inputs.ranks} "
            f"worker_started={self._l3_worker is not None} "
            f"cpu_tensor_args={len(tensor_names)} non_shared={non_shared} "
            + " ".join(parts),
            flush=True,
        )
        if os.getenv("PYPTO_DSV4_DEBUG_ARGS") == "1":
            for name in _DECODE_FWD_TENSOR_ORDER:
                tensor = named_args[name]
                if isinstance(tensor, torch.Tensor):
                    print(
                        "DeepSeekV4 decode arg "
                        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                        f"device={tensor.device} shared={tensor.is_shared()}",
                        flush=True,
                    )
                    self._debug_tensor_stats(f"dispatch.fwd.{name}", tensor)

    @staticmethod
    def _is_layer_weight_name(name: str) -> bool:
        runtime_names = {
            "x_hc",
            "freqs_cos",
            "freqs_sin",
            "hca_compress_state_block_table",
            "csa_compress_state_block_table",
            "csa_inner_compress_state_block_table",
            "kv_cache",
            "ori_block_table",
            "block_table",
            "ori_slot_mapping",
            "cmp_kv",
            "cmp_block_table",
            "idx_kv_cache",
            "idx_kv_scale",
            "idx_block_table",
            "position_ids",
            "window_swa_indices",
            "window_swa_lens",
            "swa_slot_mapping",
            "swa_indices",
            "swa_lens",
            "hca_cmp_slot_mapping",
            "hca_state_slot_mapping",
            "csa_cmp_slot_mapping",
            "csa_idx_slot_mapping",
            "csa_state_slot_mapping",
            "csa_inner_state_slot_mapping",
            "hca_compress_state",
            "csa_compress_state",
            "csa_inner_compress_state",
            "kv_seq_lens",
            "input_ids",
            "x_next",
        }
        return name not in runtime_names

    def _ensure_decode_buffers(self, hidden_size: int) -> _DeepSeekV4DecodeSharedBuffers:
        buffers = self._decode_buffers
        if buffers is None:
            self._ensure_shared_host_allocation_before_worker("decode inputs")
            layout = self._compiled.layout
            ranks = layout.ranks
            batch = layout.decode_batch
            tokens = layout.decode_tokens
            buffers = _DeepSeekV4DecodeSharedBuffers(
                x_hc_a=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.float32,
                    name="decode_x_hc",
                ),
                x_hc_b=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.float32,
                    name="decode_x_hc_next",
                ),
                pre_hc_hidden_out=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.float32,
                    name="decode_pre_hc_hidden_out",
                ),
                x_out=self._shared_empty(
                    (ranks, tokens, int(hidden_size)),
                    torch.bfloat16,
                    name="decode_x_out",
                ),
                tensors={
                    "input_ids": self._shared_empty((ranks, tokens), torch.long, name="decode_input_ids"),
                    "position_ids": self._shared_empty((ranks, tokens), torch.int32, name="decode_position_ids"),
                    "kv_seq_lens": self._shared_empty((ranks, batch), torch.int32, name="decode_kv_seq_lens"),
                    "block_table": self._shared_empty(
                        (ranks, batch, layout.ori_table_max_blocks),
                        torch.int32,
                        name="decode_block_table",
                    ),
                    "ori_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_ori_slot_mapping",
                    ),
                    "window_swa_indices": self._shared_empty(
                        (ranks, tokens, layout.sliding_window),
                        torch.int32,
                        name="decode_window_swa_indices",
                    ),
                    "window_swa_lens": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name="decode_window_swa_lens",
                    ),
                    "swa_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_swa_slot_mapping",
                    ),
                    "swa_indices": self._shared_empty(
                        (ranks, tokens, layout.sliding_window),
                        torch.int32,
                        name="decode_swa_indices",
                    ),
                    "swa_lens": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name="decode_swa_lens",
                    ),
                    "cmp_block_table": self._shared_empty(
                        (ranks, batch, layout.cmp_max_blocks),
                        torch.int32,
                        name="decode_cmp_block_table",
                    ),
                    "idx_block_table": self._shared_empty(
                        (ranks, batch, layout.idx_max_blocks),
                        torch.int32,
                        name="decode_idx_block_table",
                    ),
                    "hca_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_hca_state_max_blocks),
                        torch.int32,
                        name="decode_hca_compress_state_block_table",
                    ),
                    "csa_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_csa_state_max_blocks),
                        torch.int32,
                        name="decode_csa_compress_state_block_table",
                    ),
                    "csa_inner_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_csa_inner_state_max_blocks),
                        torch.int32,
                        name="decode_csa_inner_compress_state_block_table",
                    ),
                    "hca_cmp_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_hca_cmp_slot_mapping",
                    ),
                    "hca_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_hca_state_slot_mapping",
                    ),
                    "csa_cmp_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_cmp_slot_mapping",
                    ),
                    "csa_idx_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_idx_slot_mapping",
                    ),
                    "csa_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_state_slot_mapping",
                    ),
                    "csa_inner_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_inner_state_slot_mapping",
                    ),
                    "num_tokens_per_owner": self._shared_empty(
                        (ranks,),
                        torch.int32,
                        name="decode_num_tokens_per_owner",
                    ),
                    "logit_row_indices": self._shared_empty(
                        (ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS),
                        torch.int32,
                        name="decode_logit_row_indices",
                    ),
                },
            )
            self._decode_buffers = buffers
        return buffers

    def _ensure_mtp_buffers(self, hidden_size: int) -> _DeepSeekV4MtpSharedBuffers | None:
        """Load immutable MTP weights and allocate mutable shared buffers before worker fork."""
        if self._compiled.mtp_prefill is None or self._compiled.mtp_decode is None:
            return None
        if self._mtp_buffers is not None:
            return self._mtp_buffers
        self._ensure_shared_host_allocation_before_worker("mtp buffers")
        layout = self._compiled.layout
        ranks = layout.ranks
        tokens = layout.decode_tokens
        hidden = int(hidden_size)
        loaded = self.load_mtp_weights()
        weights = dict(loaded.tensors)
        mtp_kv_cache = self._shared_empty(
            (ranks, layout.prefill_ori_max_blocks, layout.block_size, 1, DEEPSEEK_V4_HEAD_DIM),
            torch.bfloat16,
            name="mtp_unified_kv_cache",
        )
        self._mtp_buffers = _DeepSeekV4MtpSharedBuffers(
            weights=weights,
            prefill_hidden_in=self._shared_empty(
                (ranks, layout.prefill_seq, hidden), torch.bfloat16, name="mtp_prefill_hidden_in"
            ),
            prefill_prev_hidden_in=self._shared_empty(
                (ranks, layout.prefill_seq, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_prefill_prev_hidden_in",
            ),
            prefill_input_ids=self._shared_empty(
                (ranks, layout.prefill_seq), torch.long, name="mtp_prefill_input_ids"
            ),
            prefill_position_ids=self._shared_empty(
                (ranks, layout.prefill_seq), torch.int32, name="mtp_prefill_position_ids"
            ),
            prefill_block_table=self._shared_empty(
                (ranks, layout.prefill_ori_max_blocks), torch.int32, name="mtp_prefill_block_table"
            ),
            prefill_slot_mapping=self._shared_empty(
                (ranks, layout.prefill_seq), torch.long, name="mtp_prefill_slot_mapping"
            ),
            prefill_kv_cache=mtp_kv_cache,
            decode_hidden_in=self._shared_empty(
                (ranks, tokens, hidden), torch.bfloat16, name="mtp_decode_hidden_in"
            ),
            decode_prev_hidden_in=self._shared_empty(
                (ranks, tokens, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_decode_prev_hidden_in",
            ),
            decode_input_ids=self._shared_empty(
                (ranks, tokens), torch.long, name="mtp_decode_input_ids"
            ),
            decode_position_ids=self._shared_empty(
                (ranks, tokens), torch.int32, name="mtp_decode_position_ids"
            ),
            decode_slot_mapping=self._shared_empty(
                (ranks, tokens), torch.long, name="mtp_decode_slot_mapping"
            ),
            decode_swa_indices=self._shared_empty(
                (ranks, tokens, layout.sliding_window), torch.int32, name="mtp_decode_swa_indices"
            ),
            decode_swa_lens=self._shared_empty(
                (ranks, tokens), torch.int32, name="mtp_decode_swa_lens"
            ),
            decode_kv_cache=mtp_kv_cache,
            prefill_hidden_out=self._shared_empty(
                (ranks, layout.prefill_seq, hidden), torch.bfloat16, name="mtp_prefill_hidden_out"
            ),
            prefill_pre_hc_out=self._shared_empty(
                (ranks, layout.prefill_seq, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_prefill_pre_hc_out",
            ),
            prefill_logits=self._shared_empty(
                (ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS, DEEPSEEK_V4_VOCAB_SIZE),
                torch.float32,
                name="mtp_prefill_logits",
            ),
            prefill_logit_row_indices=self._shared_empty(
                (ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS), torch.int32, name="mtp_prefill_logit_row_indices"
            ),
            decode_hidden_out=self._shared_empty(
                (ranks, tokens, hidden), torch.bfloat16, name="mtp_decode_hidden_out"
            ),
            decode_pre_hc_out=self._shared_empty(
                (ranks, tokens, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_decode_pre_hc_out",
            ),
            decode_logits=self._shared_empty(
                (ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS, DEEPSEEK_V4_VOCAB_SIZE),
                torch.float32,
                name="mtp_decode_logits",
            ),
            decode_logit_row_indices=self._shared_empty(
                (ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS), torch.int32, name="mtp_decode_logit_row_indices"
            ),
        )
        self._mtp_buffers.prefill_kv_cache.zero_()
        return self._mtp_buffers

    def _stage_decode_inputs(self, inputs: DeepSeekV4PreparedDecodeInputs) -> DeepSeekV4PreparedDecodeInputs:
        buffers = self._ensure_decode_buffers(inputs.x_hc.shape[-1])
        self._copy_shared(buffers.x_hc_a, inputs.x_hc, name="decode_x_hc")
        staged_values: dict[str, torch.Tensor] = {}
        for name in _DECODE_INPUT_TENSOR_FIELDS:
            dst = buffers.tensors[name]
            self._copy_shared(dst, getattr(inputs, name), name=f"decode_{name}")
            staged_values[name] = dst
        return replace(inputs, x_hc=buffers.x_hc_a, **staged_values)

    def _ensure_prefill_fwd_buffers(self, hidden_size: int) -> _DeepSeekV4PrefillFwdSharedBuffers:
        """Allocate the layer-stacked shared buffers for the packed prefill dispatch."""
        buffers = self._prefill_fwd_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("prefill_fwd buffers")
        layout = self._compiled.layout
        ranks = layout.ranks
        seq = layout.prefill_seq
        hidden = int(hidden_size)
        rope_dim = self._compiled.freqs_cos.shape[-1] if self._compiled.freqs_cos is not None else 0
        max_seq_len = self._compiled.freqs_cos.shape[0] if self._compiled.freqs_cos is not None else 0

        def shared(shape, dtype, name):
            return self._shared_empty(shape, dtype, name=name)

        tensors: dict[str, torch.Tensor] = {
            "hca_compress_state_block_table": shared(
                (ranks, layout.prefill_hca_state_max_blocks), torch.int32, "prefill_fwd_hca_state_block_table"
            ),
            "csa_compress_state_block_table": shared(
                (ranks, layout.prefill_csa_state_max_blocks), torch.int32, "prefill_fwd_csa_state_block_table"
            ),
            "csa_inner_compress_state_block_table": shared(
                (ranks, layout.prefill_csa_inner_state_max_blocks), torch.int32, "prefill_fwd_csa_inner_state_block_table"
            ),
            # Shared single per-rank metadata (the kernel passes each whole tensor
            # to every layer).
            "ori_block_table": shared(
                (ranks, layout.prefill_ori_max_blocks), torch.int32, "prefill_fwd_ori_block_table"
            ),
            "ori_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_ori_slot_mapping"),
            "cmp_block_table": shared((ranks, layout.prefill_cmp_max_blocks), torch.int32, "prefill_fwd_cmp_block_table"),
            "idx_block_table": shared((ranks, layout.prefill_idx_max_blocks), torch.int32, "prefill_fwd_idx_block_table"),
            "position_ids": shared((ranks, seq), torch.int32, "prefill_fwd_position_ids"),
            "hca_cmp_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_hca_cmp_slot_mapping"),
            "hca_state_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_hca_state_slot_mapping"),
            "csa_cmp_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_cmp_slot_mapping"),
            "csa_idx_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_idx_slot_mapping"),
            "csa_state_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_state_slot_mapping"),
            "csa_inner_state_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_inner_state_slot_mapping"),
            "input_ids": shared((ranks, seq), torch.long, "prefill_fwd_input_ids"),
            "num_tokens_per_owner": shared(
                (ranks,), torch.int32, "prefill_fwd_num_tokens_per_owner"
            ),
            "logit_row_indices": shared(
                (ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS),
                torch.int32,
                "prefill_fwd_logit_row_indices",
            ),
        }
        buffers = _DeepSeekV4PrefillFwdSharedBuffers(
            x_hc=shared((ranks, seq, layout.hc_mult, hidden), torch.float32, "prefill_fwd_x_hc"),
            freqs_cos=shared((ranks, max_seq_len, rope_dim), torch.bfloat16, "prefill_fwd_freqs_cos"),
            freqs_sin=shared((ranks, max_seq_len, rope_dim), torch.bfloat16, "prefill_fwd_freqs_sin"),
            tensors=tensors,
        )
        self._prefill_fwd_buffers = buffers
        return buffers

    def _require_prefill_fwd_buffers(self) -> _DeepSeekV4PrefillFwdSharedBuffers:
        if self._prefill_fwd_buffers is None:
            raise RuntimeError("DeepSeekV4 packed prefill shared buffers were not staged")
        return self._prefill_fwd_buffers

    def _stage_prefill_fwd_inputs(self, inputs: DeepSeekV4PreparedPrefillInputs) -> None:
        """Copy one prefill chunk's mutable metadata into shared host buffers.

        The per-request metadata (slot mappings, block tables, position/input
        ids), the RoPE tables and the compressor-state block tables
        are shared single per-rank copies (the kernel slices them per layer
        internally). Cache pools are worker-resident and are not staged here.
        """
        buffers = self._require_prefill_fwd_buffers()

        # x_hc / output collapse weights.
        self._copy_shared(buffers.x_hc, inputs.x_hc, name="prefill_fwd_x_hc")
        self._copy_shared(
            buffers.freqs_cos,
            self._static_freqs_cos_table(),
            name="prefill_fwd_freqs_cos",
        )
        self._copy_shared(
            buffers.freqs_sin,
            self._static_freqs_sin_table(),
            name="prefill_fwd_freqs_sin",
        )

        # Shared single per-rank metadata (the kernel slices it per layer).
        shared_metadata = {
            "ori_block_table": inputs.ori_block_table,
            "ori_slot_mapping": inputs.ori_slot_mapping,
            "cmp_block_table": inputs.cmp_block_table,
            "idx_block_table": inputs.idx_block_table,
            "position_ids": inputs.position_ids,
            "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
            "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
            "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
            "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
            "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
            "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
            "input_ids": inputs.input_ids,
            "hca_compress_state_block_table": inputs.hca_compress_state_block_table,
            "csa_compress_state_block_table": inputs.csa_compress_state_block_table,
            "csa_inner_compress_state_block_table": inputs.csa_inner_compress_state_block_table,
            "num_tokens_per_owner": inputs.num_tokens_per_owner,
            "logit_row_indices": inputs.logit_row_indices,
        }
        for name, tensor in shared_metadata.items():
            self._copy_shared(buffers.tensors[name], tensor, name=f"prefill_fwd_{name}")

    def _static_freqs_cos_table(self) -> torch.Tensor:
        if self._compiled.freqs_cos is None:
            raise RuntimeError("DeepSeekV4 RoPE cosine table is not initialized")
        return self._rank_stack(self._compiled.freqs_cos)

    def _static_freqs_sin_table(self) -> torch.Tensor:
        if self._compiled.freqs_sin is None:
            raise RuntimeError("DeepSeekV4 RoPE sine table is not initialized")
        return self._rank_stack(self._compiled.freqs_sin)

    def _retain_stacked_host_weights(
        self,
        weights: DeepSeekV4StackedLayerWeights,
    ) -> DeepSeekV4StackedLayerWeights:
        """Retain immutable layer-stacked weights for fork inheritance and resident upload."""
        host_weights = self._stacked_host_weights
        if host_weights is None:
            self._ensure_shared_host_allocation_before_worker("stacked layer weights")
            host_weights = dict(weights.tensors)
            self._stacked_host_weights = host_weights

        missing = sorted(set(weights.tensors) - set(host_weights))
        if missing:
            raise KeyError(f"DeepSeekV4 stacked Host weights are missing: {', '.join(missing)}")

        return DeepSeekV4StackedLayerWeights(tensors=host_weights)

    def _hc_head_tensors(self) -> dict[str, torch.Tensor]:
        """Return rank-replicated hc_head weights for the decode_fwd output collapse."""
        buffers = self._hc_head_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("hc_head weights")
        global_weights = self.load_packed_global_weights()
        ranks = self._compiled.layout.ranks
        # The kernel hc_head_fn is [HC_MULT, HC_DIM]; the checkpoint stores it as
        # [HC_MULT, hidden*HC_MULT] (== [HC_MULT, HC_DIM]). Scale/base are scalars
        # per HC_MULT row, rank-replicated.
        hc_head_fn = global_weights.hc_head_fn.to(torch.float32).contiguous().cpu()
        hc_head_scale = global_weights.hc_head_scale.to(torch.float32).contiguous().cpu()
        hc_head_base = global_weights.hc_head_base.to(torch.float32).contiguous().cpu()
        buffers = {
            "hc_head_fn": self._static_device_tensor(self._rank_stack(hc_head_fn)),
            "hc_head_scale": self._static_device_tensor(self._rank_stack(hc_head_scale)),
            "hc_head_base": self._static_device_tensor(self._rank_stack(hc_head_base)),
        }
        self._hc_head_buffers = buffers
        return buffers

    def _require_decode_buffers(self) -> _DeepSeekV4DecodeSharedBuffers:
        if self._decode_buffers is None:
            raise RuntimeError("DeepSeekV4 decode shared buffers were not staged")
        return self._decode_buffers

    def _require_decode_output_buffer(self, hidden_size: int) -> torch.Tensor:
        return self._ensure_decode_buffers(int(hidden_size)).x_out

    def _require_decode_logits_buffer(self, vocab_size: int) -> torch.Tensor:
        """Return shared device LM-head logits for every decode owner rank."""
        layout = self._compiled.layout
        logits_shape = (layout.ranks, layout.decode_tokens, int(vocab_size))
        if self._decode_logits_buffer is None:
            self._ensure_shared_host_allocation_before_worker("decode_logits")
            self._decode_logits_buffer = self._shared_empty(logits_shape, torch.float32, name="decode_logits")
        return self._decode_logits_buffer

    def _require_prefill_output_buffer(self, hidden_size: int) -> torch.Tensor:
        """Return the shared ``[ranks, prefill_seq, hidden]`` prefill hidden output."""
        layout = self._compiled.layout
        output_shape = (layout.ranks, layout.prefill_seq, int(hidden_size))
        if self._prefill_output_buffer is None:
            self._ensure_shared_host_allocation_before_worker("prefill_output")
            self._prefill_output_buffer = self._shared_empty(output_shape, torch.bfloat16, name="prefill_output")
        return self._prefill_output_buffer

    def _require_prefill_pre_hc_output_buffer(self, hidden_size: int) -> torch.Tensor:
        """Return the main-model final pre-HC rows used to seed MTP prefill."""
        layout = self._compiled.layout
        output_shape = (layout.ranks, layout.prefill_seq, layout.hc_mult, int(hidden_size))
        if self._prefill_pre_hc_output_buffer is None:
            self._ensure_shared_host_allocation_before_worker("prefill_pre_hc_output")
            self._prefill_pre_hc_output_buffer = self._shared_empty(
                output_shape,
                torch.float32,
                name="prefill_pre_hc_output",
            )
        return self._prefill_pre_hc_output_buffer

    def _require_prefill_logits_buffer(self, vocab_size: int) -> torch.Tensor:
        """Return shared owner-major selected-row logits for packed prefill."""
        layout = self._compiled.layout
        logits_shape = (layout.ranks, DEEPSEEK_V4_MAX_LOGIT_ROWS, int(vocab_size))
        if self._prefill_logits_buffer is None:
            self._ensure_shared_host_allocation_before_worker("prefill_logits")
            self._prefill_logits_buffer = self._shared_empty(
                logits_shape,
                torch.float32,
                name="prefill_logits",
            )
        return self._prefill_logits_buffer

    def _static_final_norm_weight_tensor(self) -> torch.Tensor:
        """Return the worker-resident per-rank final RMSNorm weight ``[ranks, D]``.

        Uses the model's final RMSNorm weight, rank-replicated and cast to bf16.
        """
        if self._static_final_norm_weight is None:
            global_weights = self.load_packed_global_weights()
            self._ensure_shared_host_allocation_before_worker("final_norm_w")
            final_norm_w = global_weights.final_norm_weight.to(torch.bfloat16).contiguous().cpu()
            self._static_final_norm_weight = self._static_device_tensor(self._rank_stack(final_norm_w))
        return self._static_final_norm_weight

    def _static_lm_head_weight_tensor(self) -> torch.Tensor:
        """Return the worker-visible LM-head weight, one vocab shard per DP rank.

        The kernel groups the DP world into ``ranks // tp`` independent TP groups,
        so every card consumes shard ``rank % tp``. Resident arguments are handed
        out per rank, so the shard has to be replicated here rather than indexed
        inside the kernel.
        """
        if self._static_lm_head_weight is None:
            global_weights = self.load_packed_global_weights()
            self._ensure_shared_host_allocation_before_worker("lm_head_weight")
            packed = global_weights.lm_head_weight.to(torch.bfloat16).contiguous().cpu()
            tp_size = packed.shape[0]
            ranks = self._compiled.layout.ranks
            rank_shards = [packed[rank % tp_size] for rank in range(ranks)]
            self._static_lm_head_weight = self._static_device_tensor(
                torch.stack(rank_shards, dim=0).contiguous()
            )
        return self._static_lm_head_weight

    def _static_freqs_cos_tensor(self) -> torch.Tensor:
        if self._static_freqs_cos is None:
            if self._compiled.freqs_cos is None:
                raise RuntimeError("DeepSeekV4 RoPE cosine table is not initialized")
            self._ensure_shared_host_allocation_before_worker("freqs_cos")
            self._static_freqs_cos = self._static_device_tensor(self._rank_stack(self._compiled.freqs_cos))
        return self._static_freqs_cos

    def _static_freqs_sin_tensor(self) -> torch.Tensor:
        if self._static_freqs_sin is None:
            if self._compiled.freqs_sin is None:
                raise RuntimeError("DeepSeekV4 RoPE sine table is not initialized")
            self._ensure_shared_host_allocation_before_worker("freqs_sin")
            self._static_freqs_sin = self._static_device_tensor(self._rank_stack(self._compiled.freqs_sin))
        return self._static_freqs_sin

    def _seed_decode_work_cache_from_group_ids(
        self,
        request_ids: Sequence[str],
        ranks: Sequence[int],
        block_ids_by_group: Sequence[dict[str, tuple[int, ...]]],
    ) -> None:
        """Initialize scheduler-grown blocks; prefill already wrote the cache in place."""
        self._initialize_decode_cache_blocks(request_ids, ranks, block_ids_by_group)

    def _initialize_decode_cache_blocks(
        self,
        request_ids: Sequence[str],
        ranks: Sequence[int],
        block_ids_by_group: Sequence[dict[str, tuple[int, ...]]],
    ) -> None:
        """Clear newly owned physical blocks without disturbing resident cache state."""
        if len(request_ids) != len(ranks) or len(ranks) != len(block_ids_by_group):
            raise ValueError("decode request IDs, ranks, and grouped KV rows must have the same length")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("decode request IDs must be unique")
        cache = self._require_decode_work_cache()
        dirty: dict[int, dict[str, set[int]]] = {}
        for request_id, rank, groups in zip(
            request_ids,
            ranks,
            block_ids_by_group,
            strict=True,
        ):
            initialized = self._decode_cache_block_ids.setdefault(request_id, {})
            new_blocks = {
                name: tuple(
                    block_id
                    for block_id in (int(value) for value in block_ids)
                    if block_id not in initialized.get(name, set())
                )
                for name, block_ids in groups.items()
            }
            self._zero_decode_work_cache_blocks(cache, int(rank), new_blocks)
            rank_dirty = dirty.setdefault(int(rank), {})
            for name, block_ids in new_blocks.items():
                rank_dirty.setdefault(name, set()).update(block_ids)
            for name, block_ids in groups.items():
                initialized.setdefault(name, set()).update(int(value) for value in block_ids)
        self._sync_decode_work_cache_blocks(cache, dirty)

    def _zero_decode_work_cache_blocks(
        self,
        cache: DeepSeekV4LayerCache,
        rank: int,
        block_ids_by_group: dict[str, Sequence[int]],
    ) -> None:
        """Clear newly assigned rank-local pages across every stacked layer."""
        layout = self._compiled.layout
        if not 0 <= rank < layout.ranks:
            raise ValueError(f"DeepSeekV4 cache rank must be in [0, {layout.ranks - 1}]")
        tensors_by_group = {
            "ori": ((cache.kv_cache, DEEPSEEK_V4_FWD_NUM_LAYERS, layout.decode_ori_max_blocks),),
            "cmp": ((cache.cmp_kv, DEEPSEEK_V4_FWD_NUM_LAYERS, layout.cmp_max_blocks),),
            "idx": (
                (cache.idx_kv_cache, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.idx_max_blocks),
                (cache.idx_kv_scale, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.idx_max_blocks),
            ),
            "hca_state": (
                (cache.hca_compress_state, DEEPSEEK_V4_HCA_NUM_LAYERS, layout.hca_state_max_blocks),
            ),
            "csa_state": (
                (cache.csa_compress_state, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.csa_state_max_blocks),
            ),
            "csa_inner_state": (
                (
                    cache.csa_inner_compress_state,
                    DEEPSEEK_V4_CSA_NUM_LAYERS,
                    layout.csa_inner_state_max_blocks,
                ),
            ),
        }
        unknown = sorted(set(block_ids_by_group) - set(tensors_by_group))
        if unknown:
            raise ValueError("unknown DeepSeekV4 cache groups: " + ", ".join(unknown))
        for name, block_ids in block_ids_by_group.items():
            ids = tuple(dict.fromkeys(int(block_id) for block_id in block_ids))
            if not ids:
                continue
            for tensor, layer_count, blocks_per_layer in tensors_by_group[name]:
                if any(block_id < 0 or block_id >= blocks_per_layer for block_id in ids):
                    raise ValueError(
                        f"DeepSeekV4 {name} block IDs must be in [0, {blocks_per_layer})"
                    )
                pages = tensor[rank].reshape(
                    layer_count,
                    blocks_per_layer,
                    *tensor.shape[2:],
                )
                pages.index_fill_(1, torch.tensor(ids, dtype=torch.long), 0)

    def _sync_decode_work_cache_blocks(
        self,
        host: DeepSeekV4LayerCache,
        dirty_by_rank: dict[int, dict[str, set[int]]],
    ) -> None:
        """Upload only newly cleared pages when the cache is already resident."""
        device = self._decode_device_cache
        worker = self._l3_worker
        if device is None:
            return
        if worker is None:
            raise RuntimeError("DeepSeekV4 resident cache exists without an L3 worker")
        layout = self._compiled.layout
        tensors_by_group = {
            "ori": ((host.kv_cache, device.kv_cache, DEEPSEEK_V4_FWD_NUM_LAYERS, layout.decode_ori_max_blocks),),
            "cmp": ((host.cmp_kv, device.cmp_kv, DEEPSEEK_V4_FWD_NUM_LAYERS, layout.cmp_max_blocks),),
            "idx": (
                (host.idx_kv_cache, device.idx_kv_cache, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.idx_max_blocks),
                (host.idx_kv_scale, device.idx_kv_scale, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.idx_max_blocks),
            ),
            "hca_state": ((
                host.hca_compress_state,
                device.hca_compress_state,
                DEEPSEEK_V4_HCA_NUM_LAYERS,
                layout.hca_state_max_blocks,
            ),),
            "csa_state": ((
                host.csa_compress_state,
                device.csa_compress_state,
                DEEPSEEK_V4_CSA_NUM_LAYERS,
                layout.csa_state_max_blocks,
            ),),
            "csa_inner_state": ((
                host.csa_inner_compress_state,
                device.csa_inner_compress_state,
                DEEPSEEK_V4_CSA_NUM_LAYERS,
                layout.csa_inner_state_max_blocks,
            ),),
        }
        for rank, groups in dirty_by_rank.items():
            for name, block_ids in groups.items():
                for host_tensor, stacked, layer_count, blocks_per_layer in tensors_by_group[name]:
                    shard = stacked.shards[rank]
                    worker_id = stacked.worker_ids[rank]
                    page_nbytes = host_tensor[rank, 0].numel() * host_tensor.element_size()
                    for layer in range(layer_count):
                        for block_id in block_ids:
                            flat_index = layer * blocks_per_layer + int(block_id)
                            page = host_tensor[rank, flat_index]
                            worker.copy_to(
                                shard.data_ptr + flat_index * page_nbytes,
                                page.data_ptr(),
                                page_nbytes,
                                worker_id=worker_id,
                            )

    def _logits_for_hidden(
        self,
        x_hc: torch.Tensor,
        *,
        owner_rows: Sequence[tuple[int, int]] | None = None,
        active_rows: Sequence[int] | None = None,
        label: str = "unknown",
    ) -> torch.Tensor:
        global_weights = self.load_packed_global_weights()
        if x_hc.ndim == 3:
            # Decode output is already collapsed and final-normalized by
            # ``l3_decode_fwd``; host LM-head consumes it directly.
            hidden = x_hc
        else:
            hidden = self._final_hidden(x_hc)
        if owner_rows is None:
            owner_rows = tuple((0, int(row)) for row in (active_rows or ()))
        owners = tuple((int(rank), int(row)) for rank, row in owner_rows)
        if not owners:
            raise ValueError("DeepSeekV4 LM-head requires at least one active row")
        if any(rank < 0 or rank >= hidden.shape[0] for rank, _ in owners):
            raise ValueError(
                f"DeepSeekV4 LM-head owner ranks exceed hidden ranks={hidden.shape[0]}: {owners}"
            )
        if any(row < 0 or row >= hidden.shape[1] for _, row in owners):
            raise ValueError(
                f"DeepSeekV4 LM-head owner rows {owners} exceed hidden rows={hidden.shape[1]}"
            )
        if self._debug_tensor_stats_enabled():
            print(f"DSV4_DEBUG lm_head.label={label} owner_rows={owners}", flush=True)

        layout = global_weights.lm_head_layout
        if global_weights.lm_head_weight.shape[0] != layout.ranks:
            raise ValueError(
                "DeepSeekV4 packed LM-head rank count mismatch: "
                f"weight ranks={global_weights.lm_head_weight.shape[0]} layout ranks={layout.ranks}"
            )
        if global_weights.lm_head_weight.shape[1] < layout.vocab_per_rank:
            raise ValueError(
                "DeepSeekV4 packed LM-head shard is smaller than the real vocab shard: "
                f"shape={tuple(global_weights.lm_head_weight.shape)} vocab_per_rank={layout.vocab_per_rank}"
            )

        selected = torch.stack(
            [hidden[rank, row] for rank, row in owners],
            dim=0,
        ).detach().cpu().to(torch.float32).contiguous()
        if self._debug_tensor_stats_enabled():
            self._debug_tensor_stats("lm_head.hidden.active", selected)
        logits_parts = []
        for rank in range(layout.ranks):
            shard = global_weights.lm_head_weight[rank, : layout.vocab_per_rank, :]
            shard = shard.detach().cpu().to(torch.float32).contiguous()
            logits_parts.append(torch.matmul(selected, shard.t()))
        logits = torch.cat(logits_parts, dim=-1)
        if logits.shape[-1] != layout.vocab_size:
            logits = logits[:, : layout.vocab_size].contiguous()
        else:
            logits = logits.contiguous()
        self._debug_tensor_stats("lm_head.logits.returned", logits)
        return logits

    @staticmethod
    def _debug_tensor_stats_enabled() -> bool:
        return os.getenv("PYPTO_DSV4_LOGIT_DEBUG") == "1"

    @staticmethod
    def _debug_tensor_stats(name: str, tensor: torch.Tensor, *, per_rank: bool = False) -> None:
        if not DeepSeekV4ModelRunner._debug_tensor_stats_enabled():
            return
        data = tensor.detach().cpu().to(torch.float32)
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        total = data.numel()
        nan_count = int(torch.isnan(data).sum().item())
        pos_inf_count = int(torch.isposinf(data).sum().item())
        neg_inf_count = int(torch.isneginf(data).sum().item())
        if finite_count:
            finite_values = data[finite]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            absmax_value = float(finite_values.abs().max().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
            absmax_value = float("nan")
        print(
            "DSV4_DEBUG "
            f"{name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"finite={finite_count}/{total} nan={nan_count} "
            f"+inf={pos_inf_count} -inf={neg_inf_count} "
            f"min={min_value:.6g} max={max_value:.6g} absmax={absmax_value:.6g}",
            flush=True,
        )
        if per_rank and data.ndim >= 1:
            rank_view = data.reshape(data.shape[0], -1)
            rank_finite = torch.isfinite(rank_view)
            rank_counts = (rank_view.shape[1] - rank_finite.sum(dim=1)).tolist()
            print(f"DSV4_DEBUG {name} nonfinite_by_rank={rank_counts}", flush=True)

    @staticmethod
    def _tensor_is_finite(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor.detach().cpu().to(torch.float32)).all().item())

    def _final_hidden(self, x_hc: torch.Tensor) -> torch.Tensor:
        """Collapse a ``[ranks, T, HC_MULT, D]`` HC stack and apply the final norm."""
        weights = self.load_packed_global_weights()
        x_hc = x_hc.to(torch.bfloat16).cpu()
        x_float = x_hc.float()
        flat = x_float.flatten(2)
        rms = torch.sqrt(flat.double().square().mean(dim=-1, keepdim=True) + DEEPSEEK_V4_RMS_NORM_EPS)
        normed_flat = flat / rms.to(torch.float32)
        mixes = torch.matmul(normed_flat, weights.hc_head_fn.t())
        pre = torch.sigmoid(mixes * weights.hc_head_scale + weights.hc_head_base) + DEEPSEEK_V4_HC_EPS
        collapsed = torch.sum(pre.unsqueeze(-1).double() * x_float.double(), dim=2)
        return self._final_norm(collapsed)

    def _final_norm(self, collapsed: torch.Tensor) -> torch.Tensor:
        """Apply the final RMS norm to an already-collapsed ``[ranks, T, D]`` hidden.

        The packed ``l3_decode_fwd`` kernel collapses HC_MULT in-kernel via
        ``hc_head`` and returns the collapsed (pre-final-norm) hidden, so decode
        only needs the model's final RMS norm before the LM head.
        """
        collapsed = collapsed.cpu().double()
        weights = self.load_packed_global_weights()
        norm_inv = torch.rsqrt(collapsed.square().mean(dim=-1, keepdim=True) + DEEPSEEK_V4_RMS_NORM_EPS)
        normed = collapsed * norm_inv * weights.final_norm_weight.double()
        return normed.to(torch.float32).to(torch.bfloat16).contiguous()

    def _scope_stats_run_config(self) -> Any:
        """Optional per-dispatch RunConfig that captures device scope stats.

        Enabled with ``PYPTO_DSV4_SCOPE_STATS=1`` to dump per-scope
        heap / task_window / tensormap peaks under ``<dir>/dfx_outputs/``.
        """
        if os.getenv("PYPTO_DSV4_SCOPE_STATS") != "1":
            return None
        from pypto.runtime import RunConfig  # noqa: PLC0415

        out_dir = os.getenv("PYPTO_DSV4_SCOPE_STATS_DIR", "/data/liuxu/pypto-serving/dsv4_scope_stats")
        return RunConfig(
            platform=self._compiled.platform,
            device_id=self._compiled.device_id,
            enable_scope_stats=True,
            save_kernels=True,
            save_kernels_dir=out_dir,
        )

    def _run_l3(self, callable_spec: DeepSeekV4L3Callable, *args: Any) -> Any:
        """Dispatch one DeepSeek L3 program and emit Qwen-compatible timing traces."""
        if self._l3_worker is None:
            self._assert_l3_args_shared_before_worker(callable_spec, args)
        trace_name = _kernel_trace_name(callable_spec.name)
        span_args = {
            "kernel": callable_spec.name,
            "block_dim": callable_spec.block_dim,
            "aicpu_thread_num": callable_spec.aicpu_thread_num,
        }
        with profile_span(trace_name, cat="kernel", level="kernel", args=span_args):
            worker = self._shared_l3_worker()
            run_config = self._scope_stats_run_config()
            uploaded: list[DeviceTensor] = []
            try:
                l3_args = tuple(self._coerce_l3_arg(worker, arg, uploaded) for arg in args)
                worker_run_args = dict(span_args)
                with profile_span(
                    f"{trace_name}.worker_run",
                    cat="kernel",
                    level="kernel",
                    args=worker_run_args,
                ):
                    if run_config is not None:
                        timing = worker.run(callable_spec.compiled, *l3_args, config=run_config)
                    else:
                        timing = worker.run(callable_spec.compiled, *l3_args)
                    _add_run_timing_args(worker_run_args, timing)
                _add_run_timing_args(span_args, timing)
                return timing
            finally:
                for tensor in uploaded:
                    worker.free_tensor(tensor)

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not tensor.is_shared():
            tensor = tensor.share_memory_()
        return tensor

    @staticmethod
    def _shared_empty(shape: Sequence[int], dtype: torch.dtype, *, name: str) -> torch.Tensor:
        del name
        return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype).share_memory_()

    @staticmethod
    def _copy_shared(dst: torch.Tensor, src: torch.Tensor, *, name: str) -> None:
        if src.device.type != "cpu":
            src = src.cpu()
        if not src.is_contiguous():
            src = src.contiguous()
        if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype:
            raise ValueError(
                f"{name} shared buffer shape/dtype mismatch: "
                f"buffer shape={tuple(dst.shape)} dtype={dst.dtype}, "
                f"source shape={tuple(src.shape)} dtype={src.dtype}"
            )
        dst.copy_(src)

    @staticmethod
    def _int32_scalar(value: int) -> int:
        return int(value)

    def _ensure_shared_host_allocation_before_worker(self, name: str) -> None:
        if self._l3_worker is not None:
            raise RuntimeError(
                f"DeepSeekV4 shared host buffer '{name}' must be allocated before the L3 worker starts"
            )

    def _assert_l3_args_shared_before_worker(
        self,
        callable_spec: DeepSeekV4L3Callable,
        args: Sequence[Any],
    ) -> None:
        for index, arg in enumerate(args):
            self._assert_l3_arg_shared(arg, name=f"{callable_spec.name}[{index}]")

    def _assert_l3_arg_shared(self, arg: Any, *, name: str) -> None:
        if isinstance(arg, (_StaticDeviceTensor, _TransientDeviceTensor)):
            self._assert_l3_arg_shared(arg.tensor, name=f"{name}.tensor")
            return
        if isinstance(arg, torch.Tensor) and arg.device.type == "cpu" and not arg.is_shared():
            raise TypeError(
                "DeepSeekV4 L3 dispatch requires shared-memory CPU tensors allocated before "
                f"the L3 worker starts; got {name} shape={tuple(arg.shape)} dtype={arg.dtype}"
            )
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, bytearray)):
            for index, item in enumerate(arg):
                self._assert_l3_arg_shared(item, name=f"{name}[{index}]")
            return
        if isinstance(arg, dict):
            for key, item in arg.items():
                self._assert_l3_arg_shared(item, name=f"{name}[{key!r}]")

    def _coerce_l3_arg(self, worker: Any, arg: Any, uploaded: list[DeviceTensor]) -> Any:
        if isinstance(arg, _StaticDeviceTensor):
            self._assert_l3_arg_shared(arg, name="static")
            tensor = arg.tensor
            key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
            cached = self._l3_static_tensors.get(key)
            if cached is None:
                cached = worker.alloc_stacked_tensor(tensor)
                self._l3_static_tensors[key] = cached
            if arg.cache_state:
                self._l3_cache_tensor_keys.add(key)
            return cached
        if isinstance(arg, _TransientDeviceTensor):
            tensor = arg.tensor
            self._assert_l3_arg_shared(arg, name="transient")
            dev = worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
            uploaded.append(dev)
            return dev
        if isinstance(arg, torch.Tensor) and arg.device.type == "cpu" and not arg.is_shared():
            raise TypeError(
                "DeepSeekV4 L3 dispatch requires shared-memory CPU tensors allocated before "
                f"the worker starts; got non-shared tensor shape={tuple(arg.shape)} dtype={arg.dtype}"
            )
        return arg

    @staticmethod
    def _mark_resident_args(
        values: dict[str, Any],
        policy: dict[str, bool],
    ) -> dict[str, Any]:
        """Mark policy-selected arguments for lazy one-time Device upload."""
        missing = [name for name in policy if name not in values]
        if missing:
            raise KeyError(f"DeepSeekV4 resident argument policy is missing values: {missing}")
        for name, cache_state in policy.items():
            tensor = values[name]
            if isinstance(tensor, StackedDeviceTensor):
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"DeepSeekV4 resident argument {name!r} must be a torch.Tensor or "
                    f"StackedDeviceTensor, got {type(tensor).__name__}"
                )
            if tensor.device.type != "cpu" or not tensor.is_contiguous() or not tensor.is_shared():
                raise ValueError(
                    f"DeepSeekV4 resident argument {name!r} must be a contiguous shared-memory "
                    "CPU tensor"
                )
            values[name] = _StaticDeviceTensor(tensor=tensor, cache_state=cache_state)
        return values

    @staticmethod
    def _upload_weight_group(
        worker: Any,
        host_weights: dict[str, torch.Tensor],
    ) -> dict[str, StackedDeviceTensor]:
        """Upload a rank-stacked weight group, rolling back partial allocation."""
        device_weights: dict[str, StackedDeviceTensor] = {}
        try:
            for name, tensor in host_weights.items():
                device_weights[name] = worker.alloc_stacked_tensor(tensor)
        except Exception:
            for tensor in device_weights.values():
                worker.free_stacked_tensor(tensor)
            raise
        return device_weights

    def _materialize_resident_weights(self) -> None:
        """Upload inherited weights once and release their parent-process Host references."""
        worker = self._shared_l3_worker()
        if self._stacked_device_weights is None:
            host_weights = self._stacked_host_weights
            if not host_weights:
                raise RuntimeError("DeepSeekV4 stacked Host weights are not retained")
            parent_host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_weights.values())
            with profile_span("DeepSeekV4ModelRunner.upload_resident_main_weights", cat="executor"):
                self._stacked_device_weights = self._upload_weight_group(worker, host_weights)
            self._stacked_host_weights = None
            logger.info(
                "DeepSeekV4 resident main weights uploaded; released_parent_host_bytes=%d",
                parent_host_bytes,
            )

        buffers = self._mtp_buffers
        if buffers is not None and self._mtp_device_weights is None:
            if not buffers.weights:
                raise RuntimeError("DeepSeekV4 MTP Host weights are not staged")
            parent_host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers.weights.values())
            with profile_span("DeepSeekV4ModelRunner.upload_resident_mtp_weights", cat="executor"):
                self._mtp_device_weights = self._upload_weight_group(worker, buffers.weights)
            buffers.weights.clear()
            logger.info(
                "DeepSeekV4 resident MTP weights uploaded; released_parent_host_bytes=%d",
                parent_host_bytes,
            )
        worker.release_inherited_host_tensor_refs()

    def _invalidate_resident_cache_tensors(self) -> None:
        """Free resident KV/compressor state so the next request starts clean."""
        worker = self._l3_worker
        if worker is None:
            self._l3_cache_tensor_keys.clear()
            return
        for key in tuple(self._l3_cache_tensor_keys):
            tensor = self._l3_static_tensors.pop(key, None)
            if tensor is not None:
                worker.free_stacked_tensor(tensor)
        self._l3_cache_tensor_keys.clear()

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            self._assert_l3_shared_buffers_preallocated()
            compiled_callables = self._compiled.l3_callables()
            if not compiled_callables:
                raise RuntimeError("DeepSeekV4 L3 callables are not compiled")
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            compiled = [callable_spec.compiled for callable_spec in compiled_callables]
            with profile_span(
                "DeepSeekV4ModelRunner.create_persistent_l3_worker",
                cat="executor",
                args={"callable_count": len(compiled)},
            ):
                worker = DistributedWorker(
                    compiled,
                    persistent=True,
                    inherited_host_tensors=self._inherited_host_weights(),
                )
            self._l3_worker = worker
        return worker

    def _inherited_host_weights(self) -> list[torch.Tensor]:
        """Return immutable main and MTP weights that must be visible at worker fork."""
        tensors = list(self._stacked_host_weights.values()) if self._stacked_host_weights else []
        if self._mtp_buffers is not None:
            tensors.extend(self._mtp_buffers.weights.values())
        return tensors

    def _ensure_decode_work_cache(self) -> DeepSeekV4LayerCache:
        cache = self._decode_work_cache
        if cache is not None:
            return cache
        self._ensure_shared_host_allocation_before_worker("decode work cache")
        layout = self._compiled.layout
        fwd_layers = DEEPSEEK_V4_FWD_NUM_LAYERS
        csa_layers = DEEPSEEK_V4_CSA_NUM_LAYERS
        hca_layers = DEEPSEEK_V4_HCA_NUM_LAYERS
        cache = DeepSeekV4LayerCache(
            kv_cache=self._shared_empty(
                (
                    layout.ranks,
                    fwd_layers * layout.decode_ori_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_kv_cache",
            ),
            cmp_kv=self._shared_empty(
                (
                    layout.ranks,
                    fwd_layers * layout.cmp_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_cmp_kv",
            ),
            idx_kv_cache=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.idx_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_IDX_HEAD_DIM,
                ),
                torch.int8,
                name="decode_work_idx_kv_cache",
            ),
            idx_kv_scale=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.idx_max_blocks,
                    layout.block_size,
                    1,
                    1,
                ),
                torch.float32,
                name="decode_work_idx_kv_scale",
            ),
            hca_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    hca_layers * layout.hca_state_max_blocks,
                    layout.c128_state_block_size,
                    DEEPSEEK_V4_HCA_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_hca_compress_state",
            ),
            csa_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.csa_state_max_blocks,
                    layout.c4_state_block_size,
                    DEEPSEEK_V4_CSA_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_csa_compress_state",
            ),
            csa_inner_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.csa_inner_state_max_blocks,
                    layout.c4_state_block_size,
                    DEEPSEEK_V4_CSA_INNER_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_csa_inner_compress_state",
            ),
        )
        for tensor in (
            cache.kv_cache,
            cache.cmp_kv,
            cache.idx_kv_cache,
            cache.idx_kv_scale,
            cache.hca_compress_state,
            cache.csa_compress_state,
            cache.csa_inner_compress_state,
        ):
            tensor.zero_()
        self._decode_work_cache = cache
        return cache

    def _require_decode_work_cache(self) -> DeepSeekV4LayerCache:
        if self._decode_work_cache is None:
            raise RuntimeError("DeepSeekV4 decode work cache was not allocated before the L3 worker started")
        return self._decode_work_cache

    def _materialize_decode_device_cache(self) -> DeepSeekV4DeviceCache:
        """Upload one shard per rank once and keep all cache pools worker-resident."""
        cache = self._decode_device_cache
        if cache is not None:
            return cache
        host = self._require_decode_work_cache()
        worker = self._shared_l3_worker()
        allocated: list[StackedDeviceTensor] = []

        def resident(tensor: torch.Tensor) -> StackedDeviceTensor:
            stacked = worker.alloc_stacked_tensor(
                tensor,
                worker_ids=range(self._compiled.layout.ranks),
            )
            allocated.append(stacked)
            return stacked

        try:
            cache = DeepSeekV4DeviceCache(
                kv_cache=resident(host.kv_cache),
                cmp_kv=resident(host.cmp_kv),
                idx_kv_cache=resident(host.idx_kv_cache),
                idx_kv_scale=resident(host.idx_kv_scale),
                hca_compress_state=resident(host.hca_compress_state),
                csa_compress_state=resident(host.csa_compress_state),
                csa_inner_compress_state=resident(host.csa_inner_compress_state),
            )
        except Exception:
            for tensor in allocated:
                worker.free_stacked_tensor(tensor)
            raise
        self._decode_device_cache = cache
        return cache

    def _materialize_mtp_device_kv_cache(self) -> StackedDeviceTensor | None:
        """Materialize the optional MTP cache once for both MTP kernels."""
        cache = self._mtp_device_kv_cache
        if cache is not None:
            return cache
        buffers = self._mtp_buffers
        if buffers is None:
            return None
        cache = self._shared_l3_worker().alloc_stacked_tensor(
            buffers.prefill_kv_cache,
            worker_ids=range(self._compiled.layout.ranks),
        )
        self._mtp_device_kv_cache = cache
        return cache

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return DeepSeekV4ModelRunner._share_cpu_tensor(tensor)

    def _reset_l3_worker(self) -> None:
        worker = self._l3_worker
        if worker is None:
            return
        try:
            worker.close()
        finally:
            self._l3_worker = None
            self._l3_static_tensors.clear()
            self._l3_cache_tensor_keys.clear()

    def close(self) -> None:
        worker = self._l3_worker
        try:
            if worker is not None:
                worker.close()
        finally:
            self._l3_worker = None
            self._decode_work_cache = None
            self._stacked_host_weights = None
            self._stacked_device_weights = None
            self._mtp_device_weights = None
            self._mtp_buffers = None
            self._global_weights = None
            self._decode_device_cache = None
            self._mtp_device_kv_cache = None
            self._decode_cache_block_ids.clear()
            self._mtp_request_states.clear()
            self._l3_static_tensors.clear()
            self._l3_cache_tensor_keys.clear()

    def _require_input_builder(self) -> DeepSeekV4InputBuilder:
        if self.input_builder is None:
            raise RuntimeError("DeepSeekV4 input builder is not initialized")
        return self.input_builder

    def _rank_stack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0).expand(self._compiled.layout.ranks, *tensor.shape).contiguous()

    def _rank_scatter(
        self,
        tensors: Sequence[torch.Tensor],
        ranks: Sequence[int],
    ) -> torch.Tensor:
        """Place distinct local tensors on their owning ranks, filling inactive ranks."""
        if not tensors or len(tensors) != len(ranks):
            raise ValueError("rank-scattered tensors and ranks must be non-empty and aligned")
        reference = tensors[0]
        if any(tensor.shape != reference.shape or tensor.dtype != reference.dtype for tensor in tensors):
            raise ValueError("rank-scattered tensors must have identical shapes and dtypes")
        result = reference.unsqueeze(0).expand(
            self._compiled.layout.ranks,
            *reference.shape,
        ).clone()
        for rank, tensor in zip(ranks, tensors, strict=True):
            result[int(rank)].copy_(tensor)
        return result.contiguous()

    def _rank_scatter_mappings(
        self,
        tensors: Sequence[torch.Tensor],
        ranks: Sequence[int],
    ) -> torch.Tensor:
        """Scatter cache-write mappings and disable every inactive rank."""
        if not tensors or len(tensors) != len(ranks):
            raise ValueError("rank-scattered mappings and ranks must be non-empty and aligned")
        reference = tensors[0]
        if any(tensor.shape != reference.shape or tensor.dtype != reference.dtype for tensor in tensors):
            raise ValueError("rank-scattered mappings must have identical shapes and dtypes")
        result = torch.full(
            (self._compiled.layout.ranks, *reference.shape),
            -1,
            dtype=reference.dtype,
        )
        for rank, tensor in zip(ranks, tensors, strict=True):
            result[int(rank)].copy_(tensor)
        return result.contiguous()

    def _prefill_kernel_tokens(self, actual_tokens: int) -> int:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")
        return self._compiled.layout.prefill_seq

    @staticmethod
    def _prefill_kernel_positions(
        positions: Sequence[int],
        *,
        kernel_tokens: int,
        max_seq_len: int,
    ) -> list[int]:
        if len(positions) <= 0:
            raise ValueError("positions must not be empty")
        if kernel_tokens < len(positions):
            raise ValueError("kernel_tokens must cover all active positions")
        start = int(positions[0])
        kernel_positions = list(range(start, start + kernel_tokens))
        if kernel_positions[-1] >= max_seq_len:
            raise ValueError(
                f"prefill static kernel position {kernel_positions[-1]} exceeds max_seq_len={max_seq_len}"
            )
        return kernel_positions

    @staticmethod
    def _padded_rows(values: torch.Tensor, length: int) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"values must be rank-2, got shape={tuple(values.shape)}")
        if values.shape[0] <= 0:
            raise ValueError("values must not be empty")
        if values.shape[0] > length:
            raise ValueError(f"values rows {values.shape[0]} exceed padded length {length}")
        out = torch.empty((length, values.shape[1]), dtype=values.dtype, device=values.device)
        out[: values.shape[0]].copy_(values)
        if values.shape[0] < length:
            pad_rows = torch.arange(values.shape[0], length, device=values.device) % values.shape[0]
            out[values.shape[0] :].copy_(values.index_select(0, pad_rows))
        return out

    @staticmethod
    def _padded_vector(values: torch.Tensor, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        if values.numel() <= 0:
            raise ValueError("values must not be empty")
        if values.numel() > length:
            raise ValueError(f"values length {values.numel()} exceeds padded length {length}")
        out = torch.empty((length,), dtype=dtype)
        out[: values.numel()] = values.to(dtype=dtype)
        if values.numel() < length:
            pad_rows = torch.arange(values.numel(), length) % values.numel()
            out[values.numel() :] = values.to(dtype=dtype).index_select(0, pad_rows)
        return out

    @staticmethod
    def _prefill_position_ids(positions: Sequence[int], length: int) -> torch.Tensor:
        if len(positions) <= 0:
            raise ValueError("positions must not be empty")
        if len(positions) > length:
            raise ValueError(f"positions length {len(positions)} exceeds padded length {length}")
        out = torch.arange(length, dtype=torch.int32)
        out[: len(positions)] = torch.tensor(tuple(int(pos) for pos in positions), dtype=torch.int32)
        return out

    @staticmethod
    def _pad_prefill_mapping(mapping: torch.Tensor, length: int) -> torch.Tensor:
        if mapping.ndim != 1:
            raise ValueError(f"prefill mapping must be rank-1, got shape={tuple(mapping.shape)}")
        if mapping.numel() > length:
            raise ValueError(f"prefill mapping length {mapping.numel()} exceeds padded length {length}")
        out = torch.full((length,), -1, dtype=mapping.dtype)
        out[: mapping.numel()].copy_(mapping.to(dtype=mapping.dtype))
        return out

    def _decode_assignment(self, batch: DecodeBatch) -> _DeepSeekV4DecodeAssignment:
        """Assign scheduler rows to the fixed rank-local decode tile."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if actual_batch <= 0:
            raise ValueError("DeepSeekV4 decode batch must not be empty")
        if len(batch.cache_partitions) != actual_batch:
            raise ValueError("DeepSeekV4 decode requires one cache partition per request")
        ranks = tuple(int(rank) for rank in batch.cache_partitions)
        if min(ranks) < 0 or max(ranks) >= layout.ranks:
            raise ValueError(f"DeepSeekV4 decode cache partitions must be in [0, {layout.ranks - 1}]")
        indices_by_rank: list[list[int]] = [[] for _ in range(layout.ranks)]
        local_rows = [0] * actual_batch
        for request_index, rank in enumerate(ranks):
            local_row = len(indices_by_rank[rank])
            if local_row >= layout.decode_batch:
                raise ValueError(
                    f"DeepSeekV4 rank {rank} decode batch exceeds local capacity {layout.decode_batch}"
                )
            local_rows[request_index] = local_row
            indices_by_rank[rank].append(request_index)
        return _DeepSeekV4DecodeAssignment(
            ranks=ranks,
            local_rows=tuple(local_rows),
            per_rank_counts=tuple(len(indices) for indices in indices_by_rank),
            indices_by_rank=tuple(tuple(indices) for indices in indices_by_rank),
        )

    def _autoregressive_decode_positions(
        self,
        batch: DecodeBatch,
        actual_batch: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the single current position for each autoregressive request."""
        decode_seq = self._compiled.layout.decode_seq
        positions = []
        for row in range(actual_batch):
            seq_len = int(batch.seq_lens[row].item())
            if seq_len < 1:
                raise ValueError("decode seq_lens must be positive")
            positions.append((seq_len - 1,) * decode_seq)
        return tuple(positions)

    def _autoregressive_decode_token_rows(
        self,
        token_ids: torch.Tensor,
        actual_batch: int,
    ) -> torch.Tensor:
        """Expand one current token per request to the fixed sequence width."""
        layout = self._compiled.layout
        token_ids = token_ids.detach().cpu().to(torch.long)
        if token_ids.ndim == 1:
            active = token_ids[:actual_batch].reshape(actual_batch, 1)
        else:
            active = token_ids[:actual_batch, :1]
        return active.expand(actual_batch, layout.decode_seq).clone()

    def _pad_decode_token_rows(
        self,
        active: torch.Tensor,
        actual_batch: int,
        *,
        vocab_size: int,
    ) -> torch.Tensor:
        """Pad explicit token rows to the fixed local decode batch."""
        layout = self._compiled.layout
        if active.ndim != 2 or active.shape[1] != layout.decode_seq:
            raise ValueError("decode token rows must have shape [requests, decode_seq]")
        if actual_batch <= 0 or active.shape[0] < actual_batch:
            raise ValueError("decode token rows must cover every active request")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rows = torch.empty(layout.decode_tokens, dtype=torch.long).reshape(
            layout.decode_batch,
            layout.decode_seq,
        )
        rows.copy_(active[0].expand(layout.decode_batch, layout.decode_seq))
        rows[:actual_batch].copy_(active[:actual_batch])
        return rows.reshape(layout.decode_tokens)

    def _decode_kv_seq_lens(self, seq_lens: torch.Tensor, actual_batch: int) -> torch.Tensor:
        layout = self._compiled.layout
        # The last written KV position is ``seq_len-1``, so the valid KV history
        # is exactly ``seq_len`` entries. (yangyaodong's "seq_len+1" was relative
        # to a seq_len = prompt length, which does not count the prefill token;
        # our seq_len already does.)
        active = seq_lens[:actual_batch].detach().cpu().to(torch.int32)
        return DeepSeekV4CacheMetadataBuilder.replicate_first_row(
            active.reshape(actual_batch, 1),
            actual_rows=actual_batch,
            kernel_rows=layout.decode_batch,
        ).reshape(layout.decode_batch)
