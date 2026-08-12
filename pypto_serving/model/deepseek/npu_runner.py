# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import math
import threading
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


DEEPSEEK_V4_RANKS = 8
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_VOCAB_SIZE = 129280
DEEPSEEK_V4_BLOCK_SIZE = 128
DEEPSEEK_V4_DECODE_BATCH = 4
DEEPSEEK_V4_DECODE_SEQ = 2
DEEPSEEK_V4_DECODE_TOKENS = DEEPSEEK_V4_DECODE_BATCH * DEEPSEEK_V4_DECODE_SEQ
DEEPSEEK_V4_MTP_DECODE_TOKENS = 8
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
# Layer-stacking counts for the packed all-layer decode_fwd kernel.
DEEPSEEK_V4_FWD_NUM_LAYERS = 43
DEEPSEEK_V4_CSA_NUM_LAYERS = 21
DEEPSEEK_V4_HCA_NUM_LAYERS = 20
DEEPSEEK_V4_LM_HEAD_TP_SIZE = 4
DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS = 8
DEEPSEEK_V4_SAMPLED_IDS_PAD = 8

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
    "embed_weight": False,
}
_MTP_RESIDENT_POLICY = {
    "freqs_cos": False,
    "freqs_sin": False,
    "lm_head_weight": False,
    "kv_cache": True,
}
_MTP_DECODE_RESIDENT_POLICY = {
    **_MTP_RESIDENT_POLICY,
    "embed_weight": False,
}


def build_deepseek_v4_cache_group_specs(
    num_hidden_layers: int,
    compress_ratios: Sequence[int] | None = None,
    *,
    decode_batch: int = DEEPSEEK_V4_DECODE_TOKENS,
) -> tuple[KVCacheGroupSpec, ...]:
    """Describe cache namespaces independently from their runtime pool sizes.

    ``max_blocks_per_seq`` is a per-request ring/table limit. The number of
    physical blocks in each rank-local pool is intentionally left unset: the
    runner fills available NPU memory after weights and persistent scratch have
    been materialized, then the engine scales every group to the reported
    runtime capacity.
    """
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
        extra_row_bytes: int = 0,
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
                # One scheduler-visible block owns a page in every layer of
                # this cache family. Keep the complete physical footprint here
                # so all heterogeneous pools share one accurate HBM budget.
                page_size_bytes=(
                    len(layers)
                    * block_size
                    * (row_width * element_bytes + extra_row_bytes)
                ),
                compress_ratio=compress_ratio,
            ),
            # Each request may address this many pages through its logical ring;
            # physical capacity across concurrent requests is runtime-sized.
            max_blocks_per_seq=blocks_per_request,
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
            element_bytes=1,
            row_width=DEEPSEEK_V4_IDX_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_IDX_MAX_BLOCKS,
            compress_ratio=4,
            # One float32 scale accompanies every int8 index-cache row.
            extra_row_bytes=4,
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


DEEPSEEK_V4_CACHE_GROUP_NAMES = (
    "ori",
    "cmp",
    "idx",
    "hca_state",
    "csa_state",
    "csa_inner_state",
)


def deepseek_v4_cache_blocks_for_slots(
    group_specs: Sequence[KVCacheGroupSpec],
    capacity_slots: int,
) -> dict[str, int]:
    """Return scheduler-visible blocks per rank for ``capacity_slots`` requests."""
    capacity_slots = int(capacity_slots)
    if capacity_slots <= 0:
        raise ValueError("DeepSeekV4 cache capacity_slots must be positive")
    specs = {spec.name: spec for spec in group_specs}
    missing = [name for name in DEEPSEEK_V4_CACHE_GROUP_NAMES if name not in specs]
    if missing:
        raise ValueError("missing DeepSeekV4 cache groups: " + ", ".join(missing))
    return {
        name: capacity_slots * specs[name].max_blocks_per_seq
        for name in DEEPSEEK_V4_CACHE_GROUP_NAMES
    }


def deepseek_v4_physical_cache_blocks(
    group_specs: Sequence[KVCacheGroupSpec],
    capacity_slots: int,
    *,
    scratch_blocks: int,
) -> dict[str, int]:
    """Return physical blocks per rank, including isolated padding pages."""
    if scratch_blocks <= 0:
        raise ValueError("DeepSeekV4 scratch_blocks must be positive")
    return {
        name: num_blocks + int(scratch_blocks)
        for name, num_blocks in deepseek_v4_cache_blocks_for_slots(
            group_specs,
            capacity_slots,
        ).items()
    }


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
    "embed_weight",
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
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "block_counts",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "lm_head_weight",
    "hidden_out",
    "logits",
    "sampled_ids",
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
    "embed_weight", "main_pre_hc_hidden", "tail_pre_hc_pool",
    "accepted_counts", "tail_slot_ids", "position_ids",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "ori_block_table",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "lm_head_weight", "hidden_out", "next_pre_hc_hidden", "logits", "sampled_ids",
    "logit_row_indices",
)

# The fused K=1 kernel owns persistent draft state, while the standalone MTP
# kernel used to build additional K>1 drafts keeps the original stateless ABI.
_FUSED_MTP_DECODE_TENSOR_ORDER = (
    *_MTP_DECODE_TENSOR_ORDER[:5],
    "state_generations", "state_tokens", "state_meta",
    *_MTP_DECODE_TENSOR_ORDER[5:],
)

_FUSED_MTP_SHARED_TENSORS = frozenset(
    {
        "embed_weight",
        "main_pre_hc_hidden",
        "freqs_cos",
        "freqs_sin",
        "ori_block_table",
        "lm_head_weight",
    }
)

_MTP_DEVICE_STATE_TOKEN_WIDTH = 2
_MTP_DEVICE_STATE_META_WIDTH = 4
_MTP_STATE_VALID = 0
_MTP_STATE_GENERATION = 1
_MTP_STATE_TAIL_POSITION = 2
_MTP_STATE_COMMITTED_COUNT = 3

@dataclass(frozen=True)
class DeepSeekV4CacheLayout:
    """Kernel table depths and fixed execution dimensions.

    Physical cache-pool sizes are runtime state on ``DeepSeekV4ModelRunner``;
    the ``*_max_blocks`` fields here only bound per-request metadata tables and
    provide compatibility defaults for shape-only callers.
    """

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
    """Select an aggressive 16-row decode tile for one MTP chunk.

    Autoregressive decode retains the established eight-row tile. MTP doubles
    the per-rank tile to 16 rows and groups them into power-of-two request-local
    sequences so target verification preserves twice the request capacity.
    Draft depths larger than seven use repeated S=8 target chunks.
    """
    num_speculative_tokens = int(num_speculative_tokens)
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")
    decode_tokens = (
        DEEPSEEK_V4_DECODE_TOKENS
        if num_speculative_tokens == 0
        else DEEPSEEK_V4_MTP_DECODE_TOKENS
    )
    if num_speculative_tokens == 0:
        decode_seq = 1
    elif num_speculative_tokens == 1:
        decode_seq = 2
    elif num_speculative_tokens <= 3:
        decode_seq = 4
    else:
        decode_seq = 8
    return DeepSeekV4CacheLayout(
        decode_batch=decode_tokens // decode_seq,
        decode_seq=decode_seq,
        decode_tokens=decode_tokens,
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
        out: torch.Tensor | None = None,
        fill_all: bool = True,
    ) -> torch.Tensor:
        """Pack one autoregressive token per request into rank-local rows."""
        token_rows = embeddings.unsqueeze(1).expand(-1, self.layout.decode_seq, -1)
        return self._pack_decode_x_hc(
            token_rows,
            ranks=ranks,
            local_rows=local_rows,
            out=out,
            fill_all=fill_all,
        )

    def decode_token_rows_x_hc(
        self,
        embeddings: torch.Tensor,
        *,
        ranks: Sequence[int],
        local_rows: Sequence[int],
        out: torch.Tensor | None = None,
        fill_all: bool = True,
    ) -> torch.Tensor:
        """Pack explicit request-local token rows for target verification."""
        return self._pack_decode_x_hc(
            embeddings,
            ranks=ranks,
            local_rows=local_rows,
            out=out,
            fill_all=fill_all,
        )

    def _pack_decode_x_hc(
        self,
        token_rows: torch.Tensor,
        *,
        ranks: Sequence[int],
        local_rows: Sequence[int],
        out: torch.Tensor | None,
        fill_all: bool,
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
        if out is None:
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

        expected_shape = (
            self.layout.ranks,
            self.layout.decode_tokens,
            self.layout.hc_mult,
            self.hidden_size,
        )
        if tuple(out.shape) != expected_shape or out.dtype != torch.float32:
            raise ValueError(
                "decode x_hc output must have "
                f"shape={expected_shape} dtype=torch.float32, "
                f"got shape={tuple(out.shape)} dtype={out.dtype}"
            )
        if out.device.type != "cpu":
            raise ValueError("decode x_hc shared output must be on CPU")

        fallback = token_rows[0, -1]
        fill_tokens = self.layout.decode_tokens
        if not fill_all:
            fill_tokens = (max(int(row) for row in local_rows) + 1) * self.layout.decode_seq
        out[:, :fill_tokens].copy_(
            fallback.reshape(1, 1, 1, -1).expand(
                self.layout.ranks,
                fill_tokens,
                self.layout.hc_mult,
                self.hidden_size,
            )
        )
        for index, (rank, local_row) in enumerate(zip(ranks, local_rows, strict=True)):
            rank = int(rank)
            local_row = int(local_row)
            if not 0 <= rank < self.layout.ranks:
                raise ValueError(f"decode rank {rank} is out of range")
            if not 0 <= local_row < self.layout.decode_batch:
                raise ValueError(f"decode local row {local_row} is out of range")
            start = local_row * self.layout.decode_seq
            out[rank, start : start + self.layout.decode_seq].copy_(
                token_rows[index]
                .unsqueeze(1)
                .expand(self.layout.decode_seq, self.layout.hc_mult, self.hidden_size)
            )
        return out

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


# Compiled HOST-dispatched program. Unified with qwen's _L3Callable in
# pypto_serving.model.common.compiler.l3_callable.
from pypto_serving.model.common.compiler.l3_callable import L3Callable as DeepSeekV4L3Callable


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
    prepacked_layer_weights: DeepSeekV4StackedLayerWeights | None = None
    runtime_model: RuntimeModel | None = None
    prefill: DeepSeekV4L3Callable | None = None
    decode: DeepSeekV4L3Callable | None = None
    mtp_prefill: DeepSeekV4L3Callable | None = None
    mtp_decode: DeepSeekV4L3Callable | None = None
    freqs_cos: torch.Tensor | None = None
    freqs_sin: torch.Tensor | None = None
    platform: str = "a2a3"
    device_id: int = 0
    device_ids: tuple[int, ...] = ()
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
    """Fixed-shape shared tensors derived from one decode scheduler batch.

    The tensor fields are views over one runner-owned execution slot and remain
    valid until that slot is reused.
    """

    request_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    local_rows: tuple[int, ...]
    per_rank_counts: tuple[int, ...]
    actual_batch: int
    x_hc: torch.Tensor | None
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    block_counts: torch.Tensor
    block_ids_by_group: tuple[dict[str, tuple[int, ...]], ...]
    num_tokens_per_owner: torch.Tensor
    # Main-model LM-head routing.  Each non-negative entry is a flattened
    # decode hidden-row index whose logits must be computed; -1 disables that
    # output row.  For MTP S=2, active request k normally contributes rows
    # 2*k and 2*k+1 so the verifier can compare both main-model predictions.
    logit_row_indices: torch.Tensor
    buffer_slot: int = 0
    # Per-batch-row mapping to the request's stable device-state/tail-pool
    # slot.  Unlike local_row, this slot survives batch reorder and compaction;
    # -1 marks padding or an inactive row.
    mtp_tail_slot_ids: torch.Tensor | None = None
    # Expected allocation generation for mtp_tail_slot_ids.  A kernel may use
    # a slot only when this value matches state_meta[slot].generation, which
    # prevents a stale prepared step from touching a slot recycled to another
    # request (the classic ABA reuse case).
    mtp_state_generations: torch.Tensor | None = None
    # MTP-draft LM-head routing.  There is one next-draft prediction per active
    # request, selected from the last row of its committed S=2 window; unused
    # entries are -1.
    mtp_logit_row_indices: torch.Tensor | None = None
    # Fully bound L3 arguments for the steady fused-MTP path.  Preparing this
    # tuple on the command lane keeps Python argument binding off the device
    # lane, including the first decode prepared while terminal prefill runs.
    dispatch_args: tuple[Any, ...] | None = None


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
    pre_hc_hidden: torch.Tensor | StackedDeviceTensor
    logits: torch.Tensor
    sampled_ids: torch.Tensor


@dataclass(frozen=True)
class _DeepSeekV4PendingMtpDecode:
    """Completed device dispatch whose small host outputs await reclaim."""

    inputs: DeepSeekV4PreparedDecodeInputs
    sampled_ids: torch.Tensor
    accepted_counts: torch.Tensor
    mtp_sampled_ids: torch.Tensor
    committed_input_ids: torch.Tensor
    committed_position_ids: torch.Tensor
    states: tuple["_DeepSeekV4MtpRequestState", ...]


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

    pre_hc_hidden_out: torch.Tensor
    x_out: torch.Tensor
    sampled_ids: torch.Tensor
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
    decode_input_ids: torch.Tensor
    decode_position_ids: torch.Tensor
    decode_accepted_counts: torch.Tensor
    decode_tail_slot_ids: torch.Tensor
    decode_tail_token_ids: torch.Tensor
    decode_tail_positions: torch.Tensor
    decode_state_generations: torch.Tensor
    state_init_tokens: torch.Tensor
    state_init_meta: torch.Tensor
    tail_init_hidden: torch.Tensor
    decode_kv_cache: torch.Tensor
    prefill_hidden_out: torch.Tensor
    prefill_pre_hc_out: torch.Tensor
    prefill_logits: torch.Tensor
    prefill_logit_row_indices: torch.Tensor
    decode_hidden_out: torch.Tensor
    decode_pre_hc_out: torch.Tensor
    decode_logits: torch.Tensor
    decode_sampled_ids: torch.Tensor
    decode_logit_row_indices: torch.Tensor


@dataclass
class _DeepSeekV4MtpPrefillDeviceOutputs:
    """Worker-resident MTP prefill outputs shared across requests."""

    hidden_out: StackedDeviceTensor
    pre_hc_hidden_out: StackedDeviceTensor
    logits: StackedDeviceTensor


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
    tail_token_id: int | None = None
    tail_rank: int | None = None
    tail_slot_id: int | None = None
    tail_position: int | None = None
    generation: int = 0
    device_state_initialized: bool = False
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    # Running count of committed output tokens (advances by 1 or 2 per decode
    # step, unlike tail_position which stalls on draft rejection). Used to
    # correct seq_len under async scheduling.
    committed_count: int = 0
    # Prompt length, captured at initialization for seq_len correction.
    prompt_len: int = 0


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

    def __init__(
        self,
        *,
        compiled: DeepSeekV4CompiledKernels,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self.cache_metadata = DeepSeekV4CacheMetadataBuilder(layout=compiled.layout)
        self.input_builder: DeepSeekV4InputBuilder | None = None
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[
            tuple[int, tuple[int, ...], torch.dtype], StackedDeviceTensor
        ] = {}
        self._l3_cache_tensor_keys: set[tuple[int, tuple[int, ...], torch.dtype]] = set()
        self._cache_group_specs: tuple[KVCacheGroupSpec, ...] = ()
        self._cache_group_num_blocks: dict[str, int] = {}
        self._decode_device_cache: DeepSeekV4DeviceCache | None = None
        self._global_weights: DeepSeekV4GlobalWeights | None = None
        self._static_final_norm_weight: torch.Tensor | None = None
        self._static_lm_head_weight: torch.Tensor | None = None
        self._static_freqs_cos: torch.Tensor | None = None
        self._static_freqs_sin: torch.Tensor | None = None
        self._prefill_fwd_buffers: _DeepSeekV4PrefillFwdSharedBuffers | None = None
        self._decode_buffers: _DeepSeekV4DecodeSharedBuffers | None = None
        self._decode_input_slots: list[dict[str, torch.Tensor]] = []
        self._decode_static_metadata_keys: list[tuple[object, ...] | None] = [
            None
        ] * compiled.layout.ranks
        self._decode_assignment_cache_key: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        self._decode_assignment_cache: _DeepSeekV4DecodeAssignment | None = None
        self._decode_logit_row_range = torch.arange(
            compiled.layout.decode_tokens,
            dtype=torch.int32,
        )
        self._decode_fwd_args_cache: tuple[Any, ...] | None = None
        self._mtp_decode_args_cache: tuple[Any, ...] | None = None
        self._fused_mtp_main_args: tuple[Any, ...] | None = None
        self._fused_mtp_args_cache: dict[int, tuple[Any, ...]] = {}
        self._stacked_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._embedding_device_weight: StackedDeviceTensor | None = None
        self._main_pre_hc_device: StackedDeviceTensor | None = None
        self._mtp_tail_pre_hc_pool: StackedDeviceTensor | None = None
        self._mtp_device_state_tokens: StackedDeviceTensor | None = None
        self._mtp_device_state_meta: StackedDeviceTensor | None = None
        self._l3_shared_buffers_ready = False
        self._mtp_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._prefill_output_buffer: torch.Tensor | None = None
        self._prefill_pre_hc_output_buffer: torch.Tensor | None = None
        self._prefill_logits_buffer: torch.Tensor | None = None
        self._decode_logits_buffer: torch.Tensor | None = None
        self._mtp_buffers: _DeepSeekV4MtpSharedBuffers | None = None
        self._mtp_decode_inputs_initialized = False
        self._mtp_device_kv_cache: StackedDeviceTensor | None = None
        self._mtp_prefill_device_outputs: _DeepSeekV4MtpPrefillDeviceOutputs | None = None
        self._mtp_request_states: dict[str, _DeepSeekV4MtpRequestState] = {}
        self._mtp_state_lock = threading.RLock()
        self._mtp_free_tail_slots: list[list[int]] = [
            list(range(compiled.layout.decode_batch - 1, -1, -1))
            for _ in range(compiled.layout.ranks)
        ]
        self._mtp_slot_generations = [
            [0] * compiled.layout.decode_batch for _ in range(compiled.layout.ranks)
        ]
        self._mtp_proposed_tokens = 0
        self._mtp_accepted_tokens = 0
        if compiled.num_speculative_tokens:
            self._decode_flow = self._run_mtp_decode
            self._prefill_completion = self._capture_mtp_prefill_context
        else:
            self._decode_flow = self._run_autoregressive_decode
            self._prefill_completion = self._ignore_prefill_context

    @property
    def supports_async_decode_reclaim(self) -> bool:
        """Fused MTP owns stable device state and ping-ponged host outputs."""
        return self._compiled.num_speculative_tokens == 1

    @staticmethod
    def prepared_decode_requires_token(prepared: object) -> bool:
        """Fused MTP consumes persistent state initialized during prefill."""
        return not isinstance(prepared, DeepSeekV4PreparedDecodeInputs)

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Allocate heterogeneous cache pools from the post-weight HBM budget.

        The returned value is the scheduler-visible ``ori`` block count in one
        rank-local partition. Other group capacities are derived from the same
        number of maximum-sequence cache slots by ``KvCacheManager``.
        """
        self.input_builder = DeepSeekV4InputBuilder(
            layout=self._compiled.layout,
            hidden_size=config.hidden_size,
        )
        self._cache_group_specs = self._resolve_cache_group_specs(config, runtime)

        # Shape/metadata-only unit tests construct a runner without compiled
        # callables. Preserve a deterministic fixed fallback for that path; the
        # production executor always attaches the RuntimeModel and callables.
        model = self._compiled.runtime_model
        if model is None or not self._compiled.l3_callables():
            self._cache_group_num_blocks = self._fallback_cache_group_num_blocks()
            return self._cache_group_num_blocks["ori"]

        logger.info("[init_kv_cache] preparing DeepSeekV4 worker and resident weights …")
        self._ensure_l3_shared_buffers(model)

        requested_slots = self._compute_kv_cache_capacity_slots(runtime)
        allocated_slots = self._alloc_kv_cache_with_retry(requested_slots)
        self._log_kv_cache_allocation(runtime, requested_slots, allocated_slots)
        return self._cache_group_num_blocks["ori"]

    def _resolve_cache_group_specs(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
    ) -> tuple[KVCacheGroupSpec, ...]:
        specs = runtime.kv_cache_groups or build_deepseek_v4_cache_group_specs(
            config.num_hidden_layers,
            self._compiled.compress_ratios,
            decode_batch=self._compiled.layout.decode_batch,
        )
        names = tuple(spec.name for spec in specs)
        if names != DEEPSEEK_V4_CACHE_GROUP_NAMES:
            raise ValueError(
                "DeepSeekV4 KV cache groups must be ordered as "
                + ", ".join(DEEPSEEK_V4_CACHE_GROUP_NAMES)
                + f"; got {names}"
            )
        if any(spec.num_partitions != self._compiled.layout.ranks for spec in specs):
            raise ValueError(
                f"DeepSeekV4 KV cache groups must use {self._compiled.layout.ranks} partitions"
            )
        return tuple(specs)

    def _fallback_cache_group_num_blocks(self) -> dict[str, int]:
        """Return aligned fixed capacities for shape-only/non-runtime callers."""
        layout = self._compiled.layout
        physical_limits = {
            "ori": layout.decode_ori_max_blocks,
            "cmp": layout.cmp_max_blocks,
            "idx": layout.idx_max_blocks,
            "hca_state": layout.hca_state_max_blocks,
            "csa_state": layout.csa_state_max_blocks,
            "csa_inner_state": layout.csa_inner_state_max_blocks,
        }
        specs = {spec.name: spec for spec in self._cache_group_specs}
        capacity_slots = min(
            (physical_limits[name] - layout.decode_batch)
            // specs[name].max_blocks_per_seq
            for name in DEEPSEEK_V4_CACHE_GROUP_NAMES
        )
        if capacity_slots <= 0:
            raise RuntimeError(
                "DeepSeekV4 fixed fallback cache cannot hold one maximum-length "
                "sequence after reserving decode scratch blocks"
            )
        return deepseek_v4_cache_blocks_for_slots(
            self._cache_group_specs,
            capacity_slots,
        )

    def _compute_kv_cache_capacity_slots(self, runtime: RuntimeConfig) -> int:
        """Compute common cache slots using Qwen's utilization-budget formula."""
        ori_spec = self._cache_group_specs[0]
        if runtime.total_kv_pages is not None:
            requested_pages = int(runtime.total_kv_pages)
            if requested_pages < ori_spec.max_blocks_per_seq:
                raise ValueError(
                    "DeepSeekV4 total_kv_pages must hold at least one maximum ring: "
                    f"expected >= {ori_spec.max_blocks_per_seq}, got {requested_pages}"
                )
            return requested_pages // ori_spec.max_blocks_per_seq

        device_ids = self._compiled.device_ids or (self._compiled.device_id,)
        utilization = float(getattr(runtime, "npu_memory_utilization", 0.90))
        budgets = []
        memory_rows = []
        for device_id in device_ids:
            free_bytes, total_bytes = torch.npu.mem_get_info(f"npu:{device_id}")
            peak_non_kv = int(total_bytes) - int(free_bytes)
            budget = int(int(total_bytes) * utilization - peak_non_kv)
            budgets.append(budget)
            memory_rows.append((int(device_id), int(free_bytes), int(total_bytes), budget))

        bytes_per_slot = sum(
            spec.max_blocks_per_seq * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        scratch_bytes = sum(
            self._compiled.layout.decode_batch * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        # The optional MTP layer addresses the same ori IDs but owns a separate
        # one-layer BF16 pool.
        mtp_page_bytes = (
            self._compiled.layout.block_size * DEEPSEEK_V4_HEAD_DIM * torch.bfloat16.itemsize
            if self._compiled.num_speculative_tokens
            else 0
        )
        bytes_per_slot += ori_spec.max_blocks_per_seq * mtp_page_bytes
        scratch_bytes += self._compiled.layout.decode_batch * mtp_page_bytes

        kv_budget = min(budgets)
        minimum_bytes = scratch_bytes + bytes_per_slot
        if kv_budget < minimum_bytes:
            device_id, free_bytes, total_bytes, budget = min(
                memory_rows,
                key=lambda row: row[3],
            )
            raise RuntimeError(
                "DeepSeekV4 KV cache cannot fit one capacity slot within "
                f"npu_memory_utilization={utilization:.2f} on npu:{device_id}: "
                f"post-weight budget={budget} bytes, requires at least "
                f"{minimum_bytes} bytes ({scratch_bytes} scratch + "
                f"{bytes_per_slot} one slot); total={total_bytes} bytes, "
                f"free={free_bytes} bytes"
            )
        capacity_slots = (kv_budget - scratch_bytes) // bytes_per_slot
        logger.info(
            "DeepSeekV4 KV cache sizing: utilization=%.2f, limiting_budget=%.2f GB, "
            "slot=%.1f MB, scratch=%.1f MB, requested_slots=%d",
            utilization,
            kv_budget / 1e9,
            bytes_per_slot / 1e6,
            scratch_bytes / 1e6,
            capacity_slots,
        )
        for device_id, free_bytes, total_bytes, budget in memory_rows:
            logger.info(
                "  npu:%d total=%.2f GB free=%.2f GB post-weight KV budget=%.2f GB",
                device_id,
                total_bytes / 1e9,
                free_bytes / 1e9,
                budget / 1e9,
            )
        return int(capacity_slots)

    def _alloc_kv_cache_with_retry(self, requested_slots: int) -> int:
        """Allocate every cache family atomically, halving capacity on OOM."""
        capacity_slots = max(int(requested_slots), 1)
        while capacity_slots >= 1:
            self._cache_group_num_blocks = deepseek_v4_cache_blocks_for_slots(
                self._cache_group_specs,
                capacity_slots,
            )
            try:
                self._materialize_decode_device_cache()
                self._materialize_mtp_device_kv_cache()
                return capacity_slots
            except (RuntimeError, MemoryError) as exc:
                self._free_device_caches()
                if capacity_slots == 1:
                    raise RuntimeError(
                        "DeepSeekV4 KV cache allocation failed at the one-slot minimum"
                    ) from exc
                previous = capacity_slots
                capacity_slots = max(capacity_slots // 2, 1)
                logger.warning(
                    "DeepSeekV4 KV cache allocation failed (%s); retrying slots %d -> %d",
                    exc,
                    previous,
                    capacity_slots,
                )
        raise RuntimeError("DeepSeekV4 KV cache allocation failed")

    def _log_kv_cache_allocation(
        self,
        runtime: RuntimeConfig,
        requested_slots: int,
        allocated_slots: int,
    ) -> None:
        main_bytes = sum(
            self._physical_cache_num_blocks(spec.name) * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        mtp_bytes = 0
        if self._compiled.num_speculative_tokens:
            mtp_bytes = (
                self._physical_cache_num_blocks("ori")
                * self._compiled.layout.block_size
                * DEEPSEEK_V4_HEAD_DIM
                * torch.bfloat16.itemsize
            )
        logger.info(
            "[init_kv_cache] allocated DeepSeekV4 cache: slots=%d (requested=%d), "
            "per-rank=%.2f GB, ori_blocks=%d, max_seq_len=%d",
            allocated_slots,
            requested_slots,
            (main_bytes + mtp_bytes) / 1e9,
            self._cache_group_num_blocks["ori"],
            runtime.max_seq_len,
        )

    def _physical_cache_num_blocks(self, group_name: str) -> int:
        try:
            return self._cache_group_num_blocks[group_name] + self._compiled.layout.decode_batch
        except KeyError as exc:
            raise RuntimeError("DeepSeekV4 KV cache capacity is not initialized") from exc

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Discard request-local MTP state for finished or preempted requests."""
        request_ids = tuple(request_ids)
        for request_id in request_ids:
            with self._mtp_state_lock:
                state = self._mtp_request_states.pop(request_id, None)
                if state is not None and state.tail_rank is not None and state.tail_slot_id is not None:
                    # Lifecycle release runs on the FIFO device lane. A future
                    # owner receives a new generation and overwrites the complete
                    # meta row before publishing its descriptor. The kernel also
                    # requires STATE_VALID and an exact generation match.
                    self._mtp_free_tail_slots[state.tail_rank].append(state.tail_slot_id)
            if state is None:
                continue
            if state.proposed_tokens:
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
            embed_weight = loaded.embed_weight.to(
                device="cpu",
                dtype=torch.bfloat16,
            ).contiguous()
            exact_weight = loaded.lm_head_weight[:, : loaded.lm_head_layout.vocab_per_rank, :].contiguous()
            self._global_weights = replace(
                loaded,
                embed_weight=embed_weight,
                lm_head_weight=exact_weight,
            )
            self._compiled.embedding_weight = embed_weight
        return self._global_weights

    def load_stacked_layer_weights(self) -> DeepSeekV4StackedLayerWeights:
        """Load and stack all hidden-layer weights for the packed decode_fwd kernel."""
        if self._compiled.prepacked_layer_weights is not None:
            return self._compiled.prepacked_layer_weights
        compress_ratios = tuple(int(layer.compress_ratio) for layer in self._compiled.layer_plan)
        return self._compiled.weight_store.load_stacked_layer_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=self._compiled.num_hash_layers,
            use_prepacked=False,
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
            (layout.ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS),
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

    def prepare_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Prepare an isolated decode metadata snapshot while the device is busy.

        Token IDs and exact speculative sequence lengths are placeholders. The
        fused MTP kernel replaces active rows from stable device-state slots.
        """
        assignment = self._decode_assignment(batch)
        fused_mtp = self._compiled.num_speculative_tokens == 1
        if fused_mtp:
            if not batch.allow_device_greedy_sampling:
                raise RuntimeError("DeepSeekV4 MTP decode currently requires greedy device sampling")
            for request_id, rank in zip(batch.request_ids, assignment.ranks, strict=True):
                self._reserve_mtp_request_state(request_id, rank)
        actual_batch = len(batch.request_ids)
        active_seq = self._compiled.layout.decode_seq if fused_mtp else 1
        if not fused_mtp and max(assignment.per_rank_counts) > 1:
            raise ValueError(
                "DeepSeekV4 non-MTP decode supports at most one request per DP rank"
            )
        # Exact positions depend on step N acceptance. Use valid filler values;
        # the fused kernel binds active rows from device state before main decode.
        positions = tuple((0,) * self._compiled.layout.decode_seq for _ in range(actual_batch))
        placeholder_rows = torch.zeros(
            (actual_batch, self._compiled.layout.decode_seq), dtype=torch.long
        )
        with profile_span("DeepSeekV4ModelRunner.decode.prepare_early", cat="executor"):
            prepared = self._prepare_decode_inputs(
                model,
                batch,
                assignment=assignment,
                active_seq=active_seq,
                positions=positions,
                token_rows=placeholder_rows,
                x_hc=None,
                buffer_slot=buffer_slot,
            )
            prepared = self._prepare_mtp_device_state_descriptors(prepared)
            if (
                fused_mtp
                and prepared.mtp_tail_slot_ids is not None
                and self._l3_shared_buffers_ready
            ):
                prepared = self._bind_prepared_mtp_dispatch(
                    prepared,
                    model.config.hidden_size,
                    model.config.vocab_size,
                )
            if fused_mtp and self._l3_shared_buffers_ready:
                if prepared.mtp_tail_slot_ids is None or prepared.dispatch_args is None:
                    raise RuntimeError("DeepSeekV4 MTP prepare did not produce a complete dispatch")
            return prepared

    def run_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> DecodeResult:
        """Attach stable state descriptors and execute an early-prepared slot."""
        if not isinstance(prepared, DeepSeekV4PreparedDecodeInputs):
            raise TypeError("DeepSeekV4 prepared decode has an unexpected type")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("prepared decode request order changed before execution")
        if tuple(int(rank) for rank in batch.cache_partitions) != prepared.ranks:
            raise ValueError("prepared decode cache partitions changed before execution")
        # Production preflight prepares every shared tensor before the L3
        # worker forks.  Keep a guarded late fallback for shape-only/unit-test
        # runners, but do not put the idempotent readiness walk on every steady
        # decode step.
        if not self._l3_shared_buffers_ready:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.ensure_ready_late",
                cat="executor",
            ):
                self._ensure_l3_shared_buffers(model)
        if self._compiled.num_speculative_tokens == 1:
            return self.reclaim_prepared_decode(
                self.dispatch_prepared_decode(model, batch, prepared)
            )
        return self._run_autoregressive_decode(model, batch, prepared=prepared)

    def dispatch_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> object:
        """Run the blocking PyPTO dispatch, leaving host reclaim to another lane."""
        if self._compiled.num_speculative_tokens != 1:
            raise RuntimeError("split decode reclaim requires fused DeepSeekV4 MTP")
        if not isinstance(prepared, DeepSeekV4PreparedDecodeInputs):
            raise TypeError("DeepSeekV4 prepared decode has an unexpected type")
        if prepared.dispatch_args is None:
            raise RuntimeError("DeepSeekV4 fused decode was not fully bound during prepare")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("prepared decode request order changed before execution")
        if tuple(int(rank) for rank in batch.cache_partitions) != prepared.ranks:
            raise ValueError("prepared decode cache partitions changed before execution")
        states = tuple(
            self._require_mtp_request_state(request_id)
            for request_id in prepared.request_ids
        )
        for request_id, state in zip(prepared.request_ids, states, strict=True):
            if not state.device_state_initialized:
                raise RuntimeError(
                    f"DeepSeekV4 MTP device state was not finalized during prefill for {request_id!r}"
                )
        return self._launch_prepared_mtp_decode(prepared, states)

    def reclaim_prepared_decode(self, pending: object) -> DecodeResult:
        """Read ping-ponged sampled outputs after a fused dispatch completes."""
        if not isinstance(pending, _DeepSeekV4PendingMtpDecode):
            raise TypeError("DeepSeekV4 pending decode has an unexpected type")
        return self._reclaim_mtp_decode(pending)

    def prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build inputs for the single-token autoregressive decode flow."""
        assignment = self._decode_assignment(batch)
        if self._compiled.layout.decode_seq != 1 and max(assignment.per_rank_counts) > 1:
            raise ValueError(
                "DeepSeekV4 non-MTP decode supports at most one request per DP rank; "
                "the fixed S=2 kernel can expose only one cache-safe active token per rank"
            )
        actual_batch = len(batch.request_ids)
        positions = self._autoregressive_decode_positions(batch, actual_batch)
        token_rows = self._autoregressive_decode_token_rows(batch.token_ids, actual_batch)
        with profile_span("DeepSeekV4ModelRunner.decode.prepare_metadata_sources", cat="executor"):
            return self._prepare_decode_inputs(
                model,
                batch,
                assignment=assignment,
                active_seq=1,
                positions=positions,
                token_rows=token_rows,
                x_hc=None,
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
        if active_width < layout.decode_seq and any(
            count > 1 for count in assignment.per_rank_counts
        ):
            raise ValueError(
                "partial MTP target chunks require at most one request per rank"
            )
        token_rows = token_rows.detach().cpu().to(torch.long)
        with profile_span("DeepSeekV4ModelRunner.decode.build_metadata", cat="executor"):
            prepared = self._prepare_decode_inputs(
                model,
                batch,
                assignment=assignment,
                active_seq=active_width,
                positions=position_rows,
                token_rows=token_rows,
                x_hc=None,
            )
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
        x_hc: torch.Tensor | None,
        buffer_slot: int = 0,
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

        self._ensure_decode_buffers(model.config.hidden_size)
        if not 0 <= buffer_slot < len(self._decode_input_slots):
            raise ValueError(f"decode buffer_slot must be 0 or 1, got {buffer_slot}")
        staged = self._decode_input_slots[buffer_slot]
        staged["num_tokens_per_owner"].zero_()
        staged["logit_row_indices"].fill_(-1)
        self._stage_decode_dynamic_inputs(
            staged,
            batch,
            assignment=assignment,
            positions=positions,
            token_rows=token_rows,
        )

        for rank, request_indices in enumerate(indices_by_rank):
            if request_indices:
                local_groups = [active_group_ids[index] for index in request_indices]
            else:
                # All ranks must enter the distributed program with the common
                # scalar num_tokens. This rank contributes filler rows whose
                # cache mappings cover otherwise-unowned scratch blocks.
                local_groups = []

            padded_group_ids = {}
            for name in DEEPSEEK_V4_CACHE_GROUP_NAMES:
                if local_groups:
                    padded_group_ids[name] = self._pad_group_block_ids(
                        [groups[name] for groups in local_groups],
                        group_name=name,
                        kernel_rows=layout.decode_batch,
                    )
                else:
                    padded_group_ids[name] = self._scratch_group_block_ids(
                        group_name=name,
                        kernel_rows=layout.decode_batch,
                    )
            static_values = {
                "block_table": self.cache_metadata.paged_ori_block_table_from_ids(
                    padded_group_ids["ori"]
                ),
                "cmp_block_table": self.cache_metadata.block_table_from_ids(
                    padded_group_ids["cmp"],
                    max_blocks=layout.cmp_max_blocks,
                ),
                "idx_block_table": self.cache_metadata.block_table_from_ids(
                    padded_group_ids["idx"],
                    max_blocks=layout.idx_max_blocks,
                ),
                "hca_compress_state_block_table": self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["hca_state"],
                    max_blocks=layout.prefill_hca_state_max_blocks,
                ),
                "csa_compress_state_block_table": self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["csa_state"],
                    max_blocks=layout.prefill_csa_state_max_blocks,
                ),
                "csa_inner_compress_state_block_table": (
                    self.cache_metadata.ring_block_table_from_ids(
                        padded_group_ids["csa_inner_state"],
                        max_blocks=layout.prefill_csa_inner_state_max_blocks,
                    )
                ),
                "block_counts": torch.tensor(
                    [
                        [
                            len(padded_group_ids[name][row])
                            for name in DEEPSEEK_V4_CACHE_GROUP_NAMES
                        ]
                        for row in range(layout.decode_batch)
                    ],
                    dtype=torch.int32,
                ),
            }
            for name, value in static_values.items():
                self._copy_shared(
                    staged[name][rank],
                    value,
                    name=f"decode_{name}_rank{rank}",
                )

        for rank, count in enumerate(per_rank_counts):
            row_count = count * active_seq
            if row_count > layout.decode_tokens:
                raise ValueError(
                    f"rank {rank} requires {row_count} logit rows, "
                    f"capacity is {layout.decode_tokens}"
                )
            staged["num_tokens_per_owner"][rank] = row_count
            if row_count:
                staged["logit_row_indices"][rank, :row_count].copy_(
                    torch.arange(row_count, dtype=torch.int32)
                )

        return DeepSeekV4PreparedDecodeInputs(
            request_ids=tuple(batch.request_ids),
            ranks=ranks,
            local_rows=tuple(local_rows),
            per_rank_counts=per_rank_counts,
            actual_batch=actual_batch,
            x_hc=x_hc,
            input_ids=staged["input_ids"],
            position_ids=staged["position_ids"],
            kv_seq_lens=staged["kv_seq_lens"],
            block_table=staged["block_table"],
            cmp_block_table=staged["cmp_block_table"],
            idx_block_table=staged["idx_block_table"],
            hca_compress_state_block_table=staged["hca_compress_state_block_table"],
            csa_compress_state_block_table=staged["csa_compress_state_block_table"],
            csa_inner_compress_state_block_table=staged[
                "csa_inner_compress_state_block_table"
            ],
            block_counts=staged["block_counts"],
            block_ids_by_group=active_group_ids,
            num_tokens_per_owner=staged["num_tokens_per_owner"],
            logit_row_indices=staged["logit_row_indices"],
            buffer_slot=buffer_slot,
        )

    def _stage_decode_dynamic_inputs(
        self,
        staged: dict[str, torch.Tensor],
        batch: DecodeBatch,
        *,
        assignment: _DeepSeekV4DecodeAssignment,
        positions: tuple[tuple[int, ...], ...],
        token_rows: torch.Tensor,
    ) -> None:
        """Write only token/position/length fields into one execution slot."""
        layout = self._compiled.layout
        input_rows = staged["input_ids"].view(
            layout.ranks, layout.decode_batch, layout.decode_seq
        )
        position_rows = staged["position_ids"].view(
            layout.ranks, layout.decode_batch, layout.decode_seq
        )
        first_tokens = token_rows[0]
        first_positions = torch.tensor(positions[0], dtype=torch.int32)
        input_rows.copy_(
            first_tokens.view(1, 1, layout.decode_seq).expand(
                layout.ranks, layout.decode_batch, layout.decode_seq
            )
        )
        position_rows.copy_(
            first_positions.view(1, 1, layout.decode_seq).expand(
                layout.ranks, layout.decode_batch, layout.decode_seq
            )
        )
        staged["kv_seq_lens"].fill_(int(batch.seq_lens[0].item()))
        for rank, request_indices in enumerate(assignment.indices_by_rank):
            if not request_indices:
                continue
            first_request = request_indices[0]
            first_local_positions = torch.tensor(positions[first_request], dtype=torch.int32)
            input_rows[rank].copy_(
                token_rows[first_request].view(1, layout.decode_seq).expand(
                    layout.decode_batch, layout.decode_seq
                )
            )
            position_rows[rank].copy_(
                first_local_positions.view(1, layout.decode_seq).expand(
                    layout.decode_batch, layout.decode_seq
                )
            )
            staged["kv_seq_lens"][rank].fill_(int(batch.seq_lens[first_request].item()))
            for local_row, request_index in enumerate(request_indices):
                input_rows[rank, local_row].copy_(token_rows[request_index])
                position_rows[rank, local_row].copy_(
                    torch.tensor(positions[request_index], dtype=torch.int32)
                )
                staged["kv_seq_lens"][rank, local_row] = batch.seq_lens[request_index]

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

    def _pad_group_block_ids(
        self,
        active_rows: Sequence[Sequence[int]],
        *,
        group_name: str,
        kernel_rows: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Pad inactive rows with pages at the tail of a dynamic cache pool."""
        if not active_rows or len(active_rows) > kernel_rows:
            raise ValueError("active grouped KV rows must fit the kernel batch")
        normalized = [tuple(int(block_id) for block_id in row) for row in active_rows]
        if any(not row for row in normalized):
            raise ValueError("active grouped KV rows must not be empty")
        used = [block_id for row in normalized for block_id in row]
        if len(used) != len(set(used)):
            raise ValueError("active grouped KV rows must not share physical blocks")
        scratch = self._scratch_group_block_ids(
            group_name=group_name,
            kernel_rows=kernel_rows,
        )
        try:
            allocator_blocks = self._cache_group_num_blocks[group_name]
        except KeyError as exc:
            raise ValueError(f"unknown DeepSeekV4 cache group: {group_name}") from exc
        if any(block_id < 0 or block_id >= allocator_blocks for block_id in used):
            raise ValueError(
                f"grouped KV block IDs must be in [0, {allocator_blocks}); "
                f"[{allocator_blocks}, {allocator_blocks + kernel_rows}) is reserved "
                "for kernel padding"
            )
        padded = list(normalized)
        # Attention and compressor cache writes are fixed-B and are not fully
        # gated by num_tokens. Give every inactive row a distinct reserved page
        # instead of mirroring a live request's metadata.
        padded.extend(scratch[: kernel_rows - len(normalized)])
        return tuple(padded)

    def _scratch_group_block_ids(
        self,
        *,
        group_name: str,
        kernel_rows: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return one isolated physical scratch page for every fixed kernel row."""
        if kernel_rows <= 0:
            raise ValueError("kernel_rows must be positive")
        try:
            first = self._cache_group_num_blocks[group_name]
        except KeyError as exc:
            raise ValueError(f"unknown DeepSeekV4 cache group: {group_name}") from exc
        if self._physical_cache_num_blocks(group_name) < first + kernel_rows:
            raise ValueError("cache pool must provide one scratch page per kernel row")
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

        logits = torch.stack(
            tuple(logits_buffer[rank, 0] for rank in inputs.ranks),
        ).float()
        return PrefillResult(last_hidden=None, logits=logits)

    def finalize_prefill(
        self,
        model: RuntimeModel,
        request_ids: Sequence[str],
        sampled_token_ids: Sequence[int],
    ) -> None:
        """Run terminal MTP prefill and publish persistent decode state."""
        if not self._compiled.num_speculative_tokens:
            return
        if len(request_ids) != len(sampled_token_ids):
            raise ValueError("MTP prefill requires one sampled token per request")
        with profile_span(
            "DeepSeekV4ModelRunner.prefill.mtp_initialize",
            cat="executor",
            args={"batch_size": len(request_ids)},
        ):
            for request_id, token_id in zip(request_ids, sampled_token_ids, strict=True):
                state = self._require_mtp_request_state(request_id)
                if state.draft_token_id is None:
                    self._initialize_mtp_draft(request_id, state, int(token_id))

    def run_decode(self, model, batch: DecodeBatch) -> DecodeResult:
        """Dispatch to the decode flow selected when the model was compiled."""
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        with profile_span("DeepSeekV4ModelRunner.decode.prepare", cat="executor"):
            self._ensure_l3_shared_buffers(model)
        return self._decode_flow(model, batch)

    def _run_autoregressive_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        prepared: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> DecodeResult:
        """Run the single-token autoregressive decode flow."""
        if prepared is None:
            inputs = self.prepare_decode_inputs(model, batch)
        else:
            assignment = _DeepSeekV4DecodeAssignment(
                ranks=prepared.ranks,
                local_rows=prepared.local_rows,
                per_rank_counts=prepared.per_rank_counts,
                indices_by_rank=self._indices_by_rank(prepared.ranks),
            )
            positions = self._autoregressive_decode_positions(batch, len(batch.request_ids))
            token_rows = self._autoregressive_decode_token_rows(
                batch.token_ids, len(batch.request_ids)
            )
            self._stage_decode_dynamic_inputs(
                self._decode_input_slots[prepared.buffer_slot],
                batch,
                assignment=assignment,
                positions=positions,
                token_rows=token_rows,
            )
            inputs = prepared
        output = self._execute_main_decode(
            model,
            inputs,
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
            raise ValueError(f"MTP drafts must have shape [batch, K], got {tuple(drafts.shape)}")

        current = (
            batch.token_ids[:actual_batch]
            .detach()
            .cpu()
            .to(torch.long)
            .reshape(actual_batch, -1)[:, 0]
        )
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
                    (active_rows, active_rows[:, -1:].expand(-1, width - real_width)),
                    dim=1,
                )
            position_rows = []
            for request_index in active:
                first_position = int(base_seq_lens[request_index]) - 1 + input_offset
                real_positions = tuple(first_position + offset for offset in range(real_width))
                position_rows.append(real_positions + (real_positions[-1],) * (width - real_width))

            continuing = []
            if real_width == width:
                waves = [list(range(len(active)))]
            else:
                active_batch = self._select_decode_batch_rows(batch, active)
                active_assignment = self._decode_assignment(active_batch)
                waves = [
                    [
                        request_indices[wave]
                        for request_indices in active_assignment.indices_by_rank
                        if wave < len(request_indices)
                    ]
                    for wave in range(max(active_assignment.per_rank_counts))
                ]

            for wave_local_indices in waves:
                wave_request_indices = [active[index] for index in wave_local_indices]
                wave_batch = self._select_decode_batch_rows(batch, wave_request_indices)
                wave_batch.seq_lens = torch.tensor(
                    [
                        int(base_seq_lens[request_index])
                        + input_offset
                        + real_width
                        - 1
                        for request_index in wave_request_indices
                    ],
                    dtype=torch.int32,
                )
                wave_position_rows = [position_rows[index] for index in wave_local_indices]
                output = self._execute_main_decode(
                    model,
                    self.prepare_mtp_target_inputs(
                        model,
                        wave_batch,
                        token_rows=active_rows[wave_local_indices],
                        positions=wave_position_rows,
                        active_width=real_width,
                    ),
                    active_seq=real_width,
                )

                for chunk_index, request_index in enumerate(wave_request_indices):
                    rank = output.inputs.ranks[chunk_index]
                    row_start = output.inputs.local_rows[chunk_index] * width
                    row_logits = output.logits[rank, row_start : row_start + real_width].float()
                    row_predictions = output.sampled_ids[
                        rank, row_start : row_start + real_width, 0
                    ].to(torch.long)
                    if first_logits[request_index] is None:
                        first_logits[request_index] = row_logits[0].clone()

                    rejected = False
                    for offset in range(real_width - 1):
                        predicted = int(row_predictions[offset])
                        accepted[request_index].append(predicted)
                        draft_index = input_offset + offset
                        if predicted != int(drafts[request_index, draft_index]):
                            tail_tokens[request_index] = predicted
                            tail_hidden[request_index] = self._copy_main_pre_hc_row(
                                output.pre_hc_hidden,
                                rank=rank,
                                row=row_start + offset,
                                hidden_size=model.config.hidden_size,
                            )
                            tail_positions[request_index] = (
                                wave_position_rows[chunk_index][offset] + 1
                            )
                            rejected = True
                            break
                    if rejected:
                        continue

                    last_offset = real_width - 1
                    predicted = int(row_predictions[last_offset])
                    previous_hidden = self._copy_main_pre_hc_row(
                        output.pre_hc_hidden,
                        rank=rank,
                        row=row_start + last_offset,
                        hidden_size=model.config.hidden_size,
                    )
                    position = wave_position_rows[chunk_index][last_offset] + 1
                    if input_offset + last_offset == drafts.shape[1]:
                        accepted[request_index].append(predicted)
                        tail_tokens[request_index] = predicted
                        tail_hidden[request_index] = previous_hidden
                        tail_positions[request_index] = position
                    else:
                        pending[request_index] = predicted, previous_hidden, position
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

    def _copy_main_pre_hc_row(
        self,
        source: torch.Tensor | StackedDeviceTensor,
        *,
        rank: int,
        row: int,
        hidden_size: int,
    ) -> torch.Tensor:
        """Copy one target-model pre-HC row needed by recurrent MTP."""
        if not isinstance(source, torch.Tensor):
            raise RuntimeError("MTP verification requires captured host pre-HC output")
        hidden = source[rank, row]
        expected_shape = (self._compiled.layout.hc_mult, int(hidden_size))
        if tuple(hidden.shape) != expected_shape:
            raise ValueError(
                f"captured pre-HC row must have shape {expected_shape}, got {tuple(hidden.shape)}"
            )
        return hidden.detach().cpu().clone()

    def _run_mtp_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        prepared: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> DecodeResult:
        """Run fused K=1 or synchronous arbitrary-depth MTP decode."""
        if self._compiled.num_speculative_tokens != 1:
            if prepared is not None:
                raise ValueError("arbitrary-depth MTP does not consume async prepared inputs")
            return self._run_chunked_mtp_decode(model, batch)
        pending = self._dispatch_mtp_decode(model, batch, prepared=prepared)
        return self._reclaim_mtp_decode(pending)

    def _dispatch_mtp_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        prepared: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> _DeepSeekV4PendingMtpDecode:
        """Prepare and run fused MTP device work for synchronous callers."""
        if self._compiled.num_speculative_tokens != 1:
            raise RuntimeError("split decode dispatch requires fused DeepSeekV4 K=1 MTP")
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError("DeepSeekV4 MTP decode currently requires greedy device sampling")
        if prepared is None:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.prepare_inputs_fallback",
                cat="executor",
            ):
                speculative_batch = self._device_state_placeholder_batch(batch)
                inputs = self.prepare_mtp_decode_inputs(model, speculative_batch)
        else:
            inputs = prepared
        if inputs.mtp_tail_slot_ids is None:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.bind_device_state_late",
                cat="executor",
            ):
                inputs = self._prepare_mtp_device_state_descriptors(inputs, require_ready=True)
        if getattr(inputs, "dispatch_args", None) is None:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.bind_fwd_args_late",
                cat="executor",
                args={"actual_tokens": max(inputs.per_rank_counts) * self._compiled.layout.decode_seq},
            ):
                inputs = self._bind_prepared_mtp_dispatch(
                    inputs,
                    model.config.hidden_size,
                    model.config.vocab_size,
                )
        if getattr(inputs, "dispatch_args", None) is None:
            raise RuntimeError("DeepSeekV4 fused decode arguments were not bound")
        return self.dispatch_prepared_decode(model, batch, inputs)

    def _launch_prepared_mtp_decode(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        states: tuple[_DeepSeekV4MtpRequestState, ...],
    ) -> _DeepSeekV4PendingMtpDecode:
        """Launch a command-lane-complete fused MTP snapshot."""
        if inputs.dispatch_args is None:
            raise RuntimeError("DeepSeekV4 fused decode snapshot is not ready to launch")
        active_tokens = max(inputs.per_rank_counts) * self._compiled.layout.decode_seq
        try:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.l3_dispatch",
                cat="executor",
                args={"actual_tokens": active_tokens, "fused_mtp": True},
            ):
                self._run_l3(self._require_decode_callable(), *inputs.dispatch_args)
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 fused main/MTP decode dispatch failed "
                f"(actual_batch={inputs.actual_batch}, ranks={inputs.ranks})"
            ) from exc
        staged = self._decode_input_slots[inputs.buffer_slot]
        return _DeepSeekV4PendingMtpDecode(
            inputs=inputs,
            sampled_ids=staged["sampled_ids"],
            accepted_counts=staged["mtp_accepted_counts"],
            mtp_sampled_ids=staged["mtp_sampled_ids"],
            committed_input_ids=staged["mtp_committed_input_ids"],
            committed_position_ids=staged["mtp_committed_position_ids"],
            states=states,
        )

    def _reclaim_mtp_decode(self, pending: _DeepSeekV4PendingMtpDecode) -> DecodeResult:
        """Convert one completed output slot into scheduler-visible tokens."""
        inputs = pending.inputs
        layout = self._compiled.layout
        decode_seq = layout.decode_seq
        with profile_span("DeepSeekV4ModelRunner.decode.mtp_reclaim", cat="executor"):
            accepted_counts_list = []
            accepted = []
            states = pending.states
            for state, rank, local_row in zip(
                states,
                inputs.ranks,
                inputs.local_rows,
                strict=True,
            ):
                row_start = local_row * decode_seq
                main_tokens = pending.sampled_ids[
                    rank,
                    row_start : row_start + decode_seq,
                    0,
                ].tolist()
                accepted_count = int(
                    pending.accepted_counts[rank, local_row].item()
                )
                accepted_counts_list.append(accepted_count)
                accepted.append([int(token) for token in main_tokens[:accepted_count]])
            accepted_counts = tuple(accepted_counts_list)
            self._mtp_proposed_tokens += inputs.actual_batch
            self._mtp_accepted_tokens += sum(count == decode_seq for count in accepted_counts)
            for state, tokens in zip(states, accepted, strict=True):
                state.proposed_tokens += 1
                state.accepted_tokens += int(len(tokens) == decode_seq)
                state.committed_count += len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "DeepSeekV4 MTP acceptance progress: accepted=%d proposed=%d rate=%.2f%%",
                    self._mtp_accepted_tokens,
                    self._mtp_proposed_tokens,
                    100.0 * self._mtp_accepted_tokens / self._mtp_proposed_tokens,
                )
        # The fused kernel has already committed draft/tail/position to the
        # stable device-state slot.  Host mirrors are diagnostics for the
        # legacy split-MTP path and must not delay the next steady dispatch.
        if logger.isEnabledFor(logging.DEBUG):
            for state, rank, local_row in zip(
                states,
                inputs.ranks,
                inputs.local_rows,
                strict=True,
            ):
                row_end = (local_row + 1) * decode_seq - 1
                state.draft_token_id = int(
                    pending.mtp_sampled_ids[rank, local_row, 0].item()
                )
                state.tail_token_id = int(pending.committed_input_ids[rank, row_end].item())
                state.tail_position = int(
                    pending.committed_position_ids[rank, row_end].item()
                )
        return DecodeResult(
            hidden_states=None,
            logits=None,
            accepted_token_ids=accepted,
        )

    def _run_chunked_mtp_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> DecodeResult:
        """Generate and verify the configured number of request-local MTP drafts."""
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError("DeepSeekV4 MTP decode currently requires greedy device sampling")
        batch = replace(batch, seq_lens=self._correct_mtp_seq_lens(batch))
        num_drafts = self._mtp_draft_count(model, batch)
        drafts = self._propose_mtp_tokens(model, batch, num_drafts=num_drafts)
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
            state.committed_count += len(tokens)
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
                model,
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
            inputs = prepared
        decode_buffers = self._require_decode_buffers()
        active_decode_tokens = max(inputs.per_rank_counts) * active_seq

        hidden_buffer = self._require_decode_output_buffer(model.config.hidden_size)
        pre_hc_hidden_buffer = self._materialize_main_pre_hc_device(model.config.hidden_size)
        logits_buffer = self._require_decode_logits_buffer(model.config.vocab_size)
        sampled_ids_buffer = self._decode_input_slots[inputs.buffer_slot]["sampled_ids"]
        # The active hidden/pre-HC rows are pl.Out tensors and the grouped LM
        # head overwrites every logits row, including rows selected with -1.
        # Pre-clearing these reusable buffers only adds host memory bandwidth.
        num_tokens = active_decode_tokens
        with profile_span(
            "DeepSeekV4ModelRunner.decode.prepare_fwd_args",
            cat="executor",
            args={"actual_tokens": num_tokens},
        ):
            args = self._decode_fwd_args(
                inputs,
                pre_hc_hidden_buffer,
                hidden_buffer,
                logits_buffer,
                sampled_ids_buffer,
            )
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

        captured_pre_hc: torch.Tensor | StackedDeviceTensor = pre_hc_hidden_buffer
        if self._compiled.mtp_decode is not None:
            # Arbitrary-depth verification needs selected target pre-HC rows on
            # the host, while recurrent MTP still consumes the device-resident
            # output. Read back complete shards from their allocation base; a
            # raw D2H copy from an offset device pointer is not supported by the
            # distributed control path.
            self._shared_l3_worker().copy_stacked_from(
                pre_hc_hidden_buffer,
                decode_buffers.pre_hc_hidden_out,
            )
            captured_pre_hc = decode_buffers.pre_hc_hidden_out

        return _DeepSeekV4MainDecodeOutput(
            inputs=inputs,
            hidden=hidden_buffer,
            pre_hc_hidden=captured_pre_hc,
            logits=logits_buffer,
            sampled_ids=sampled_ids_buffer,
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
            state = self._reserve_mtp_request_state(request_id, rank)
            state.prefill_context = _DeepSeekV4MtpPrefillContext(
                rank=rank,
                actual_tokens=actual_tokens,
                hidden_states=inputs.x_hc[rank, :, 0].detach().cpu().clone(),
                prev_hidden_states=pre_hc_hidden[rank].detach().cpu().clone(),
                input_ids=inputs.input_ids[rank].detach().cpu().clone(),
                position_ids=inputs.position_ids[rank].detach().cpu().clone(),
                block_table=inputs.ori_block_table[rank].detach().cpu().clone(),
                slot_mapping=inputs.ori_slot_mapping[rank].detach().cpu().clone(),
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
        if self._l3_shared_buffers_ready:
            return
        with profile_span("DeepSeekV4ModelRunner.prepare.load_global_weights", cat="executor"):
            self.load_packed_global_weights()
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_rope_tables", cat="executor"):
            self._static_freqs_cos_tensor()
            self._static_freqs_sin_tensor()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_decode_buffers", cat="executor"):
            self._ensure_decode_buffers(model.config.hidden_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_mtp_buffers", cat="executor"):
            self._ensure_mtp_buffers(model.config.hidden_size)
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
                del stacked_weights
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_hc_head", cat="executor"):
            self._hc_head_tensors()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_prefill_fwd_buffers", cat="executor"):
            self._ensure_prefill_fwd_buffers(model.config.hidden_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.validate_shared_buffers", cat="executor"):
            self._assert_l3_shared_buffers_preallocated()
        with profile_span("DeepSeekV4ModelRunner.upload_resident_weights", cat="executor"):
            self._materialize_resident_weights()
        self._l3_shared_buffers_ready = True

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
        pre_hc_hidden_out: torch.Tensor | StackedDeviceTensor,
        hidden_out: torch.Tensor,
        logits: torch.Tensor,
        sampled_ids: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Build the single packed ``l3_decode_fwd`` argument tuple."""
        cache = self._materialize_decode_device_cache()
        stacked = self._require_stacked_weights()
        hc_head = self._hc_head_tensors()
        values = dict(stacked.tensors)
        values.update(
            {
                "embed_weight": self._materialize_embedding_device_weight(),
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": cache.kv_cache,
                "block_table": inputs.block_table,
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
                "block_counts": inputs.block_counts,
                "input_ids": inputs.input_ids,
                "hc_head_fn": hc_head["hc_head_fn"],
                "hc_head_scale": hc_head["hc_head_scale"],
                "hc_head_base": hc_head["hc_head_base"],
                "final_norm_w": self._static_final_norm_weight_tensor(),
                "pre_hc_hidden_out": pre_hc_hidden_out,
                "lm_head_weight": self._static_lm_head_weight_tensor(),
                "hidden_out": hidden_out,
                "logits": logits,
                "sampled_ids": sampled_ids,
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
        outputs = self._materialize_mtp_prefill_device_outputs(
            int(buffers.prefill_hidden_in.shape[-1])
        )
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
                "hidden_out": outputs.hidden_out,
                "pre_hc_hidden_out": outputs.pre_hc_hidden_out,
                "logits": outputs.logits,
                "logit_row_indices": buffers.prefill_logit_row_indices,
            }
        )
        values = self._mark_resident_args(values, _MTP_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _MTP_PREFILL_TENSOR_ORDER)

    def _mtp_decode_args(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs | None = None,
        *,
        include_device_state: bool = False,
    ) -> tuple[Any, ...]:
        buffer_slot = inputs.buffer_slot if inputs is not None else -1
        buffers = self._require_mtp_buffers()
        staged = self._decode_input_slots[buffer_slot] if inputs is not None else None
        tail_slot_ids = (
            inputs.mtp_tail_slot_ids
            if inputs is not None and inputs.mtp_tail_slot_ids is not None
            else buffers.decode_tail_slot_ids
        )
        logit_row_indices = (
            inputs.mtp_logit_row_indices
            if inputs is not None and inputs.mtp_logit_row_indices is not None
            else buffers.decode_logit_row_indices
        )
        kv_cache = self._materialize_mtp_device_kv_cache()
        if kv_cache is None:
            raise RuntimeError("DeepSeekV4 MTP KV cache is unavailable")
        values = dict(self._mtp_device_weights or buffers.weights)
        values.update(
            {
                "embed_weight": self._materialize_embedding_device_weight(),
                "main_pre_hc_hidden": self._materialize_main_pre_hc_device(
                    int(self.load_packed_global_weights().embed_weight.shape[1])
                ),
                "tail_pre_hc_pool": self._materialize_mtp_tail_pre_hc_pool(
                    int(self.load_packed_global_weights().embed_weight.shape[1])
                ),
                "accepted_counts": (
                    staged["mtp_accepted_counts"] if staged is not None else buffers.decode_accepted_counts
                ),
                "tail_slot_ids": tail_slot_ids,
                "position_ids": (
                    staged["mtp_committed_position_ids"]
                    if staged is not None
                    else buffers.decode_position_ids
                ),
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": kv_cache,
                "ori_block_table": (
                    inputs.block_table
                    if inputs is not None
                    else self._require_decode_buffers().tensors["block_table"]
                ),
                "input_ids": (
                    staged["mtp_committed_input_ids"]
                    if staged is not None
                    else buffers.decode_input_ids
                ),
                "lm_head_weight": self._static_lm_head_weight_tensor(),
                "hidden_out": buffers.decode_hidden_out,
                "next_pre_hc_hidden": buffers.decode_pre_hc_out,
                "logits": buffers.decode_logits,
                "sampled_ids": (
                    staged["mtp_sampled_ids"] if staged is not None else buffers.decode_sampled_ids
                ),
                "logit_row_indices": logit_row_indices,
            }
        )
        if include_device_state:
            values.update(
                {
                    "state_generations": (
                        inputs.mtp_state_generations
                        if inputs is not None and inputs.mtp_state_generations is not None
                        else buffers.decode_state_generations
                    ),
                    "state_tokens": self._materialize_mtp_device_state_tokens(),
                    "state_meta": self._materialize_mtp_device_state_meta(),
                }
            )
        values = self._mark_resident_args(values, _MTP_DECODE_RESIDENT_POLICY)
        tensor_order = (
            _FUSED_MTP_DECODE_TENSOR_ORDER
            if include_device_state
            else _MTP_DECODE_TENSOR_ORDER
        )
        return self._ordered_layer_args(values, tensor_order)

    def _fused_mtp_decode_args(
        self,
        main_args: tuple[Any, ...],
        inputs: DeepSeekV4PreparedDecodeInputs,
        active_tokens: int,
    ) -> tuple[Any, ...]:
        """Append the non-shared MTP arguments to the main decode arguments."""
        buffers = self._require_mtp_buffers()
        mtp_args = self._mtp_decode_args(inputs, include_device_state=True)
        fused_mtp_args = tuple(
            arg
            for name, arg in zip(_FUSED_MTP_DECODE_TENSOR_ORDER, mtp_args, strict=True)
            if name not in _FUSED_MTP_SHARED_TENSORS
        )
        return (
            *main_args,
            self._decode_input_slots[inputs.buffer_slot]["mtp_tail_token_ids"],
            self._decode_input_slots[inputs.buffer_slot]["mtp_tail_positions"],
            *fused_mtp_args,
            self._int32_scalar(active_tokens),
        )

    def _bind_prepared_mtp_dispatch(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        hidden_size: int,
        vocab_size: int,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Bind all steady L3 arguments on the prepare lane."""
        staged = self._decode_input_slots[inputs.buffer_slot]
        hidden_buffer = self._require_decode_output_buffer(hidden_size)
        pre_hc_hidden_buffer = self._materialize_main_pre_hc_device(hidden_size)
        logits_buffer = self._require_decode_logits_buffer(vocab_size)
        main_args = self._decode_fwd_args(
            inputs,
            pre_hc_hidden_buffer,
            hidden_buffer,
            logits_buffer,
            staged["sampled_ids"],
        )
        active_tokens = max(inputs.per_rank_counts) * self._compiled.layout.decode_seq
        return replace(
            inputs,
            dispatch_args=self._fused_mtp_decode_args(main_args, inputs, active_tokens),
        )

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

    def _main_speculative_batch(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        draft_token_ids: torch.Tensor,
    ) -> DecodeBatch:
        actual_batch = len(batch.request_ids)
        draft = draft_token_ids[:actual_batch].detach().cpu().to(torch.long)
        current = batch.token_ids[:actual_batch].detach().cpu().to(torch.long).reshape(-1)
        corrected_seq_lens = self._correct_mtp_seq_lens(batch)
        return replace(
            batch,
            token_ids=draft.reshape(actual_batch, 1),
            hidden_states=None,
            prev_token_ids=current,
            prev_hidden_states=None,
            seq_lens=corrected_seq_lens + 1,
        )

    def _device_state_placeholder_batch(self, batch: DecodeBatch) -> DecodeBatch:
        """Build valid fallback rows; the fused kernel replaces every active row."""
        actual_batch = len(batch.request_ids)
        current = batch.token_ids[:actual_batch].detach().cpu().to(torch.long).reshape(-1)
        for request_id in batch.request_ids:
            state = self._require_mtp_request_state(request_id)
            if not state.device_state_initialized:
                raise RuntimeError(f"DeepSeekV4 MTP device state is missing for {request_id!r}")
        return replace(
            batch,
            token_ids=current.reshape(actual_batch, 1),
            hidden_states=None,
            prev_token_ids=current,
            prev_hidden_states=None,
            seq_lens=torch.clamp(batch.seq_lens.detach().cpu().to(torch.int32), min=2),
        )

    def _correct_mtp_seq_lens(self, batch: DecodeBatch) -> torch.Tensor:
        """Return request lengths corrected from committed speculative output."""
        actual_batch = len(batch.request_ids)
        corrected = batch.seq_lens[:actual_batch].detach().cpu().to(torch.int32).clone()
        # Async scheduling reserves the maximum accepted width before the worker
        # knows which drafts match. After the first MTP step, the runner's
        # committed count is therefore the authoritative request length.
        for index, request_id in enumerate(batch.request_ids):
            state = self._mtp_request_states.get(request_id)
            if state is not None and state.proposed_tokens > 0:
                corrected[index] = state.prompt_len + state.committed_count + 1
        return corrected

    def _require_mtp_request_state(self, request_id: str) -> _DeepSeekV4MtpRequestState:
        with self._mtp_state_lock:
            state = self._mtp_request_states.get(request_id)
        if state is None:
            raise RuntimeError(f"DeepSeekV4 MTP state is missing for request {request_id!r}")
        return state

    def _reserve_mtp_request_state(
        self,
        request_id: str,
        rank: int,
    ) -> _DeepSeekV4MtpRequestState:
        """Publish a stable slot identity before terminal prefill completes."""
        rank = int(rank)
        with self._mtp_state_lock:
            state = self._mtp_request_states.get(request_id)
            if state is None:
                state = _DeepSeekV4MtpRequestState()
                self._mtp_request_states[request_id] = state
            self._reserve_mtp_state_slot(state, rank, request_id=request_id)
            return state

    def _reserve_mtp_state_slot(
        self,
        state: _DeepSeekV4MtpRequestState,
        rank: int,
        *,
        request_id: str,
    ) -> None:
        """Assign one host-visible slot identity without initializing its device contents."""
        rank = int(rank)
        with self._mtp_state_lock:
            if state.tail_rank is not None:
                if state.tail_rank != rank:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP state rank changed for {request_id!r}: "
                        f"slot rank={state.tail_rank}, decode rank={rank}"
                    )
                return
            free_slots = self._mtp_free_tail_slots[rank]
            if not free_slots:
                raise RuntimeError(f"DeepSeekV4 MTP tail slots are exhausted on rank {rank}")
            slot = free_slots.pop()
            generation = self._mtp_slot_generations[rank][slot] + 1
            if generation >= 2**31:
                generation = 1
            self._mtp_slot_generations[rank][slot] = generation
            state.tail_rank = rank
            state.tail_slot_id = slot
            state.generation = generation

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
        model: RuntimeModel,
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
                model,
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
        model: RuntimeModel,
        batch: DecodeBatch,
        token_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one recurrent MTP token per request using device-side inputs."""
        actual_batch = len(batch.request_ids)
        if token_ids.numel() != actual_batch or positions.numel() != actual_batch:
            raise ValueError("MTP token IDs and positions must align with active requests")
        if previous_hidden.shape[0] != actual_batch:
            raise ValueError("MTP recurrent hidden rows must align with active requests")

        buffers = self._require_mtp_buffers()
        assignment = self._decode_assignment(batch)
        token_ids = token_ids.detach().cpu().to(torch.long).reshape(actual_batch)
        positions = positions.detach().cpu().to(torch.int32).reshape(actual_batch)
        previous_hidden = previous_hidden.detach().cpu().to(torch.float32)
        next_tokens: list[int | None] = [None] * actual_batch
        next_hidden: list[torch.Tensor | None] = [None] * actual_batch
        wave_count = max(assignment.per_rank_counts)

        for wave in range(wave_count):
            wave_items = [
                (rank, request_indices[wave])
                for rank, request_indices in enumerate(assignment.indices_by_rank)
                if wave < len(request_indices)
            ]
            wave_ranks = tuple(rank for rank, _request_index in wave_items)
            wave_indices = tuple(request_index for _rank, request_index in wave_items)
            self._stage_recurrent_mtp_block_tables(
                batch,
                request_indices=wave_indices,
                ranks=wave_ranks,
            )

            fallback_index = wave_indices[0]
            buffers.decode_input_ids.fill_(int(token_ids[fallback_index]))
            buffers.decode_position_ids.fill_(int(positions[fallback_index]))
            buffers.decode_accepted_counts.fill_(1)
            buffers.decode_tail_slot_ids.fill_(-1)
            buffers.decode_logit_row_indices.fill_(-1)
            for rank, request_index in wave_items:
                state = self._require_mtp_request_state(batch.request_ids[request_index])
                self._write_mtp_tail_hidden(
                    state,
                    rank,
                    previous_hidden[request_index],
                )
                buffers.decode_input_ids[rank, 0] = int(token_ids[request_index])
                buffers.decode_position_ids[rank, 0] = int(positions[request_index])
                buffers.decode_tail_slot_ids[rank, 0] = int(state.tail_slot_id)
                buffers.decode_logit_row_indices[rank, 0] = 0

            with profile_span(
                "DeepSeekV4ModelRunner.mtp.decode.l3_dispatch",
                cat="executor",
                args={"actual_tokens": 1},
            ):
                self._run_l3(
                    self._require_mtp_decode_callable(),
                    *self._mtp_decode_args(),
                    self._int32_scalar(1),
                )

            for rank, request_index in wave_items:
                next_tokens[request_index] = int(
                    buffers.decode_sampled_ids[rank, 0, 0].item()
                )
                next_hidden[request_index] = (
                    buffers.decode_pre_hc_out[rank, 0].detach().cpu().clone()
                )

        if any(token is None for token in next_tokens) or any(
            hidden is None for hidden in next_hidden
        ):
            raise RuntimeError("recurrent MTP decoding left incomplete request state")
        self._mtp_decode_inputs_initialized = True
        return (
            torch.tensor(next_tokens, dtype=torch.long),
            torch.stack(next_hidden),
        )

    def _stage_recurrent_mtp_block_tables(
        self,
        batch: DecodeBatch,
        *,
        request_indices: Sequence[int],
        ranks: Sequence[int],
    ) -> None:
        """Stage only the ori tables consumed by one recurrent MTP wave."""
        if not request_indices or len(request_indices) != len(ranks):
            raise ValueError("recurrent MTP requests and ranks must be non-empty and aligned")
        if len(set(int(rank) for rank in ranks)) != len(ranks):
            raise ValueError("recurrent MTP waves support at most one request per rank")

        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if len(batch.block_ids_by_group) != actual_batch:
            raise ValueError(
                "grouped KV metadata has "
                f"{len(batch.block_ids_by_group)} rows, expected decode batch {actual_batch}"
            )
        requests_by_rank = {
            int(rank): int(request_index)
            for rank, request_index in zip(ranks, request_indices, strict=True)
        }
        if any(not 0 <= rank < layout.ranks for rank in requests_by_rank):
            raise ValueError("recurrent MTP rank is outside the compiled world")
        if any(not 0 <= index < actual_batch for index in requests_by_rank.values()):
            raise ValueError("recurrent MTP request index is outside the active batch")
        active_ori_ids = {}
        for request_index in requests_by_rank.values():
            ori_ids = tuple(
                int(block_id)
                for block_id in batch.block_ids_by_group[request_index].get("ori", ())
            )
            if not ori_ids:
                raise ValueError(f"decode row {request_index} is missing grouped KV blocks: ori")
            active_ori_ids[request_index] = ori_ids

        block_tables = self._require_decode_buffers().tensors["block_table"]
        for rank in range(layout.ranks):
            request_index = requests_by_rank.get(rank)
            if request_index is None:
                cache_key = (("mtp_recurrent_scratch",),)
                if self._decode_static_metadata_keys[rank] == cache_key:
                    continue
                padded_ids = self._scratch_group_block_ids(
                    group_name="ori",
                    kernel_rows=layout.decode_batch,
                )
            else:
                active_ids = active_ori_ids[request_index]
                cache_key = (("mtp_recurrent_ori", active_ids),)
                if self._decode_static_metadata_keys[rank] == cache_key:
                    continue
                padded_ids = self._pad_group_block_ids(
                    [active_ids],
                    group_name="ori",
                    kernel_rows=layout.decode_batch,
                )
            self._copy_shared(
                block_tables[rank],
                self.cache_metadata.paged_ori_block_table_from_ids(padded_ids),
                name=f"mtp_recurrent_block_table_rank{rank}",
            )
            # Mark the other fixed decode metadata stale. The next main-model
            # decode will rebuild its full static set before dispatch.
            self._decode_static_metadata_keys[rank] = cache_key

    def _initialize_mtp_drafts(self, batch: DecodeBatch) -> None:
        """Initialize every request's first draft without sharing mutable state."""
        for batch_index, request_id in enumerate(batch.request_ids):
            state = self._require_mtp_request_state(request_id)
            if state.draft_token_id is None:
                first_token_id = int(
                    batch.token_ids[batch_index].detach().cpu().to(torch.long).reshape(-1)[0]
                )
                self._initialize_mtp_draft(request_id, state, first_token_id)

    def _initialize_mtp_draft(
        self,
        request_id: str,
        state: _DeepSeekV4MtpRequestState,
        first_token_id: int,
    ) -> None:
        context = state.prefill_context
        if context is None:
            raise RuntimeError(f"DeepSeekV4 MTP prefill context is missing for {request_id!r}")
        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        n = context.actual_tokens
        owner_rank = context.rank
        first_token = torch.tensor([first_token_id], dtype=torch.long)
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
            with profile_span(
                "DeepSeekV4ModelRunner.mtp.prefill.read_logits",
                cat="executor",
                args={"bytes": DEEPSEEK_V4_VOCAB_SIZE * torch.float32.itemsize},
            ):
                logits = self._read_mtp_prefill_logits(owner_rank)
        state.draft_token_id = int(logits.argmax().item())
        state.tail_token_id = int(first_token[0].item())
        self._write_mtp_tail_hidden(
            state,
            owner_rank,
            context.prev_hidden_states[n - 1],
        )
        state.tail_position = int(context.position_ids[n - 1].item())
        self._initialize_mtp_device_state(state)
        state.prompt_len = n
        state.draft_pre_hc_hidden = (
            buffers.prefill_pre_hc_out[owner_rank, n - 1].detach().cpu().clone()
        )
        state.draft_position = int(context.position_ids[n - 1].item()) + 1
        state.prefill_context = None

    def _write_mtp_tail_hidden(
        self,
        state: _DeepSeekV4MtpRequestState,
        rank: int,
        hidden: torch.Tensor,
    ) -> None:
        """Initialize the preallocated persistent tail slot contents."""
        self._reserve_mtp_state_slot(state, rank, request_id="<prefill>")
        if state.tail_rank != rank or state.tail_slot_id is None:
            raise RuntimeError("DeepSeekV4 MTP tail slot reservation is inconsistent")
        slot = state.tail_slot_id
        buffers = self._require_mtp_buffers()
        staged = buffers.tail_init_hidden[rank, slot]
        staged.copy_(hidden.to(dtype=torch.float32, device="cpu"))
        pool = self._materialize_mtp_tail_pre_hc_pool(int(staged.shape[-1]))
        shard = pool.shards[rank]
        row_nbytes = staged.numel() * staged.element_size()
        dst = shard.data_ptr + slot * row_nbytes
        self._shared_l3_worker().copy_to(
            dst,
            staged.data_ptr(),
            row_nbytes,
            worker_id=rank,
        )

    def _initialize_mtp_device_state(self, state: _DeepSeekV4MtpRequestState) -> None:
        """Seed one stable device slot after the first MTP draft is available."""
        if state.device_state_initialized:
            return
        if (
            state.tail_rank is None
            or state.tail_slot_id is None
            or state.tail_token_id is None
            or state.draft_token_id is None
            or state.tail_position is None
        ):
            raise RuntimeError("DeepSeekV4 MTP device state cannot be initialized from partial state")
        buffers = self._require_mtp_buffers()
        rank = state.tail_rank
        slot = state.tail_slot_id
        token_row = buffers.state_init_tokens[rank, slot]
        token_row[0] = state.tail_token_id
        token_row[1] = state.draft_token_id
        meta_row = buffers.state_init_meta[rank, slot]
        meta_row.zero_()
        # Publish a completely initialized payload.  STATE_VALID is an
        # initialization guard; slot reuse safety comes from generation.
        meta_row[_MTP_STATE_VALID] = 1
        meta_row[_MTP_STATE_GENERATION] = state.generation
        meta_row[_MTP_STATE_TAIL_POSITION] = state.tail_position
        meta_row[_MTP_STATE_COMMITTED_COUNT] = state.committed_count
        worker = self._shared_l3_worker()
        for device_tensor, source in (
            (self._materialize_mtp_device_state_tokens(), token_row),
            (self._materialize_mtp_device_state_meta(), meta_row),
        ):
            shard = device_tensor.shards[rank]
            row_nbytes = source.numel() * source.element_size()
            worker.copy_to(
                shard.data_ptr + slot * row_nbytes,
                source.data_ptr(),
                row_nbytes,
                worker_id=rank,
            )
        state.device_state_initialized = True

    def _mtp_committed_window(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        main_ids: torch.Tensor,
        *,
        request_index: int,
        accepted_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one request's fixed MTP window from committed outputs only."""
        layout = self._compiled.layout
        request_id = inputs.request_ids[request_index]
        state = self._require_mtp_request_state(request_id)
        owner_rank = inputs.ranks[request_index]
        row_start = inputs.local_rows[request_index] * layout.decode_seq
        row_slice = slice(row_start, row_start + layout.decode_seq)
        if accepted_count == layout.decode_seq:
            return (
                main_ids[request_index, :layout.decode_seq].detach().cpu().to(torch.long),
                inputs.position_ids[owner_rank, row_slice].detach().cpu().to(torch.int32),
            )
        if accepted_count != 1 or layout.decode_seq != 2:
            raise ValueError(
                "DeepSeekV4 MTP currently supports next_n=1 acceptance only; "
                f"got accepted_count={accepted_count}, decode_seq={layout.decode_seq}"
            )
        if (
            state.tail_token_id is None
            or state.tail_slot_id is None
            or state.tail_position is None
        ):
            raise RuntimeError(f"DeepSeekV4 MTP committed tail is not initialized for {request_id!r}")
        return (
            torch.tensor(
                (state.tail_token_id, int(main_ids[request_index, 0].item())),
                dtype=torch.long,
            ),
            torch.tensor(
                (state.tail_position, int(inputs.position_ids[owner_rank, row_start].item())),
                dtype=torch.int32,
            ),
        )

    def _advance_mtp_drafts(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        main_ids: torch.Tensor,
        *,
        accepted_counts: Sequence[int],
    ) -> None:
        """Run one packed MTP decode and scatter new draft state by request ID."""
        if len(accepted_counts) != inputs.actual_batch:
            raise ValueError("MTP accepted counts must align with active requests")
        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        with profile_span("DeepSeekV4ModelRunner.mtp.commit_windows", cat="executor"):
            committed = [
                self._mtp_committed_window(
                    inputs,
                    main_ids,
                    request_index=index,
                    accepted_count=int(accepted_counts[index]),
                )
                for index in range(inputs.actual_batch)
            ]
        active_tokens = self._stage_mtp_decode_inputs(inputs, committed, accepted_counts)
        # The MTP pl.Out tensors follow the same overwrite contract as the main
        # decode outputs, so no host-side clear is required before dispatch.
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
        for request_index, request_id in enumerate(inputs.request_ids):
            state = self._require_mtp_request_state(request_id)
            committed_ids, committed_positions = committed[request_index]
            state.draft_token_id = int(
                buffers.decode_sampled_ids[
                    inputs.ranks[request_index],
                    inputs.local_rows[request_index],
                    0,
                ].item()
            )
            state.tail_token_id = int(committed_ids[-1].item())
            state.tail_position = int(committed_positions[-1].item())

    def _stage_mtp_decode_inputs(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        committed: Sequence[tuple[torch.Tensor, torch.Tensor]],
        accepted_counts: Sequence[int],
    ) -> int:
        """Stage compact committed windows into persistent MTP decode buffers."""
        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        fallback_ids, fallback_positions = committed[0]
        active_tokens = max(inputs.per_rank_counts) * layout.decode_seq
        fill_tokens = (
            layout.decode_tokens
            if not self._mtp_decode_inputs_initialized
            else active_tokens
        )
        fill_rows = fill_tokens // layout.decode_seq

        with profile_span("DeepSeekV4ModelRunner.mtp.pack_inputs", cat="executor"):
            fallback_id_rows = fallback_ids.repeat(fill_rows)
            fallback_position_rows = fallback_positions.repeat(fill_rows)
            buffers.decode_input_ids[:, :fill_tokens].copy_(
                fallback_id_rows.unsqueeze(0).expand(layout.ranks, -1)
            )
            buffers.decode_position_ids[:, :fill_tokens].copy_(
                fallback_position_rows.unsqueeze(0).expand(layout.ranks, -1)
            )
            buffers.decode_accepted_counts.fill_(1)
            buffers.decode_tail_slot_ids.fill_(-1)
            for request_index, (rank, local_row) in enumerate(
                zip(inputs.ranks, inputs.local_rows, strict=True)
            ):
                row_start = local_row * layout.decode_seq
                row_slice = slice(row_start, row_start + layout.decode_seq)
                ids, positions = committed[request_index]
                buffers.decode_input_ids[rank, row_slice].copy_(ids)
                buffers.decode_position_ids[rank, row_slice].copy_(positions)
                state = self._require_mtp_request_state(inputs.request_ids[request_index])
                if state.tail_slot_id is None:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP tail slot is missing for {inputs.request_ids[request_index]!r}"
                    )
                buffers.decode_accepted_counts[rank, local_row] = int(
                    accepted_counts[request_index]
                )
                buffers.decode_tail_slot_ids[rank, local_row] = state.tail_slot_id
        buffers.decode_logit_row_indices.fill_(-1)
        for rank, local_row in zip(inputs.ranks, inputs.local_rows, strict=True):
            buffers.decode_logit_row_indices[rank, local_row] = (
                local_row * layout.decode_seq + layout.decode_seq - 1
            )
        self._mtp_decode_inputs_initialized = True
        return active_tokens

    def _stage_fused_mtp_metadata(self, inputs: DeepSeekV4PreparedDecodeInputs) -> int:
        """Bind active rows to stable device-state generations."""
        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        buffers.decode_tail_slot_ids.fill_(-1)
        buffers.decode_state_generations.zero_()
        buffers.decode_logit_row_indices.fill_(-1)
        for request_id, rank, local_row in zip(
            inputs.request_ids,
            inputs.ranks,
            inputs.local_rows,
            strict=True,
        ):
            state = self._require_mtp_request_state(request_id)
            if (
                state.tail_slot_id is None
                or not state.device_state_initialized
            ):
                raise RuntimeError(
                    f"DeepSeekV4 MTP committed tail is not initialized for {request_id!r}"
                )
            if state.tail_rank != rank:
                raise RuntimeError(
                    f"DeepSeekV4 MTP state rank changed for {request_id!r}: "
                    f"slot rank={state.tail_rank}, decode rank={rank}"
                )
            buffers.decode_tail_slot_ids[rank, local_row] = state.tail_slot_id
            buffers.decode_state_generations[rank, local_row] = state.generation
            buffers.decode_logit_row_indices[rank, local_row] = (
                local_row * layout.decode_seq + layout.decode_seq - 1
            )
        return max(inputs.per_rank_counts) * layout.decode_seq

    def _prepare_mtp_device_state_descriptors(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        *,
        require_ready: bool = False,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Bind batch rows to stable state slots in the input ping-pong slot.

        Stable slot ids and generations do not depend on step-N acceptance, so
        steady-state N+1 preparation can write them while N executes.  The
        first MTP decode still uses the guarded late path because it creates the
        device state itself.

        ``tail_slot_ids[rank, local_row]`` maps the transient scheduler row to
        the persistent request slot.  ``state_generations`` carries the slot's
        allocation tag, while ``logit_row_indices`` selects the final row of
        each S-token committed window for the next-draft LM head.
        """
        if self._compiled.num_speculative_tokens != 1:
            return inputs
        staged = self._decode_input_slots[inputs.buffer_slot]
        tail_slot_ids = staged["mtp_tail_slot_ids"]
        state_generations = staged["mtp_state_generations"]
        logit_row_indices = staged["mtp_logit_row_indices"]
        tail_slot_ids.fill_(-1)
        state_generations.zero_()
        logit_row_indices.fill_(-1)
        for request_id, rank, local_row in zip(
            inputs.request_ids,
            inputs.ranks,
            inputs.local_rows,
            strict=True,
        ):
            with self._mtp_state_lock:
                state = self._mtp_request_states.get(request_id)
                if state is None:
                    if require_ready:
                        raise RuntimeError(
                            f"DeepSeekV4 MTP state is missing for request {request_id!r}"
                        )
                    return inputs
                if state.tail_slot_id is None:
                    if require_ready:
                        raise RuntimeError(
                            f"DeepSeekV4 MTP tail slot is not reserved for {request_id!r}"
                        )
                    return inputs
                if state.tail_rank != rank:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP state rank changed for {request_id!r}: "
                        f"slot rank={state.tail_rank}, decode rank={rank}"
                    )
                tail_slot_ids[rank, local_row] = state.tail_slot_id
                state_generations[rank, local_row] = state.generation
            # The MTP model emits one next draft per request.  It must consume
            # the hidden state at the end of that request's committed window,
            # not every main-model verifier row.
            logit_row_indices[rank, local_row] = (
                local_row * self._compiled.layout.decode_seq
                + self._compiled.layout.decode_seq
                - 1
            )
        return replace(
            inputs,
            mtp_tail_slot_ids=tail_slot_ids,
            mtp_state_generations=state_generations,
            mtp_logit_row_indices=logit_row_indices,
        )

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

    def _ensure_decode_buffers(self, hidden_size: int) -> _DeepSeekV4DecodeSharedBuffers:
        buffers = self._decode_buffers
        if buffers is None:
            self._ensure_shared_host_allocation_before_worker("decode inputs")
            layout = self._compiled.layout
            ranks = layout.ranks
            batch = layout.decode_batch
            tokens = layout.decode_tokens

            def allocate_input_slot(slot: int) -> dict[str, torch.Tensor]:
                prefix = f"decode_slot{slot}"
                return {
                    "input_ids": self._shared_empty(
                        (ranks, tokens), torch.long, name=f"{prefix}_input_ids"
                    ),
                    "position_ids": self._shared_empty(
                        (ranks, tokens), torch.int32, name=f"{prefix}_position_ids"
                    ),
                    "kv_seq_lens": self._shared_empty(
                        (ranks, batch), torch.int32, name=f"{prefix}_kv_seq_lens"
                    ),
                    "block_table": self._shared_empty(
                        (ranks, batch, layout.ori_table_max_blocks),
                        torch.int32,
                        name=f"{prefix}_block_table",
                    ),
                    "cmp_block_table": self._shared_empty(
                        (ranks, batch, layout.cmp_max_blocks),
                        torch.int32,
                        name=f"{prefix}_cmp_block_table",
                    ),
                    "idx_block_table": self._shared_empty(
                        (ranks, batch, layout.idx_max_blocks),
                        torch.int32,
                        name=f"{prefix}_idx_block_table",
                    ),
                    "hca_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_hca_state_max_blocks),
                        torch.int32,
                        name=f"{prefix}_hca_compress_state_block_table",
                    ),
                    "csa_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_csa_state_max_blocks),
                        torch.int32,
                        name=f"{prefix}_csa_compress_state_block_table",
                    ),
                    "csa_inner_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_csa_inner_state_max_blocks),
                        torch.int32,
                        name=f"{prefix}_csa_inner_compress_state_block_table",
                    ),
                    "block_counts": self._shared_empty(
                        (ranks, batch, len(DEEPSEEK_V4_CACHE_GROUP_NAMES)),
                        torch.int32,
                        name=f"{prefix}_block_counts",
                    ),
                    "num_tokens_per_owner": self._shared_empty(
                        (ranks,), torch.int32, name=f"{prefix}_num_tokens_per_owner"
                    ),
                    "logit_row_indices": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name=f"{prefix}_logit_row_indices",
                    ),
                    "mtp_tail_slot_ids": self._shared_empty(
                        (ranks, batch),
                        torch.int32,
                        name=f"{prefix}_mtp_tail_slot_ids",
                    ),
                    "mtp_state_generations": self._shared_empty(
                        (ranks, batch),
                        torch.int32,
                        name=f"{prefix}_mtp_state_generations",
                    ),
                    "mtp_logit_row_indices": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name=f"{prefix}_mtp_logit_row_indices",
                    ),
                    # Host-reclaimed outputs must follow the same ping-pong
                    # lifetime as inputs.  Step N+1 may write slot 1 while the
                    # output lane is still reading step N from slot 0.
                    "sampled_ids": self._shared_empty(
                        (ranks, tokens, DEEPSEEK_V4_SAMPLED_IDS_PAD),
                        torch.int32,
                        name=f"{prefix}_sampled_ids",
                    ),
                    "mtp_accepted_counts": self._shared_empty(
                        (ranks, batch),
                        torch.int32,
                        name=f"{prefix}_mtp_accepted_counts",
                    ),
                    "mtp_sampled_ids": self._shared_empty(
                        (ranks, tokens, DEEPSEEK_V4_SAMPLED_IDS_PAD),
                        torch.int32,
                        name=f"{prefix}_mtp_sampled_ids",
                    ),
                    "mtp_committed_input_ids": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name=f"{prefix}_mtp_committed_input_ids",
                    ),
                    "mtp_committed_position_ids": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name=f"{prefix}_mtp_committed_position_ids",
                    ),
                    "mtp_tail_token_ids": self._shared_empty(
                        (ranks, batch),
                        torch.long,
                        name=f"{prefix}_mtp_tail_token_ids",
                    ),
                    "mtp_tail_positions": self._shared_empty(
                        (ranks, batch),
                        torch.int32,
                        name=f"{prefix}_mtp_tail_positions",
                    ),
                }

            # Two execution snapshots let the command thread prepare step N+1
            # while the device thread consumes step N. Worker slot ownership is
            # released only after output reclaim has finished reading the slot.
            self._decode_input_slots = [allocate_input_slot(0), allocate_input_slot(1)]
            buffers = _DeepSeekV4DecodeSharedBuffers(
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
                sampled_ids=self._decode_input_slots[0]["sampled_ids"],
                tensors=self._decode_input_slots[0],
            )
            self._decode_buffers = buffers
        return buffers

    def _ensure_mtp_buffers(self, hidden_size: int) -> _DeepSeekV4MtpSharedBuffers | None:
        """Load immutable MTP weights and allocate mutable shared buffers before worker fork."""
        legacy_mtp = (
            self._compiled.mtp_prefill is not None
            and self._compiled.mtp_decode is not None
        )
        if not self._compiled.num_speculative_tokens and not legacy_mtp:
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
        # The real MTP pool is allocated directly on each worker after runtime
        # HBM sizing. Keep only a one-page shared placeholder here so legacy
        # buffer consumers retain the unified prefill/decode object contract.
        mtp_kv_cache = self._shared_empty(
            (ranks, 1, layout.block_size, 1, DEEPSEEK_V4_HEAD_DIM),
            torch.bfloat16,
            name="mtp_kv_cache_placeholder",
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
            decode_input_ids=self._shared_empty(
                (ranks, tokens), torch.long, name="mtp_decode_input_ids"
            ),
            decode_position_ids=self._shared_empty(
                (ranks, tokens), torch.int32, name="mtp_decode_position_ids"
            ),
            decode_accepted_counts=self._shared_empty(
                (ranks, layout.decode_batch), torch.int32, name="mtp_decode_accepted_counts"
            ),
            decode_tail_slot_ids=self._shared_empty(
                (ranks, layout.decode_batch), torch.int32, name="mtp_decode_tail_slot_ids"
            ),
            decode_tail_token_ids=self._shared_empty(
                (ranks, layout.decode_batch), torch.long, name="mtp_decode_tail_token_ids"
            ),
            decode_tail_positions=self._shared_empty(
                (ranks, layout.decode_batch), torch.int32, name="mtp_decode_tail_positions"
            ),
            decode_state_generations=self._shared_empty(
                (ranks, layout.decode_batch), torch.int32, name="mtp_decode_state_generations"
            ),
            state_init_tokens=self._shared_empty(
                (ranks, layout.decode_batch, _MTP_DEVICE_STATE_TOKEN_WIDTH),
                torch.long,
                name="mtp_state_init_tokens",
            ),
            state_init_meta=self._shared_empty(
                (ranks, layout.decode_batch, _MTP_DEVICE_STATE_META_WIDTH),
                torch.int32,
                name="mtp_state_init_meta",
            ),
            tail_init_hidden=self._shared_empty(
                (ranks, layout.decode_batch, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_tail_init_hidden",
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
                (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS, DEEPSEEK_V4_VOCAB_SIZE),
                torch.float32,
                name="mtp_prefill_logits",
            ),
            prefill_logit_row_indices=self._shared_empty(
                (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS),
                torch.int32,
                name="mtp_prefill_logit_row_indices",
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
                (ranks, tokens, DEEPSEEK_V4_VOCAB_SIZE),
                torch.float32,
                name="mtp_decode_logits",
            ),
            decode_sampled_ids=self._shared_empty(
                (ranks, tokens, DEEPSEEK_V4_SAMPLED_IDS_PAD),
                torch.int32,
                name="mtp_decode_sampled_ids",
            ),
            decode_logit_row_indices=self._shared_empty(
                (ranks, tokens), torch.int32, name="mtp_decode_logit_row_indices"
            ),
        )
        self._mtp_buffers.prefill_kv_cache.zero_()
        self._mtp_decode_inputs_initialized = False
        return self._mtp_buffers

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
                (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS),
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
        logits_shape = (layout.ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS, int(vocab_size))
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

    def _run_l3(self, callable_spec: DeepSeekV4L3Callable, *args: Any) -> None:
        """Dispatch one DeepSeek L3 program and emit Qwen-compatible timing traces."""
        if self._l3_worker is None:
            self._assert_l3_args_shared_before_worker(callable_spec, args)
        span_args = {
            "block_dim": callable_spec.block_dim,
            "aicpu_thread_num": callable_spec.aicpu_thread_num,
        }
        with profile_span(callable_spec.name, cat="kernel", level="kernel", args=span_args):
            worker = self._shared_l3_worker()
            uploaded: list[DeviceTensor] = []
            try:
                l3_args = tuple(self._coerce_l3_arg(worker, arg, uploaded) for arg in args)
                worker_run_args = dict(span_args)
                with profile_span(
                    f"{callable_spec.name}.worker_run",
                    cat="kernel",
                    level="kernel",
                    args=worker_run_args,
                ):
                    worker.run(callable_spec.compiled, *l3_args)
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
        if dst.data_ptr() == src.data_ptr():
            return
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
        if self._global_weights is not None:
            hidden_size = int(self.load_packed_global_weights().embed_weight.shape[1])
            self._materialize_embedding_device_weight()
            self._materialize_main_pre_hc_device(hidden_size)
            if self._compiled.num_speculative_tokens:
                self._materialize_mtp_tail_pre_hc_pool(hidden_size)
                self._materialize_mtp_prefill_device_outputs(hidden_size)
                self._materialize_mtp_device_state_tokens()
                self._materialize_mtp_device_state_meta()
        if self._stacked_device_weights is None:
            host_weights = self._stacked_host_weights
            if not host_weights:
                raise RuntimeError("DeepSeekV4 stacked Host weights are not retained")
            parent_host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_weights.values())
            with profile_span("DeepSeekV4ModelRunner.upload_resident_main_weights", cat="executor"):
                self._stacked_device_weights = self._upload_weight_group(worker, host_weights)
            self._compiled.prepacked_layer_weights = None
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
                    reset_persistent_windows=False,
                    inherited_host_tensors=self._inherited_host_weights(),
                )
            self._l3_worker = worker
        return worker

    def _inherited_host_weights(self) -> list[torch.Tensor]:
        """Return immutable main and MTP weights that must be visible at worker fork."""
        tensors = list(self._stacked_host_weights.values()) if self._stacked_host_weights else []
        global_weights = getattr(self, "_global_weights", None)
        if global_weights is not None:
            tensors.append(global_weights.embed_weight)
        if self._mtp_buffers is not None:
            tensors.extend(self._mtp_buffers.weights.values())
        return tensors

    def _alloc_empty_stacked_tensor(
        self,
        full_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> StackedDeviceTensor:
        """Allocate an uninitialized shard directly on every chip worker."""
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards: list[DeviceTensor] = []
        try:
            for worker_id in worker_ids:
                shards.append(
                    worker.alloc_tensor(
                        full_shape[1:],
                        dtype,
                        worker_id=worker_id,
                    )
                )
        except Exception:
            for shard, worker_id in zip(shards, worker_ids, strict=False):
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        return StackedDeviceTensor(shards, full_shape, worker_ids)

    def _materialize_embedding_device_weight(self) -> StackedDeviceTensor:
        """Upload one full embedding table to every decode rank."""
        stacked = self._embedding_device_weight
        if stacked is not None:
            return stacked
        source = self.load_packed_global_weights().embed_weight
        if source.device.type != "cpu" or source.dtype != torch.bfloat16 or not source.is_contiguous():
            raise ValueError("DeepSeekV4 embedding weight must be contiguous BF16 CPU storage before worker fork")
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards: list[DeviceTensor] = []
        try:
            for worker_id in worker_ids:
                shards.append(worker.alloc_tensor(source.shape, source.dtype, init=source, worker_id=worker_id))
        except Exception:
            for shard, worker_id in zip(shards, worker_ids, strict=False):
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        stacked = StackedDeviceTensor(
            shards,
            (self._compiled.layout.ranks, *source.shape),
            worker_ids,
        )
        self._embedding_device_weight = stacked
        return stacked

    def _materialize_main_pre_hc_device(self, hidden_size: int) -> StackedDeviceTensor:
        """Allocate the main decode pre-HC output used directly by MTP."""
        stacked = self._main_pre_hc_device
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_tokens, layout.hc_mult, int(hidden_size)),
                torch.float32,
            )
            self._main_pre_hc_device = stacked
        return stacked

    def _materialize_mtp_tail_pre_hc_pool(self, hidden_size: int) -> StackedDeviceTensor:
        """Allocate persistent request tail hidden slots on every rank."""
        stacked = self._mtp_tail_pre_hc_pool
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_batch, layout.hc_mult, int(hidden_size)),
                torch.float32,
            )
            self._mtp_tail_pre_hc_pool = stacked
        return stacked

    def _materialize_mtp_prefill_device_outputs(
        self,
        hidden_size: int,
    ) -> _DeepSeekV4MtpPrefillDeviceOutputs:
        """Allocate uninitialized MTP prefill outputs before KV-cache sizing."""
        outputs = self._mtp_prefill_device_outputs
        if outputs is not None:
            return outputs

        layout = self._compiled.layout
        allocated: list[StackedDeviceTensor] = []

        def resident(shape: tuple[int, ...], dtype: torch.dtype) -> StackedDeviceTensor:
            tensor = self._alloc_empty_stacked_tensor(shape, dtype)
            allocated.append(tensor)
            return tensor

        try:
            outputs = _DeepSeekV4MtpPrefillDeviceOutputs(
                hidden_out=resident(
                    (layout.ranks, layout.prefill_seq, int(hidden_size)),
                    torch.bfloat16,
                ),
                pre_hc_hidden_out=resident(
                    (
                        layout.ranks,
                        layout.prefill_seq,
                        layout.hc_mult,
                        int(hidden_size),
                    ),
                    torch.float32,
                ),
                logits=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS,
                        DEEPSEEK_V4_VOCAB_SIZE,
                    ),
                    torch.float32,
                ),
            )
        except Exception:
            worker = self._shared_l3_worker()
            for tensor in reversed(allocated):
                worker.free_stacked_tensor(tensor)
            raise
        self._mtp_prefill_device_outputs = outputs
        return outputs

    def _read_mtp_prefill_logits(self, owner_rank: int) -> torch.Tensor:
        """Read only the valid owner logits row needed for Host argmax."""
        buffers = self._require_mtp_buffers()
        outputs = self._materialize_mtp_prefill_device_outputs(
            int(buffers.prefill_hidden_in.shape[-1])
        )
        host_row = buffers.prefill_logits[owner_rank, 0]
        shard = outputs.logits.shards[owner_rank]
        worker_id = outputs.logits.worker_ids[owner_rank]
        self._shared_l3_worker().copy_from(
            host_row.data_ptr(),
            shard.data_ptr,
            host_row.numel() * host_row.element_size(),
            worker_id=worker_id,
        )
        return host_row

    def _materialize_mtp_device_state_tokens(self) -> StackedDeviceTensor:
        """Allocate persistent tail/draft token slots on every rank."""
        stacked = self._mtp_device_state_tokens
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_batch, _MTP_DEVICE_STATE_TOKEN_WIDTH),
                torch.long,
            )
            self._mtp_device_state_tokens = stacked
        return stacked

    def _materialize_mtp_device_state_meta(self) -> StackedDeviceTensor:
        """Allocate persistent generation, position, and commit metadata."""
        stacked = self._mtp_device_state_meta
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_batch, _MTP_DEVICE_STATE_META_WIDTH),
                torch.int32,
            )
            self._mtp_device_state_meta = stacked
        return stacked

    def _materialize_decode_device_cache(self) -> DeepSeekV4DeviceCache:
        """Allocate dynamically sized cache shards directly on each NPU."""
        cache = self._decode_device_cache
        if cache is not None:
            return cache
        worker = self._shared_l3_worker()
        allocated: list[StackedDeviceTensor] = []
        layout = self._compiled.layout

        def resident(shape: tuple[int, ...], dtype: torch.dtype) -> StackedDeviceTensor:
            stacked = self._alloc_empty_stacked_tensor(shape, dtype)
            allocated.append(stacked)
            return stacked

        try:
            cache = DeepSeekV4DeviceCache(
                kv_cache=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_FWD_NUM_LAYERS * self._physical_cache_num_blocks("ori"),
                        layout.block_size,
                        1,
                        DEEPSEEK_V4_HEAD_DIM,
                    ),
                    torch.bfloat16,
                ),
                cmp_kv=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_FWD_NUM_LAYERS * self._physical_cache_num_blocks("cmp"),
                        layout.block_size,
                        1,
                        DEEPSEEK_V4_HEAD_DIM,
                    ),
                    torch.bfloat16,
                ),
                idx_kv_cache=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS * self._physical_cache_num_blocks("idx"),
                        layout.block_size,
                        1,
                        DEEPSEEK_V4_IDX_HEAD_DIM,
                    ),
                    torch.int8,
                ),
                idx_kv_scale=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS * self._physical_cache_num_blocks("idx"),
                        layout.block_size,
                        1,
                        1,
                    ),
                    torch.float32,
                ),
                hca_compress_state=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_HCA_NUM_LAYERS * self._physical_cache_num_blocks("hca_state"),
                        layout.c128_state_block_size,
                        DEEPSEEK_V4_HCA_STATE_DIM,
                    ),
                    torch.float32,
                ),
                csa_compress_state=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS * self._physical_cache_num_blocks("csa_state"),
                        layout.c4_state_block_size,
                        DEEPSEEK_V4_CSA_STATE_DIM,
                    ),
                    torch.float32,
                ),
                csa_inner_compress_state=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS
                        * self._physical_cache_num_blocks("csa_inner_state"),
                        layout.c4_state_block_size,
                        DEEPSEEK_V4_CSA_INNER_STATE_DIM,
                    ),
                    torch.float32,
                ),
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
        layout = self._compiled.layout
        cache = self._alloc_empty_stacked_tensor(
            (
                layout.ranks,
                self._physical_cache_num_blocks("ori"),
                layout.block_size,
                1,
                DEEPSEEK_V4_HEAD_DIM,
            ),
            torch.bfloat16,
        )
        self._mtp_device_kv_cache = cache
        return cache

    def _free_device_caches(self) -> None:
        worker = self._l3_worker
        if worker is None:
            self._decode_device_cache = None
            self._mtp_device_kv_cache = None
            return
        cache = self._decode_device_cache
        if cache is not None:
            for tensor in (
                cache.kv_cache,
                cache.cmp_kv,
                cache.idx_kv_cache,
                cache.idx_kv_scale,
                cache.hca_compress_state,
                cache.csa_compress_state,
                cache.csa_inner_compress_state,
            ):
                worker.free_stacked_tensor(tensor)
        if self._mtp_device_kv_cache is not None:
            worker.free_stacked_tensor(self._mtp_device_kv_cache)
        self._decode_device_cache = None
        self._mtp_device_kv_cache = None

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return DeepSeekV4ModelRunner._share_cpu_tensor(tensor)

    def close(self) -> None:
        worker = self._l3_worker
        try:
            if worker is not None:
                worker.close()
        finally:
            self._l3_worker = None
            self._cache_group_num_blocks.clear()
            self._stacked_host_weights = None
            self._stacked_device_weights = None
            self._embedding_device_weight = None
            self._main_pre_hc_device = None
            self._mtp_tail_pre_hc_pool = None
            self._mtp_device_state_tokens = None
            self._mtp_device_state_meta = None
            self._l3_shared_buffers_ready = False
            self._mtp_device_weights = None
            self._mtp_prefill_device_outputs = None
            self._mtp_buffers = None
            self._mtp_decode_inputs_initialized = False
            self._global_weights = None
            self._decode_device_cache = None
            self._mtp_device_kv_cache = None
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

    def _indices_by_rank(self, ranks: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Reconstruct scheduler-row ownership for an immutable prepared slot."""
        indices: list[list[int]] = [[] for _ in range(self._compiled.layout.ranks)]
        for request_index, rank in enumerate(ranks):
            indices[int(rank)].append(request_index)
        return tuple(tuple(rows) for rows in indices)

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

    def _mtp_decode_token_rows(
        self,
        token_ids: torch.Tensor,
        prev_token_ids: torch.Tensor,
        actual_batch: int,
    ) -> torch.Tensor:
        """Build explicit [committed token, draft token] verification rows."""
        layout = self._compiled.layout
        if layout.decode_seq != 2:
            raise ValueError("DeepSeekV4 MTP verification requires decode_seq=2")
        current = token_ids.detach().cpu().to(torch.long)[:actual_batch].reshape(actual_batch, -1)[:, 0]
        previous = prev_token_ids.detach().cpu().to(torch.long)[:actual_batch].reshape(actual_batch, -1)[:, 0]
        return torch.stack((previous, current), dim=1)

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
