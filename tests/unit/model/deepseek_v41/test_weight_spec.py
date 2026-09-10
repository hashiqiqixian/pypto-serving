# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Storage-contract tests; no PyTorch, checkpoint download, or device required."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[4] / "pypto_serving/model/deepseek_v41/weight_spec.py"
MODULE_SPEC = importlib.util.spec_from_file_location("v41_weight_spec_under_test", MODULE_PATH)
weight_spec = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = weight_spec
MODULE_SPEC.loader.exec_module(weight_spec)


def released_config():
    """Fields from pinned dba1be0 config; expectations below also use shard headers."""
    return {
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "head_dim": 512,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hc_mult": 4,
            "vocab_size": 129280,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_head_dim": 256,
            "engram_max_ngram_size": 4,
            "engram_n_heads": 8,
        },
    }


@pytest.fixture(scope="module")
def specs():
    return weight_spec.backbone_weight_specs(released_config())


class TestWeightSpec:
    def test_observed_dense_and_expert_storage(self, specs):
        observed = {
            "embed.weight": ((129280, 5120), "BF16"),
            "head.weight": ((129280, 5120), "BF16"),
            "layers.0.attn.wq_b.weight": ((32768, 1280), "F8_E4M3"),
            "layers.0.attn.wq_b.scale": ((1024, 40), "F8_E8M0"),
            "layers.0.attn.wo_a.weight": ((8192, 4096), "F8_E4M3"),
            "layers.0.attn.wo_a.scale": ((256, 128), "F8_E8M0"),
            "layers.0.ffn.experts.0.w1.weight": ((2304, 2560), "I8"),
            "layers.0.ffn.experts.0.w1.scale": ((2304, 160), "F8_E8M0"),
            "layers.0.ffn.experts.0.w2.weight": ((5120, 1152), "I8"),
            "layers.0.ffn.experts.0.w2.scale": ((5120, 72), "F8_E8M0"),
            "layers.2.attn.compressor.wgate.weight": ((512, 5120), "BF16"),
            "layers.0.hc_attn_fn": ((24, 20480), "F32"),
        }
        for name, expected in observed.items():
            spec = specs[name]
            assert (spec.shape, spec.dtype) == expected, name

    def test_source_layers_own_only_the_required_projections(self, specs):
        assert "layers.2.attn.compressor.wgate.weight" in specs
        assert "layers.3.attn.compressor.wkv.weight" not in specs
        assert "layers.20.attn.compressor.wkv.weight" in specs
        assert "layers.20.attn.compressor.wgate.weight" not in specs
        assert "layers.24.attn.indexer.wq_b.weight" in specs
        assert "layers.24.attn.indexer.wk.weight" not in specs
        assert "layers.25.attn.indexer.wq_b.weight" not in specs

    def test_conversions_do_not_confuse_checkpoint_and_runtime_dtype(self, specs):
        assert "cast_bf16_to_fp32" in specs["layers.2.attn.compressor.wkv.weight"].conversion
        assert specs["layers.20.attn.compressor.wkv.weight"].conversion == "identity"
        assert "dequantize_fp8_32x32" in specs["layers.0.attn.wo_a.weight"].conversion
        assert specs["layers.0.attn.wo_a.scale"].runtime_name == "layers.0.attn.wo_a.weight"
        assert "cast_bf16_to_fp32" in specs["head.weight"].conversion

    def test_engram_is_required_but_vision_and_draft_are_out_of_scope(self, specs):
        for layer, rows in ((1, 384006168), (14, 384016682)):
            assert specs[f"layers.{layer}.engram.embed.weight"].shape == (rows, 256)
            assert specs[f"layers.{layer}.engram.embed.scale"].shape == (rows, 8)
            assert specs[f"layers.{layer}.engram.wkv.weight"].shape == (25600, 6144)
        assert not any(name.startswith(("mtp.", "vision.", "layers.40.")) for name in specs)
        assert "layers.39.ffn.experts.383.w3.weight" in specs
        assert specs["layers.0.ffn.gate.bias_vl"].conversion == "retain_unused_text_bias"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("weight_block_size", [128, 128]),
            ("expert_dtype", "int4"),
            ("scale_fmt", "float32"),
            ("quant_method", "none"),
        ],
    )
    def test_unsupported_quantization_fails_before_producing_specs(self, field, value):
        config = released_config()
        config["quantization_config"][field] = value
        with pytest.raises(ValueError, match="FP8 32x32"):
            weight_spec.backbone_weight_specs(config)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("kv_source_layer_ids", [3]),
            ("engram_layer_ids", [1, 1]),
            ("engram_num_embeddings", [12]),
            ("compress_ratios", [0]),
            ("index_source_layer_ids", [40]),
        ],
    )
    def test_invalid_ownership_and_engram_metadata_are_rejected(self, field, value):
        config = released_config()
        config["text_config"][field] = value
        with pytest.raises(ValueError):
            weight_spec.backbone_weight_specs(config)
