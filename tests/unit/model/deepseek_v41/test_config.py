# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU configuration tests without importing the serving stack."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "tests/fixtures/deepseek_v41/config.json"
# Fixture copied verbatim from deepseek-ai/DeepSeek-V4.1-Flash/config.json at
# dba1be0a40aa45a94ad051997016db3960a90277. No weights are part of this fixture.
MODULE_PATH = ROOT / "pypto_serving/model/deepseek_v41/config.py"
MODULE_SPEC = importlib.util.spec_from_file_location("_deepseek_v41_config_test_target", MODULE_PATH)
config_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = config_module
MODULE_SPEC.loader.exec_module(config_module)
DeepSeekV41Config = config_module.DeepSeekV41Config
DeepSeekV41Profile = config_module.DeepSeekV41Profile
ConfigError = config_module.ConfigError


def _read_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def raw_config():
    return _read_config()


def test_published_shape_and_ep8_ownership():
    config = DeepSeekV41Config.from_json(CONFIG_PATH)
    assert config.hidden_size == 5120
    assert config.num_hidden_layers == 40
    assert config.num_nextn_predict_layers == 3
    assert config.n_routed_experts == 384
    assert config.num_experts_per_tok == 6
    owners = [config.expert_ownership(8, rank) for rank in range(8)]
    assert [(owner.start, owner.stop) for owner in owners] == [
        (rank * 48, (rank + 1) * 48) for rank in range(8)
    ]
    assert all((owner.local_expert_count == 48 and owner.shared_experts == 1 for owner in owners))
    assert owners[7].local_index(383) == 47
    with pytest.raises(ConfigError, match="not owned"):
        owners[7].local_index(335)
    profile = DeepSeekV41Profile(config)
    assert profile.model_type == "deepseek_v41"
    assert profile.local_expert_count == 48


def test_source_maps_drive_full_reindex_reuse_and_private_swa(raw_config):
    plans = DeepSeekV41Config.from_dict(raw_config).layer_plan
    assert len(plans) == 40
    assert [p.layer_id for p in plans if p.attention_kind == "full"] == [2, 8, 14, 20]
    assert [p.layer_id for p in plans if p.attention_kind == "reindex"] == [24, 28, 32, 36]
    assert [p.layer_id for p in plans if p.attention_kind == "swa"] == [0, 1]
    assert plans[7].kv_source_layer_id == 2
    assert plans[8].kv_source_layer_id == 8
    assert plans[19].kv_source_layer_id == 14
    assert plans[23].index_source_layer_id == 20
    assert plans[24].kv_source_layer_id == 20
    assert plans[24].index_source_layer_id == 24
    assert plans[39].index_source_layer_id == 36
    assert plans[39].kv_source_layer_id == 20
    assert plans[39].candidate_source_layer_id == 20
    assert not plans[24].owns_main_kv
    assert not plans[24].owns_index_keys
    assert plans[24].owns_index_results
    assert all((p.swa_owner_layer_id == p.layer_id for p in plans))
    assert plans[0].kv_source_layer_id is None
    assert plans[19].candidate_source_layer_id is None


def test_source_change_updates_mapping_without_fixed_layer_constants(raw_config):
    raw_config["text_config"]["kv_source_layer_ids"][1] = 9
    raw_config["text_config"]["index_source_layer_ids"][1] = 9
    plans = DeepSeekV41Config.from_dict(raw_config).layer_plan
    assert plans[8].attention_kind == "reuse"
    assert plans[8].kv_source_layer_id == 2
    assert plans[9].attention_kind == "full"
    assert plans[10].kv_source_layer_id == 9


def test_engram_is_a_required_text_dependency_and_no_invented_ced_split(raw_config):
    config = DeepSeekV41Config.from_dict(raw_config)
    assert config.text_requires_engram
    assert [p.layer_id for p in config.layer_plan if p.requires_engram] == [1, 14]
    assert config.engram_num_embeddings == (384006168, 384016682)
    assert config.ced_split is None
    assert config.layer_plan[19].compress_ratio == 2
    assert config.layer_plan[20].compress_ratio == 1


def test_quantization_is_preserved_as_distinct_weight_contract(raw_config):
    quantization = DeepSeekV41Config.from_dict(raw_config).quantization
    assert quantization.weight_block_size == (32, 32)
    assert quantization.expert_block_size == 32
    assert quantization.expert_dtype == "fp4"
    assert quantization.scale_fmt == "ue8m0"


def test_duplicate_json_metadata_is_rejected_before_interpretation(tmp_path):
    content = CONFIG_PATH.read_text(encoding="utf-8")
    content = content.replace('"hidden_size": 5120,', '"hidden_size": 1, "hidden_size": 5120,')
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate config key.*hidden_size"):
        DeepSeekV41Config.from_json(path)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("quant_method", "int8"),
        ("expert_dtype", "fp8"),
        ("scale_fmt", "e4m3"),
        ("weight_block_size", [128, 128]),
        ("weight_block_size", [True, 32]),
        ("activation_scheme", "static"),
        ("expert_block_size", 16),
    ],
)
def test_quantization_mismatch_or_unknown_is_rejected(raw_config, key, bad):
    raw_config["quantization_config"][key] = bad
    with pytest.raises(ConfigError):
        DeepSeekV41Config.from_dict(raw_config)


def test_missing_quantization_metadata_is_rejected(raw_config):
    del raw_config["quantization_config"]["scale_fmt"]
    with pytest.raises(ConfigError):
        DeepSeekV41Config.from_dict(raw_config)


def test_v4_and_flat_inference_config_are_not_misidentified(raw_config):
    raw_config["model_type"] = "deepseek_v4"
    with pytest.raises(ConfigError, match="model_type"):
        DeepSeekV41Config.from_dict(raw_config)
    with pytest.raises(ConfigError, match="model_type"):
        DeepSeekV41Config.from_dict({"dim": 5120, "n_layers": 40})


@pytest.mark.parametrize("bad", [None, "5120", True, 0, -1, 5120.0])
def test_required_fields_are_not_defaulted_or_coerced(raw_config, bad):
    raw_config["text_config"]["hidden_size"] = bad
    with pytest.raises(ConfigError, match="hidden_size"):
        DeepSeekV41Config.from_dict(raw_config)


@pytest.mark.parametrize("sources", [[2, 8, 8, 20], [8, 2, 14, 20], [2, 8, 14, 40], [-1, 8, 14, 20]])
def test_source_ids_must_be_sorted_unique_and_in_backbone(raw_config, sources):
    raw_config["text_config"]["kv_source_layer_ids"] = sources
    with pytest.raises(ConfigError):
        DeepSeekV41Config.from_dict(raw_config)


def test_sources_cannot_be_missing_or_pure_swa(raw_config):
    raw_config["text_config"]["kv_source_layer_ids"] = [8, 14, 20]
    with pytest.raises(ConfigError, match="before a source"):
        DeepSeekV41Config.from_dict(raw_config)
    raw_config = _read_config()
    raw_config["text_config"]["kv_source_layer_ids"] = [0, 2, 8, 14, 20]
    raw_config["text_config"]["index_source_layer_ids"] = [0, 2, 8, 14, 20, 24, 28, 32, 36]
    with pytest.raises(ConfigError, match="compress_ratio=0"):
        DeepSeekV41Config.from_dict(raw_config)


def test_kv_source_requires_fresh_index_results(raw_config):
    raw_config["text_config"]["index_source_layer_ids"].remove(8)
    with pytest.raises(ConfigError, match="each KV source"):
        DeepSeekV41Config.from_dict(raw_config)


def test_ratio_transition_cannot_reuse_incompatible_cache(raw_config):
    raw_config["text_config"]["compress_ratios"][19] = 1
    with pytest.raises(ConfigError, match="incompatible compress_ratio"):
        DeepSeekV41Config.from_dict(raw_config)


def test_v4_ratios_and_incorrect_draft_coverage_rejected(raw_config):
    for ratios in ([0, 0] + [4] * 38 + [0] * 3, raw_config["text_config"]["compress_ratios"][:40]):
        raw_config["text_config"]["compress_ratios"] = ratios
        with pytest.raises(ConfigError):
            DeepSeekV41Config.from_dict(raw_config)
    raw_config = _read_config()
    raw_config["text_config"]["compress_ratios"][40] = 1
    with pytest.raises(ConfigError, match="draft layers"):
        DeepSeekV41Config.from_dict(raw_config)


def test_hierarchical_candidates_require_valid_owner_and_geometry(raw_config):
    raw_config["text_config"]["candidate_source_layer_id"] = 21
    with pytest.raises(ConfigError, match="index source"):
        DeepSeekV41Config.from_dict(raw_config)
    raw_config["text_config"]["candidate_source_layer_id"] = 14
    with pytest.raises(ConfigError, match="invalidate candidate ownership"):
        DeepSeekV41Config.from_dict(raw_config)
    raw_config["text_config"]["candidate_source_layer_id"] = 20
    raw_config["text_config"]["candidate_block_size"] = 0
    with pytest.raises(ConfigError, match="positive"):
        DeepSeekV41Config.from_dict(raw_config)


def test_engram_configuration_cannot_silently_drop_a_required_table(raw_config):
    raw_config["text_config"]["engram_num_embeddings"] = [384006168]
    with pytest.raises(ConfigError, match="one entry"):
        DeepSeekV41Config.from_dict(raw_config)
    raw_config = _read_config()
    raw_config["text_config"]["engram_num_embeddings"][0] = 0
    with pytest.raises(ConfigError, match="positive"):
        DeepSeekV41Config.from_dict(raw_config)


@pytest.mark.parametrize("ep_size,rank", [(0, 0), (7, 0), (8, -1), (8, 8), (True, 0)])
def test_ep_rejects_partial_or_invalid_rank_partitions(raw_config, ep_size, rank):
    config = DeepSeekV41Config.from_dict(raw_config)
    with pytest.raises(ConfigError):
        config.expert_ownership(ep_size, rank)


def test_profile_rejects_incomplete_ep_partitions(raw_config):
    config = DeepSeekV41Config.from_dict(raw_config)
    with pytest.raises(ConfigError):
        DeepSeekV41Profile(config, expert_parallel_size=7)


def test_moe_topk_cannot_exceed_routed_expert_count(raw_config):
    raw_config["text_config"]["num_experts_per_tok"] = 385
    with pytest.raises(ConfigError, match="exceeds"):
        DeepSeekV41Config.from_dict(raw_config)


def test_metadata_is_detached_from_caller_mutation(raw_config):
    config = DeepSeekV41Config.from_dict(raw_config)
    raw_config["text_config"]["hidden_size"] = 1
    raw_config["text_config"]["rope_scaling"]["factor"] = 99
    assert config.hidden_size == 5120
    assert config.text_config["rope_scaling"]["factor"] == 16
    with pytest.raises(TypeError):
        config.text_config["rope_scaling"]["factor"] = 99
