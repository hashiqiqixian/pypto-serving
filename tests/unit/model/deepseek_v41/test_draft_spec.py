# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark source metadata is independently inspectable without Torch imports."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def schema(monkeypatch):
    package = ModuleType("v41_draft_schema_test")
    package.__path__ = [str(ROOT / "pypto_serving/model/deepseek_v41")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for suffix in ("weight_spec", "draft_spec"):
        name = package.__name__ + "." + suffix
        spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0]) / (suffix + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw():
    return json.loads((ROOT / "tests/fixtures/deepseek_v41/config.json").read_text())


def test_full_draft_contract(schema, raw):
    specs = schema.draft_weight_specs(raw)
    assert len(specs) == 2401
    assert specs["mtp.0.main_proj.weight"].shape == (5120, 15360)
    assert specs["mtp.2.markov_head.head.weight"].shape == (129280, 256)
    assert "cast_bf16_to_fp32" in specs["mtp.2.confidence_head.proj.weight"].conversion
    assert specs["mtp.1.attn.wo_a.scale"].runtime_name == "mtp.1.attn.wo_a.weight"
    assert specs["mtp.0.ffn.experts.0.w1.weight"].shape == (2304, 2560)
    assert all(name.startswith("mtp.") for name in specs)


def test_optional_actual_checkpoint_metadata(schema, raw):
    reference = ROOT / "artifacts/deepseek-v41-p0/reference"
    index_path = reference / "model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip("optional downloaded checkpoint index absent")
    index = json.loads(index_path.read_text())["weight_map"]
    keys = {name for name in index if name.startswith("mtp.")}
    specs = schema.draft_weight_specs(raw)
    assert keys == specs.keys()
    for shard in {index[key] for key in keys}:
        path = reference / (shard + ".header.json")
        if not path.exists():
            pytest.skip("optional downloaded shard header absent")
        header = json.loads(path.read_text())
        for name in keys.intersection(header):
            assert specs[name].shape == tuple(header[name]["shape"])
            assert specs[name].dtype == header[name]["dtype"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dspark_target_layer_ids", []),
        ("dspark_target_layer_ids", [37, 37]),
        ("num_nextn_predict_layers", 0),
        ("n_shared_experts", 2),
    ],
)
def test_invalid_draft_contract_is_rejected(schema, raw, field, value):
    raw = copy.deepcopy(raw)
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        schema.draft_weight_specs(raw)
