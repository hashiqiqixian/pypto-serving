# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep V4.1 checkpoint shards inside their model directory."""

from pathlib import Path

import pytest

from pypto_serving.model.deepseek_v41.weight_loader import _validate_shard_paths


def test_external_shard_symlink_is_rejected(tmp_path):
    model_dir = tmp_path / "metadata"
    shard_dir = tmp_path / "shards"
    model_dir.mkdir()
    shard_dir.mkdir()
    (shard_dir / "model-00001.safetensors").touch()
    (model_dir / "model-00001.safetensors").symlink_to(shard_dir / "model-00001.safetensors")
    weight_map = {"weight": "model-00001.safetensors"}

    with pytest.raises(ValueError, match="model directory"):
        _validate_shard_paths(model_dir, weight_map)

def test_shard_path_must_stay_inside_model(tmp_path):
    model_dir = tmp_path / "metadata"
    shard_dir = tmp_path / "shards"
    model_dir.mkdir()
    shard_dir.mkdir()
    with pytest.raises(ValueError, match="stay inside"):
        _validate_shard_paths(model_dir, {"weight": "../shards/model-00001.safetensors"})
    with pytest.raises(ValueError, match="stay inside"):
        _validate_shard_paths(model_dir, {"weight": str(Path(tmp_path) / "outside.safetensors")})
    (model_dir / "model-00001.safetensors").touch()
    _validate_shard_paths(model_dir, {"weight": "model-00001.safetensors"})
