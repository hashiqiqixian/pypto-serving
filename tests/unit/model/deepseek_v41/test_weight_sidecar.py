# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read real v2 sidecars with and without the retired MXFP4 lookup table."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from pypto_serving.model.common.weights.nz import pack_nz
from pypto_serving.model.deepseek_v41 import weight_loader as weights


def _old_lut(world):
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    codes = values.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)
    packed = torch.arange(256)
    pairs = (codes[packed % 16] + 256 * codes[packed // 16]).to(torch.int16)
    return pairs.repeat(world, 2, 1)


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    # Fingerprints use real files and production serialization. Tensor sizes
    # are small here; the tested contract is slot/rank ownership and mapping.
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}', encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "part.safetensors"}}), encoding="utf-8")
    (tmp_path / "part.safetensors").write_bytes(b"checkpoint identity")
    layout = weights.DecodeWeightLayout(2, 1, 2, 1, world=2)
    tensors = {}
    for name in weights._ALL:
        prefix = (layout.world,) if name in weights._SINGLE else (layout.slots(name), layout.world)
        tensors[name] = torch.arange(32 * 32, dtype=torch.float32).to(torch.bfloat16).reshape(
            32, 32).expand(*prefix, 32, 32).contiguous()
    logical = (torch.arange(2 * 2 * 32 * 64) % 120).to(torch.uint8).reshape(2, 2, 32, 64)
    tensors["wq_a"] = pack_nz(logical)
    tensors["wq_a_scale"] = torch.full((2, 2, 2, 64), 127, dtype=torch.uint8)
    metadata = {
        "format": weights._FORMAT,
        "source_fingerprint": weights.source_fingerprint(tmp_path, layout),
        "float8_fields": json.dumps({"wq_a": "float8_e4m3fn", "wq_a_scale": "float8_e8m0fnu"}),
    }
    monkeypatch.setattr(weights, "_sample_file_page_cache_residency", lambda *args: 1.0)
    return tmp_path, layout, tensors, metadata


@pytest.mark.parametrize("with_lut", [False, True])
def test_v2_sidecar_maps_only_live_fields_without_repacking(sidecar, with_lut, monkeypatch):
    directory, layout, tensors, metadata = sidecar
    if with_lut:
        tensors["mxfp4_pair_lut"] = _old_lut(layout.world)
    path = directory / "decode.safetensors"
    save_file(tensors, path, metadata=metadata)
    original = hashlib.sha256(path.read_bytes()).digest()
    mapped_names = []
    map_tensors = weights._map_shared_prepacked_tensors

    def record_map(fd, names):
        mapped_names.append(set(names))
        return map_tensors(fd, names)

    monkeypatch.setattr(weights, "_map_shared_prepacked_tensors", record_map)
    bank = weights.load_prepacked_bank(directory, layout, weights._ALL, path=path)

    assert bank is not None and bank.prepacked
    assert len(bank.tensors) == 44
    assert mapped_names == [weights._ALL]
    assert set(bank.tensors) == weights._ALL
    assert "mxfp4_pair_lut" not in bank.tensors
    assert bank.tensors["wq_a"].dtype == torch.float8_e4m3fn
    assert bank.tensors["wq_a_scale"].dtype == torch.float8_e8m0fnu
    assert torch.equal(bank.tensors["wq_a"].view(torch.uint8), tensors["wq_a"])
    assert bank.allocated_bytes == sum(value.numel() * value.element_size()
                                       for name, value in tensors.items() if name in weights._ALL)
    assert hashlib.sha256(path.read_bytes()).digest() == original


@pytest.mark.parametrize("change", ["extra", "missing", "stale", "v1"])
def test_legacy_lut_exception_does_not_accept_other_incompatible_sidecars(sidecar, change, monkeypatch):
    directory, layout, tensors, metadata = sidecar
    tensors["mxfp4_pair_lut"] = _old_lut(layout.world)
    if change == "extra":
        tensors["unknown_extra"] = torch.zeros(1)
    elif change == "missing":
        del tensors["routed_w1"]
    elif change == "stale":
        metadata["source_fingerprint"] = "stale"
    else:
        metadata["format"] = "pypto-deepseek-v41-decode-slot-major-v1"
    path = directory / "incompatible.safetensors"
    save_file(tensors, path, metadata=metadata)

    def unexpected_map(*args):
        pytest.fail("incompatible sidecar reached the mapping path")

    monkeypatch.setattr(weights, "_map_shared_prepacked_tensors", unexpected_map)
    assert weights.load_prepacked_bank(directory, layout, weights._ALL, path=path) is None


@pytest.mark.parametrize("change", ["shape", "dtype", "value"])
def test_legacy_lut_requires_known_shape_dtype_and_values(sidecar, change, monkeypatch):
    directory, layout, tensors, metadata = sidecar
    lut = _old_lut(layout.world)
    if change == "shape":
        lut = lut[:, :1].contiguous()
    elif change == "dtype":
        lut = lut.to(torch.int32)
    else:
        lut[0, 0, 1] ^= 1
    tensors["mxfp4_pair_lut"] = lut
    path = directory / "bad-lut.safetensors"
    save_file(tensors, path, metadata=metadata)

    def unexpected_map(*args):
        pytest.fail("invalid legacy LUT reached the mapping path")

    monkeypatch.setattr(weights, "_map_shared_prepacked_tensors", unexpected_map)
    with pytest.raises(ValueError, match="invalid legacy MXFP4 LUT"):
        weights.load_prepacked_bank(directory, layout, weights._ALL, path=path)


@pytest.mark.parametrize("change", ["rank", "float8_dtype"])
def test_legacy_lut_preserves_live_tensor_validation(sidecar, change):
    directory, layout, tensors, metadata = sidecar
    tensors["mxfp4_pair_lut"] = _old_lut(layout.world)
    if change == "rank":
        tensors["wq_a"] = tensors["wq_a"][:, :1].contiguous()
    else:
        tensors["wq_a"] = tensors["wq_a"].to(torch.int16)
    path = directory / "bad-live-tensor.safetensors"
    save_file(tensors, path, metadata=metadata)
    with pytest.raises(ValueError, match="invalid rank/slot shape|not byte-backed"):
        weights.load_prepacked_bank(directory, layout, weights._ALL, path=path)
