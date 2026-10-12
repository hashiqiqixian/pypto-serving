"""V4.1 prepacked weights must preserve bytes and reject stale sources."""

import json

import torch

from pypto_serving.model.deepseek_v41.decode_weight_resources import DecodeWeightBank, DecodeWeightLayout, _ALL
from pypto_serving.model.deepseek_v41.decode_weights import DecodeWeightSlice
from pypto_serving.model.deepseek_v41 import weight_sidecar


def test_sidecar_roundtrip_and_source_invalidation(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"test": "source.safetensors"},
    }))
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source")
    layout = DecodeWeightLayout(2, 1, 2, 1, world=2)
    bank = DecodeWeightBank(layout, max_host_bytes=1 << 20)
    for name in sorted(_ALL):
        for slot in range(layout.slots(name)):
            bits = torch.full((2, 2), slot + 17, dtype=torch.uint8)
            value = bits.view(torch.float8_e8m0fnu) if name == "wq_a_scale" else bits
            bank.append(DecodeWeightSlice(name, slot, value))
    path = tmp_path / "packed.safetensors"
    weight_sidecar.write_prepacked_bank(tmp_path, bank, path=path)
    monkeypatch.setattr(weight_sidecar, "_sample_file_page_cache_residency", lambda *_: 1.0)
    loaded = weight_sidecar.load_prepacked_bank(tmp_path, layout, _ALL, path=path)
    assert loaded.prepacked
    assert loaded.tensors["wq_a_scale"].dtype == torch.float8_e8m0fnu
    for name in _ALL:
        assert torch.equal(loaded.tensors[name].view(torch.uint8),
                           bank.tensors[name].view(torch.uint8))
    source.write_bytes(b"changed source")
    assert weight_sidecar.load_prepacked_bank(tmp_path, layout, _ALL, path=path) is None
