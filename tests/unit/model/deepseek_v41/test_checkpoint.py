# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint corruption and coverage tests using small actual safetensors files."""

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "pypto_serving/model/deepseek_v41/checkpoint.py"
MODULE_SPEC = importlib.util.spec_from_file_location("_v41_checkpoint_test", SOURCE)
checkpoint = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(checkpoint)


@pytest.fixture
def case(tmp_path):
    name = "layers.0.attn.wq_a.weight"
    header = {name: {"dtype": "F8_E4M3", "shape": [2, 2], "data_offsets": [0, 4]}}
    specs = {name: SimpleNamespace(dtype="F8_E4M3", shape=(2, 2))}
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {name: "model.safetensors"}}))
    return SimpleNamespace(root=tmp_path, name=name, header=header, specs=specs, index=index)


def _write_tensor_file(case, payload=b"\x00\x01\x02\x03"):
    raw = json.dumps(case.header).encode()
    path = case.root / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def test_complete_metadata_does_not_claim_numerics(case):
    _write_tensor_file(case)
    report = checkpoint.inspect_checkpoint(case.index, case.specs, checkpoint_dir=case.root)
    assert report["inspection_status"] == "PASS"
    assert report["available_header_status"] == "PASS"
    assert report["metadata_coverage_complete"]
    assert not report["tensor_payload_read"]
    assert report["tensor_numerics"] == "NOT_RUN"
    assert report["npu_upload"] == "NOT_RUN"


def test_header_snapshot_partial_coverage_is_explicit(case):
    (case.root / "model.safetensors.header.json").write_text(json.dumps(case.header))
    case.index.write_text(
        json.dumps({"weight_map": {case.name: "model.safetensors", "vision.weight": "vision.safetensors"}})
    )
    report = checkpoint.inspect_checkpoint(case.index, case.specs, header_dir=case.root)
    assert report["inspection_status"] == "BLOCKED"
    assert report["available_header_status"] == "PASS"
    assert not report["metadata_coverage_complete"]
    assert report["unavailable_shards"] == ["vision.safetensors"]
    assert report["excluded_tensor_counts"] == {"vision": 1}


@pytest.mark.parametrize("source_mode", ["checkpoint_dir", "header_dir"])
def test_missing_required_text_shard_cannot_report_pass(case, source_mode):
    missing_name = "layers.1.attn.wq_a.weight"
    case.specs[missing_name] = SimpleNamespace(dtype="F8_E4M3", shape=(2, 2))
    case.index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {case.name: "model.safetensors", missing_name: "missing.safetensors"},
            }
        )
    )
    if source_mode == "checkpoint_dir":
        _write_tensor_file(case)
    else:
        (case.root / "model.safetensors.header.json").write_text(json.dumps(case.header))
    report = checkpoint.inspect_checkpoint(case.index, case.specs, **{source_mode: case.root})
    assert report["inspection_status"] == "BLOCKED"
    assert report["available_header_status"] == "PASS"
    assert not report["metadata_coverage_complete"]
    assert report["validated_text_tensor_count"] == 1
    assert report["required_text_tensor_count"] == 2
    assert report["unavailable_shards"] == ["missing.safetensors"]
    assert report["errors"] == []


@pytest.mark.parametrize("total_size,expected_status", [(4, "PASS"), (5, "FAIL")])
def test_complete_shard_sizes_are_compared_with_index_total_size(case, total_size, expected_status):
    _write_tensor_file(case)
    case.index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": {case.name: "model.safetensors"},
            }
        )
    )
    report = checkpoint.inspect_checkpoint(case.index, case.specs, checkpoint_dir=case.root)
    assert report["inspection_status"] == expected_status
    assert report["available_header_status"] == expected_status
    assert report["checkpoint_index_declared_bytes"] == total_size
    if expected_status == "FAIL":
        assert report["errors"] == ["index total_size mismatch: declared 5, headers 4"]
        assert not report["metadata_coverage_complete"]
    else:
        assert report["errors"] == []
        assert report["metadata_coverage_complete"]


@pytest.mark.parametrize(
    "metadata,message",
    [
        (None, "metadata must be an object"),
        ([], "metadata must be an object"),
        ("invalid", "metadata must be an object"),
        ({"total_size": None}, "total_size must be a nonnegative integer"),
        ({"total_size": True}, "total_size must be a nonnegative integer"),
        ({"total_size": -1}, "total_size must be a nonnegative integer"),
        ({"total_size": "4"}, "total_size must be a nonnegative integer"),
        ({"total_size": 4.0}, "total_size must be a nonnegative integer"),
    ],
)
def test_malformed_index_metadata_is_rejected(case, metadata, message):
    case.index.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "weight_map": {case.name: "model.safetensors"},
            }
        )
    )
    with pytest.raises(ValueError, match=message):
        checkpoint.read_index(case.index)


def test_missing_shards_are_blocked(case):
    report = checkpoint.inspect_checkpoint(case.index, case.specs, checkpoint_dir=case.root)
    assert report["inspection_status"] == "BLOCKED"
    assert not report["metadata_coverage_complete"]


def test_truncated_tensor_payload_is_rejected_without_loading_it(case):
    path = _write_tensor_file(case, b"\x00")
    with pytest.raises(ValueError, match="buffer size mismatch"):
        checkpoint.read_header(path)


def test_oversized_header_rejected_before_read(case):
    path = case.root / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", checkpoint.MAX_HEADER_BYTES + 1))
    with pytest.raises(ValueError, match="header size"):
        checkpoint.read_header(path)


def test_duplicate_index_keys_rejected(case):
    case.index.write_text('{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        checkpoint.read_index(case.index)


@pytest.mark.parametrize(
    "shard", ["../a.safetensors", "C:\\a.safetensors", "/a.safetensors", ".hidden.safetensors"]
)
def test_unsafe_shard_paths_rejected(case, shard):
    case.index.write_text(json.dumps({"weight_map": {"x": shard}}))
    with pytest.raises(ValueError, match="unsafe"):
        checkpoint.read_index(case.index)


def test_overlapping_offsets_rejected(case):
    case.header["y"] = {"dtype": "I8", "shape": [2], "data_offsets": [2, 4]}
    with pytest.raises(ValueError, match="overlapping"):
        checkpoint.validate_header(case.header)


@pytest.mark.parametrize("shape", [[True, 4], [-1, 4], [5], [1.0, 4]])
def test_shape_bytes_and_boolean_dimensions_rejected(case, shape):
    case.header[case.name]["shape"] = shape
    with pytest.raises(ValueError):
        checkpoint.validate_header(case.header)


def test_wrong_shard_and_missing_expected_tensor_reported(case):
    case.header = {"unknown.weight": case.header[case.name]}
    _write_tensor_file(case)
    report = checkpoint.inspect_checkpoint(case.index, case.specs, checkpoint_dir=case.root)
    assert report["inspection_status"] == "FAIL"
    assert any(("absent from shard" in error for error in report["errors"]))
    assert any(("wrong/unindexed shard" in error for error in report["errors"]))


def test_dtype_mismatch_is_not_format_renamed(case):
    case.header[case.name]["dtype"] = "I8"
    _write_tensor_file(case)
    report = checkpoint.inspect_checkpoint(case.index, case.specs, checkpoint_dir=case.root)
    assert report["inspection_status"] == "FAIL"
    assert "contract mismatch" in report["errors"][0]


def test_required_index_tensor_missing(case):
    case.index.write_text(json.dumps({"weight_map": {"unknown.weight": "model.safetensors"}}))
    report = checkpoint.inspect_checkpoint(case.index, case.specs, checkpoint_dir=case.root)
    assert any(("missing required tensor" in error for error in report["errors"]))
    assert any(("unrecognized text tensor" in error for error in report["errors"]))
