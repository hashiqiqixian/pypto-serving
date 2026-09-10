# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded, metadata-only inspection of the independent V4.1 checkpoint contract.

Inspection never allocates tensors or treats header validation as numerical validation.
The caller supplies an explicit weight spec; V4's layout and constants are not used.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Protocol

MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
}


class WeightSpec(Protocol):
    """Metadata required from a source weight specification."""

    shape: tuple[int, ...]
    dtype: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def read_header(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read a safetensors header without reading its potentially huge data buffer."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors length: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header size: {length}")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"truncated safetensors header: {path.name}")
    header = json.loads(raw, object_pairs_hook=_unique_object)
    validate_header(header, path.stat().st_size - 8 - length)
    return header, hashlib.sha256(raw).hexdigest()


def validate_header(header: Any, data_size: int | None = None) -> None:
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ValueError("safetensors metadata must contain strings")
            continue
        if not isinstance(tensor, dict):
            raise ValueError(f"invalid tensor descriptor: {name}")
        shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"invalid shape: {name}")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported storage dtype: {name}: {dtype}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ValueError(f"invalid data offsets: {name}")
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected:
            raise ValueError(f"shape/dtype byte length mismatch: {name}")
        spans.append((offsets[0], offsets[1], name))
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ValueError(f"non-contiguous or overlapping data offsets: {name}")
        cursor = end
    if data_size is not None and cursor != data_size:
        raise ValueError(f"tensor buffer size mismatch: expected {cursor}, actual {data_size}")


def read_index(path: str | Path) -> dict[str, Any]:
    index = read_json(path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index requires a nonempty weight_map")
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint index metadata must be an object")
    if "total_size" in metadata and (type(metadata["total_size"]) is not int or metadata["total_size"] < 0):
        raise ValueError("checkpoint index total_size must be a nonnegative integer")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str):
            raise ValueError("weight_map requires nonempty names and string filenames")
        if (
            not shard.endswith(".safetensors")
            or "/" in shard
            or "\\" in shard
            or ":" in shard
            or shard.startswith(".")
        ):
            raise ValueError(f"unsafe checkpoint shard filename: {shard}")
    return index


def inspect_checkpoint(
    index_path: str | Path,
    specs: Mapping[str, WeightSpec],
    *,
    checkpoint_dir: str | Path | None = None,
    header_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate index coverage and every supplied header against an explicit text spec.

    Vision/image and draft tensors are counted separately and are never called validated.
    In header_dir mode, files are named <shard>.header.json (HTTP range snapshots).
    Missing snapshots block full coverage but do not invalidate a successful subset check.
    """
    if (checkpoint_dir is None) == (header_dir is None):
        raise ValueError("provide exactly one of checkpoint_dir or header_dir")
    index_path = Path(index_path)
    index = read_index(index_path)
    weight_map = index["weight_map"]
    errors = []
    excluded = Counter()
    for name in sorted(weight_map.keys() - specs.keys()):
        if name.startswith(("vision.", "aligner.", "image_")):
            excluded["vision"] += 1
        elif name.startswith("mtp."):
            excluded["draft"] += 1
        else:
            errors.append(f"unrecognized text tensor: {name}")
    errors.extend(f"missing required tensor: {name}" for name in sorted(specs.keys() - weight_map.keys()))
    root = Path(checkpoint_dir or header_dir).resolve()
    by_shard = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, set()).add(name)
    hashes, unavailable, checked, checked_bytes = {}, [], 0, 0
    all_tensor_bytes = 0
    for shard, indexed_names in sorted(by_shard.items()):
        path = root / (shard if checkpoint_dir else shard + ".header.json")
        if not path.is_file():
            unavailable.append(shard)
            continue
        if path.resolve().parent != root:
            errors.append(f"shard resolves outside checkpoint directory: {shard}")
            continue
        try:
            if checkpoint_dir:
                header, digest = read_header(path)
            else:
                if path.stat().st_size > MAX_HEADER_BYTES:
                    raise ValueError("oversized header snapshot")
                raw = path.read_bytes()
                header = json.loads(raw, object_pairs_hook=_unique_object)
                validate_header(header)
                digest = hashlib.sha256(raw).hexdigest()
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append(f"invalid shard {shard}: {exc}")
            continue
        hashes[shard] = digest
        actual_names = set(header) - {"__metadata__"}
        all_tensor_bytes += sum(
            header[name]["data_offsets"][1] - header[name]["data_offsets"][0] for name in actual_names
        )
        errors.extend(
            f"indexed tensor absent from shard {shard}: {name}"
            for name in sorted(indexed_names - actual_names)
        )
        errors.extend(
            f"tensor in wrong/unindexed shard {shard}: {name}"
            for name in sorted(actual_names - indexed_names)
        )
        for name in sorted(actual_names & specs.keys()):
            spec, tensor = specs[name], header[name]
            if tuple(tensor["shape"]) != spec.shape or tensor["dtype"] != spec.dtype:
                errors.append(
                    f"contract mismatch {name}: expected {spec.shape}/{spec.dtype}, "
                    f"got {tensor['shape']}/{tensor['dtype']}"
                )
            else:
                checked += 1
                checked_bytes += tensor["data_offsets"][1] - tensor["data_offsets"][0]
    declared_bytes = index.get("metadata", {}).get("total_size")
    if len(hashes) == len(by_shard) and declared_bytes is not None and declared_bytes != all_tensor_bytes:
        errors.append(f"index total_size mismatch: declared {declared_bytes}, headers {all_tensor_bytes}")
    return {
        "schema_version": 1,
        "model_type": "deepseek_v41",
        "inspection_status": "FAIL" if errors else ("PASS" if checked and not unavailable else "BLOCKED"),
        "available_header_status": "FAIL" if errors else ("PASS" if checked else "NOT_RUN"),
        "validation_level": "checkpoint_index_and_available_header_metadata",
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "index_tensor_count": len(weight_map),
        "required_text_tensor_count": len(specs),
        "validated_text_tensor_count": checked,
        "validated_text_storage_bytes": checked_bytes,
        "excluded_tensor_counts": dict(excluded),
        "header_sha256": hashes,
        "unavailable_shards": unavailable,
        "metadata_coverage_complete": not unavailable and not errors and checked == len(specs),
        "tensor_payload_read": False,
        "full_shard_file_sizes_checked": checkpoint_dir is not None and len(hashes) == len(by_shard),
        "tensor_numerics": "NOT_RUN",
        "npu_upload": "NOT_RUN",
        "checkpoint_index_declared_bytes": index.get("metadata", {}).get("total_size"),
        "errors": errors,
    }
