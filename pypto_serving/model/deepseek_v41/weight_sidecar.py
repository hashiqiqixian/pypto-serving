# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional, fingerprinted V4.1 resident-weight sidecar, following V4's path."""

from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import tempfile

import torch

from pypto_serving.model.deepseek.weight_loader import (
    _map_shared_prepacked_tensors,
    _sample_file_page_cache_residency,
    _PREPACKED_MIN_CACHE_RESIDENCY,
)

from .decode_weight_resources import DecodeWeightBank, _ALL, _SINGLE


_LOG = logging.getLogger(__name__)
_FORMAT = "pypto-deepseek-v41-decode-slot-major-v1"
_FLOAT8 = {"float8_e4m3fn": torch.float8_e4m3fn,
           "float8_e8m0fnu": torch.float8_e8m0fnu}


def sidecar_path(model_dir):
    override = os.environ.get("PYPTO_V41_PREPACKED_WEIGHTS")
    return Path(override) if override else Path(model_dir) / "pypto-deepseek-v41-decode-r8.safetensors"


def source_fingerprint(model_dir, layout):
    model_dir = Path(model_dir)
    config = (model_dir / "config.json").read_bytes()
    index = (model_dir / "model.safetensors.index.json").read_bytes()
    weight_map = json.loads(index)["weight_map"]
    source_files = []
    for filename in sorted(set(weight_map.values())):
        source = (model_dir / filename).stat()
        source_files.append((filename, source.st_size, source.st_mtime_ns))
    payload = json.dumps({
        "format": _FORMAT,
        "layout": asdict(layout),
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "source_files": source_files,
    }, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_prepacked_bank(model_dir, layout, parameter_names, *, path=None):
    path = sidecar_path(model_dir) if path is None else Path(path)
    if not path.is_file():
        return None
    from safetensors import safe_open

    fd = os.open(path, os.O_RDONLY)
    try:
        residency = _sample_file_page_cache_residency(fd, path)
        if residency is None or residency < _PREPACKED_MIN_CACHE_RESIDENCY:
            _LOG.info("Skipping cold V4.1 weight sidecar %s (residency %.1f%%)",
                      path, 0 if residency is None else 100 * residency)
            return None
        with safe_open(f"/proc/self/fd/{fd}", framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            if metadata.get("format") != _FORMAT or metadata.get("source_fingerprint") != source_fingerprint(
                    model_dir, layout):
                _LOG.warning("Ignoring stale V4.1 weight sidecar %s", path)
                return None
            names = frozenset(reader.keys())
            if names != _ALL:
                _LOG.warning("Ignoring V4.1 weight sidecar with wrong fields: %s", path)
                return None
            float8_fields = json.loads(metadata.get("float8_fields", "{}"))
            if any(name not in _ALL or dtype not in _FLOAT8 for name, dtype in float8_fields.items()):
                raise ValueError("V4.1 weight sidecar has invalid float8 metadata")
            for name in names:
                source = reader.get_slice(name)
                shape = tuple(source.get_shape())
                if (shape[0] != (layout.world if name in _SINGLE else layout.slots(name))
                        or (name not in _SINGLE and (len(shape) < 3 or shape[1] != layout.world))):
                    raise ValueError(f"V4.1 sidecar weight has invalid rank/slot shape: {name}")
                if name in float8_fields and source.get_dtype() != "U8":
                    raise ValueError(f"V4.1 sidecar float8 weight is not byte-backed: {name}")
        tensors = _map_shared_prepacked_tensors(fd, names)
        for name, dtype in float8_fields.items():
            tensors[name] = tensors[name].view(_FLOAT8[dtype])
        bank = DecodeWeightBank(layout)
        bank.tensors = tensors
        bank.allocated_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        bank._seen = {(name, slot) for name in _ALL for slot in range(layout.slots(name))}
        bank.prepacked = True
        bank.finish(parameter_names)
        _LOG.info("Mapped prepacked V4.1 weight bank: %.1f GiB from %s",
                  bank.allocated_bytes / (1 << 30), path)
        return bank
    finally:
        os.close(fd)


def write_prepacked_bank(model_dir, bank, *, path=None, force=False):
    """Publish a completed bank atomically; never overwrite a valid file implicitly."""
    from safetensors.torch import save_file

    bank.finish(_ALL)
    path = sidecar_path(model_dir) if path is None else Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"V4.1 weight sidecar already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    reference = Path(model_dir) / sorted(set(index["weight_map"].values()))[0]
    output_mode = stat.S_IMODE(reference.stat().st_mode) & 0o666
    float8_fields = {name: str(value.dtype).removeprefix("torch.")
                     for name, value in bank.tensors.items() if value.dtype in _FLOAT8.values()}
    tensors = {name: value.view(torch.uint8) if name in float8_fields else value
               for name, value in bank.tensors.items()}
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        save_file(tensors, str(temporary), metadata={
            "format": _FORMAT,
            "source_fingerprint": source_fingerprint(model_dir, bank.layout),
            "float8_fields": json.dumps(float8_fields, sort_keys=True),
        })
        os.chmod(temporary, output_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
