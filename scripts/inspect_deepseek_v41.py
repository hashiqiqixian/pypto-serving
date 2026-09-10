# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inspect V4.1 metadata without importing torch or initializing a serving runtime."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _load(name: str):
    # The package root imports the runtime. Load this dependency-free inspection tool
    # explicitly so checkpoint metadata can be examined before NPU dependencies exist.
    path = Path(__file__).resolve().parents[1] / "pypto_serving" / "model" / "deepseek_v41" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_v41_inspect_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint-dir", type=Path)
    inputs.add_argument("--header-dir", type=Path)
    parser.add_argument("--revision", required=True, help="Exact source checkpoint revision for provenance")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.report.resolve()
    source_root = (args.checkpoint_dir or args.header_dir).resolve()
    if output in (args.config.resolve(), args.index.resolve()) or output.is_relative_to(source_root):
        parser.error("--report must be outside the checkpoint/header directory and must not overwrite inputs")
    checkpoint, config_module, weights = (_load(n) for n in ("checkpoint", "config", "weight_spec"))
    try:
        raw_config = checkpoint.read_json(args.config)
        config_module.DeepSeekV41Config.from_dict(raw_config)
        specs = weights.backbone_weight_specs(raw_config)
        report = checkpoint.inspect_checkpoint(
            args.index, specs, checkpoint_dir=args.checkpoint_dir, header_dir=args.header_dir
        )
        report["checkpoint_revision"] = args.revision
        report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = {"inspection_status": "FAIL", "errors": [str(exc)], "checkpoint_revision": args.revision}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.report.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(report, indent=2) + "\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, args.report)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in report.items() if key != "header_sha256"}, indent=2))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[report["inspection_status"]]


if __name__ == "__main__":
    raise SystemExit(main())
