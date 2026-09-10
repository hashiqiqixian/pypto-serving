# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the actual inspection CLI's output protection and failure reports."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/inspect_deepseek_v41.py"


@pytest.fixture
def case(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    preserved = source / "existing-source-file.json"
    preserved.write_bytes(b"source data must remain unchanged\n")
    return SimpleNamespace(
        root=tmp_path,
        config=config,
        index=index,
        source=source,
        preserved=preserved,
        revision="test-fixture-revision",
    )


def _run(case, report, source_option):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(case.config),
            "--index",
            str(case.index),
            source_option,
            str(case.source),
            "--revision",
            case.revision,
            "--report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("source_option", ["--checkpoint-dir", "--header-dir"])
@pytest.mark.parametrize("destination", ["config", "index", "preserved", "new_source_file"])
def test_report_cannot_overwrite_inputs_or_write_inside_source_directory(case, source_option, destination):
    before = {path: path.read_bytes() for path in (case.config, case.index, case.preserved)}
    report = (
        case.source / "nested" / "new-report.json"
        if destination == "new_source_file"
        else getattr(case, destination)
    )
    result = _run(case, report, source_option)
    assert result.returncode == 2
    assert "--report must be outside" in result.stderr
    assert "must not overwrite inputs" in result.stderr
    assert {path: path.read_bytes() for path in before} == before
    if destination == "new_source_file":
        assert not report.exists()
        assert not report.parent.exists()


@pytest.mark.parametrize("source_option", ["--checkpoint-dir", "--header-dir"])
def test_minimal_invalid_config_writes_explicit_fail_report(case, source_option):
    before = {path: path.read_bytes() for path in (case.config, case.index, case.preserved)}
    path = case.root / "reports" / "inspection.json"
    result = _run(case, path, source_option)
    assert result.returncode == 1
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["inspection_status"] == "FAIL"
    assert report["checkpoint_revision"] == case.revision
    assert any("architectures" in message for message in report["errors"])
    assert json.loads(result.stdout) == report
    assert "Traceback" not in result.stderr
    assert {path: path.read_bytes() for path in before} == before


def test_atomic_report_does_not_truncate_hardlinked_input(case):
    report = case.root / "report.json"
    report.hardlink_to(case.config)
    before = case.config.read_bytes()
    result = _run(case, report, "--header-dir")
    assert result.returncode == 1
    assert case.config.read_bytes() == before
    assert json.loads(report.read_text())["inspection_status"] == "FAIL"
    assert not report.samefile(case.config)
