# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Measure cold Palace 64->128 K=7 acceptance using engine metrics, not runner logs.

Run under task-submit with TASK_DEVICE, PYPTO_DSPARK_EP_SIZE, PYPTO_LIB_ROOT,
and optionally PYPTO_DSV4_DSPARK_MODEL_DIR. --output-dir must not already exist.
No acceptance-rate threshold is imposed; compare identical settings across revisions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepseek_dspark_accuracy import (  # noqa: E402
    DEFAULT_MODEL_DIR, GREEDY_CASES, MODEL_ID, OVERALL_TIMEOUT_SECONDS, ROOT,
    _request_completion, _server_command, _stop_process_group, _task_devices,
    _unused_local_port, _wait_for_device_reclaim, _wait_for_health,
)
from test_deepseek_v4_accuracy import LOCAL_URL_OPENER  # noqa: E402


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30,
    ).strip()


def _summary(metrics: dict) -> dict:
    replicas = metrics["replicas"]
    totals = {
        key: sum(replica["counters"][key] for replica in replicas)
        for key in ("speculative_drafts", "draft_tokens", "accepted_tokens", "speculative_fallbacks")
    }
    rounds, drafted, accepted = (totals[key] for key in (
        "speculative_drafts", "draft_tokens", "accepted_tokens",
    ))
    assert rounds > 0 and drafted > 0, "no speculative verification recorded by engine metrics"
    assert 0 <= accepted <= drafted
    positions = [
        sum(replica["accepted_tokens_per_pos"][position] for replica in replicas
            if position < len(replica["accepted_tokens_per_pos"]))
        for position in range(7)
    ]
    assert sum(positions) == accepted, "per-position counts disagree with accepted draft count"
    return {
        **totals, "draft_acceptance_rate": accepted / drafted,
        "mean_acceptance_length": 1 + accepted / rounds,
        "accepted_tokens_per_pos": positions,
        "counting": (
            "Actual verified drafts before EOS/length truncation, including the final round; "
            "accepted_tokens excludes bonus; mean_acceptance_length includes one bonus per verify"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("PYPTO_LIB_ROOT"):
        parser.error("set PYPTO_LIB_ROOT explicitly to the kernel checkout being measured")
    lib_root = Path(os.environ["PYPTO_LIB_ROOT"]).resolve()
    model = Path(os.environ.get("PYPTO_DSV4_DSPARK_MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()
    if not model.is_dir():
        parser.error(f"model directory does not exist: {model}")
    devices = _task_devices()
    case = next(case for case in GREEDY_CASES if case.case_id == "palace-64-128-k7")
    port = _unused_local_port()
    command = _server_command(model, devices, port, num_speculative_tokens=7, enable_prefix_caching=False)
    metadata = {
        "case": case.case_id, "model": str(model), "devices": devices, "server_command": command,
        "serving_sha": _git_sha(ROOT), "pypto_lib_sha": _git_sha(lib_root),
        "pypto_lib_root": str(lib_root), "prefix_caching": False, "num_speculative_tokens": 7,
        "prompt": case.prompt, "prompt_tokens": case.prompt_tokens, "max_new_tokens": case.max_new_tokens,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    with (args.output_dir / "server.log").open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        try:
            _wait_for_health(process, port, deadline)
            response = _request_completion(
                process, port, deadline, prompt=case.prompt,
                max_new_tokens=case.max_new_tokens, model=MODEL_ID,
            )
            (args.output_dir / "response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("overall timeout reached before metrics capture")
            with LOCAL_URL_OPENER.open(
                f"http://127.0.0.1:{port}/metrics/json", timeout=min(30, remaining),
            ) as reply:
                metrics = json.loads(reply.read())
            (args.output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
            )
            assert response.get("model") == MODEL_ID
            choices = response.get("choices", [])
            assert len(choices) == 1 and choices[0].get("finish_reason") == "length"
            usage = response.get("usage", {})
            assert usage.get("prompt_tokens") == case.prompt_tokens
            assert usage.get("completion_tokens") == case.max_new_tokens
            summary = _summary(metrics)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8",
            )
            print(json.dumps(summary), flush=True)
        finally:
            try:
                _stop_process_group(process)
            finally:
                _wait_for_device_reclaim(devices)


if __name__ == "__main__":
    main()
