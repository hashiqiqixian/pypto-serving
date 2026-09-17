# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Evidence gate tests with synthetic artifacts; these do not establish device acceptance."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[4] / "pypto_serving/tools/validate_deepseek_v41_m0.py"
    spec = importlib.util.spec_from_file_location("_v41_m0_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def reference(module):
    return {"reference_revision": module.REFERENCE_REVISION, "checkpoint_revision": "test-checkpoint",
            "config_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "cases": [{"case_id": name, "prompt_token_ids": [token] * 8192, "token_ids": [token + 10] * 128}
                      for name, token in (("A", 1), ("B", 2))]}


@pytest.fixture
def actual(reference):
    result = {key: reference[key] for key in ("checkpoint_revision", "config_sha256", "tokenizer_sha256")}
    result.update(schema="pypto_v41_greedy_trace_v1", platform="a5", world_size=8, cases=[])
    for index, case in enumerate([*reference["cases"], reference["cases"][0]]):
        result["cases"].append({**deepcopy(case), "temperature": 0, "num_speculative_tokens": 0,
                                "host_requests_after": [], "scheduler_events": [
                                    {"request_id": str(index), "phase": "prefill", "tokens": 8192},
                                    *[{"request_id": str(index), "phase": "decode", "tokens": 1} for _ in range(127)]]})
    return result


def test_token_trace_checks_complete_ids_and_reports_first_mismatch(module, reference, actual):
    assert module.compare_traces(reference, actual, "a5")["status"] == "PASS"
    actual["cases"][1]["token_ids"][90] = 999
    result = module.compare_traces(reference, actual, "a5")
    assert result["status"] == "FAIL"
    assert result["cases"][1]["first_mismatch"] == 90
    assert result["cases"][1]["actual_token"] == [999]


@pytest.mark.parametrize("change", ["chunked", "short", "missing_repeat", "retained_host", "wrong_platform", "malformed"])
def test_trace_does_not_pass_incomplete_or_wrong_workload(module, reference, actual, change):
    if change == "chunked":
        actual["cases"][0]["scheduler_events"][0]["tokens"] = 4096
    elif change == "short":
        actual["cases"][0]["token_ids"].pop()
    elif change == "missing_repeat":
        actual["cases"].pop()
    elif change == "retained_host":
        actual["cases"][0]["host_requests_after"] = ["request-0"]
    elif change == "wrong_platform":
        actual["platform"] = "a2a3"
    else:
        actual["cases"][0]["scheduler_events"] = None
    assert module.compare_traces(reference, actual, "a5")["status"] == "FAIL"


def test_inspection_requires_measured_coverage_not_claimed_pass(module, reference):
    assert module.check_inspection({"inspection_status": "PASS"}, reference)["status"] == "FAIL"
    artifact = {"schema_version": 1, "model_type": "deepseek_v41", "errors": [], "index_sha256": "c" * 64,
                "checkpoint_revision": reference["checkpoint_revision"], "config_sha256": reference["config_sha256"],
                "required_text_tensor_count": 10, "validated_text_tensor_count": 10,
                "header_sha256": {"part.safetensors": "d" * 64}, "unavailable_shards": []}
    assert module.check_inspection(artifact, reference)["status"] == "PASS"
    artifact["validated_text_tensor_count"] = 5
    artifact["unavailable_shards"] = ["missing.safetensors"]
    assert module.check_inspection(artifact, reference)["status"] == "NOT_RUN"


def test_http_gate_recomputes_measurements_and_repeatability(module, reference):
    artifact = {"schema": "pypto_http_benchmark_v1", "concurrency": 1, "max_tokens": 128,
                "workload_sha256": "c" * 64, "workload_rows": 2,
                "server_manifest": {**reference, "platform": "a5", "world_size": 8},
                "requests": [{"index": i, "request_id": str(i), "success": True, "prompt_tokens": 8192,
                              "completion_tokens": 128, "finish_reason": "length",
                              "output_sha256": ("a" if i % 2 == 0 else "b") * 64} for i in range(3)]}
    assert module.check_http(artifact, reference, "a5")["status"] == "PASS"
    artifact["workload_checks"] = {"sequential_repeatability": {"passed": True}}
    artifact["requests"][2]["output_sha256"] = "c" * 64
    result = module.check_http(artifact, reference, "a5")
    assert result["status"] == "FAIL" and result["failed_request_indices"] == [2]


def test_report_keeps_missing_hardware_evidence_not_run(module, reference, actual, tmp_path):
    ref, trace, report = [tmp_path / name for name in ("reference.json", "actual.json", "report.json")]
    ref.write_text(json.dumps(reference))
    trace.write_text(json.dumps(actual))
    assert module.main(["--reference", str(ref), "--actual", str(trace), "--platform", "a5",
                        "--output", str(report)]) == 2
    value = json.loads(report.read_text())
    assert value["acceptance_status"] == "NOT_RUN"
    assert value["checks"]["greedy_8k_128"]["status"] == "PASS"
    assert value["checks"]["device_cache_rollback_and_reuse"]["status"] == "NOT_RUN"
    assert value["checks"]["per_rank_peak_hbm"]["status"] == "NOT_RUN"
    assert len(value["artifact_sha256"]["actual"]) == 64


@pytest.mark.parametrize("fail", [False, True])
def test_engine_capture_uses_pretokenized_input_and_restores_observer(module, reference, monkeypatch, fail):
    @dataclass
    class PreparedPrompt:
        text: str
        token_ids: list[int]

    monkeypatch.setitem(sys.modules, "pypto_serving.model.tokenizer", SimpleNamespace(PreparedPrompt=PreparedPrompt))
    monkeypatch.setitem(sys.modules, "pypto_serving.config.types", SimpleNamespace(GenerateConfig=SimpleNamespace))
    pending = []
    schedule = lambda: SimpleNamespace(scheduled_requests=list(pending))
    engine = SimpleNamespace(scheduler=SimpleNamespace(schedule=schedule, requests={}), starts=0, stops=0)

    async def start():
        engine.starts += 1

    async def stop():
        engine.stops += 1

    async def generate(prompt, config):
        assert isinstance(prompt, PreparedPrompt) and len(prompt.token_ids) == 8192
        assert config.max_new_tokens == 128 and config.temperature == 0 and not config.stream
        if fail:
            raise RuntimeError("device failure")
        pending[:] = [SimpleNamespace(request=SimpleNamespace(request_id="observed"), is_prefill=True, num_new_tokens=8192)]
        engine.scheduler.schedule()
        return SimpleNamespace(token_ids=[prompt.token_ids[0] + 10] * 128, finish_reason="length")

    engine.start, engine.stop, engine.generate_result = start, stop, generate
    if fail:
        with pytest.raises(RuntimeError, match="device failure"):
            asyncio.run(module.capture_engine(engine, reference))
    else:
        cases = asyncio.run(module.capture_engine(engine, reference))
        assert [case["case_id"] for case in cases] == ["A", "B", "A"]
        assert cases[0]["token_ids"] == reference["cases"][0]["token_ids"]
        assert cases[0]["scheduler_events"][0]["tokens"] == 8192
    assert engine.starts == 1 and engine.stops == 1 and engine.scheduler.schedule is schedule
