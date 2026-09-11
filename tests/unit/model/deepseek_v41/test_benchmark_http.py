# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Streaming benchmark accounting tests; protocol fixtures are not model inference."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "pypto_serving/tools/benchmark_http.py"
    spec = importlib.util.spec_from_file_location("_http_benchmark_test", path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    return result


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def test_stream_uses_server_token_usage_and_first_content_time(module):
    times = iter([12., 15., 18.])
    lines = [event({"choices": [{"delta": {"role": "assistant"}}]}),
             event({"choices": [{"delta": {"content": "first"}}]}),
             event({"choices": [{"delta": {"content": " rest"}, "finish_reason": "length"}]}),
             event({"usage": {"prompt_tokens": 8192, "completion_tokens": 4}, "choices": []}), b"data: [DONE]\n"]
    result = module.measure_stream(lines, 10., clock=lambda: next(times))
    assert result == {"latency_s": 8., "ttft_s": 2., "tpot_s": 1., "prompt_tokens": 8192,
                      "completion_tokens": 4, "finish_reason": "length"}


@pytest.mark.parametrize("lines", [[], [b"data: [DONE]\n"], [event({"error": {"message": "failed"}})],
                                  [event({"choices": [{"text": "a", "finish_reason": "stop"}]})]])
def test_incomplete_or_failed_stream_is_not_a_success(module, lines):
    with pytest.raises(ValueError):
        module.measure_stream(lines, 0., clock=lambda: 1.)


def test_benchmark_success_throughput_goodput_and_null_memory(module):
    rows = [module.Measurement(0, True, 5., 1., 2., 8192, 3, "length", None),
            module.Measurement(1, True, 2., .5, .75, 8192, 3, "length", None),
            module.Measurement(2, False, 4., None, None, None, None, None, "timeout")]
    result = module.summarize(rows, 10., ttft_slo=2., tpot_slo=1.)
    assert result["success_rate"] == 2 / 3
    assert result["output_tokens_per_s"] == .6
    assert result["goodput_requests_per_s"] == .1
    assert result["ttft_s"]["p95"] == 1.
    assert "peak_hbm" not in result
    assert module.summarize(rows, 10.)["goodput_requests_per_s"] is None


def test_stream_byte_limit(module):
    with pytest.raises(ValueError, match="byte limit"):
        module.measure_stream([b"data: " + b"x" * 20], 0., max_bytes=10)


@pytest.mark.parametrize("reason", ["aborted", "error", "content_filter"])
def test_aborted_stream_with_usage_is_not_success(module, reason):
    lines = [event({"choices": [{"text": "partial", "finish_reason": reason}]}),
             event({"usage": {"prompt_tokens": 4, "completion_tokens": 1}}), b"data: [DONE]\n"]
    with pytest.raises(ValueError, match="successful"):
        module.measure_stream(lines, 0., clock=lambda: 1.)


def test_serving_eos_finish_is_success(module):
    lines = [event({"choices": [{"text": "done", "finish_reason": "eos"}]}),
             event({"usage": {"prompt_tokens": 4, "completion_tokens": 1}}), b"data: [DONE]\n"]
    assert module.measure_stream(lines, 0., clock=lambda: 1.)["finish_reason"] == "eos"


def test_immediate_eos_has_no_invented_visible_token_time(module):
    lines = [event({"choices": [{"text": "", "finish_reason": "eos"}]}),
             event({"usage": {"prompt_tokens": 4, "completion_tokens": 1}}), b"data: [DONE]\n"]
    result = module.measure_stream(lines, 0., clock=lambda: 1.)
    assert result["ttft_s"] is None and result["tpot_s"] is None
    assert result["completion_tokens"] == 1 and result["finish_reason"] == "eos"
