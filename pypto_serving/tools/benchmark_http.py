# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded streaming HTTP benchmark with measured usage, latency and explicit provenance.

This client performs real requests to an explicitly supplied server. Missing
stream usage is a failed measurement, not an estimated tokenizer count. Device
memory/OOM observations belong in the accompanying server manifest; client
latencies do not establish accelerator memory use or an OOM capacity boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_STREAM_BYTES = 64 << 20


@dataclass(frozen=True)
class Measurement:
    index: int
    success: bool
    latency_s: float
    ttft_s: float | None
    tpot_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    error: str | None
    request_id: str | None = None
    output_sha256: str | None = None


def measure_stream(lines, started: float, *, clock=time.perf_counter, max_bytes=MAX_STREAM_BYTES,
                   capture_identity=False) -> dict:
    """Read OpenAI SSE events; require a finish event, [DONE], and authoritative usage."""
    first = last = None
    usage = reason = None
    total = 0
    done = False
    request_id = None
    digests = {}
    for raw in lines:
        total += len(raw)
        if total > max_bytes:
            raise ValueError("stream exceeded the configured byte limit")
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if data == b"[DONE]":
            done = True
            break
        event = json.loads(data)
        if "error" in event:
            raise ValueError("server returned a streaming error")
        event_id = event.get("id")
        if event_id is not None:
            if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
                raise ValueError("stream request ID must be a bounded nonempty string")
            if request_id is not None and event_id != request_id:
                raise ValueError("stream changed request ID")
            request_id = event_id
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if capture_identity:
                index = choice.get("index", 0)
                if type(index) is not int or index < 0 or index > 64:
                    raise ValueError("stream choice index is invalid")
                for channel, text in (("text", choice.get("text")), ("content", delta.get("content")),
                                      ("reasoning_content", delta.get("reasoning_content"))):
                    if text is not None:
                        if not isinstance(text, str):
                            raise ValueError("stream output must be text")
                        digests.setdefault((index, channel), hashlib.sha256()).update(text.encode("utf-8"))
            content = choice.get("text") or delta.get("content") or delta.get("reasoning_content")
            if content:
                last = clock()
                if first is None:
                    first = last
            if choice.get("finish_reason") is not None:
                reason = choice["finish_reason"]
    ended = clock()
    if not done or reason is None or not isinstance(usage, dict):
        raise ValueError("stream is missing completion, finish reason or usage")
    if reason not in ("stop", "length", "eos"):
        raise ValueError("stream did not finish with a successful text-generation reason")
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(type(count) is not int or count < 0 for count in (prompt, completion)):
        raise ValueError("stream usage token counts must be nonnegative integers")
    if completion and first is None and reason != "eos":
        raise ValueError("stream usage reports tokens without a timed output event")
    result = {"latency_s": ended - started, "ttft_s": None if first is None else first - started,
            "tpot_s": (last - first) / (completion - 1) if completion > 1 and first is not None else None,
            "prompt_tokens": prompt, "completion_tokens": completion, "finish_reason": reason}
    if capture_identity:
        content = [(index, channel, digest.hexdigest()) for (index, channel), digest in sorted(digests.items())]
        result.update(request_id=request_id, output_sha256=hashlib.sha256(json.dumps(content).encode()).hexdigest())
    return result


def request_once(index: int, endpoint: str, payload: dict, timeout: float) -> Measurement:
    started = time.perf_counter()
    try:
        request = Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            def lines():
                total = 0
                while True:
                    if time.perf_counter() - started > timeout:
                        raise TimeoutError("request exceeded its elapsed-time budget")
                    line = response.readline(min(1 << 20, MAX_STREAM_BYTES - total + 1))
                    if not line:
                        return
                    total += len(line)
                    if total > MAX_STREAM_BYTES or (len(line) == 1 << 20 and not line.endswith(b"\n")):
                        raise ValueError("stream event or response exceeds byte budget")
                    yield line
            result = measure_stream(lines(), started, capture_identity=True)
        return Measurement(index, True, **result, error=None)
    except Exception as error:
        # Do not include response bodies, URLs, or prompts in error diagnostics.
        return Measurement(index, False, time.perf_counter() - started, None, None, None, None, None,
                           f"{type(error).__name__}: {str(error)[:160]}")


def summarize(measurements: list[Measurement], elapsed: float, *, ttft_slo=None, tpot_slo=None) -> dict:
    if not measurements or elapsed <= 0:
        raise ValueError("summary requires measured requests and a positive elapsed window")
    for slo in (ttft_slo, tpot_slo):
        if slo is not None and (not math.isfinite(slo) or slo <= 0):
            raise ValueError("SLOs must be finite positive seconds")
    success = [value for value in measurements if value.success]
    def distribution(field):
        values = sorted(getattr(value, field) for value in success if getattr(value, field) is not None)
        if not values:
            return None
        return {"mean": statistics.mean(values), "p50": values[math.ceil(.5 * len(values)) - 1],
                "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1]}
    good = None
    if ttft_slo is not None or tpot_slo is not None:
        good = sum((ttft_slo is None or (value.ttft_s is not None and value.ttft_s <= ttft_slo))
                   and (tpot_slo is None or (value.tpot_s is not None and value.tpot_s <= tpot_slo))
                   for value in success) / elapsed
    return {"requests": len(measurements), "successful_requests": len(success),
            "success_rate": len(success) / len(measurements), "elapsed_s": elapsed,
            "output_tokens_per_s": sum(value.completion_tokens for value in success) / elapsed,
            "successful_requests_per_s": len(success) / elapsed, "goodput_requests_per_s": good,
            "ttft_slo": ttft_slo, "tpot_slo": tpot_slo,
            **{field: distribution(field) for field in ("latency_s", "ttft_s", "tpot_s")}}


def check_workload(measurements, workload_rows, *, expected_prompt=None, expected_completion=None,
                   repeat=False):
    """Check measured counts and sequential repeatability without inferring cache internals."""
    if type(workload_rows) is not int or workload_rows < 1:
        raise ValueError("workload_rows must be a positive integer")
    checks = {}
    for field, expected in (("prompt_tokens", expected_prompt), ("completion_tokens", expected_completion)):
        if expected is not None:
            failed = [row.index for row in measurements if not row.success or getattr(row, field) != expected]
            checks[field] = {"passed": bool(measurements) and not failed, "expected": expected,
                             "failed_request_indices": failed}
    if repeat:
        previous = {}
        failures = []
        repeats = 0
        request_ids = set()
        for row in measurements:
            key = row.index % workload_rows
            if (not row.success or not row.request_id or not row.output_sha256
                    or row.request_id in request_ids):
                failures.append(row.index)
            request_ids.add(row.request_id)
            signature = (row.output_sha256, row.prompt_tokens, row.completion_tokens, row.finish_reason)
            if key in previous:
                repeats += 1
                if previous[key] != signature:
                    failures.append(row.index)
            previous[key] = signature
        checks["sequential_repeatability"] = {
            "passed": repeats > 0 and not failures, "repeated_requests": repeats,
            "failed_request_indices": sorted(set(failures)),
            "scope": "HTTP text/reasoning digests and usage; no token-ID or internal cache comparison",
        }
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Explicit /v1/completions or /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL objects containing prompt or messages")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-manifest", type=Path, help="Recorded model/config/commit/device memory provenance JSON")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--ttft-slo", type=float)
    parser.add_argument("--tpot-slo", type=float)
    parser.add_argument("--expect-prompt-tokens", type=int, help="Fail if authoritative prompt usage differs")
    parser.add_argument("--expect-completion-tokens", type=int, help="Fail if output stops before this count")
    parser.add_argument("--check-repeats", action="store_true",
                        help="Check sequential repeats by output digest; does not establish token golden or rollback")
    args = parser.parse_args(argv)
    parsed = urlsplit(args.endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        parser.error("endpoint must be an explicit HTTP(S) URL without credentials")
    if not 1 <= args.concurrency <= 64 or not 1 <= args.requests <= 10000 or args.max_tokens < 1:
        parser.error("concurrency must be 1..64, requests 1..10000 and max-tokens positive")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        parser.error("timeout must be finite and in (0,3600] seconds")
    if any(value is not None and value < 1 for value in (args.expect_prompt_tokens, args.expect_completion_tokens)):
        parser.error("expected token counts must be positive")
    if args.check_repeats and args.concurrency != 1:
        parser.error("--check-repeats requires --concurrency 1")
    if args.prompts.stat().st_size > 64 << 20:
        parser.error("prompt file exceeds 64 MiB")
    raw = args.prompts.read_bytes()
    prompts = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not prompts or any(not isinstance(value, dict) or (set(value) != {"prompt"} and set(value) != {"messages"})
                          for value in prompts):
        parser.error("each workload row must contain exactly prompt or messages")
    if args.check_repeats and args.requests <= len(prompts):
        parser.error("--check-repeats requires more requests than workload rows")
    manifest = None
    if args.server_manifest:
        if args.server_manifest.stat().st_size > 1 << 20:
            parser.error("server manifest exceeds 1 MiB")
        manifest = json.loads(args.server_manifest.read_text())
    for slo in (args.ttft_slo, args.tpot_slo):
        if slo is not None and (not math.isfinite(slo) or slo <= 0):
            parser.error("SLOs must be finite positive seconds")
    def execute(index):
        payload = {**prompts[index % len(prompts)], "model": args.model, "max_tokens": args.max_tokens,
                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        return request_once(index, args.endpoint, payload, args.timeout)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        measurements = list(pool.map(execute, range(args.requests)))
    elapsed = time.perf_counter() - started
    checks = check_workload(measurements, len(prompts), expected_prompt=args.expect_prompt_tokens,
                            expected_completion=args.expect_completion_tokens, repeat=args.check_repeats)
    artifact = {"schema": "pypto_http_benchmark_v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "concurrency": args.concurrency, "max_tokens": args.max_tokens,
                "workload_rows": len(prompts), "workload_sha256": hashlib.sha256(raw).hexdigest(),
                "server_manifest": manifest,
                "measurement_scope": "HTTP client; no inferred HBM/OOM capacity or real-model golden acceptance",
                "workload_checks": checks,
                "summary": summarize(measurements, elapsed, ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo),
                "requests": [asdict(result) for result in measurements]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    return 0 if all(result.success for result in measurements) and all(value["passed"] for value in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
