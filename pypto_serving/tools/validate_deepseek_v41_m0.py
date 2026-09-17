# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture or compare V4.1 M0 token traces without treating missing device evidence as PASS.

The optional --run path uses the normal serving CLI configuration and engine.
The default only reads bounded JSON artifacts and never imports the NPU runtime.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


REFERENCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
MAX_JSON_BYTES = 64 << 20


def read_artifact(path):
    raw = path.read_bytes() if path.stat().st_size <= MAX_JSON_BYTES else None
    if raw is None:
        raise ValueError("artifact exceeds 64 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("artifact must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def write_artifact(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ids(value, count):
    return isinstance(value, list) and len(value) == count and all(type(token) is int and token >= 0 for token in value)


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_reference(reference):
    if reference.get("reference_revision") != REFERENCE_REVISION:
        raise ValueError("reference must identify the pinned independent DeepSeek reference revision")
    for key in ("checkpoint_revision", "config_sha256", "tokenizer_sha256"):
        if not isinstance(reference.get(key), str) or not reference[key]:
            raise ValueError(f"reference requires {key}")
    if any(not _sha256(reference[key]) for key in ("config_sha256", "tokenizer_sha256")):
        raise ValueError("reference config/tokenizer hashes must be SHA-256 hex strings")
    cases = reference.get("cases", [])
    if not isinstance(cases, list) or not cases or len(cases) > 16:
        raise ValueError("reference requires 1..16 cases")
    names = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
            raise ValueError("each reference case requires a case_id")
        if case["case_id"] in names:
            raise ValueError("reference case IDs must be unique")
        names.add(case["case_id"])
        if not _ids(case.get("prompt_token_ids"), 8192) or not _ids(case.get("token_ids"), 128):
            raise ValueError("each reference case requires exactly 8192 prompt IDs and 128 generated IDs")


def compare_traces(reference, actual, platform):
    """Compare tokens and observed scheduler work; no retokenization of HTTP text."""
    validate_reference(reference)
    if actual is None:
        return {"status": "NOT_RUN", "reason": "No serving token trace"}
    errors = []
    if actual.get("schema") != "pypto_v41_greedy_trace_v1":
        errors.append("unrecognized serving token trace schema")
    for key in ("checkpoint_revision", "config_sha256", "tokenizer_sha256"):
        if actual.get(key) != reference[key]:
            errors.append(f"{key} differs from reference")
    if actual.get("platform") != platform or actual.get("world_size") != 8:
        errors.append("trace must use the requested platform and eight devices")
    expected = {case["case_id"]: case for case in reference["cases"]}
    comparisons = []
    observed = actual.get("cases", [])
    if not isinstance(observed, list) or len(observed) > 17 or any(not isinstance(case, dict) for case in observed):
        return {"status": "FAIL", "errors": [*errors, "trace requires at most 17 case objects"]}
    request_ids = set()
    for case in observed:
        case_id = case.get("case_id")
        wanted = expected.get(case_id) if isinstance(case_id, str) else None
        row_errors = []
        if wanted is None:
            row_errors.append("unknown reference case")
            wanted = {"prompt_token_ids": [], "token_ids": []}
        if case.get("prompt_token_ids") != wanted["prompt_token_ids"]:
            row_errors.append("input token IDs differ")
        tokens = case.get("token_ids", [])
        if not isinstance(tokens, list):
            tokens = []
        mismatch = next((index for index in range(max(len(tokens), len(wanted["token_ids"])))
                         if tokens[index:index + 1] != wanted["token_ids"][index:index + 1]), None)
        if mismatch is not None or not _ids(tokens, 128):
            row_errors.append("greedy token IDs differ or output ended before 128 tokens")
        events = case.get("scheduler_events", [])
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            events = []
        if (not 128 <= len(events) <= 129 or events[0].get("phase") != "prefill"
                or events[0].get("tokens") != 8192
                or any(event.get("phase") != "decode" or event.get("tokens") != 1 for event in events[1:])):
            row_errors.append("missing one 8192-token prefill followed by single-token decode observations")
        event_ids = [event.get("request_id") for event in events]
        if (not event_ids or not isinstance(event_ids[0], str) or not event_ids[0]
                or any(request_id != event_ids[0] for request_id in event_ids)
                or event_ids[0] in request_ids):
            row_errors.append("scheduler observations must identify one distinct request per case")
        elif event_ids:
            request_ids.add(event_ids[0])
        if case.get("temperature") != 0 or case.get("num_speculative_tokens") != 0:
            row_errors.append("trace must use greedy target-only decoding")
        if case.get("host_requests_after") != []:
            row_errors.append("host request cleanup was not observed")
        comparisons.append({"case_id": case.get("case_id"), "status": "FAIL" if row_errors else "PASS",
                            "first_mismatch": mismatch, "expected_token": None if mismatch is None else
                            wanted["token_ids"][mismatch:mismatch + 1], "actual_token": None if mismatch is None else
                            tokens[mismatch:mismatch + 1], "errors": row_errors})
    if not observed or set(expected) - {case.get("case_id") for case in observed if isinstance(case.get("case_id"), str)}:
        errors.append("not every reference case was executed")
    repeated = len(observed) > len(expected) and observed[-1].get("case_id") == reference["cases"][0]["case_id"]
    if not repeated:
        errors.append("missing repeat of the first request after the other cases")
    return {"status": "FAIL" if errors or any(row["errors"] for row in comparisons) else "PASS",
            "errors": errors, "cases": comparisons,
            "scope": "token IDs, scheduler observations and host cleanup; device rollback and logits are separate evidence"}


async def capture_engine(engine, reference):
    """Use the same scheduler observer as the existing serving accuracy tests."""
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.model.tokenizer import PreparedPrompt

    events = []
    original = engine.scheduler.schedule

    def observe():
        output = original()
        for item in output.scheduled_requests:
            events.append({"request_id": item.request.request_id, "phase": "prefill" if item.is_prefill else "decode",
                           "tokens": item.num_new_tokens})
        return output

    engine.scheduler.schedule = observe
    captured = []
    try:
        await engine.start()
        for case in [*reference["cases"], reference["cases"][0]]:
            events.clear()
            started = time.perf_counter()
            result = await engine.generate_result(
                PreparedPrompt(text="", token_ids=list(case["prompt_token_ids"])),
                GenerateConfig(max_new_tokens=128, temperature=0.0, top_p=1.0, top_k=None, stream=False),
            )
            captured.append({"case_id": case["case_id"], "prompt_token_ids": case["prompt_token_ids"],
                             "token_ids": result.token_ids, "finish_reason": result.finish_reason,
                             "elapsed_s": time.perf_counter() - started, "scheduler_events": list(events),
                             "temperature": 0, "num_speculative_tokens": 0,
                             "host_requests_after": sorted(engine.scheduler.requests)})
    finally:
        try:
            await engine.stop()
        finally:
            engine.scheduler.schedule = original
    return captured


def run_engine(reference, platform, engine_argv):
    # Import only after --run. Reuse the production CLI's model registration,
    # topology checks, cache sizing and backend factory; do not build another executor.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    config = build_serving_engine_config(build_parser().parse_args(engine_argv))
    runtime = config.resolve_runtime_config()
    devices = config.worker_device_ids()
    if (config.platform != platform or config.executor_cls != "PyptoDeepSeekV41Executor" or len(devices) != 8
            or config.parallel_config is None or config.parallel_config.num_replicas != 1):
        raise ValueError("M0 capture requires V4.1, the requested platform, eight devices and one serving replica")
    if runtime.num_speculative_tokens or runtime.max_seq_len < 8320 or config.max_num_scheduled_tokens < 8192:
        raise ValueError("M0 capture requires target-only decoding, max-model-len >= 8320 and a token budget >= 8192")
    model = Path(config.model_dir)
    identities = {"checkpoint_revision": reference["checkpoint_revision"], "platform": platform,
                  "world_size": len(devices), "devices": list(devices), "executor": config.executor_cls,
                  "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                  "tokenizer_sha256": hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest()}
    if any(identities[key] != reference[key] for key in ("config_sha256", "tokenizer_sha256")):
        raise ValueError("local model config/tokenizer hashes differ from the reference")
    engine = AsyncLLMEngine(config=config, tokenizer=load_tokenizer(config.model_dir))
    return {"schema": "pypto_v41_greedy_trace_v1", **identities,
            "checkpoint_revision_source": "reference declaration; use checkpoint inspection to verify local payloads",
            "cases": asyncio.run(capture_engine(engine, reference))}


def check_inspection(inspection, reference):
    """Use the existing inspector's measured coverage and hashes, not a claimed PASS."""
    if inspection is None:
        return {"status": "NOT_RUN", "reason": "No checkpoint inspection artifact"}
    errors = []
    if inspection.get("schema_version") != 1 or inspection.get("model_type") != "deepseek_v41":
        errors.append("unrecognized checkpoint inspection schema")
    for key in ("checkpoint_revision", "config_sha256"):
        if inspection.get(key) != reference[key]:
            errors.append(f"inspection {key} differs from reference")
    if inspection.get("errors") != []:
        errors.append("inspection contains errors or omits error observations")
    if not _sha256(inspection.get("index_sha256")):
        errors.append("missing checkpoint index hash")
    required, checked = inspection.get("required_text_tensor_count"), inspection.get("validated_text_tensor_count")
    if type(required) is not int or type(checked) is not int or not 0 <= checked <= required or required == 0:
        errors.append("invalid measured tensor coverage")
    hashes = inspection.get("header_sha256")
    if not isinstance(hashes, dict) or any(not _sha256(value) for value in hashes.values()):
        errors.append("invalid measured header hashes")
    complete = checked == required and bool(hashes) and inspection.get("unavailable_shards") == []
    return {"status": "FAIL" if errors else ("PASS" if complete else "NOT_RUN"), "errors": errors,
            "validated_text_tensor_count": checked, "required_text_tensor_count": required,
            "scope": "checkpoint index and header metadata; tensor payload values and device upload are not checked"}


def check_http(http, reference, platform):
    """Recompute sequential request checks from measurements, ignoring summary claims."""
    if http is None:
        return {"status": "NOT_RUN", "reason": "No HTTP benchmark artifact"}
    errors = []
    rows, workload_rows = http.get("requests"), http.get("workload_rows")
    if (http.get("schema") != "pypto_http_benchmark_v1" or http.get("concurrency") != 1
            or http.get("max_tokens") != 128 or not _sha256(http.get("workload_sha256"))):
        errors.append("HTTP artifact must record sequential 128-token benchmark requests and a workload hash")
    manifest = http.get("server_manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    if manifest.get("platform") != platform or manifest.get("world_size") != 8:
        errors.append("HTTP server manifest does not identify the requested eight-device platform")
    for key in ("checkpoint_revision", "config_sha256", "tokenizer_sha256"):
        if manifest.get(key) != reference[key]:
            errors.append(f"HTTP manifest {key} differs from reference")
    if (type(workload_rows) is not int or workload_rows < 1 or not isinstance(rows, list)
            or not workload_rows < len(rows) <= 10000 or any(not isinstance(row, dict) for row in rows)):
        return {"status": "FAIL", "errors": [*errors, "missing bounded request measurements and a repeated workload row"]}
    previous, request_ids = {}, set()
    failures = []
    for index, row in enumerate(rows):
        request_id = row.get("request_id")
        if (row.get("index") != index or row.get("success") is not True or row.get("prompt_tokens") != 8192
                or row.get("completion_tokens") != 128 or not _sha256(row.get("output_sha256"))
                or not isinstance(request_id, str) or not request_id or request_id in request_ids
                or row.get("finish_reason") not in ("stop", "length", "eos")):
            failures.append(index)
        if isinstance(request_id, str):
            request_ids.add(request_id)
        key = index % workload_rows
        signature = (row.get("output_sha256"), row.get("prompt_tokens"), row.get("completion_tokens"), row.get("finish_reason"))
        if key in previous and previous[key] != signature:
            failures.append(index)
        previous[key] = signature
    return {"status": "FAIL" if errors or failures else "PASS", "errors": errors,
            "failed_request_indices": sorted(set(failures)),
            "scope": "HTTP output digests, measured usage and declared server provenance; device isolation is separate"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["a2a3", "a5"], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--http-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("engine_args", nargs=argparse.REMAINDER, help="after --, normal pypto-serving CLI arguments")
    args = parser.parse_args(argv)
    inputs = [path.resolve() for path in (args.reference, args.actual, args.inspection, args.http_artifact) if path]
    if args.output.resolve() in inputs or (args.run and args.actual):
        parser.error("output must not overwrite inputs; --run cannot be combined with --actual")
    artifacts, hashes = {}, {}
    for name in ("reference", "actual", "inspection", "http_artifact"):
        path = getattr(args, name)
        if path:
            artifacts[name], hashes[name] = read_artifact(path)
    reference = artifacts["reference"]
    validate_reference(reference)
    actual = artifacts.get("actual")
    if actual is not None and actual.get("schema") == "pypto_v41_m0_evidence_v1":
        actual = actual.get("actual_trace")
        if actual is not None and not isinstance(actual, dict):
            raise ValueError("actual_trace must contain a JSON object")
    run_error = None
    if args.run:
        try:
            engine_args = args.engine_args[1:] if args.engine_args[:1] == ["--"] else args.engine_args
            actual = run_engine(reference, args.platform, engine_args)
        except Exception as error:
            run_error = f"{type(error).__name__}: {error}"
    checks = {"greedy_8k_128": compare_traces(reference, actual, args.platform)}
    if run_error:
        checks["greedy_8k_128"] = {"status": "FAIL", "error": run_error}
    checks["checkpoint_metadata"] = check_inspection(artifacts.get("inspection"), reference)
    checks["http_8k_128"] = check_http(artifacts.get("http_artifact"), reference, args.platform)
    for name in ("weight_conversion_and_layer_golden", "device_cache_rollback_and_reuse", "per_rank_peak_hbm",
                 "backend_support_and_toolchain_provenance", "mismatch_logits"):
        checks[name] = {"status": "NOT_RUN", "reason": "Requires independent device/log artifacts and review"}
    report = {"schema": "pypto_v41_m0_evidence_v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "platform": args.platform, "artifact_sha256": hashes, "checks": checks,
              "acceptance_status": "FAIL" if any(value["status"] == "FAIL" for value in checks.values()) else "NOT_RUN",
              "actual_trace": actual, "scope": "Available automated checks; not complete M0 certification"}
    write_artifact(args.output, report)
    print(json.dumps({"acceptance_status": report["acceptance_status"], "checks": checks}, indent=2))
    return 1 if report["acceptance_status"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
