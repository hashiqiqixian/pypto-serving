# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce the screenshot's revisions and 16-card DSpark 64/256 shape.

Run this helper separately from the pristine production --serving-root. The
repository Palace prompt is used; the screenshot's original prompt and full
launch configuration were unavailable, so this is not an exact workload claim.
One non-streaming warmup precedes one streaming measurement, without profiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


REVISIONS = {
    "serving": "2a03e11afb8357b256c666e8b1b82487d2a0d5e7",
    "lib": "7b2c3f94020b963c1afdf3033917676adb074378",
    "pypto": "e8191e3cc9fb3c93b2e66e8329d4ce85b522cfae",
    "runtime": "22385d2b0c08b8f697fe8feede488c87cf83ef84",
}
ENV_KEYS = (
    "PATH", "PYTHONPATH", "PYPTO_LIB_ROOT", "PTOAS_ROOT", "PTO_ISA_ROOT",
    "PYPTO_DSPARK_EP_SIZE", "PYPTO_DSPARK_RING_HEAP", "PYPTO_DSPARK_DRAFTER_RING_HEAP",
    "PYPTO_DSPARK_DECODE_RING_HEAP", "PYPTO_PROG_BUILD_DIR", "PYPTO_CODEGEN_MAX_WORKERS",
    "PYPTO_BUILD_JOBS", "PYPTO_TEST_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "MAX_JOBS",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TASK_DEVICE",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "PYPTO_RUNTIME_LOG",
    "SIMPLER_DFX", "SIMPLER_ORCH_PROFILING", "SIMPLER_SCHED_PROFILING", "SIMPLER_TENSORMAP_PROFILING",
    "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES",
)
FINISHED = re.compile(
    r"DSpark speculation finished: request=(\S+) verifies=(\d+) "
    r"matched=(\d+) proposed=(\d+) accepted=(\d+) mean_len=([\d.]+) fallbacks=(\d+)"
)


def _save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, timeout=15,
    ).strip()


def _revision(root: Path, expected: str) -> dict:
    actual = _git(root, "rev-parse", "HEAD")
    changed = _git(root, "diff", "--name-only", "HEAD", "--")
    if actual != expected or changed:
        raise RuntimeError(f"source must be pristine at {expected}: {root}, HEAD={actual}, diff={changed}")
    return {"root": str(root), "commit": actual, "tracked_changes": changed}


def _validate_response(response: dict) -> None:
    expected = {"prompt_tokens": 64, "completion_tokens": 256, "total_tokens": 320}
    if response.get("usage") != expected:
        raise RuntimeError(f"expected 64/256 token accounting, got {response.get('usage')}")
    choices = response.get("choices", [])
    if (not response.get("id") or len(choices) != 1
            or choices[0].get("finish_reason") != "length" or not choices[0].get("text")):
        raise RuntimeError(f"invalid completion contract: {response}")


def _read_stream(port: int, payload: dict, deadline: float, output: Path) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    first_text = None
    parts = []
    request_id = None
    usage = None
    finish_reason = None
    done = False
    with (output / "formal-stream.sse").open("wb") as raw:
        with opener.open(request, timeout=max(1.0, deadline - started)) as response:
            for line in response:
                received = time.monotonic()
                raw.write(line)
                raw.flush()
                if received >= deadline:
                    raise TimeoutError("formal stream exceeded its deadline")
                if not line.startswith(b"data: "):
                    continue
                body = line[6:].strip()
                if body == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(body)
                if request_id is not None and chunk.get("id") != request_id:
                    raise RuntimeError("stream changed its request ID")
                request_id = chunk.get("id")
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    if len(choices) != 1:
                        raise RuntimeError("expected one streaming completion choice")
                    text = choices[0].get("text", "")
                    if text:
                        first_text = first_text if first_text is not None else received
                        parts.append(text)
                    if choices[0].get("finish_reason") is not None:
                        finish_reason = choices[0]["finish_reason"]
    ended = time.monotonic()
    completion = {
        "id": request_id, "usage": usage,
        "choices": [{"text": "".join(parts), "finish_reason": finish_reason}],
    }
    _save(output / "formal-response.json", completion)
    (output / "formal-output.txt").write_text("".join(parts), encoding="utf-8")
    if not done or first_text is None:
        raise RuntimeError("stream did not contain both text and [DONE]")
    _validate_response(completion)
    ttft = first_text - started
    total = ended - started
    tpot = (total - ttft) / (usage["completion_tokens"] - 1)
    if tpot <= 0.0:
        raise RuntimeError("non-positive decode interval; cannot compute a meaningful token rate")
    return {
        "request_id": request_id, "usage": usage, "ttft_seconds": ttft,
        "total_seconds": total, "tpot_seconds": tpot, "decode_tokens_per_second": 1.0 / tpot,
        "ttft_definition": "HTTP start to the first nonempty text SSE chunk; not token-ID timing",
        "total_definition": "HTTP start through [DONE] and closing the response",
        "tpot_formula": "(total_seconds - ttft_seconds) / (completion_tokens - 1)",
        "decode_rate_formula": "1 / tpot_seconds; excludes first-token latency",
    }


def _measure_stream(process, port: int, payload: dict, deadline: float, output: Path) -> dict:
    results = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            results.put((True, _read_stream(port, payload, deadline, output)))
        except BaseException as exc:
            results.put((False, exc))

    # The foreground owner enforces the deadline even if a socket read stalls.
    threading.Thread(target=read, daemon=True, name="dspark-formal-stream").start()
    while time.monotonic() < deadline:
        try:
            succeeded, value = results.get(timeout=min(5.0, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            if process.poll() is not None:
                raise RuntimeError(f"server exited during formal measurement: {process.returncode}")
            continue
        if not succeeded:
            raise value
        return value
    raise TimeoutError("formal measurement exceeded its deadline")


def _finished_counters(log_path: Path, request_id: str, deadline: float) -> dict:
    while True:
        for match in FINISHED.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
            if match[1] == request_id:
                fields = ("verifies", "matched", "proposed", "accepted", "mean_len", "fallbacks")
                counters = dict(zip(fields, (int(match[i]) if i != 6 else float(match[i])
                                             for i in range(2, 8))))
                return {
                    "request_id": request_id, "raw_log": match[0], "counters": counters,
                    "legacy_matched_over_proposed": (
                        counters["matched"] / counters["proposed"] if counters["proposed"] else None
                    ),
                    "acceptance_caveat": (
                        "Legacy runner proposed includes initial seeding; counters may include async "
                        "work not consumed by the finished request. This is not engine verified-draft "
                        "acceptance and the screenshot's original formula is unconfirmed."
                    ),
                }
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no complete DSpark finished counters for {request_id}")
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _cleanup(process, stop) -> None:
    stop(process)
    # The shared helper can return after the parent dies before its children.
    # Only this Popen(start_new_session=True) process group may be signalled.
    for signum, duration in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, 0)
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        until = time.monotonic() + duration
        while time.monotonic() < until:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
    raise RuntimeError(f"owned server process group {process.pid} still exists after cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--pypto-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--overall-timeout", type=int, default=2400)
    parser.add_argument("--reclaim-timeout", type=int, default=900)
    args = parser.parse_args()
    if os.name != "posix" or any(getattr(args, name) <= 0 for name in (
        "startup_timeout", "request_timeout", "overall_timeout", "reclaim_timeout",
    )):
        parser.error("requires POSIX and positive timeouts")
    if os.environ.get("PYPTO_DSPARK_EP_SIZE", "16") != "16":
        parser.error("PYPTO_DSPARK_EP_SIZE must be 16")
    if not os.environ.get("PYPTO_LIB_ROOT"):
        parser.error("PYPTO_LIB_ROOT must select the screenshot lib checkout")
    roots = {
        "serving": args.serving_root.resolve(), "lib": Path(os.environ["PYPTO_LIB_ROOT"]).resolve(),
        "pypto": args.pypto_root.resolve(), "runtime": args.runtime_root.resolve(),
    }
    versions = {name: _revision(root, REVISIONS[name]) for name, root in roots.items()}
    args.model = args.model.resolve()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(roots["serving"])
    sys.path[:0] = [str(roots["serving"] / "tests"), str(roots["serving"])]
    import test_deepseek_dspark_accuracy as dspark
    import test_deepseek_v4_accuracy as shared

    devices = dspark._task_devices()
    if devices != tuple(range(16)):
        parser.error("TASK_DEVICE must be 0,1,...,15 in ascending order")
    shared.STARTUP_TIMEOUT_SECONDS = args.startup_timeout
    shared.HEARTBEAT_SECONDS = 5
    port = shared._unused_local_port()
    command = dspark._server_command(args.model.resolve(), devices, port, num_speculative_tokens=7)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (
        str(roots["serving"]), os.environ.get("PYTHONPATH", ""),
    )))}
    payload = {
        "model": dspark.MODEL_ID, "prompt": dspark.GREEDY_CASES[1].prompt,
        "max_tokens": 256, "temperature": 0.0, "top_p": 1.0,
        "top_k": None, "seed": None, "stop": [], "stream": False,
    }
    formal = {**payload, "stream": True}
    _save(args.output_dir / "warmup-request.json", payload)
    _save(args.output_dir / "formal-request.json", formal)
    metadata = {
        "source_versions": versions, "helper_commit": _git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "python": sys.executable, "python_version": sys.version, "command": command,
        "cwd": str(roots["serving"]), "environment": {key: env[key] for key in ENV_KEYS if key in env},
        "model": str(args.model.resolve()), "devices": devices, "cli_profiling": False,
        "warmup_requests": 1, "formal_requests": 1, "endpoint": "/v1/completions",
        "prompt_sha256": hashlib.sha256(payload["prompt"].encode("utf-8")).hexdigest(),
        "limitations": (
            "Reproduces screenshot revisions/topology/lengths; original screenshot artifact, prompt, "
            "launch options and metric formula unavailable. Uses repository Palace64 prompt and "
            "default server settings. No HC-mean fix or production source edits."
        ),
        "timeouts": {name: getattr(args, name) for name in (
            "startup_timeout", "request_timeout", "overall_timeout", "reclaim_timeout",
        )},
    }
    _save(args.output_dir / "run.json", metadata)
    log_path = args.output_dir / "server.log"
    deadline = time.monotonic() + args.overall_timeout
    process = None
    failure = None
    performance = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command, cwd=roots["serving"], env=env, stdout=server_log,
                stderr=subprocess.STDOUT, start_new_session=True, text=True,
            )
            metadata["server_pid"] = process.pid
            _save(args.output_dir / "run.json", metadata)
            started = time.monotonic()
            shared._wait_for_health(process, port, min(deadline, started + args.startup_timeout))
            metadata["startup_seconds"] = time.monotonic() - started
            _save(args.output_dir / "run.json", metadata)
            warmup = shared._request_json(
                process, port, min(deadline, time.monotonic() + args.request_timeout),
                endpoint="/v1/completions", request_kind="screenshot warmup", payload=payload,
            )
            _save(args.output_dir / "warmup-response.json", warmup)
            _validate_response(warmup)
            _save(args.output_dir / "warmup-counters.json", _finished_counters(
                log_path, warmup["id"], min(deadline, time.monotonic() + 30),
            ))
            performance = _measure_stream(
                process, port, formal, min(deadline, time.monotonic() + args.request_timeout), args.output_dir,
            )
            _save(args.output_dir / "performance.json", performance)
            performance["speculation"] = _finished_counters(
                log_path, performance["request_id"], min(deadline, time.monotonic() + 30),
            )
            formal_response = json.loads((args.output_dir / "formal-response.json").read_text(encoding="utf-8"))
            performance["same_text_as_warmup"] = (
                formal_response["choices"][0]["text"] == warmup["choices"][0]["text"]
            )
            _save(args.output_dir / "performance.json", performance)
            print(json.dumps(performance, ensure_ascii=False), flush=True)
    except BaseException as exc:
        failure = exc
        _save(args.output_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        cleanup_errors = []
        if process is not None:
            for action in (
                lambda: _cleanup(process, shared._stop_process_group),
                lambda: dspark._wait_for_device_reclaim(devices, timeout_s=args.reclaim_timeout),
            ):
                try:
                    action()
                except BaseException as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        for sig, handler in prior.items():
            signal.signal(sig, handler)
        _save(args.output_dir / "cleanup.json", {
            "server_pid": process.pid if process else None, "errors": cleanup_errors,
            "server_returncode": process.returncode if process else None,
            "measurement_complete": performance is not None and failure is None,
        })
        if cleanup_errors and failure is None:
            raise RuntimeError(f"cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    main()
