# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the fixed DeepSeek V4 DSpark single-batch workload under serving profiling."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pypto_serving.model.tokenizer import load_tokenizer


REQUEST_COUNT = 1
OUTPUT_TOKENS = 128
SPECULATIVE_TOKENS = 7
DEVICE_COUNT = 16
ALIGNED_PROMPT = (
    "<｜begin▁of▁sentence｜><｜User｜>"
    "请用中文详细介绍北京故宫，分为历史沿革、整体布局、主要宫殿、建筑特色、重要馆藏、"
    "文化价值和参观建议七节，每节至少一百字，内容准确连贯，不要省略。请使用清晰的小标题，"
    "并说明关键年代、人物与用途。"
    "<｜Assistant｜></think>"
)
EXPECTED_PROMPT_IDS = [
    0, 128803, 2788, 642, 21134, 87336, 6127, 74437, 303, 9969, 5163,
    8689, 4155, 410, 10319, 17996, 410, 2897, 64474, 410, 6786, 10716,
    410, 3036, 6071, 5376, 410, 3415, 87482, 23177, 7383, 3958, 2045,
    303, 1833, 2045, 11732, 21080, 2024, 303, 3975, 12963, 95512, 303,
    4916, 62186, 320, 2788, 2541, 17165, 5968, 24153, 303, 1380, 6977,
    7511, 10776, 410, 13320, 947, 27917, 320, 128804, 128822,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--served-model-name", default="dsv4-flash-dspark-w8a8")
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    url: str,
    *,
    payload: dict | None = None,
    timeout: float,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    if not body:
        return {}
    return json.loads(body)


def wait_for_health(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"DSpark server exited during startup with code {code}")
        try:
            payload = request_json(f"{base_url}/health", timeout=5.0)
            if payload.get("status") == "ok":
                return
        except (OSError, RuntimeError, json.JSONDecodeError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"DSpark server did not become healthy in {timeout:.0f}s")


def completion_payload(model: str) -> dict:
    return {
        "model": model,
        "prompt": ALIGNED_PROMPT,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
        "stream": False,
        "ignore_eos": True,
    }


def generate_single(base_url: str, model: str, timeout: float) -> list[dict]:
    payload = completion_payload(model)
    response = request_json(
        f"{base_url}/v1/completions", payload=payload, timeout=timeout
    )
    usage = response.get("usage", {})
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise RuntimeError(f"single request returned {len(choices)} choices")
    completion_tokens = int(usage.get("completion_tokens", -1))
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    if prompt_tokens != len(EXPECTED_PROMPT_IDS) or completion_tokens != OUTPUT_TOKENS:
        raise RuntimeError(
            f"single request token mismatch: prompt={prompt_tokens}, "
            f"completion={completion_tokens}"
        )
    choice = choices[0]
    if choice.get("finish_reason") != "length":
        raise RuntimeError(
            f"single request finish_reason={choice.get('finish_reason')!r}"
        )
    return [{
        "index": 0,
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }]


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if (
        len(devices) != DEVICE_COUNT
        or len(set(devices)) != DEVICE_COUNT
        or any(not item.isdigit() for item in devices)
    ):
        raise ValueError(
            f"--devices must contain exactly {DEVICE_COUNT} unique device IDs: "
            f"{args.devices!r}"
        )

    tokenizer = load_tokenizer(model_dir)
    prompt_ids = tokenizer.encode(ALIGNED_PROMPT)
    if prompt_ids != EXPECTED_PROMPT_IDS:
        raise RuntimeError(f"aligned prompt token mismatch: {prompt_ids}")

    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m", "pypto_serving.cli",
        "--model", str(model_dir),
        "--served-model-name", args.served_model_name,
        "--backend", "npu",
        "--platform", "a2a3",
        "--devices", ",".join(devices),
        "--dp", "4",
        "--ep", "16",
        "--tp", "4",
        "--block-size", "32",
        "--max-model-len", "1024",
        "--max-num-seqs", "8",
        "--max-num-batched-tokens", "8192",
        "--long-prefill-token-threshold", "128",
        "--speculative-config", '{"method":"dspark","num_speculative_tokens":7}',
        "--generate-config",
        '{"max_new_tokens":128,"temperature":0,"top_p":1,"top_k":null,'
        '"stream":false,"ignore_eos":true}',
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--ring-heap", "2147483648,2147483648,4294967296,8589934592",
        "--profile",
        "--profile-output", str(artifact_dir / "serving-trace"),
        "--profile-level", "verbose",
        "--port", str(port),
        "--show-startup-logs",
    ]
    if args.use_compile_cache:
        command.append("--use-compile-cache")

    print("SERVER_COMMAND=" + json.dumps(command, ensure_ascii=False), flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        wait_for_health(base_url, process, args.startup_timeout)
        print("DSpark server is healthy", flush=True)

        warmup = generate_single(base_url, args.served_model_name, args.request_timeout)
        (artifact_dir / "warmup-responses.json").write_text(
            json.dumps(warmup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("Unprofiled DSpark single-batch warmup completed", flush=True)

        request_json(f"{base_url}/start_profile", payload={}, timeout=60.0)
        batch_started = time.perf_counter()
        try:
            responses = generate_single(base_url, args.served_model_name, args.request_timeout)
        finally:
            request_json(f"{base_url}/stop_profile", payload={}, timeout=180.0)
        elapsed = time.perf_counter() - batch_started

        (artifact_dir / "responses.json").write_text(
            json.dumps(responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        total_tokens = REQUEST_COUNT * OUTPUT_TOKENS
        summary = {
            "batch_elapsed_seconds": elapsed,
            "request_count": REQUEST_COUNT,
            "prompt_tokens_per_request": len(prompt_ids),
            "tokens_per_request": [item["completion_tokens"] for item in responses],
            "total_output_tokens": total_tokens,
            "throughput_tokens_per_second": total_tokens / elapsed,
            "effective_tpot_ms": elapsed * 1000.0 / OUTPUT_TOKENS,
            "profiler_enabled": True,
            "profile_level": "verbose",
            "speculative_method": "dspark",
            "num_speculative_tokens": SPECULATIVE_TOKENS,
        }
        (artifact_dir / "performance_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        stop_process_group(process)


if __name__ == "__main__":
    run(parse_args())
