# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private real-engine V4.1 ASGI acceptance, one fresh process per K0/K5 mode.

No TCP listener is opened. HTTPX ASGITransport buffers SSE responses, so this
checks SSE content/usage parity, not network backpressure or disconnect timing.
Cancellation uses the engine's public abort API after scheduler admission.
Run only inside an exclusive, bounded four-device validation task.
"""

import argparse
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback


for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"


class _BoundedNativeCompilePool(ThreadPoolExecutor):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(4 if max_workers is None else min(max_workers, 4), *args, **kwargs)


def _install_native_compile_bound():
    from pypto.runtime import device_runner

    tree = ast.parse(Path(device_runner.__file__).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_compile_and_assemble_locked"]
    if len(functions) != 1 or device_runner.ThreadPoolExecutor is not ThreadPoolExecutor:
        raise RuntimeError("Expected unpatched be23606/950a30e native compiler")
    bounds = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "max_workers"
                      for target in node.targets)]
    expected = ast.parse("max_workers = min(64, 1 + len(code_units))").body[0]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "ThreadPoolExecutor"]
    expected_call = ast.parse("ThreadPoolExecutor(max_workers=max_workers)", mode="eval").body
    if (len(bounds) != 1 or ast.dump(bounds[0]) != ast.dump(expected)
            or len(calls) != 1 or ast.dump(calls[0]) != ast.dump(expected_call)
            or not any(node is calls[0] for node in ast.walk(functions[0]))):
        raise RuntimeError("Pinned native compilation pool contract changed")
    device_runner.ThreadPoolExecutor = _BoundedNativeCompilePool


# Spawn imports this file as __mp_main__; preserve the same compiler cap there.
_install_native_compile_bound()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _chat_payload(content, *, stream=False, max_tokens=96, **extra):
    return {"model": "v41-private-acceptance", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "stream": stream,
            "chat_template_kwargs": {"thinking_mode": "chat"}, **extra}


def _render(tokenizer, payload):
    return tokenizer.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True,
                                         **payload["chat_template_kwargs"])


def _exact_prompt(tokenizer, target, *, question=None):
    """Find bounded natural text with exactly the requested rendered token count."""
    prefix = "Read this background and answer the final question. Background:"
    suffix = "\nFinal question: " + (question or "What is 17 times 19? Answer with the number only.")
    for unit in (" x", " a", " 0", " item", " text", "\nx"):
        low, high = 0, target + 16
        while low < high:
            middle = (low + high) // 2
            content = prefix + unit * middle + suffix
            count = len(tokenizer.encode(_render(tokenizer, _chat_payload(content))))
            if count < target:
                low = middle + 1
            else:
                high = middle
        for repeat in range(max(0, low - 8), min(target + 16, low + 8) + 1):
            content = prefix + unit * repeat + suffix
            if len(tokenizer.encode(_render(tokenizer, _chat_payload(content)))) == target:
                return content
    raise RuntimeError(f"Could not construct an exact {target}-token chat prompt")


def _parse_response(response, stream):
    if not stream:
        body = response.json()
        choice = body["choices"][0]
        return {"request_id": body["id"], "text": choice["message"].get("content") or "",
                "reasoning": choice["message"].get("reasoning") or "",
                "finish_reason": choice["finish_reason"], "usage": body["usage"], "chunks": 1}
    if not response.headers.get("content-type", "").startswith("text/event-stream"):
        raise AssertionError("Streaming response is not SSE")
    request_id = None
    text, reasoning, finishes, usages = [], [], [], []
    done = 0
    chunks = 0
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done += 1
            continue
        if done:
            raise AssertionError("SSE contains data after DONE")
        value = json.loads(data)
        if "error" in value or value.get("object") == "error":
            raise AssertionError(f"SSE error: {value}")
        chunks += 1
        if request_id is not None and request_id != value["id"]:
            raise AssertionError("SSE request ID changed")
        request_id = value["id"]
        if value.get("usage") is not None:
            usages.append(value["usage"])
        for choice in value.get("choices", []):
            if choice["index"] != 0:
                raise AssertionError("Unexpected additional choice")
            text.append(choice.get("delta", {}).get("content") or "")
            reasoning.append(choice.get("delta", {}).get("reasoning") or "")
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    if done != 1 or len(finishes) != 1 or len(usages) != 1:
        raise AssertionError(f"Invalid terminal SSE contract: done={done}, finishes={finishes}, usages={usages}")
    return {"request_id": request_id, "text": "".join(text), "reasoning": "".join(reasoning),
            "finish_reason": finishes[0], "usage": usages[0], "chunks": chunks}


async def _benchmark_engine(engine, tokenizer, args, destination, event, idle):
    """Time token-bearing engine events; ASGI buffering is outside this metric."""
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

    result = {"scope": "single active request; engine admission to token events; no TCP",
              "warmups_per_prompt": args.benchmark_warmups, "repeats": args.benchmark_repeats,
              "output_tokens": args.benchmark_output_tokens, "scenarios": {}}
    question = ("Explain step by step how to multiply 17 by 19, then describe several ways to "
                "verify the result. Write a detailed tutorial of at least 300 words.")
    for length in args.benchmark_prompt_tokens:
        prompt = _render(tokenizer, _chat_payload(_exact_prompt(tokenizer, length, question=question)))
        runs = []
        for iteration in range(args.benchmark_warmups + args.benchmark_repeats):
            await idle()
            config = GenerateConfig(max_new_tokens=args.benchmark_output_tokens, temperature=0.0,
                                    stream=True, ignore_eos=True)
            request_id = f"bench-{length}-{iteration}"
            events, terminal = [], None
            observed = 0
            start = time.perf_counter()
            outputs = AsyncLLMEngine.add_request(engine, request_id, prompt, config)
            async with contextlib.aclosing(outputs):
                async for output in outputs:
                    now = time.perf_counter()
                    if output.completion_tokens > observed:
                        events.append({"seconds": now - start, "tokens": output.completion_tokens})
                        observed = output.completion_tokens
                    if output.finished:
                        terminal = output
            if terminal is None or terminal.completion_tokens != args.benchmark_output_tokens:
                raise AssertionError(f"Benchmark did not generate the fixed output length: {terminal}")
            if len(events) < 2:
                raise AssertionError("Engine events were buffered; TTFT/TPOT cannot be measured")
            first, last = events[0]["seconds"], events[-1]["seconds"]
            tpot = (last - first) / (terminal.completion_tokens - 1)
            warmup = iteration < args.benchmark_warmups
            record = {"warmup": warmup, "iteration": iteration, "prompt_tokens": length,
                      "output_tokens": terminal.completion_tokens, "ttft_ms": first * 1000,
                      "tpot_ms": tpot * 1000, "decode_tokens_per_second": 1 / tpot,
                      "last_token_seconds": last, "events": events, "text": terminal.text,
                      "token_ids": list(terminal.token_ids),
                      "eos_tokens": list(terminal.token_ids).count(tokenizer.eos_token_id)}
            runs.append(record)
            event("benchmark_sample", prompt_tokens=length, warmup=warmup,
                  ttft_ms=record["ttft_ms"], tpot_ms=record["tpot_ms"])
        measured = [row for row in runs if not row["warmup"]]
        result["scenarios"][str(length)] = {
            "median_ttft_ms": statistics.median(row["ttft_ms"] for row in measured),
            "median_tpot_ms": statistics.median(row["tpot_ms"] for row in measured),
            "median_decode_tokens_per_second": statistics.median(row["decode_tokens_per_second"] for row in measured),
            "runs": runs,
        }
        _write_json(destination / "benchmark.json", result)
    return result


async def _run(args):
    serving_root = args.serving_root.resolve(strict=True)
    sys.path.insert(0, str(serving_root))
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
    from pypto_serving.serving.server.server import create_serving_app
    import httpx
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    destination = args.results_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    summary = {"mode": args.mode, "max_model_len": args.max_model_len, "cases": {}, "passed": False}
    tail_counts = (args.max_model_len - 5, args.max_model_len - 6)
    standard_cases = {
        "natural_eos", "heterogeneous", "repeat_after_heterogeneous", "stream_parity",
        "prompt_81", "prompt_129", "prompt_161", "repeat_after_long_context", "concurrent_two",
        "queued_abort", "active_abort", "repeat_after_abort", "final_repeat",
        *[f"context_tail_{count}" for count in tail_counts],
    }
    if args.mode == "K5":
        standard_cases.update(("non_greedy_400_plain", "non_greedy_400_stream"))
    if args.long_prompt_tokens is not None:
        standard_cases.add("long_prompt")
    selected = standard_cases if args.cases == "all" else set(args.cases.split(","))
    if not selected or selected - standard_cases:
        raise ValueError(f"Unknown/unavailable cases: {sorted(selected - standard_cases)}")
    if any(name.startswith("repeat_") or name in ("stream_parity", "final_repeat") for name in selected):
        selected.add("natural_eos")
    summary["selected_cases"] = sorted(selected)

    def event(event_type, **values):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event_type, **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        print(line, flush=True)
        with (destination / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    devices = args.devices.split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(item.isdecimal() for item in devices):
        raise ValueError("Exactly four distinct numeric --devices are required")
    cli = ["--model", str(args.model.resolve(strict=True)), "--served-model-name", "v41-private-acceptance",
           "--backend", "npu", "--platform", "a5", "--devices", args.devices,
           "--tp", "1", "--dp", "4", "--ep", "4", "--block-size", "128",
           "--max-model-len", str(args.max_model_len), "--max-num-seqs", "4", "--max-num-batched-tokens", "80",
           "--long-prefill-token-threshold", "80", "--enable-chunked-prefill",
           "--no-enable-prefix-caching", "--show-startup-logs"]
    if args.mode == "K5":
        cli += ["--speculative-config", '{"method":"dspark","num_speculative_tokens":5}']
    if args.use_compile_cache:
        cli.append("--use-compile-cache")
    config = build_serving_engine_config(build_parser().parse_args(cli))
    tokenizer = load_tokenizer(config.model_dir)
    needed_counts = {count for count in (81, 129, 161) if f"prompt_{count}" in selected}
    needed_counts.update(count for count in tail_counts if f"context_tail_{count}" in selected)
    if selected.intersection(("concurrent_two", "queued_abort")):
        needed_counts.add(161)
    if "active_abort" in selected:
        needed_counts.add(128)
    if "long_prompt" in selected:
        needed_counts.add(args.long_prompt_tokens)
    questions = {
        "arithmetic": "What is 17 times 19? Answer with the number and one sentence.",
        "heterogeneous": "What is 2 plus 2? Answer with the number only.",
        **{f"tokens_{count}": _exact_prompt(tokenizer, count) for count in sorted(needed_counts)},
    }
    rendered_counts = {name: len(tokenizer.encode(_render(tokenizer, _chat_payload(content))))
                       for name, content in questions.items()}
    _write_json(destination / "manifest.json", {
        "mode": args.mode, "cli": cli, "serving_root": str(serving_root), "devices": devices,
        "prompt_token_counts": rendered_counts, "questions": questions,
        "transport": "httpx.ASGITransport; no listener; SSE buffered by transport",
        "native_compile_pool_max_workers": 4,
    })

    class ObservedEngine(AsyncLLMEngine):
        """Read-only observation of terminal TokenOutput; all calls delegate unchanged."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.observed = {}
            self.observed_active = set()
            self.max_observed_active = 0

        async def add_request(self, request_id, prompt, config, **kwargs):
            record = {"request_id": request_id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "rendered_prompt_tokens": len(self.tokenizer.encode(prompt)), "finished": False}
            self.observed[request_id] = record
            self.observed_active.add(request_id)
            self.max_observed_active = max(self.max_observed_active, len(self.observed_active))
            try:
                outputs = super().add_request(request_id, prompt, config, **kwargs)
                async with contextlib.aclosing(outputs):
                    async for output in outputs:
                        if output.finished:
                            record.update(finished=True, finish_reason=output.finish_reason, text=output.text,
                                          reasoning=output.reasoning, token_ids=list(output.token_ids),
                                          prompt_tokens=output.prompt_tokens,
                                          completion_tokens=output.completion_tokens)
                        yield output
            except BaseException as error:
                record["exception"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                self.observed_active.discard(request_id)

    engine = ObservedEngine(config=config, tokenizer=tokenizer)
    app = create_serving_app(engine, config.model_id, GenerateConfig(max_new_tokens=96))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                               base_url="http://asgi.local", timeout=None)

    async def idle():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not engine.is_ready():
                raise AssertionError("Engine lost readiness")
            unfinished = [key for key, request in engine.scheduler.requests.items() if not request.status.is_finished]
            if not unfinished and not engine.observed_active and engine.pending_token_load() == 0:
                response = await client.get("/health")
                if response.status_code != 200:
                    raise AssertionError(f"Health returned {response.status_code}")
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Requests did not drain after the case")

    async def request(payload, *, expected_status=200, natural_eos=False, answer=None, exact_count=None):
        response = await asyncio.wait_for(client.post("/v1/chat/completions", json=payload), args.case_timeout)
        if response.status_code != expected_status:
            raise AssertionError(f"HTTP {response.status_code}, expected {expected_status}: {response.text[:3000]}")
        if expected_status != 200:
            if not response.headers.get("content-type", "").startswith("application/json"):
                raise AssertionError("Rejected request must return JSON before SSE headers")
            body = response.json()
            if "greedy sampling only" not in body.get("message", ""):
                raise AssertionError(f"Unexpected admission error: {body}")
            return {"status": response.status_code, "body": body}
        result = _parse_response(response, payload["stream"])
        observed = engine.observed[result["request_id"]]
        result["engine"] = dict(observed)
        usage = result["usage"]
        count = len(tokenizer.encode(_render(tokenizer, payload)))
        if not observed["finished"] or observed["finish_reason"] not in ("FINISHED_EOS", "FINISHED_STOP", "FINISHED_LENGTH"):
            raise AssertionError(f"Unsuccessful engine termination: {observed}")
        if (usage["prompt_tokens"] != count or usage["prompt_tokens"] != observed["prompt_tokens"]
                or usage["completion_tokens"] != observed["completion_tokens"]
                or usage["completion_tokens"] != len(observed["token_ids"])
                or usage["total_tokens"] != count + usage["completion_tokens"]):
            raise AssertionError(f"Token accounting mismatch: {usage}, observed={observed}")
        if not 0 < usage["completion_tokens"] <= min(payload["max_tokens"], args.max_model_len - count):
            raise AssertionError(f"Invalid generated length: {usage}")
        if result["text"] != observed["text"] or result["reasoning"] != observed["reasoning"]:
            raise AssertionError("HTTP text does not match final engine text")
        if natural_eos and (observed["finish_reason"] != "FINISHED_EOS" or result["finish_reason"] != "stop"):
            raise AssertionError(f"Expected natural EOS, got {result['finish_reason']} / {observed['finish_reason']}")
        if answer is not None and re.search(r"\b" + re.escape(answer) + r"\b", result["text"]) is None:
            raise AssertionError(f"Expected answer {answer!r}: {result['text']!r}")
        if exact_count is not None and (usage["completion_tokens"] != exact_count
                                        or observed["finish_reason"] != "FINISHED_LENGTH"):
            raise AssertionError(f"Context-tail did not reach the exact limit: {result}")
        return result

    def same_output(left, right):
        if left["engine"]["prompt_sha256"] != right["engine"]["prompt_sha256"]:
            raise AssertionError("Comparison inputs have different rendered chat prompts")
        for key in ("text", "reasoning", "finish_reason", "usage"):
            if left[key] != right[key]:
                raise AssertionError(f"Repeated/streamed output differs in {key}: {left[key]!r} vs {right[key]!r}")
        if left["engine"]["token_ids"] != right["engine"]["token_ids"]:
            raise AssertionError("Repeated/streamed token IDs differ")

    async def case(name, operation):
        if name not in selected:
            return None
        started = time.monotonic()
        event("case_start", name=name)
        try:
            result = await operation()
            await idle()
            record = {"passed": True, "seconds": time.monotonic() - started, "result": result}
        except Exception as error:
            record = {"passed": False, "seconds": time.monotonic() - started,
                      "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-6000:]}
            # Preserve partial terminal observations even when an assertion fails.
            record["observed"] = dict(engine.observed)
            for request_id in tuple(engine.observed_active):
                await engine.abort_request(request_id)
            try:
                await idle()
            except Exception as cleanup_error:
                record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        summary["cases"][name] = record
        _write_json(destination / f"{name}.json", record)
        _write_json(destination / "summary.json", summary)
        event("case_end", name=name, passed=record["passed"], seconds=record["seconds"], error=record.get("error"))
        if not engine.is_ready() or record.get("cleanup_error"):
            raise RuntimeError("Engine is unhealthy; remaining cases were not run")
        return record.get("result")

    try:
        event("engine_start", mode=args.mode, prompt_token_counts=rendered_counts)
        startup_started = time.perf_counter()
        await asyncio.wait_for(engine.start(), args.startup_timeout)
        summary["startup_seconds"] = time.perf_counter() - startup_started
        await idle()
        event("engine_ready", mode=args.mode)
        baseline = await case("natural_eos", lambda: request(_chat_payload(questions["arithmetic"]), natural_eos=True, answer="323"))
        await case("heterogeneous", lambda: request(_chat_payload(questions["heterogeneous"]), natural_eos=True, answer="4"))

        async def repeat(stream=False):
            value = await request(_chat_payload(questions["arithmetic"], stream=stream), natural_eos=True, answer="323")
            if baseline is None:
                raise AssertionError("Natural-EOS baseline failed; parity cannot be established")
            same_output(baseline, value)
            return value

        await case("repeat_after_heterogeneous", repeat)
        await case("stream_parity", lambda: repeat(True))
        for count in (81, 129, 161):
            await case(f"prompt_{count}", lambda count=count: request(
                _chat_payload(questions[f"tokens_{count}"], max_tokens=24), answer="323"))
        for count in tail_counts:
            await case(f"context_tail_{count}", lambda count=count: request(
                _chat_payload(questions[f"tokens_{count}"], max_tokens=16, ignore_eos=True),
                exact_count=args.max_model_len - count))
        if args.long_prompt_tokens is not None:
            await case("long_prompt", lambda: request(
                _chat_payload(questions[f"tokens_{args.long_prompt_tokens}"], max_tokens=128),
                natural_eos=True, answer="323"))
        await case("repeat_after_long_context", repeat)

        async def concurrent():
            engine.max_observed_active = 0
            tasks = [asyncio.create_task(request(_chat_payload(questions["arithmetic"]), natural_eos=True, answer="323")),
                     asyncio.create_task(request(_chat_payload(questions["tokens_161"], max_tokens=24), answer="323"))]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            if engine.max_observed_active < 2:
                raise AssertionError("Concurrent HTTP requests did not overlap at the engine boundary")
            if baseline is not None:
                same_output(baseline, results[0])
            previous = summary["cases"].get("prompt_161", {}).get("result")
            if previous is not None:
                same_output(previous, results[1])
            return {"max_active": engine.max_observed_active, "requests": results}

        await case("concurrent_two", concurrent)

        async def abort_queued():
            before = set(engine.observed)
            payload = _chat_payload(questions["tokens_161"], max_tokens=128, ignore_eos=True)
            pending = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
            try:
                deadline = time.monotonic() + 30
                request_id = None
                while time.monotonic() < deadline:
                    if pending.done():
                        raise AssertionError("Request completed before queued abort could be exercised")
                    added = set(engine.observed) - before
                    if len(added) == 1:
                        candidate = next(iter(added))
                        queued = engine.scheduler.requests.get(candidate)
                        if queued is not None and not queued.status.is_finished:
                            request_id = candidate
                            status_before = queued.status.name
                            break
                    await asyncio.sleep(0.001)
                if request_id is None:
                    raise AssertionError("Request was not admitted before abort deadline")
                await engine.abort_request(request_id)
                response = await asyncio.wait_for(pending, args.case_timeout)
                if response.status_code != 200:
                    raise AssertionError(f"Abort response returned HTTP {response.status_code}: {response.text[:2000]}")
                result = _parse_response(response, False)
                observed = engine.observed[request_id]
                if not observed["finished"] or observed["finish_reason"] != "FINISHED_ABORTED":
                    raise AssertionError(f"Abort did not reach the consumer: {observed}")
                if result["finish_reason"] != "aborted":
                    raise AssertionError(f"Abort HTTP finish reason differs: {result}")
                return {"status_before_abort": status_before, "http": result, "engine": dict(observed)}
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

        await case("queued_abort", abort_queued)

        async def active_abort():
            # Prefill writes 0..127. The first decode writes position 128 on
            # the next cache page; its output must be consumed before abort.
            request_id = "private-active-abort-reuse"
            prompt = _render(tokenizer, _chat_payload(questions["tokens_128"]))
            stream_config = GenerateConfig(max_new_tokens=64, temperature=0.0, top_p=1.0,
                                           stream=True, ignore_eos=True)

            async def consume_and_abort():
                observed_steps = []
                aborted = False
                terminal = None
                outputs = engine.add_request(request_id, prompt, stream_config)
                async with contextlib.aclosing(outputs):
                    async for output in outputs:
                        if output.finished:
                            terminal = output.finish_reason
                            break
                        queued = engine.scheduler.requests.get(request_id)
                        computed = None if queued is None else queued.num_computed_tokens
                        observed_steps.append({"completion_tokens": output.completion_tokens,
                                               "computed_tokens": computed})
                        if len(observed_steps) == 2:
                            if computed is None or computed <= 128:
                                raise AssertionError(f"Second output does not prove decode crossed page 128: {observed_steps}")
                            await engine.abort_request(request_id)
                            aborted = True
                if not aborted or terminal != "FINISHED_ABORTED":
                    raise AssertionError(f"Active abort did not terminate as aborted: steps={observed_steps}, reason={terminal}")
                return {"steps": observed_steps, "engine": dict(engine.observed[request_id])}

            first = await asyncio.wait_for(consume_and_abort(), args.case_timeout)
            # Finished output has passed the core's worker_release_done ACK.
            await idle()
            if request_id in engine.scheduler.requests:
                raise AssertionError("Aborted request remains registered in the scheduler")
            second_prompt = _render(tokenizer, _chat_payload(questions["heterogeneous"]))
            if second_prompt == prompt:
                raise AssertionError("ID reuse must change the prompt")

            async def reuse_id():
                terminal = None
                outputs = engine.add_request(
                    request_id, second_prompt,
                    GenerateConfig(max_new_tokens=96, temperature=0.0, top_p=1.0, stream=False),
                )
                async with contextlib.aclosing(outputs):
                    async for output in outputs:
                        if output.finished:
                            terminal = output
                if terminal is None or terminal.finish_reason != "FINISHED_EOS":
                    raise AssertionError("Reused request ID did not complete naturally")
                if re.search(r"\b4\b", terminal.text) is None:
                    raise AssertionError(f"Reused ID answer is incorrect: {terminal.text!r}")
                if terminal.prompt_tokens != len(tokenizer.encode(second_prompt)):
                    raise AssertionError("Reused ID retained the previous prompt length")
                if terminal.completion_tokens != len(terminal.token_ids) or not terminal.token_ids:
                    raise AssertionError("Reused ID token accounting is inconsistent")
                return dict(engine.observed[request_id])

            second = await asyncio.wait_for(reuse_id(), args.case_timeout)
            return {"aborted": first, "same_id_reused": second,
                    "coverage": "prefill to 127, decode across 128, public abort, ACK, same ID with different prompt"}

        await case("active_abort", active_abort)
        await case("repeat_after_abort", repeat)
        if args.mode == "K5":
            for stream in (False, True):
                async def reject(stream=stream):
                    before = set(engine.observed)
                    result = await request(_chat_payload(questions["arithmetic"], temperature=0.7, stream=stream), expected_status=400)
                    if set(engine.observed) != before:
                        raise AssertionError("Non-greedy request reached engine admission")
                    return result
                await case(f"non_greedy_400_{'stream' if stream else 'plain'}", reject)
        await case("final_repeat", repeat)
        if args.benchmark_repeats:
            summary["benchmark"] = await asyncio.wait_for(
                _benchmark_engine(engine, tokenizer, args, destination, event, idle),
                args.benchmark_timeout,
            )
        summary["passed"] = all(record["passed"] for record in summary["cases"].values())
        if args.compare_results is not None:
            other = json.loads(args.compare_results.read_text(encoding="utf-8"))
            if other["mode"] == args.mode or not other.get("passed"):
                raise AssertionError("Cross-mode comparison requires a passing result from the opposite mode")
            if other["max_model_len"] != args.max_model_len:
                raise AssertionError("Cross-mode results must use the same maximum model length")
            compared = []
            for name, record in summary["cases"].items():
                if not record["passed"] or name.startswith("non_greedy") or name in ("queued_abort", "active_abort"):
                    continue
                previous = other["cases"][name]
                if not previous["passed"]:
                    raise AssertionError(f"Other mode failed {name}")
                if name == "concurrent_two":
                    for left, right in zip(previous["result"]["requests"], record["result"]["requests"]):
                        same_output(left, right)
                else:
                    same_output(previous["result"], record["result"])
                compared.append(name)
            summary["cross_mode"] = {"other_mode": other["mode"], "compared": compared, "passed": True}
    except BaseException as error:
        summary["passed"] = False
        summary["fatal_error"] = f"{type(error).__name__}: {error}"
        event("fatal", error=summary["fatal_error"])
        raise
    finally:
        await client.aclose()
        event("engine_stop")
        try:
            stopped = await asyncio.wait_for(engine.stop(), args.shutdown_timeout)
            summary["workers_reaped"] = stopped
            if not stopped:
                summary["passed"] = False
        except BaseException as error:
            summary["workers_reaped"] = False
            summary["passed"] = False
            summary["shutdown_error"] = f"{type(error).__name__}: {error}"
        _write_json(destination / "summary.json", summary)
        event("complete", passed=summary["passed"], workers_reaped=summary["workers_reaped"])
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE", ""))
    parser.add_argument("--mode", choices=("K0", "K5"), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--long-prompt-tokens", type=int,
                        help="Add an exact-length chat prompt with a 128-token natural-EOS generation budget")
    parser.add_argument("--cases", default="all",
                        help="Comma-separated case names, e.g. long_prompt for an isolated 8K check")
    parser.add_argument("--compare-results", type=Path)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=2400)
    parser.add_argument("--case-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=180)
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--benchmark-warmups", type=int, default=1)
    parser.add_argument("--benchmark-prompt-tokens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--benchmark-output-tokens", type=int, default=64)
    parser.add_argument("--benchmark-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.benchmark_repeats < 0 or args.benchmark_warmups < 0 or args.benchmark_output_tokens < 2:
        parser.error("Invalid benchmark sample counts or output length")
    if any(length < 128 or length + args.benchmark_output_tokens > args.max_model_len
           for length in args.benchmark_prompt_tokens):
        parser.error("Benchmark prompt/output length is outside the configured model capacity")
    if min(args.startup_timeout, args.case_timeout, args.shutdown_timeout) <= 0:
        parser.error("Timeouts must be positive")
    if args.max_model_len < 192:
        parser.error("--max-model-len must be at least 192 for the standard boundary fixtures")
    if args.long_prompt_tokens is not None and not 1 <= args.long_prompt_tokens <= args.max_model_len - 128:
        parser.error("--long-prompt-tokens must leave at least 128 positions within --max-model-len")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
