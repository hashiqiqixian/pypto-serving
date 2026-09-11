# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inline images through real tokenization, scheduling, IPC, packing, and HTTP accounting."""

from __future__ import annotations

import asyncio
import base64
import copy
from dataclasses import replace
import importlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("msgspec")
pytest.importorskip("fastapi")
tokenizers = pytest.importorskip("tokenizers")
Image = pytest.importorskip("PIL.Image")
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def modules(monkeypatch):
    previous = set(sys.modules)
    for name in ("pypto_serving", "pypto_serving.model"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)
    names = {
        "tokenizer": "model.deepseek_v41.tokenizer",
        "encoding": "model.deepseek_v41.encoding",
        "vision": "model.deepseek_v41.vision",
        "common": "model.tokenizer",
        "types": "config.types",
        "engine": "serving.engine.async_engine",
        "scheduler": "serving.sched.scheduler",
        "ipc": "serving.server.ipc",
        "worker": "serving.server.serving_worker",
        "server": "serving.server.server",
        "memory": "serving.memory.kv_cache",
    }
    loaded = {name: importlib.import_module("pypto_serving." + path) for name, path in names.items()}
    yield SimpleNamespace(**loaded)
    for name in set(sys.modules) - previous:
        if name.startswith("pypto_serving."):
            sys.modules.pop(name, None)


@pytest.fixture
def adapter(modules):
    names = [
        modules.encoding.BOS_TOKEN,
        modules.encoding.EOS_TOKEN,
        "[UNK]",
        modules.encoding.USER_TOKEN,
        modules.encoding.ASSISTANT_TOKEN,
        modules.encoding.SYSTEM_TOKEN,
        modules.encoding.LATEST_REMINDER_TOKEN,
        modules.encoding.IMAGE_TOKEN,
    ]
    native = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({name: i for i, name in enumerate(names)}, unk_token="[UNK]")
    )
    native.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    native.add_special_tokens(names)

    class LocalTokenizer:
        bos_token_id, eos_token_id = 0, 1

        def encode(self, text, *, add_special_tokens=False):
            return native.encode(text, add_special_tokens=add_special_tokens).ids

    raw = {
        "image_token_id": 7,
        "text_config": {"hidden_size": 6},
        "vision_config": {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 12,
            "patch_size": 2,
            "downsample_ratio": 2,
            "max_image_tokens": 40,
            "min_pixels": 0,
        },
    }
    return modules.tokenizer.DeepSeekV41TokenizerAdapter(LocalTokenizer(), raw_config=raw)


@pytest.fixture
def messages():
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color=(20, 50, 100)).save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "again"},
            ],
        }
    ]


def test_real_tokenization_wire_roundtrip_and_rejected_metadata(modules, adapter, messages):
    before = copy.deepcopy(messages)
    prepared = adapter.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert isinstance(prepared, modules.common.PreparedPrompt)
    assert "look\n\n" + modules.encoding.IMAGE_TOKEN + "\n\nagain" in prepared.text
    config = modules.vision.VisionConfig.from_config(adapter.raw_config)
    result = modules.vision.from_wire(prepared.multimodal, config)
    assert list(result.tokens) == prepared.token_ids
    assert result.token_types.count(modules.vision.IMAGE) == 1
    assert prepared.token_ids.count(config.image_token_id) == 4
    assert messages == before
    for mutate in (
        lambda p: p.update(version=2),
        lambda p: p.update(first_chunk_end=0),
        lambda p: p["images"][0].update(patches_bf16=b"x"),
        lambda p: p["token_types"].__setitem__(0, modules.vision.IMAGE),
        lambda p: p["tokens"].__setitem__(0, config.image_token_id),
    ):
        invalid = copy.deepcopy(prepared.multimodal)
        mutate(invalid)
        with pytest.raises(ValueError):
            modules.vision.from_wire(invalid, config)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/image.png",
        "C:/secret.png",
        "data:image/png,raw",
        "data:image/png;base64,bm90IGFuIGltYWdl",
    ],
)
def test_image_sources_fail_explicitly_without_resource_access(adapter, messages, url):
    messages[0]["content"][1]["image_url"]["url"] = url
    with pytest.raises(ValueError):
        adapter.apply_chat_template(messages)


def make_scheduler(modules, limit):
    manager = modules.memory.KvCacheManager(num_blocks=64, block_size=2, enable_prefix_cache=False)
    config = modules.scheduler.SchedulerConfig(
        max_seq_len=128,
        enable_prefix_cache=False,
        max_num_scheduled_tokens=limit,
        long_prefill_token_threshold=limit,
    )
    return modules.scheduler.Scheduler(config, manager)


def test_scheduler_waits_for_full_image_span_and_rejects_impossible_first_chunk(modules, adapter, messages):
    prepared = adapter.apply_chat_template(messages)
    boundary = prepared.multimodal["first_chunk_end"]
    request = modules.scheduler.Request("image", prepared.token_ids, 2, multimodal=prepared.multimodal)
    scheduler = make_scheduler(modules, boundary)
    scheduler.add_request(request)
    assert scheduler._limit_scheduled_tokens(request, boundary - 1) == 0
    assert scheduler._limit_scheduled_tokens(request, boundary) == boundary
    small = make_scheduler(modules, boundary - 1)
    with pytest.raises(ValueError, match="first prefill chunk"):
        small.add_request(
            modules.scheduler.Request("too-large", prepared.token_ids, 2, multimodal=prepared.multimodal)
        )
    assert not small.requests


def test_image_data_is_registered_once_then_reaches_worker_prefill(modules, adapter, messages):
    prepared = adapter.apply_chat_template(messages)
    request = modules.scheduler.Request("image", prepared.token_ids, 2, multimodal=prepared.multimodal)
    boundary = prepared.multimodal["first_chunk_end"]
    scheduled = modules.scheduler.ScheduledRequest(request=request, num_new_tokens=boundary, is_prefill=True)
    output = modules.scheduler.SchedulerOutput(scheduled_requests=[scheduled])
    core = modules.engine.ReplicaEngineCore.__new__(modules.engine.ReplicaEngineCore)
    core._worker_known_req_ids = set()
    first = modules.ipc.decode_command(modules.ipc.encode_command(core._build_step_command(output, [])))
    assert first.new_requests[0].multimodal == prepared.multimodal
    second = core._build_step_command(output, [])
    assert second.new_requests == []
    assert len(modules.ipc.encode_command(second)) < len(modules.ipc.encode_command(first))
    batches = []
    executor = SimpleNamespace(
        supports_device_embedding=True,
        supports_device_sampling=False,
        device_topk_sampling_k=0,
        run_prefill=lambda model, batch: batches.append(batch) or None,
    )
    worker = modules.worker.WorkerProcess.__new__(modules.worker.WorkerProcess)
    worker.executor, worker._req_cache = executor, {"image": first.new_requests[0]}
    worker._batch_prefill(first.prefill_requests, SimpleNamespace(runtime=SimpleNamespace(device="cpu")), {})
    assert batches[0].multimodal == [prepared.multimodal]
    assert batches[0].token_ids.tolist() == prepared.token_ids[:boundary]
    assert batches[0].chunk_starts == [0]


def test_actual_executor_decodes_wire_before_work_dispatch_and_rejects_missing_data(
    modules, adapter, messages
):
    executor_module = importlib.import_module("pypto_serving.model.deepseek_v41.npu_executor")
    prepared = adapter.apply_chat_template(messages)
    received = []

    class RecordingRunner:
        def execute(self, items, *, validate_outputs):
            received.extend(items)
            return validate_outputs([torch.zeros(8) for _ in items])

    executor = executor_module.DeepSeekV41PyptoExecutor.__new__(executor_module.DeepSeekV41PyptoExecutor)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=8), extra={"config_data": adapter.raw_config})
    executor._model, executor._runner, executor._page_limits = model, RecordingRunner(), {}
    length = len(prepared.token_ids)
    batch = modules.types.PrefillBatch(
        request_ids=["image"],
        token_ids=torch.tensor(prepared.token_ids),
        input_embeddings=None,
        seq_lens=[length],
        chunk_lens=[length],
        chunk_offsets=[0],
        chunk_starts=[0],
        block_ids_by_group=[{}],
        multimodal=[prepared.multimodal],
    )
    result = executor.run_prefill(model, batch)
    assert result.logits.shape == (1, 8)
    assert len(received) == 1
    assert received[0].multimodal.tokens == tuple(prepared.token_ids)
    assert len(received[0].multimodal.images) == 1
    assert received[0].multimodal.images[0].patches.dtype == torch.bfloat16
    wrong_tokens = batch.token_ids.clone()
    wrong_tokens[0] += 1
    with pytest.raises(ValueError, match="tokens disagree"):
        executor.run_prefill(model, replace(batch, token_ids=wrong_tokens))
    with pytest.raises(ValueError, match="image|multimodal"):
        executor.run_prefill(model, replace(batch, multimodal=[]))
    split = prepared.multimodal["first_chunk_end"] - 1
    with pytest.raises(ValueError, match="first prefill chunk"):
        executor.run_prefill(
            model, replace(batch, token_ids=batch.token_ids[:split], seq_lens=[split], chunk_lens=[split])
        )
    assert len(received) == 1


def test_prepared_prompt_direct_engine_path_keeps_expanded_counts(modules, adapter, messages):
    prepared = adapter.apply_chat_template(messages)
    received = []

    class Core:
        def pending_token_load(self):
            return 0

        async def add_request(self, rid, prompt, config, **kwargs):
            received.append((prompt, kwargs["prompt_token_ids"], kwargs["multimodal"]))
            kwargs["on_queued"]()
            yield modules.engine.TokenOutput(finished=True, prompt_tokens=len(kwargs["prompt_token_ids"]))

    engine = modules.engine.AsyncLLMEngine.__new__(modules.engine.AsyncLLMEngine)
    engine._cores, engine._route_extra_load, engine._route_counter, engine._request_to_replica = (
        [Core()],
        [0],
        0,
        {},
    )
    engine.tokenizer = adapter

    async def collect():
        return [item async for item in engine.add_request("image", prepared, modules.types.GenerateConfig())]

    outputs = asyncio.run(collect())
    assert received == [(prepared.text, prepared.token_ids, prepared.multimodal)]
    assert outputs[0].prompt_tokens == len(prepared.token_ids)
    assert engine._route_extra_load == [0]


@pytest.mark.parametrize("stream", [False, True])
def test_http_inline_image_preserves_usage_in_both_response_modes(modules, adapter, messages, stream):
    seen = []
    scheduler = make_scheduler(modules, 32)

    class Engine:
        tokenizer = adapter

        def validate_prepared_prompt(self, prompt):
            scheduler.validate_multimodal(len(prompt.token_ids), prompt.multimodal)

        async def add_request(self, rid, prompt, config):
            seen.append(prompt)
            yield modules.engine.TokenOutput(
                text="answer",
                finished=True,
                finish_reason="FINISHED_LENGTH",
                prompt_tokens=len(prompt.token_ids),
                completion_tokens=1,
            )

    server = modules.server.ServingServer(
        async_engine=Engine(), model_id="v41", generate_config=modules.types.GenerateConfig()
    )
    request = modules.server.ChatCompletionRequest(messages=messages, stream=stream)

    async def call():
        response = await server._chat_completions(request)
        if not stream:
            return json.loads(response.body)
        chunks = [chunk async for chunk in response.body_iterator]
        items = [
            json.loads(line[6:])
            for chunk in chunks
            for line in chunk.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        return next(item for item in items if item.get("usage"))

    result = asyncio.run(call())
    assert isinstance(seen[0], modules.common.PreparedPrompt)
    assert result["usage"]["prompt_tokens"] == len(seen[0].token_ids)
    assert result["usage"]["total_tokens"] == len(seen[0].token_ids) + 1


def test_streaming_image_chunk_rejection_happens_before_headers(modules, adapter, messages):
    scheduler = make_scheduler(modules, 3)

    class Engine:
        tokenizer = adapter

        def validate_prepared_prompt(self, prompt):
            scheduler.validate_multimodal(len(prompt.token_ids), prompt.multimodal)

    server = modules.server.ServingServer(
        async_engine=Engine(), model_id="v41", generate_config=modules.types.GenerateConfig()
    )
    request = modules.server.ChatCompletionRequest(messages=messages, stream=True)
    with pytest.raises(ValueError, match="first prefill chunk"):
        asyncio.run(server._chat_completions(request))
