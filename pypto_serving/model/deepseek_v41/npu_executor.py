# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving adapter for the V4.1 transactional runner and Ascend arithmetic."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeModel,
    SamplingParams,
)
from pypto_serving.model.common.executor.executor import ModelExecutor

from .cache import V41CacheState, build_v41_cache_group_specs
from .engram import EngramHashState, EngramLayout, build_compressed_token_map
from .npu_runner import DeepSeekV41Runner, V41WorkItem
from .weight_loader import DeepSeekV41WeightLoader


@dataclass(frozen=True)
class V41BackendCapabilities:
    """Concrete limits of a kernel bundle, checked before registering a model.

    v41_packed_v1 is this integration's payload/scale cache ABI, not the published
    reference's fake-quantized BF16 storage. reference_tp shards attention and
    dense projections across the same ranks that own the routed experts.
    """

    world_size: int
    max_chunk_tokens: int
    max_batch_size: int
    max_seq_len: int
    num_layers: int
    abi_version: int = 1
    platform: str = "a2a3"
    cache_format: str = "v41_packed_v1"
    parallel_mode: str = "reference_tp"
    supports_engram: bool = True
    transactional: bool = True

    def __post_init__(self) -> None:
        for name in (
            "world_size",
            "max_chunk_tokens",
            "max_batch_size",
            "max_seq_len",
            "num_layers",
            "abi_version",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"V4.1 capability {name} must be a positive integer")
        if type(self.supports_engram) is not bool or type(self.transactional) is not bool:
            raise ValueError("V4.1 capability flags must be boolean")


def _resolve_factory(factory: str | Callable[..., Any] | None) -> Callable[..., Any]:
    if factory is None:
        from .backend import create_backend
        return create_backend
    if isinstance(factory, str):
        module, separator, name = factory.partition(":")
        if not separator or not module or not name or ":" in name:
            raise ValueError("V4.1 kernel factory must be MODULE:CALLABLE")
        factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError("V4.1 kernel factory must be callable or MODULE:CALLABLE")
    return factory


class DeepSeekV41PyptoExecutor(ModelExecutor):
    """Use the existing worker's batching and sampling with isolated V4.1 state."""

    def __init__(
        self,
        *,
        platform: str = "a2a3",
        device_ids: Sequence[int] = (0,),
        pypto_build_dir: str | None = None,
        kernel_factory: str | Callable[..., Any] | None = None,
        use_compile_cache: bool = False,
        max_load_bytes: int = 256 << 20,
        draft_confidence_threshold: float | None = None,
    ) -> None:
        super().__init__()
        if platform not in ("a2a3", "a5"):
            raise ValueError("DeepSeek V4.1 execution requires platform='a2a3' or 'a5'")
        self._platform = platform
        self._factory = _resolve_factory(kernel_factory)
        self._device_ids = tuple(device_ids)
        if not self._device_ids or any(type(d) is not int or d < 0 for d in self._device_ids):
            raise ValueError("V4.1 requires nonnegative integer device IDs")
        if len(set(self._device_ids)) != len(self._device_ids):
            raise ValueError("V4.1 device IDs must be unique")
        self._build_dir, self._use_compile_cache = pypto_build_dir, use_compile_cache
        self._max_load_bytes = max_load_bytes
        if draft_confidence_threshold is not None and not math.isfinite(draft_confidence_threshold):
            raise ValueError("DSpark confidence threshold must be finite")
        self._draft_confidence_threshold = draft_confidence_threshold
        self._runner = self._model = None

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def register_model(self, model_id: str, record: ModelRecord) -> int:
        if self._runner is not None:
            raise ValueError("a V4.1 executor owns exactly one model")
        model, runtime = record.runtime_model, record.runtime
        if model.config.model_id != model_id or model.extra.get("family") != "deepseek_v41":
            raise ValueError("V4.1 executor/model registration mismatch")
        config = model.extra["v41_config"]
        if not 0 <= runtime.num_speculative_tokens <= config.text_config["dspark_block_size"]:
            raise ValueError("DSpark token reservation exceeds the checkpoint draft block")
        world = len(self._device_ids)
        config.expert_ownership(world, 0)
        if (
            config.num_attention_heads % world
            or config.o_groups % world
            or config.vocab_size % world
            or config.index_n_heads % world
        ):
            raise ValueError("V4.1 reference TP+EP ranks must divide heads, output groups and vocabulary")
        chunk = runtime.max_prefill_tokens_per_request or runtime.max_num_batched_tokens
        expected_groups = build_v41_cache_group_specs(
            config, runtime.page_size, runtime.max_seq_len, runtime.total_kv_pages, max_chunk_tokens=chunk
        )
        if runtime.kv_cache_groups != expected_groups:
            raise ValueError("V4.1 requires its config-derived cache groups on both scheduler and worker")
        cache = V41CacheState(
            config, page_size=runtime.page_size, max_seq_len=runtime.max_seq_len, max_chunk_tokens=chunk
        )
        loader = DeepSeekV41WeightLoader(
            model.extra["model_dir"], model.extra["config_data"], max_load_bytes=self._max_load_bytes
        )
        backend = self._factory(
            config=config,
            runtime=runtime,
            cache_layouts=cache.groups,
            weight_loader=loader,
            device_ids=self._device_ids,
            platform=self._platform,
            pypto_build_dir=self._build_dir,
            use_compile_cache=self._use_compile_cache,
        )
        try:
            caps = backend.capabilities
            if runtime.num_speculative_tokens and any(
                not callable(getattr(backend, name, None))
                for name in ("checkpoint", "finish_checkpoint", "propose")
            ):
                raise ValueError("DSpark requires checkpoint, finish_checkpoint and propose backend methods")
            if not isinstance(caps, V41BackendCapabilities):
                raise ValueError("kernel bundle must return V41BackendCapabilities")
            if (
                caps.abi_version != 1
                or caps.platform != self._platform
                or caps.cache_format != "v41_packed_v1"
                or caps.parallel_mode != "reference_tp"
                or not caps.supports_engram
                or not caps.transactional
                or caps.world_size != world
                or caps.num_layers != config.num_hidden_layers
            ):
                raise ValueError("kernel bundle does not implement the complete V4.1 text execution contract")
            if (
                caps.max_chunk_tokens < chunk
                or caps.max_batch_size < runtime.max_batch_size
                or caps.max_seq_len < runtime.max_seq_len
            ):
                raise ValueError("configured runtime exceeds the kernel bundle's supported limits")
            pages = backend.num_pages
            stride = expected_groups[0].max_blocks_per_seq
            if type(pages) is not int or pages < stride or pages % stride:
                raise ValueError("backend num_pages must cover integral primary-cache request slots")
            if runtime.total_kv_pages is not None and pages != runtime.total_kv_pages:
                raise ValueError("backend page allocation disagrees with runtime.total_kv_pages")
            tokenizer = getattr(record.tokenizer, "tokenizer", record.tokenizer)
            token_map, compressed_size = build_compressed_token_map(tokenizer)
            if len(token_map) != config.vocab_size:
                raise ValueError("Engram tokenizer vocabulary does not match model config")
            if compressed_size != config.text_config["engram_compressed_vocab_size"]:
                raise ValueError("Engram compressed vocabulary does not match checkpoint hash multipliers")
            layout = EngramLayout.from_config(config.text_config)
            if layout is None:
                raise ValueError("V4.1 text executor requires configured Engram layers")

            # Build once before registration to reject a mismatched tokenizer immediately.
            def history_factory() -> EngramHashState:
                return EngramHashState(
                    layout,
                    token_map,
                    compressed_vocab_size=compressed_size,
                    pad_token_id=config.text_config["engram_pad_token_id"],
                )

            prototype = history_factory()
            self._runner = DeepSeekV41Runner(
                config,
                cache,
                backend,
                history_factory,
                max_requests=runtime.max_batch_size,
            )
            self._model = model
            model.extra["engram_token_map_sha256"] = prototype.token_map_sha256
            self._page_limits = {
                group.name: pages // stride * group.max_blocks_per_seq for group in expected_groups
            }
            return pages
        except BaseException:
            backend.close()
            raise

    def _require_model(self, model: RuntimeModel) -> DeepSeekV41Runner:
        if self._runner is None or model is not self._model:
            raise ValueError("model has not been registered with this V4.1 executor")
        return self._runner

    def _logits(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        vocab = self._model.config.vocab_size
        if any(
            not isinstance(row, torch.Tensor) or row.shape != (vocab,) or not row.is_floating_point()
            for row in outputs
        ):
            raise ValueError("V4.1 head must return one floating vocabulary row per request")
        logits = torch.stack([row.detach().to(device="cpu", dtype=torch.float32) for row in outputs])
        if not torch.isfinite(logits).all().item():
            raise ValueError("V4.1 head returned nonfinite logits")
        return logits

    def _metadata(self, batch: PrefillBatch | DecodeBatch) -> int:
        count = len(batch.request_ids)
        if len(batch.block_ids_by_group) != count:
            raise ValueError("every V4.1 request needs scheduler-owned cache group pages")
        partitions = batch.cache_partitions or [0] * count
        if len(partitions) != count or any(p not in (0, None) for p in partitions):
            raise ValueError("V4.1 reference TP+EP uses one scheduler cache partition")
        if batch.token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be an integer tensor")
        for groups in batch.block_ids_by_group:
            for name, pages in groups.items():
                if name not in self._page_limits or any(
                    type(page) is not int or not 0 <= page < self._page_limits[name] for page in pages
                ):
                    raise ValueError("V4.1 scheduler page ID exceeds the backend allocation")
        return count

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        runner = self._require_model(model)
        count = self._metadata(batch)
        if any(
            len(values) != count
            for values in (batch.chunk_lens, batch.chunk_offsets, batch.chunk_starts, batch.seq_lens)
        ):
            raise ValueError("inconsistent packed prefill metadata")
        tokens = batch.token_ids.reshape(-1).tolist()
        multimodal = getattr(batch, "multimodal", None) or [None] * count
        if len(multimodal) != count:
            raise ValueError("prefill multimodal metadata must match the request batch")
        cursor, items = 0, []
        for i in range(count):
            size, start = batch.chunk_lens[i], batch.chunk_starts[i]
            if (
                type(size) is not int
                or size < 1
                or type(start) is not int
                or start < 0
                or batch.chunk_offsets[i] != cursor
                or batch.seq_lens[i] != start + size
            ):
                raise ValueError("invalid prefill chunk offsets, lengths or positions")
            images = None
            if multimodal[i] is None and model.extra["config_data"].get("image_token_id", 129264) in tokens[cursor:cursor + size]:
                raise ValueError("image placeholder tokens require multimodal image metadata")
            if multimodal[i] is not None:
                from .vision import VisionConfig, from_wire
                images = from_wire(multimodal[i], VisionConfig.from_config(model.extra["config_data"]))
                if images.tokens[start:start + size] != tuple(tokens[cursor:cursor + size]):
                    raise ValueError("multimodal prompt tokens disagree with scheduler input tokens")
                boundary = max(image.start + len(image.types) for image in images.images)
                if boundary > (size if start == 0 else start):
                    raise ValueError("all image spans must fit the first prefill chunk")
            items.append(
                V41WorkItem(
                    batch.request_ids[i],
                    tuple(tokens[cursor : cursor + size]),
                    start,
                    batch.block_ids_by_group[i],
                    multimodal=images,
                )
            )
            cursor += size
        if cursor != len(tokens):
            raise ValueError("packed prefill token count disagrees with chunk lengths")
        return PrefillResult(None, runner.execute(items, validate_outputs=self._logits))

    def finalize_prefill(
        self,
        model: RuntimeModel,
        request_ids: list[str],
        sampled_token_ids: list[int],
        sampling_params: list[SamplingParams] | None = None,
    ) -> None:
        if len(request_ids) != len(sampled_token_ids):
            raise ValueError("terminal-prefill token count mismatch")
        self._require_model(model).finalize_prefill(request_ids)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        runner = self._require_model(model)
        count = self._metadata(batch)
        if (
            batch.token_ids.numel() != count
            or batch.seq_lens.numel() != count
            or batch.seq_lens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("decode needs one token and integer sequence length per request")
        items = [
            V41WorkItem(rid, (token,), length - 1, groups, mode="decode")
            for rid, token, length, groups in zip(
                batch.request_ids,
                batch.token_ids.reshape(-1).tolist(),
                batch.seq_lens.reshape(-1).tolist(),
                batch.block_ids_by_group,
            )
        ]
        extra = model.runtime.num_speculative_tokens
        if extra and len(batch.sampling_params) == count and all(
            params.temperature == 0 for params in batch.sampling_params
        ):
            logits, emitted = runner.speculate(items, extra_tokens=extra, validate_outputs=self._logits,
                                                minimum_score=self._draft_confidence_threshold)
            return DecodeResult(None, logits, accepted_token_ids=emitted)
        return DecodeResult(None, runner.execute(items, validate_outputs=self._logits))

    def release_finished_requests(self, request_ids: list[str]) -> None:
        if self._runner is not None:
            self._runner.release(request_ids)

    def diagnostics(self) -> dict:
        """Return actual host counters; memory values are exposed by the owning backend."""
        runner = self._require_model(self._model)
        return {"speculation": dict(runner.speculation_stats),
                "backend": runner.backend.diagnostics() if hasattr(runner.backend, "diagnostics") else {}}

    def close(self) -> None:
        if self._runner is not None:
            try:
                self._runner.close()
            finally:
                self._runner = self._model = None
