# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete V4.1 arithmetic backend with explicit CPU and PyPTO Ascend providers.

The initial implementation streams checkpoint tiles and uses a real PyPTO
BF16 Cube kernel for representable normalized products. Torch Ascend executes
FP32-sensitive and vector arithmetic. Transfers are explicit; this implementation
is intended for correctness bring-up, not a performance or HBM residency claim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .attention import Attention, SharedAttention
from .npu_executor import V41BackendCapabilities
from .numerics import ModelMath, TensorOps, rms_norm
from .packed_cache import PackedPagePool, PackedRows


@dataclass
class _RequestState:
    layers: list[Any]
    draft: Any = None
    last_hidden: torch.Tensor | None = None
    position: int = 0

    def snapshot(self):
        return ([layer.snapshot() for layer in self.layers],
                self.draft.snapshot() if self.draft is not None else None, self.last_hidden, self.position)

    def restore(self, snapshot) -> None:
        layers, draft, self.last_hidden, self.position = snapshot
        for layer, old in zip(self.layers, layers):
            layer.restore(old)
        if self.draft is not None:
            self.draft.restore(draft)


@dataclass
class _Batch:
    contexts: tuple
    snapshots: dict = field(default_factory=dict)
    created: list = field(default_factory=list)
    active: bool = True


@dataclass
class _Hidden:
    value: torch.Tensor
    pre_mix: torch.Tensor
    shared: SharedAttention
    targets: list[torch.Tensor] = field(default_factory=list)
    image_mask: torch.Tensor | None = None


class DeepSeekV41Backend:
    """One tensor-parallel rank, including mandatory Engram and all backbone layers."""

    def __init__(self, config, runtime, cache_layouts, ops: TensorOps, *, vision_config=None,
                 platform: str = "a2a3"):
        self.config, self.runtime, self.ops = config, runtime, ops
        self.math = ModelMath(config, ops)
        self.attention = [Attention(config, ops, i) for i in range(config.num_hidden_layers)]
        stride = cache_layouts[0].max_blocks_per_seq
        self.num_pages = runtime.total_kv_pages or stride * runtime.max_batch_size
        self.pool = PackedPagePool(tuple(cache_layouts), self.num_pages, device=str(ops.device))
        self.cache_capacity_bytes = sum(layout.page_size_bytes * self.pool.limits[layout.name]
                                        for layout in cache_layouts)
        self.memory_preflight = self._memory_preflight()
        self.requests: dict[tuple[str, int], _RequestState] = {}
        self._ticket: _Batch | None = None
        self._closed = False
        self.capabilities = V41BackendCapabilities(
            world_size=ops.world_size,
            max_chunk_tokens=runtime.max_prefill_tokens_per_request or runtime.max_num_batched_tokens,
            max_batch_size=runtime.max_batch_size, max_seq_len=runtime.max_seq_len,
            num_layers=config.num_hidden_layers, platform=platform,
        )
        self.vision = None
        if vision_config is not None:
            from .vision import VisionTower
            self.vision = VisionTower(vision_config, ops)
        self.drafter = None
        if runtime.num_speculative_tokens:
            from .draft import DSparkDrafter
            self.drafter = DSparkDrafter(config, ops)

    @staticmethod
    def _key(context):
        return context.work.request_id, context.generation

    def _memory_preflight(self) -> dict[str, Any] | None:
        """Check known current allocations plus eventual cache; temporary workspace is excluded."""
        utilization = getattr(self.runtime, "npu_memory_utilization", None)
        if utilization is not None and (
            isinstance(utilization, bool) or not isinstance(utilization, (int, float))
            or not math.isfinite(utilization) or not 0 < utilization <= 1
        ):
            raise ValueError("npu_memory_utilization must be a finite fraction in (0,1]")
        npu = getattr(torch, "npu", None)
        if self.ops.device.type != "npu" or npu is None or not hasattr(npu, "mem_get_info"):
            return None
        free, total = (int(value) for value in npu.mem_get_info(self.ops.device))
        allocated = int(npu.memory_allocated(self.ops.device)) if hasattr(npu, "memory_allocated") else None
        limit = int(total * utilization) if utilization is not None else None
        if allocated is not None and limit is not None and allocated + self.cache_capacity_bytes > limit:
            raise ValueError("current NPU allocations plus full logical cache exceed npu_memory_utilization")
        return {"free_bytes": free, "total_bytes": total, "allocated_bytes": allocated,
                "budget_bytes": limit, "logical_cache_capacity_bytes": self.cache_capacity_bytes,
                "excludes_activation_and_jit_workspace": True}

    def _check(self, ticket, context=None) -> None:
        if self._closed or ticket is not self._ticket or not ticket.active:
            raise RuntimeError("inactive V4.1 backend transaction")
        if context is not None and not any(context is entry for entry in ticket.contexts):
            raise ValueError("execution context does not belong to this batch")

    def begin_batch(self, contexts: tuple) -> _Batch:
        if self._closed or (self._ticket is not None and self._ticket.active):
            raise RuntimeError("V4.1 backend is closed or already executing a batch")
        if not contexts or len(contexts) > self.runtime.max_batch_size:
            raise ValueError("invalid V4.1 batch size")
        ticket = _Batch(contexts)
        self._ticket = ticket
        self.pool.begin()
        try:
            for context in contexts:
                key = self._key(context)
                request = self.requests.get(key)
                if request is None:
                    if context.work.start_pos:
                        raise ValueError("new backend request must start at zero")
                    def rows(kind, layer, dim, fmt):
                        return PackedRows(self.pool, f"{kind}.{layer}", dim, fmt)
                    request = _RequestState([
                        attention.new_state(self.runtime.max_seq_len, rows_factory=rows)
                        for attention in self.attention
                    ], self.drafter.new_state(self.runtime.max_seq_len) if self.drafter else None)
                    self.requests[key] = request
                    ticket.created.append(key)
                if request.position != context.work.start_pos:
                    raise ValueError("backend and host request positions disagree")
                ticket.snapshots[key] = request.snapshot()
                for state in request.layers:
                    for store in (state.main, state.index, state.swa):
                        if store is not None:
                            store.bind(context.cache)
            return ticket
        except BaseException:
            self.abort_batch(ticket)
            raise

    @torch.inference_mode()
    def embed(self, ticket, context) -> _Hidden:
        self._check(ticket, context)
        ids = torch.tensor(context.work.token_ids, dtype=torch.long, device=self.ops.device)
        value = self.ops.embedding("embed.weight", ids)
        multimodal = getattr(context.work, "multimodal", None)
        image_mask = None
        if multimodal is not None:
            if self.vision is None:
                raise ValueError("this checkpoint has no vision tower")
            images = multimodal.images
            types = torch.tensor(multimodal.token_types, dtype=torch.int64, device=self.ops.device)
            start, end = context.work.start_pos, context.cache.end_pos
            if start == 0:
                value = self.vision.merge_embeddings(value.unsqueeze(0), [images])[0]
            elif any(image.start + len(image.types) > start for image in images):
                raise ValueError("image spans must fit the first prefill chunk")
            image_mask = types[start:end] >= 0
            if len(image_mask) != len(ids):
                raise ValueError("image token types do not cover the current chunk")
        hc = self.math.hc
        value = value.unsqueeze(-2).expand(-1, hc, -1).clone()
        pre_mix = value.new_zeros((len(ids), hc), dtype=torch.float32)
        pre_mix[:, 0] = 1
        return _Hidden(value, pre_mix, SharedAttention(context.work.start_pos, context.cache.end_pos),
                       image_mask=image_mask)

    @torch.inference_mode()
    def engram(self, ticket, layer, state: _Hidden, context) -> _Hidden:
        self._check(ticket, context)
        layers = tuple(self.config.text_config["engram_layer_ids"])
        index = layers.index(layer.layer_id)
        hashes = torch.tensor(context.engram_hashes, dtype=torch.long, device=self.ops.device)[:, index]
        mask = None if state.image_mask is None else ~state.image_mask
        state.value = self.math.engram(state.value, hashes, f"layers.{layer.layer_id}.engram", mask)
        return state

    @torch.inference_mode()
    def layer(self, ticket, layer, state: _Hidden, context) -> _Hidden:
        self._check(ticket, context)
        layer_id = layer.layer_id
        if layer_id in self.config.text_config["dspark_target_layer_ids"]:
            state.targets.append(state.value.mean(-2))
        request = self.requests[self._key(context)]
        def attend(x):
            return self.attention[layer_id].forward(x, context.work.start_pos, request.layers[layer_id],
                                                    state.shared)
        state.value, state.pre_mix = self.math.block(state.value, state.pre_mix, f"layers.{layer_id}", attend,
                                                     image_mask=state.image_mask)
        return state

    @torch.inference_mode()
    def head(self, ticket, state: _Hidden, context) -> torch.Tensor:
        self._check(ticket, context)
        collapsed = self.math.hc_pre(state.value, state.pre_mix)
        hidden = rms_norm(collapsed[-1:], self.ops.weight("norm.weight"), self.math.eps)
        logits = self.ops.all_gather(self.ops.linear(hidden.float(), "head.weight"))[0]
        if logits.shape != (self.config.vocab_size,) or not bool(torch.isfinite(logits).all()):
            raise ValueError("V4.1 head produced invalid vocabulary logits")
        request = self.requests[self._key(context)]
        targets = torch.cat(state.targets, -1) if state.targets else None
        # Draft KV is seeded from committed target attention inputs; proposals
        # execute separately so sampled anchors may come from the shared sampler.
        if self.drafter is not None:
            self.drafter.seed(targets, context.work.start_pos, request.draft)
        request.last_hidden = targets[-1:].clone() if targets is not None else None
        request.position = context.cache.end_pos
        return logits.detach().clone()

    def commit_batch(self, ticket) -> None:
        self._check(ticket)
        ticket.active = False

    def abort_batch(self, ticket) -> None:
        if ticket is not self._ticket:
            raise RuntimeError("cannot abort an unrelated backend batch")
        self.pool.abort()
        for key, snapshot in ticket.snapshots.items():
            self.requests[key].restore(snapshot)
        for key in ticket.created:
            self.requests.pop(key, None)
        ticket.active = False

    def release_request(self, request_id: str, generation: int) -> None:
        self.requests.pop((request_id, generation), None)

    def checkpoint(self):
        if self._ticket is not None and self._ticket.active:
            raise RuntimeError("checkpoint requires an inactive backend")
        return self.pool.checkpoint(), {key: state.snapshot() for key, state in self.requests.items()}

    def finish_checkpoint(self, checkpoint, *, restore: bool = False) -> None:
        pages, snapshots = checkpoint
        self.pool.finish_checkpoint(pages, restore=restore)
        if restore:
            for key in tuple(self.requests):
                if key not in snapshots:
                    del self.requests[key]
                else:
                    self.requests[key].restore(snapshots[key])
        if self._ticket is not None:
            self._ticket.active = False

    @torch.inference_mode()
    def propose(self, request_id: str, generation: int, anchor: int):
        if self.drafter is None:
            raise RuntimeError("DSpark was not enabled for this backend")
        return self.drafter.propose(anchor, self.requests[request_id, generation].draft)

    def diagnostics(self) -> dict[str, Any]:
        """Report allocated tensors and allocator readings, without inferring weight residency."""
        def tensor_bytes(values):
            return sum(value.numel() * value.element_size() for value in values if value is not None)

        result = {
            "rank": self.ops.rank, "world_size": self.ops.world_size, "device": str(self.ops.device),
            "active_requests": len(self.requests), "cache_bytes": tensor_bytes(self.pool.pages.values()),
            "logical_cache_capacity_bytes": self.cache_capacity_bytes,
            "memory_preflight_estimate": self.memory_preflight,
            "journal_bytes": tensor_bytes((self.pool.journal or {}).values()),
            "checkpoint_journal_bytes": tensor_bytes((self.pool.outer_journal or {}).values()),
            "weight_placement": getattr(self.ops.weights, "placement", None),
            "weight_row_cache_bytes": getattr(self.ops.weights, "_cached_bytes", None),
            "npu_memory_allocated": None, "npu_memory_reserved": None, "npu_max_memory_allocated": None,
        }
        npu = getattr(torch, "npu", None)
        if self.ops.device.type == "npu" and npu is not None:
            for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
                reader = getattr(npu, name, None)
                if reader is not None:
                    result["npu_" + name] = int(reader(self.ops.device))
        return result

    def close(self) -> None:
        try:
            if self.ops.matmul_provider is not None:
                self.ops.matmul_provider.close()
        finally:
            self.ops.weights.close()
            self.pool.close()
            self.requests.clear()
            self._closed = True


def create_backend(*, config, runtime, cache_layouts, weight_loader, device_ids, platform="a2a3",
                   pypto_build_dir=None, use_compile_cache=False):
    """Built-in Ascend backend factory with explicit platform selection."""
    if platform not in ("a2a3", "a5"):
        raise ValueError("V4.1 Ascend backend requires platform='a2a3' or 'a5'")
    if len(device_ids) > 1:
        from .distributed import DistributedV41Backend
        return DistributedV41Backend(config=config, runtime=runtime, cache_layouts=cache_layouts,
                                     weight_loader=weight_loader, device_ids=device_ids,
                                     platform=platform,
                                     pypto_build_dir=pypto_build_dir, use_compile_cache=use_compile_cache)
    import torch_npu  # noqa: F401 - registers the actual torch Ascend device

    from .pypto_ops import PyptoMatmulOps
    from .tensor_store import DeepSeekV41TensorStore
    from .vision import VisionConfig

    device_id = device_ids[0]
    torch.npu.set_device(device_id)
    raw = json.loads((Path(weight_loader.model_dir) / "config.json").read_text())
    store = DeepSeekV41TensorStore(weight_loader.model_dir, raw, max_load_bytes=weight_loader.max_load_bytes)
    provider = None
    try:
        provider = PyptoMatmulOps(device_id=device_id, platform=platform, build_dir=pypto_build_dir)
        ops = TensorOps(store, device=f"npu:{device_id}", matmul_provider=provider)
        return DeepSeekV41Backend(config, runtime, cache_layouts, ops, platform=platform,
                                 vision_config=VisionConfig.from_config(raw))
    except BaseException:
        if provider is not None:
            provider.close()
        store.close()
        raise
