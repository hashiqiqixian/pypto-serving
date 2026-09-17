# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 attention tensor mathematics and append-only, request-owned state.

Derived from inference/model.py and sparse_attn in inference/kernel.py at revision
dba1be0a40aa45a94ad051997016db3960a90277. All backbone layers execute; compression
and index ownership do not introduce an encoder/decoder split. Quantizers return
the reference's dequantized values. An injected row store may additionally retain
their packed payload; this module never relabels BF16 storage as a packed cache.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import torch

from .config import DeepSeekV41Config


FP8_SWA = "fp8_e4m3_ue8m0"
FP4_INDEX = "fp4_e2m1_ue8m0"
FP4_MAIN = "fp4_e2m1_e4m3"


class AttentionOps(Protocol):
    """Linear returns a rank-local result; collectives are explicit in attention."""

    def linear(self, x: torch.Tensor, name: str) -> torch.Tensor: ...
    def weight(self, name: str) -> torch.Tensor: ...
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...
    def quantize(self, x: torch.Tensor, fmt: str, block: int) -> torch.Tensor: ...
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


class RowStore(Protocol):
    """Append/gather interface for tensor pages or scheduler-owned device pages.

    append must preserve already committed rows. truncate hides a suffix without
    changing retained rows; gather accepts only nonnegative, in-range row IDs.
    Reads return dequantized values suitable for arithmetic on the input device.
    """

    length: int

    def append(self, values: torch.Tensor) -> None: ...
    def read(self, start: int, end: int) -> torch.Tensor: ...
    def gather(self, indices: torch.Tensor) -> torch.Tensor: ...
    def truncate(self, length: int) -> None: ...


class TensorRows:
    """Allocate small tensor pages on demand, with an explicit row capacity."""

    def __init__(self, dim: int, capacity: int, page_size: int = 64) -> None:
        if any(type(v) is not int or v < 1 for v in (dim, page_size)):
            raise ValueError("row dimension and page_size must be positive integers")
        if type(capacity) is not int or capacity < 0:
            raise ValueError("row capacity must be a nonnegative integer")
        self.dim, self.capacity, self.page_size = dim, capacity, page_size
        self.length = 0
        self.pages: list[torch.Tensor] = []

    def append(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dim or self.length + len(values) > self.capacity:
            raise ValueError("cache append has the wrong shape or exceeds capacity")
        if self.pages and (values.dtype != self.pages[0].dtype or values.device != self.pages[0].device):
            raise ValueError("cache rows cannot change device or dtype")
        cursor = 0
        while cursor < len(values):
            page, offset = divmod(self.length, self.page_size)
            if page == len(self.pages):
                self.pages.append(values.new_empty((self.page_size, self.dim)))
            count = min(self.page_size - offset, len(values) - cursor)
            self.pages[page][offset : offset + count].copy_(values[cursor : cursor + count])
            self.length += count
            cursor += count

    def read(self, start: int, end: int) -> torch.Tensor:
        if not 0 <= start < end <= self.length:
            raise ValueError("cache read must name a nonempty retained interval")
        pieces = []
        while start < end:
            page, offset = divmod(start, self.page_size)
            count = min(end - start, self.page_size - offset)
            pieces.append(self.pages[page][offset : offset + count])
            start += count
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache gather requires a vector of integer row IDs")
        if not self.pages:
            raise ValueError("cannot gather from an empty cache")
        if bool(((indices < 0) | (indices >= self.length)).any()):
            raise ValueError("cache gather contains an uncommitted row")
        result = self.pages[0].new_empty((len(indices), self.dim))
        page_ids = indices // self.page_size
        for page in torch.unique(page_ids).tolist():
            select = page_ids == page
            result[select] = self.pages[page].index_select(0, (indices[select] % self.page_size).long())
        return result

    def truncate(self, length: int) -> None:
        if type(length) is not int or not 0 <= length <= self.length:
            raise ValueError("cache truncate cannot extend retained rows")
        self.length = length
        del self.pages[(length + self.page_size - 1) // self.page_size :]


@dataclass(frozen=True)
class AttentionSnapshot:
    owner: object
    position: int
    window: torch.Tensor | None
    partial_kv: torch.Tensor | None
    partial_score: torch.Tensor | None
    main_length: int
    index_length: int
    swa_length: int


class AttentionState:
    """One layer and request; snapshots keep old tails and append endpoints only."""

    def __init__(self, layer_id: int, max_seq_len: int, main: RowStore | None, index: RowStore | None,
                 swa: RowStore | None = None):
        self.layer_id, self.max_seq_len = layer_id, max_seq_len
        self.position = 0
        self.window: torch.Tensor | None = None
        self.partial_kv: torch.Tensor | None = None
        self.partial_score: torch.Tensor | None = None
        self.main, self.index, self.swa = main, index, swa
        self._owner = object()

    def snapshot(self) -> AttentionSnapshot:
        return AttentionSnapshot(
            self._owner, self.position, self.window, self.partial_kv, self.partial_score,
            self.main.length if self.main is not None else 0,
            self.index.length if self.index is not None else 0,
            self.swa.length if self.swa is not None else 0,
        )

    def restore(self, snapshot: AttentionSnapshot) -> None:
        if not isinstance(snapshot, AttentionSnapshot) or snapshot.owner is not self._owner:
            raise ValueError("attention snapshot belongs to a different request/layer")
        if snapshot.position > self.position:
            raise ValueError("attention snapshot cannot restore a discarded future")
        for rows, length in ((self.main, snapshot.main_length), (self.index, snapshot.index_length),
                             (self.swa, snapshot.swa_length)):
            if rows is not None:
                rows.truncate(length)
        self.position, self.window = snapshot.position, snapshot.window
        self.partial_kv, self.partial_score = snapshot.partial_kv, snapshot.partial_score


class SharedAttention:
    """Ephemeral publications for exactly one request's contiguous forward chunk."""

    def __init__(self, start_pos: int, end_pos: int) -> None:
        if type(start_pos) is not int or type(end_pos) is not int or not 0 <= start_pos < end_pos:
            raise ValueError("shared attention requires a nonempty contiguous interval")
        self.start_pos, self.end_pos = start_pos, end_pos
        self.sources: dict[int, AttentionState] = {}
        self.indexes: dict[int, torch.Tensor] = {}
        self.candidates: dict[int, tuple[torch.Tensor, ...]] = {}


def rotary_frequencies(config: DeepSeekV41Config, compressed: bool, device: torch.device) -> torch.Tensor:
    """Compute only the RoPE frequency vector, applying the pinned YaRN ramp."""
    text, dim = config.text_config, config.qk_rope_head_dim
    base = text["compress_rope_theta"] if compressed else text["rope_theta"]
    freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    if compressed:
        rope = text["rope_scaling"]
        original, factor = rope["original_max_position_embeddings"], rope["factor"]
        if original > 0:
            def corrected(rotations: float) -> float:
                return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

            low, high = max(math.floor(corrected(rope["beta_fast"])), 0), min(
                math.ceil(corrected(rope["beta_slow"])), dim - 1
            )
            ramp = ((torch.arange(dim // 2, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
            smooth = 1 - ramp
            freq = freq / factor * (1 - smooth) + freq * smooth
    return freq


def apply_rotary(x: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor,
                 *, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent real pairs without complex tensors, including inverse output RoPE."""
    dim = frequencies.numel() * 2
    if x.shape[0] != positions.numel() or x.shape[-1] < dim:
        raise ValueError("RoPE position/vector dimensions disagree")
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = angles.reshape(len(positions), *((1,) * (x.ndim - 2)), dim // 2)
    cosine, sine = angles.cos(), angles.sin()
    if inverse:
        sine = -sine
    tail = x[..., -dim:].float().reshape(*x.shape[:-1], dim // 2, 2)
    real, imaginary = tail[..., 0], tail[..., 1]
    rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), -1)
    return torch.cat((x[..., :-dim], rotated.flatten(-2).to(x.dtype)), -1)


def _topk_positions(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Select by score, breaking cutoff ties by the smallest absolute position.

    Reference torch.topk leaves ties unspecified. ReLU creates exact zero ties;
    its chosen positions can otherwise change when a prefill chunk adds masked
    future columns. Find the cutoff with topk, then select ties without score
    perturbations or sorting the full history. Return selected positions in order.
    """
    if scores.ndim != 1 or type(count) is not int or not 0 <= count <= len(scores):
        raise ValueError("top-k requires a score vector and a valid selection count")
    if not count:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if bool(torch.isnan(scores).any()):
        raise ValueError("top-k scores cannot contain NaN")
    cutoff = scores.topk(count, sorted=False).values.min()
    greater = (scores > cutoff).nonzero().flatten()
    tied = (scores == cutoff).nonzero().flatten()[:count - len(greater)]
    return torch.cat((greater, tied)).sort().values


def select_candidate_blocks(logits: torch.Tensor, reachable: int, topk_blocks: int,
                            block_size: int) -> torch.Tensor:
    """Return selected block IDs, including the newest reachable partial block."""
    if logits.ndim != 1 or not 0 <= reachable <= logits.numel() or topk_blocks < 1 or block_size < 1:
        raise ValueError("invalid hierarchical candidate dimensions or limits")
    width = logits.numel()
    if not width:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    scores = torch.nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.reshape(-1, block_size).amax(-1)
    if reachable:
        scores[(reachable - 1) // block_size] = torch.inf
    selected = _topk_positions(scores, min(topk_blocks, len(scores)))
    return selected[scores[selected] > -torch.inf]


def sparse_attention(q: torch.Tensor, values: torch.Tensor, sink: torch.Tensor,
                     valid: torch.Tensor, ops: AttentionOps, scale: float) -> torch.Tensor:
    """64-slot online softmax with FP32 accumulation and reference probability rounding.

    The sink contributes only to the denominator. Invalid slots never contribute
    values; an entirely empty row returns zero, as does the reference kernel.
    """
    if q.ndim != 2 or values.ndim != 2 or q.shape[1] != values.shape[1]:
        raise ValueError("sparse attention expects [heads,dim] and [slots,dim]")
    if sink.shape != (q.shape[0],) or valid.shape != (values.shape[0],) or valid.dtype != torch.bool:
        raise ValueError("sparse attention sink or slot mask shape mismatch")
    maximum = q.new_full((len(q),), -1e30, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    numerator = q.new_zeros(q.shape, dtype=torch.float32)
    for start in range(0, len(values), 64):
        mask = valid[start : start + 64]
        block = values[start : start + 64].masked_fill(~mask[:, None], 0)
        logits = ops.matmul(q.float(), block.float().T) * scale
        logits = logits.masked_fill(~mask[None, :], -torch.inf)
        updated = torch.maximum(maximum, logits.amax(-1))
        rescale = (maximum - updated).exp()
        probabilities = (logits - updated[:, None]).exp()
        denominator = denominator * rescale + probabilities.sum(-1)
        numerator = numerator * rescale[:, None] + ops.matmul(probabilities.to(q.dtype).float(), block.float())
        maximum = updated
    denominator = denominator + (sink.float() - maximum).exp()
    return (numerator / denominator[:, None]).to(q.dtype)


class Attention:
    """Full, Reindex and Reuse attention with explicit tensor operations."""

    def __init__(self, config: DeepSeekV41Config, ops: AttentionOps, layer_id: int,
                 *, index_key_tile: int = 1024) -> None:
        if type(layer_id) is not int or not 0 <= layer_id < config.num_hidden_layers:
            raise ValueError("attention layer must belong to the backbone")
        if type(index_key_tile) is not int or index_key_tile < 1:
            raise ValueError("index_key_tile must be positive")
        self.config, self.ops, self.layer_id = config, ops, layer_id
        self.plan = config.layer_plan[layer_id]
        self.prefix = f"layers.{layer_id}.attn"
        self.index_key_tile = index_key_tile

    def new_state(self, max_seq_len: int, *, cache_page_size: int = 64,
                  rows_factory: Callable[[str, int, int, str], RowStore] | None = None) -> AttentionState:
        if type(max_seq_len) is not int or not 0 < max_seq_len <= self.config.max_position_embeddings:
            raise ValueError("attention max_seq_len must fit the checkpoint")
        main = index = None
        if self.plan.owns_main_kv:
            capacity = max_seq_len // self.plan.compress_ratio
            if rows_factory is None:
                main = TensorRows(self.config.head_dim, capacity, cache_page_size)
                index = TensorRows(self.config.index_head_dim, capacity, cache_page_size)
            else:
                main = rows_factory("main_kv", self.layer_id, self.config.head_dim, FP4_MAIN)
                index = rows_factory("index_k", self.layer_id, self.config.index_head_dim, FP4_INDEX)
        swa = None if rows_factory is None else rows_factory("swa", self.layer_id, self.config.head_dim, FP8_SWA)
        return AttentionState(self.layer_id, max_seq_len, main, index, swa)

    def _linear(self, x: torch.Tensor, suffix: str) -> torch.Tensor:
        return self.ops.linear(x, f"{self.prefix}.{suffix}.weight")

    def _norm(self, x: torch.Tensor, suffix: str) -> torch.Tensor:
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.config.text_config["rms_norm_eps"])
        return (values * self.ops.weight(f"{self.prefix}.{suffix}.weight").float()).to(x.dtype)

    def _compress(self, x: torch.Tensor, state: AttentionState) -> torch.Tensor | None:
        ratio = self.plan.compress_ratio
        if ratio == 1:
            return self._norm(self._linear(x, "compressor.wkv"), "compressor.norm")
        values, scores = self._linear(x.float(), "compressor.wkv"), self._linear(x.float(), "compressor.wgate")
        if state.partial_kv is not None:
            values = torch.cat((state.partial_kv, values))
            scores = torch.cat((state.partial_score, scores))
        cutoff = len(values) // ratio * ratio
        state.partial_kv, state.partial_score = values[cutoff:].clone(), scores[cutoff:].clone()
        if not cutoff:
            return None
        values = values[:cutoff].reshape(-1, ratio, self.config.head_dim)
        scores = scores[:cutoff].reshape_as(values)
        latent = (values * scores.softmax(1)).sum(1).to(x.dtype)
        return self._norm(latent, "compressor.norm")

    def _index(self, x: torch.Tensor, qr: torch.Tensor, source: AttentionState,
               shared: SharedAttention, frequencies: torch.Tensor) -> torch.Tensor:
        cfg, ratio = self.config, self.plan.compress_ratio
        length = shared.end_pos // ratio
        topk = min(cfg.index_topk, length)
        if not length:
            if self.layer_id == cfg.candidate_source_layer_id:
                shared.candidates[self.layer_id] = tuple(
                    torch.empty(0, dtype=torch.long, device=x.device) for _ in range(len(x))
                )
            return torch.empty((len(x), 0), dtype=torch.int32, device=x.device)
        if source.index is None or source.index.length != length:
            raise ValueError("index-key source has not published this chunk")
        q = self._linear(qr, "indexer.wq_b").reshape(len(x), -1, cfg.index_head_dim)
        positions = torch.arange(shared.start_pos, shared.end_pos, device=x.device)
        q = self.ops.quantize(apply_rotary(q, positions, frequencies), FP4_INDEX, 32)
        weights = self._linear(x, "indexer.weights_proj") * (cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5)
        if q.shape[1] != weights.shape[1] or cfg.index_n_heads % q.shape[1]:
            raise ValueError("rank-local index heads disagree with the checkpoint")
        indexes, candidates = [], []
        # Share key reads/GEMMs and reduce a complete score block across ranks.
        # Keep score storage bounded for long histories and preserve per-query
        # masking, candidate selection and stable TopK after the reduction.
        query_tile = max(1, min(32, (8 << 20) // (length * 4)))
        positions = torch.arange(length, device=x.device)
        for query_begin in range(0, len(x), query_tile):
            query_end = min(len(x), query_begin + query_tile)
            queries = q[query_begin:query_end]
            pieces = []
            for begin in range(0, length, self.index_key_tile):
                keys = source.index.read(begin, min(length, begin + self.index_key_tile))
                score = self.ops.matmul(queries.flatten(0, 1), keys.T).to(q.dtype)
                score = score.reshape(len(queries), q.shape[1], len(keys))
                pieces.append((score.relu() * weights[query_begin:query_end, :, None]).sum(1))
            reduced = self.ops.all_reduce(torch.cat(pieces, dim=1))
            for offset, scores in enumerate(reduced):
                query = query_begin + offset
                reachable = (shared.start_pos + query + 1) // ratio
                scores = scores.masked_fill(positions >= reachable, -torch.inf)
                candidate_source = self.plan.candidate_source_layer_id
                if self.layer_id == cfg.candidate_source_layer_id:
                    candidates.append(select_candidate_blocks(scores, reachable, cfg.candidate_topk_blocks,
                                                               cfg.candidate_block_size))
                elif candidate_source is not None:
                    if candidate_source not in shared.candidates:
                        raise ValueError("hierarchical candidate source has not executed in this chunk")
                    blocks = shared.candidates[candidate_source][query]
                    allowed = torch.zeros((length + cfg.candidate_block_size - 1) // cfg.candidate_block_size,
                                          dtype=torch.bool, device=x.device)
                    allowed[blocks] = True
                    scores = scores.masked_fill(~allowed[positions // cfg.candidate_block_size], -torch.inf)
                chosen = _topk_positions(scores, topk)
                # The published reference filters causal positions here, not all -inf
                # candidate scores. Keep that distinction for underfilled candidate sets.
                indexes.append(torch.where(chosen < reachable, chosen, -1).to(torch.int32))
        if candidates:
            shared.candidates[self.layer_id] = tuple(candidates)
        return torch.stack(indexes)

    def forward(self, x: torch.Tensor, start_pos: int, state: AttentionState,
                shared: SharedAttention, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.config
        if x.ndim != 2 or x.shape[1] != cfg.hidden_size or not len(x) or not x.is_floating_point():
            raise ValueError("attention x must be a nonempty floating [tokens,hidden_size] tensor")
        if type(start_pos) is not int or start_pos != state.position or state.layer_id != self.layer_id:
            raise ValueError("attention state layer/contiguous position mismatch")
        end = start_pos + len(x)
        if end > state.max_seq_len or (shared.start_pos, shared.end_pos) != (start_pos, end):
            raise ValueError("attention chunk exceeds its state or shared interval")
        if token_mask is not None and (token_mask.shape != (len(x),) or token_mask.dtype != torch.bool):
            raise ValueError("attention token_mask must contain one bool per token")
        if state.window is not None and (state.window.device != x.device or state.window.dtype != x.dtype):
            raise ValueError("attention state cannot change input device/dtype")
        snapshot = state.snapshot()
        publications = dict(shared.sources), dict(shared.indexes), dict(shared.candidates)
        try:
            result = self._forward(x, start_pos, state, shared)
        except BaseException:
            state.restore(snapshot)
            shared.sources, shared.indexes, shared.candidates = publications
            raise
        return result

    def _forward(self, x: torch.Tensor, start_pos: int, state: AttentionState,
                 shared: SharedAttention) -> torch.Tensor:
        cfg, ratio = self.config, self.plan.compress_ratio
        end = start_pos + len(x)
        frequencies = rotary_frequencies(cfg, bool(ratio), x.device)
        positions = torch.arange(start_pos, end, device=x.device)
        qr = self._norm(self._linear(x, "wq_a"), "q_norm")
        q = self._linear(qr, "wq_b").reshape(len(x), -1, cfg.head_dim)
        q = apply_rotary(q, positions, frequencies)
        kv = self._norm(self._linear(x, "wkv"), "kv_norm")
        kv = self.ops.quantize(apply_rotary(kv, positions, frequencies), FP8_SWA, 32)
        source, indexes = None, None
        if ratio:
            if self.plan.owns_main_kv:
                source = state
                previous = state.main.length
                latent = self._compress(x, state)
                if latent is not None:
                    latent_positions = torch.arange(previous, previous + len(latent), device=x.device) * ratio
                    keys = self._norm(self._linear(latent, "indexer.wk"), "indexer.k_norm")
                    keys = self.ops.quantize(apply_rotary(keys, latent_positions, frequencies), FP4_INDEX, 32)
                    state.index.append(keys)
                    values = self.ops.quantize(apply_rotary(latent, latent_positions, frequencies), FP4_MAIN, 16)
                    state.main.append(values)
            else:
                source = shared.sources.get(self.plan.kv_source_layer_id)
                if source is None or source.position != end:
                    raise ValueError("compressed KV source has not executed in this chunk")
            if self.plan.owns_index_results:
                indexes = self._index(x, qr, source, shared, frequencies)
                shared.indexes[self.layer_id] = indexes
            else:
                indexes = shared.indexes.get(self.plan.index_source_layer_id)
                if indexes is None or indexes.shape[0] != len(x):
                    raise ValueError("index-result source has not executed in this chunk")
        if state.swa is not None:
            state.swa.append(kv)
            context_start = max(0, start_pos - cfg.sliding_window)
            context = state.swa.read(context_start, end)
        else:
            previous_window = state.window
            context = kv if previous_window is None else torch.cat((previous_window, kv))
            context_start = start_pos - (0 if previous_window is None else len(previous_window))
        window_width = min(end, cfg.sliding_window)
        sink = self.ops.weight(f"{self.prefix}.attn_sink")
        outputs = []
        for query in range(len(x)):
            position = start_pos + query
            begin = max(0, position + 1 - cfg.sliding_window) - context_start
            window = context[begin : position + 1 - context_start]
            values = x.new_zeros((window_width + (0 if indexes is None else indexes.shape[1]), cfg.head_dim))
            valid = torch.zeros(len(values), device=x.device, dtype=torch.bool)
            values[:len(window)], valid[:len(window)] = window, True
            if indexes is not None and indexes.shape[1]:
                selected = indexes[query]
                mask = selected >= 0
                if bool(mask.any()):
                    values[window_width:][mask] = source.main.gather(selected[mask])
                    valid[window_width:] = mask
            outputs.append(sparse_attention(q[query], values, sink, valid, self.ops, cfg.head_dim**-0.5))
        result = self._project(torch.stack(outputs), positions, frequencies)
        state.window = None if state.swa is not None else context[-cfg.sliding_window:].clone()
        state.position = end
        if self.plan.owns_main_kv:
            shared.sources[self.layer_id] = state
        return result

    def _project(self, output: torch.Tensor, positions: torch.Tensor,
                 frequencies: torch.Tensor) -> torch.Tensor:
        cfg, dtype = self.config, output.dtype
        output = apply_rotary(output, positions, frequencies, inverse=True)
        wo_a = self.ops.weight(f"{self.prefix}.wo_a.weight")
        groups = wo_a.numel() // (cfg.o_lora_rank * (cfg.num_attention_heads * cfg.head_dim // cfg.o_groups))
        if groups < 1 or cfg.o_groups % groups or output.shape[1] * cfg.o_groups != groups * cfg.num_attention_heads:
            raise ValueError("rank-local output groups and attention heads disagree")
        grouped = output.reshape(len(output), groups, -1).transpose(0, 1)
        projected = self.ops.matmul(grouped, wo_a.reshape(groups, cfg.o_lora_rank, -1).transpose(1, 2)).to(dtype)
        result = self._linear(projected.transpose(0, 1).reshape(len(output), -1), "wo_b")
        return self.ops.all_reduce(result.float()).to(dtype)


class DraftAttention(Attention):
    """DSpark's main-token ring and temporary, bidirectional draft block.

    Only main_x advances request state. A draft block sees the complete block;
    its KV is discarded after projection and never becomes committed main KV.
    """

    def __init__(self, config: DeepSeekV41Config, ops: AttentionOps, draft_id: int) -> None:
        if type(draft_id) is not int or not 0 <= draft_id < config.num_nextn_predict_layers:
            raise ValueError("draft attention index must identify a configured DSpark layer")
        self.config, self.ops = config, ops
        self.layer_id = config.num_hidden_layers + draft_id
        if config.compress_ratios[self.layer_id] != 0:
            raise ValueError("draft attention requires an uncompressed sliding window")
        self.prefix = f"mtp.{draft_id}.attn"
        self.plan = replace(
            config.layer_plan[0], layer_id=self.layer_id, compress_ratio=0,
            kv_source_layer_id=None, index_source_layer_id=None, candidate_source_layer_id=None,
            owns_main_kv=False, owns_index_keys=False, owns_index_results=False,
        )

    def new_state(self, max_seq_len: int) -> AttentionState:
        # Draft rings are private and bounded; backbone scheduler groups never alias them.
        return super().new_state(max_seq_len)

    def seed_main(self, main_x: torch.Tensor, start_pos: int, state: AttentionState) -> None:
        """Append target hidden states without evaluating a speculative draft block."""
        cfg = self.config
        if main_x.ndim != 2 or main_x.shape[1] != cfg.hidden_size or not len(main_x) or not main_x.is_floating_point():
            raise ValueError("draft main hidden states must be floating [tokens,hidden_size]")
        if type(start_pos) is not int or start_pos != state.position or state.layer_id != self.layer_id:
            raise ValueError("draft seed state layer/position mismatch")
        end = start_pos + len(main_x)
        if end > state.max_seq_len:
            raise ValueError("draft main seed exceeds state max_seq_len")
        if state.window is not None and (state.window.dtype != main_x.dtype or state.window.device != main_x.device):
            raise ValueError("draft main state cannot change device/dtype")
        snapshot = state.snapshot()
        try:
            frequencies = rotary_frequencies(cfg, False, main_x.device)
            positions = torch.arange(start_pos, end, device=main_x.device)
            main_kv = self._norm(self._linear(main_x, "wkv"), "kv_norm")
            main_kv = self.ops.quantize(apply_rotary(main_kv, positions, frequencies), FP8_SWA, 32)
            window = main_kv if state.window is None else torch.cat((state.window, main_kv))
            state.window, state.position = window[-cfg.sliding_window:].clone(), end
        except BaseException:
            state.restore(snapshot)
            raise

    def forward(self, x: torch.Tensor, main_x: torch.Tensor, start_pos: int,
                state: AttentionState) -> torch.Tensor:
        cfg = self.config
        if any(value.ndim != 2 or value.shape[1] != cfg.hidden_size or not len(value)
               or not value.is_floating_point() for value in (x, main_x)):
            raise ValueError("draft and main hidden states must be floating [tokens,hidden_size]")
        if x.dtype != main_x.dtype or x.device != main_x.device:
            raise ValueError("draft and main hidden states must share dtype/device")
        if type(start_pos) is not int or start_pos != state.position or state.layer_id != self.layer_id:
            raise ValueError("draft attention state layer/position mismatch")
        end = start_pos + len(main_x)
        if end > state.max_seq_len or (start_pos and (len(main_x) != 1 or end + len(x) > state.max_seq_len)):
            raise ValueError("draft decode needs one main token and space for the complete draft block")
        snapshot = state.snapshot()
        try:
            self.seed_main(main_x, start_pos, state)
            return x if start_pos == 0 else self.propose(x, state)
        except BaseException:
            state.restore(snapshot)
            raise

    def propose(self, x: torch.Tensor, state: AttentionState) -> torch.Tensor:
        """Read the already seeded main window; leave all committed state unchanged."""
        cfg, position = self.config, state.position
        if x.ndim != 2 or x.shape[1] != cfg.hidden_size or not len(x) or not x.is_floating_point():
            raise ValueError("draft hidden states must be floating [tokens,hidden_size]")
        if state.layer_id != self.layer_id or position < 1 or state.window is None:
            raise ValueError("draft proposal requires this layer's seeded main state")
        if position + len(x) > state.max_seq_len:
            raise ValueError("draft proposal exceeds state max_seq_len")
        if x.dtype != state.window.dtype or x.device != state.window.device:
            raise ValueError("draft proposal must share the main state's device/dtype")
        frequencies = rotary_frequencies(cfg, False, x.device)
        positions = torch.arange(position, position + len(x), device=x.device)
        qr = self._norm(self._linear(x, "wq_a"), "q_norm")
        q = self._linear(qr, "wq_b").reshape(len(x), -1, cfg.head_dim)
        q = apply_rotary(q, positions, frequencies)
        kv = self._norm(self._linear(x, "wkv"), "kv_norm")
        kv = self.ops.quantize(apply_rotary(kv, positions, frequencies), FP8_SWA, 32)
        # The reference enumerates physical ring slots before the bidirectional block.
        window = state.window
        ring = torch.roll(window, shifts=position % cfg.sliding_window, dims=0) if (
            len(window) == cfg.sliding_window
        ) else window
        values = torch.cat((ring, kv))
        valid = torch.ones(len(values), device=x.device, dtype=torch.bool)
        sink = self.ops.weight(f"{self.prefix}.attn_sink")
        outputs = torch.stack([
            sparse_attention(query, values, sink, valid, self.ops, cfg.head_dim**-0.5) for query in q
        ])
        return self._project(outputs, positions, frequencies)
