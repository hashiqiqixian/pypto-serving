# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical DSpark stages, autoregressive Markov corrections, and raw confidence scores.

Follows inference/model.py DSparkBlock/Transformer.forward_spec at DeepSeek-V4.1-Flash
revision dba1be0a40aa45a94ad051997016db3960a90277; upstream notice: encoding.LICENSE.
The caller supplies the target layers' attention-input means, concatenated in
dspark_target_layer_ids order. Image target positions may seed the draft windows,
but drafts themselves use the text router. This module does not invent a confidence
threshold or choose how many candidates to verify; dspark.py owns that host policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from .attention import DraftAttention
from .numerics import ModelMath, rms_norm


@dataclass(frozen=True)
class DraftOutput:
    output_ids: torch.Tensor
    logits: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class DraftSnapshot:
    owner: object
    attention: tuple
    rng_state: torch.Tensor
    history: tuple


class DraftState:
    """Per-request window state with bounded rollback boundaries and independent sampling RNG."""

    def __init__(self, model, attention, rollback_window, generator):
        self.model, self.attention = model, tuple(attention)
        self.rollback_window, self.generator = rollback_window, generator
        self._owner = object()
        self._history = deque(maxlen=rollback_window + 1)
        self._record_boundary()

    @property
    def position(self):
        positions = {state.position for state in self.attention}
        if len(positions) != 1:
            raise RuntimeError("DSpark stage cache positions disagree")
        return positions.pop()

    def _record_boundary(self, *, prefill=False):
        if prefill:
            self._history.clear()
        if self._history and self._history[-1][0] == self.position:
            self._history.pop()
        self._history.append(
            (
                self.position,
                tuple(state.snapshot() for state in self.attention),
                self.generator.get_state().clone(),
            )
        )

    def snapshot(self) -> DraftSnapshot:
        return DraftSnapshot(
            self._owner,
            tuple(state.snapshot() for state in self.attention),
            self.generator.get_state().clone(),
            tuple(self._history),
        )

    def restore(self, snapshot: DraftSnapshot) -> None:
        if not isinstance(snapshot, DraftSnapshot) or snapshot.owner is not self._owner:
            raise ValueError("draft snapshot belongs to a different request")
        for state, saved in zip(self.attention, snapshot.attention):
            state.restore(saved)
        self.generator.set_state(snapshot.rng_state)
        self._history = deque(snapshot.history, maxlen=self.rollback_window + 1)

    def rollback(self, position: int) -> None:
        if type(position) is not int or position < 0 or position > self.position:
            raise ValueError("draft rollback position must be a retained earlier boundary")
        if position == self.position:
            return
        boundary = next((item for item in self._history if item[0] == position), None)
        if boundary is None or self.position - position > self.rollback_window:
            raise ValueError("draft rollback exceeds the bounded pending window or crosses a prefill chunk")
        _, attention, rng_state = boundary
        for state, saved in zip(self.attention, attention):
            state.restore(saved)
        self.generator.set_state(rng_state)
        self._history = deque(
            (item for item in self._history if item[0] <= position), maxlen=self.rollback_window + 1
        )


class DSparkDrafter:
    """One-request-at-a-time tensor graph; all mutable state is in the supplied DraftState."""

    def __init__(self, config, ops, *, math=None):
        self.config, self.ops = config, ops
        self.text = config.text_config
        self.stages = int(self.text["num_nextn_predict_layers"])
        self.block_size = int(self.text["dspark_block_size"])
        self.noise_token = int(self.text["dspark_noise_token_id"])
        self.targets = tuple(self.text["dspark_target_layer_ids"])
        if self.stages <= 0 or self.block_size <= 0 or not self.targets:
            raise ValueError("DSpark requires positive stage/block counts and target hidden states")
        if not 0 <= self.noise_token < int(self.text["vocab_size"]):
            raise ValueError("DSpark noise token must belong to the tokenizer vocabulary")
        self.math = ModelMath(config, ops) if math is None else math
        self.attention = tuple(DraftAttention(config, ops, stage) for stage in range(self.stages))

    def new_state(self, max_seq_len: int, *, rollback_window: int | None = None, seed: int = 0) -> DraftState:
        if rollback_window is None:
            rollback_window = self.block_size + 1
        if type(rollback_window) is not int or rollback_window < self.block_size:
            raise ValueError("draft rollback_window must cover at least one draft block")
        if type(seed) is not int:
            raise ValueError("draft sampling seed must be an integer")
        generator = torch.Generator(device=self.ops.device).manual_seed(seed)
        return DraftState(
            self, [layer.new_state(max_seq_len) for layer in self.attention], rollback_window, generator
        )

    @staticmethod
    def _sample(logits, temperature, generator):
        if temperature == 0:
            return logits.argmax(-1)
        probabilities = (logits / max(temperature, 1e-5)).softmax(-1, dtype=torch.float32)
        noise = torch.empty_like(probabilities).exponential_(1, generator=generator)
        return (probabilities / noise).argmax(-1)

    def forward_head(
        self,
        hidden: torch.Tensor,
        anchor_token: int,
        *,
        temperature: float = 0,
        generator: torch.Generator | None = None,
    ) -> DraftOutput:
        """Project collapsed HC hidden states; the i-th Markov bias uses the previous emitted ID."""
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("DSpark temperature must be finite and nonnegative")
        if temperature > 0 and generator is None:
            raise ValueError("stochastic draft sampling requires a request-owned generator")
        if hidden.shape != (self.block_size, int(self.text["hidden_size"])):
            raise ValueError("draft head requires [block_size, hidden_size] collapsed states")
        if type(anchor_token) is not int or not 0 <= anchor_token < int(self.text["vocab_size"]):
            raise ValueError("draft anchor must belong to the tokenizer vocabulary")
        prefix = f"mtp.{self.stages - 1}"
        normalized = rms_norm(hidden, self.ops.weight(prefix + ".norm.weight"), self.math.eps)
        logits = self.ops.all_gather(self.ops.linear(normalized.float(), "head"))
        if logits.shape != (self.block_size, int(self.text["vocab_size"])):
            raise ValueError(
                "draft head collective must concatenate the complete vocabulary on the last axis"
            )
        logits = logits.clone()
        output = torch.empty(self.block_size + 1, device=hidden.device, dtype=torch.int64)
        output[0] = anchor_token
        markov_embeds = []
        for step in range(self.block_size):
            embedding = self.ops.embedding(prefix + ".markov_head.embed.weight", output[step : step + 1])[0]
            bias = self.ops.all_gather(self.ops.linear(embedding.float(), prefix + ".markov_head.head"))
            if bias.shape != logits[step].shape:
                raise ValueError("Markov head collective must produce the complete vocabulary")
            logits[step] += bias
            markov_embeds.append(embedding)
            output[step + 1] = self._sample(logits[step], temperature, generator)
        markov = torch.stack(markov_embeds)
        confidence = self.ops.linear(
            torch.cat((hidden, markov), dim=-1).float(), prefix + ".confidence_head.proj"
        ).squeeze(-1)
        return DraftOutput(output, logits, confidence)

    @torch.inference_mode()
    def seed(self, main_hidden: torch.Tensor, start_pos: int, state: DraftState) -> None:
        """Append committed target states without running draft blocks; supports prefill chunks.

        Only chunk boundaries can be rolled back. Seed accepted interior target
        tokens individually when the caller needs rollback at each such token.
        """
        if not isinstance(state, DraftState) or state.model is not self:
            raise ValueError("draft state belongs to another model/request")
        if type(start_pos) is not int or start_pos != state.position:
            raise ValueError("DSpark seed must continue the committed main-token position")
        dim = int(self.text["hidden_size"])
        if main_hidden.ndim != 2 or main_hidden.shape[-1] != len(self.targets) * dim or len(main_hidden) < 1:
            raise ValueError(
                "main_hidden must concatenate the configured target-layer inputs in [tokens, targets*D]"
            )
        before = state.snapshot()
        try:
            main_x = rms_norm(
                self.ops.linear(main_hidden, "mtp.0.main_proj"),
                self.ops.weight("mtp.0.main_norm.weight"),
                self.math.eps,
            )
            for attention, layer_state in zip(self.attention, state.attention):
                attention.seed_main(main_x, start_pos, layer_state)
            state._record_boundary(prefill=start_pos == 0)
        except BaseException:
            state.restore(before)
            raise

    @torch.inference_mode()
    def propose(self, anchor_token: int, state: DraftState, *, temperature: float = 0) -> DraftOutput:
        """Evaluate the draft block against already seeded target windows; do not append any KV."""
        if not isinstance(state, DraftState) or state.model is not self:
            raise ValueError("draft state belongs to another model/request")
        if state.position <= 0:
            raise ValueError("draft proposal requires seeded target hidden states")
        if type(anchor_token) is not int or not 0 <= anchor_token < int(self.text["vocab_size"]):
            raise ValueError("draft anchor must belong to the tokenizer vocabulary")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("DSpark temperature must be finite and nonnegative")
        before = state.snapshot()
        try:
            hc = int(self.text["hc_mult"])
            ids = torch.full((self.block_size,), self.noise_token, device=self.ops.device, dtype=torch.int64)
            ids[0] = anchor_token
            h = self.ops.embedding("embed.weight", ids).unsqueeze(1).repeat(1, hc, 1)
            pre_mix = h.new_zeros((self.block_size, hc), dtype=torch.float32)
            pre_mix[:, 0] = 1
            for stage, attention in enumerate(self.attention):
                layer_state = state.attention[stage]
                h, pre_mix = self.math.block(
                    h, pre_mix, f"mtp.{stage}", lambda x: attention.propose(x, layer_state), draft=True
                )
            output = self.forward_head(
                self.math.hc_pre(h, pre_mix), anchor_token, temperature=temperature, generator=state.generator
            )
            state._record_boundary()
            return output
        except BaseException:
            state.restore(before)
            raise

    @torch.inference_mode()
    def forward(
        self,
        anchor_token: int,
        main_hidden: torch.Tensor,
        start_pos: int,
        state: DraftState,
        *,
        temperature: float = 0,
    ) -> DraftOutput | None:
        """Reference-compatible seed plus proposal; start_pos zero performs prefill only."""
        if not isinstance(state, DraftState) or state.model is not self:
            raise ValueError("draft state belongs to another model/request")
        if start_pos and len(main_hidden) != 1:
            raise ValueError("DSpark decode consumes one committed main token at a time")
        before = state.snapshot()
        try:
            self.seed(main_hidden, start_pos, state)
            return None if start_pos == 0 else self.propose(anchor_token, state, temperature=temperature)
        except BaseException:
            state.restore(before)
            raise
