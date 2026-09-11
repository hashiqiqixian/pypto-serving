# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy DSpark prefix verification and bounded host state transactions.

The pinned V4.1 reference dba1be0a40aa45a94ad051997016db3960a90277, model.py:1137-1156,
returns an anchor followed by draft tokens and raw linear confidence scores. It
does not implement an acceptance scheduler. The operator may supply a raw-score
threshold; no calibrated probability or performance claim is inferred from it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol


def _tokens(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return result


def choose_verification_count(confidence_scores: Sequence[float], capacity: int,
                              minimum_score: float | None = None) -> int:
    """Choose a bounded consecutive prefix using an optional explicit raw-score floor.

    With no floor, verify the reserved capacity. With a floor, stop at the first
    failing confidence, including zero candidates if the first score fails.
    Threshold selection is an operator policy, not a checkpoint probability.
    """
    scores = tuple(confidence_scores)
    if type(capacity) is not int or not 0 <= capacity <= len(scores):
        raise ValueError("verification capacity must fit the generated confidence vector")
    if any(isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) for score in scores):
        raise ValueError("confidence scores must be finite real numbers")
    if minimum_score is None:
        return capacity
    if isinstance(minimum_score, bool) or not isinstance(minimum_score, Real) or not math.isfinite(minimum_score):
        raise ValueError("confidence threshold must be a finite raw score")
    for index, score in enumerate(scores[:capacity]):
        if score < minimum_score:
            return index
    return capacity


@dataclass(frozen=True)
class DraftProposal:
    anchor_token: int
    tokens: tuple[int, ...]
    confidence_scores: tuple[float, ...]
    generated_count: int


def select_draft_prefix(
    output_ids: Sequence[int],
    confidence_scores: Sequence[float],
    *,
    verify_count: int,
) -> DraftProposal:
    """Validate the reference output and use an explicitly supplied scheduling decision.

    Confidence scores are finite real logits, not calibrated probabilities. A
    confidence-based policy must supply its own verified rule and verify_count.
    """
    output = _tokens(output_ids, "draft output")
    scores = tuple(confidence_scores)
    if len(output) < 2 or len(scores) != len(output) - 1:
        raise ValueError("draft output must contain anchor + K tokens and K confidence scores")
    if any(
        not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score) for score in scores
    ):
        raise ValueError("confidence scores must be finite real numbers")
    if type(verify_count) is not int or not 1 <= verify_count <= len(scores):
        raise ValueError("verify_count must explicitly select between 1 and K draft tokens")
    return DraftProposal(
        output[0], output[1 : verify_count + 1], tuple(float(s) for s in scores[:verify_count]), len(scores)
    )


@dataclass(frozen=True)
class GreedyVerification:
    emitted_tokens: tuple[int, ...]
    accepted_draft_count: int
    compared_draft_count: int
    proposed_count: int
    pending_target_token: int | None
    finish_reason: str | None

    @property
    def discarded_draft_count(self) -> int:
        return self.proposed_count - self.accepted_draft_count


def verify_greedy(
    draft_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    stop_token_ids: Sequence[int] = (),
    max_new_tokens: int | None = None,
) -> GreedyVerification:
    """Accept a matching prefix and emit the first correction, or an all-match bonus.

    target_tokens[i] is the target greedy prediction after the already committed
    prefix and drafts[:i]. Thus K draft tokens require K+1 target predictions.
    The returned pending_target_token has been emitted but is not in the cache;
    the next model forward must consume it before the next speculative round.
    """
    drafts, target = _tokens(draft_tokens, "draft tokens"), _tokens(target_tokens, "target tokens")
    stops = set(_tokens(stop_token_ids, "stop token IDs"))
    if not drafts or len(target) != len(drafts) + 1:
        raise ValueError("greedy verification needs K>0 drafts and K+1 aligned target predictions")
    limit = len(drafts) + 1 if max_new_tokens is None else max_new_tokens
    if type(limit) is not int or limit < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    emitted, accepted, compared, pending, finish = [], 0, 0, None, None
    if limit == 0:
        return GreedyVerification((), 0, 0, len(drafts), None, "length")
    for index in range(len(drafts) + 1):
        is_draft = index < len(drafts)
        matched = is_draft and drafts[index] == target[index]
        compared += int(is_draft)
        if matched:
            emitted.append(drafts[index])
            accepted += 1
        else:
            emitted.append(target[index])
            pending = target[index]
        if emitted[-1] in stops:
            finish = "stop"
        elif max_new_tokens is not None and len(emitted) >= limit:
            finish = "length"
        if finish or not matched:
            break
    return GreedyVerification(tuple(emitted), accepted, compared, len(drafts), pending, finish)


class RollbackState(Protocol):
    """A request-owned state at an absolute cached-token position."""

    position: int

    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def rollback(self, position: int) -> None: ...


class DSparkTransaction:
    """Coordinate one pending verification over cache, Engram, and Markov states.

    base_position includes the already cached anchor. The caller advances every
    participant by the selected draft tokens before commit. Commit retains only
    accepted drafts; correction/bonus remains pending. A failed validation or
    rollback restores every participant's original snapshot. Instances represent
    one request round and cannot be reused after commit/abort.
    """

    def __init__(
        self, proposal: DraftProposal, states: Mapping[str, RollbackState], *, max_draft_tokens: int
    ):
        if type(max_draft_tokens) is not int or max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be a positive integer")
        if not isinstance(proposal, DraftProposal) or not 1 <= len(proposal.tokens) <= max_draft_tokens:
            raise ValueError("proposal exceeds the bounded draft window")
        _tokens(proposal.tokens, "proposal tokens")
        if not states or len({id(state) for state in states.values()}) != len(states):
            raise ValueError("transaction needs distinct request-owned state participants")
        self.proposal, self._states = proposal, dict(states)
        positions = {state.position for state in self._states.values()}
        if len(positions) != 1 or any(type(position) is not int or position < 0 for position in positions):
            raise ValueError("all participants must start at the same nonnegative cached position")
        self.base_position = positions.pop()
        self._snapshots = {name: state.snapshot() for name, state in self._states.items()}
        self.status = "pending"
        self.result: GreedyVerification | None = None

    def _restore(self) -> None:
        failures = []
        for name, state in self._states.items():
            try:
                state.restore(self._snapshots[name])
                if state.position != self.base_position:
                    raise ValueError("restored position differs from transaction base")
            except BaseException as error:
                failures.append((name, error))
        self.status = "failed" if failures else "aborted"
        if failures:
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(f"DSpark snapshot restoration failed for: {names}") from failures[0][1]

    def abort(self) -> None:
        if self.status != "pending":
            raise ValueError("DSpark transaction is already closed")
        self._restore()

    def commit(
        self,
        target_tokens: Sequence[int],
        *,
        stop_token_ids: Sequence[int] = (),
        max_new_tokens: int | None = None,
    ) -> GreedyVerification:
        if self.status != "pending":
            raise ValueError("DSpark transaction is already closed")
        try:
            result = verify_greedy(
                self.proposal.tokens,
                target_tokens,
                stop_token_ids=stop_token_ids,
                max_new_tokens=max_new_tokens,
            )
            expected = self.base_position + len(self.proposal.tokens)
            if any(state.position != expected for state in self._states.values()):
                raise ValueError("participants must contain exactly the selected speculative prefix")
            retained = self.base_position + result.accepted_draft_count
            for state in self._states.values():
                state.rollback(retained)
                if state.position != retained:
                    raise ValueError("participant rollback did not restore the accepted position")
        except BaseException:
            self._restore()
            raise
        self.result, self.status = result, "committed"
        return result
