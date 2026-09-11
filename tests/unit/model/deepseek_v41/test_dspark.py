# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark host decisions and transactional state tests; no draft model or device execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"v41_{name}_transaction_test", ROOT / f"pypto_serving/model/deepseek_v41/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dspark, engram = load("dspark"), load("engram")


@pytest.mark.parametrize("scores,capacity,floor,count", [
    ([3., -2., 9.], 3, None, 3), ([3., -2., 9.], 3, 0., 1),
    ([-1., 9.], 2, 0., 0), ([1., 1.], 1, 1., 1), ([], 0, None, 0),
])
def test_explicit_confidence_scheduling(scores, capacity, floor, count):
    assert dspark.choose_verification_count(scores, capacity, floor) == count


@pytest.mark.parametrize("scores,capacity,floor", [([float("nan")], 1, None), ([1.], 2, None),
                                                ([1.], 1, float("inf")), ([True], 1, None)])
def test_confidence_scheduling_rejects_invalid_measurements(scores, capacity, floor):
    with pytest.raises(ValueError):
        dspark.choose_verification_count(scores, capacity, floor)


@pytest.mark.parametrize(
    "target,emitted,accepted,pending,compared",
    [
        ([10, 11, 12, 13], (10, 11, 12, 13), 3, 13, 3),
        ([10, 99, 12, 13], (10, 99), 1, 99, 2),
        ([99, 11, 12, 13], (99,), 0, 99, 1),
    ],
)
def test_greedy_prefix_and_correction_or_bonus(target, emitted, accepted, pending, compared):
    result = dspark.verify_greedy([10, 11, 12], target)
    assert result.emitted_tokens == emitted
    assert result.accepted_draft_count == accepted
    assert result.pending_target_token == pending
    assert result.compared_draft_count == compared
    assert result.discarded_draft_count == 3 - accepted
    assert result.finish_reason is None


def test_stop_token_inside_matching_prefix_does_not_emit_a_bonus():
    result = dspark.verify_greedy([10, 11, 12], [10, 11, 12, 13], stop_token_ids=[11])
    assert result.emitted_tokens == (10, 11)
    assert result.accepted_draft_count == 2
    assert result.pending_target_token is None
    assert result.finish_reason == "stop"


def test_stop_correction_is_emitted_but_remains_uncached():
    result = dspark.verify_greedy([10, 11], [10, 99, 100], stop_token_ids=[99])
    assert result.emitted_tokens == (10, 99)
    assert result.accepted_draft_count == 1
    assert result.pending_target_token == 99
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("limit,emitted,accepted", [(0, (), 0), (1, (10,), 1), (2, (10, 11), 2)])
def test_output_budget_limits_committed_drafts(limit, emitted, accepted):
    result = dspark.verify_greedy([10, 11], [10, 11, 12], max_new_tokens=limit)
    assert result.emitted_tokens == emitted
    assert result.accepted_draft_count == accepted
    assert result.pending_target_token is None
    assert result.finish_reason == "length"


@pytest.mark.parametrize(
    "draft,target,limit", [([], [1], None), ([1], [1], None), ([1], [1, 2], -1), ([True], [1, 2], None)]
)
def test_malformed_greedy_contract_rejected(draft, target, limit):
    with pytest.raises(ValueError):
        dspark.verify_greedy(draft, target, max_new_tokens=limit)


def test_confidence_is_raw_score_and_scheduling_is_explicit():
    proposal = dspark.select_draft_prefix([7, 10, 11, 12], [-100, 0, 100], verify_count=2)
    assert proposal.anchor_token == 7
    assert proposal.tokens == (10, 11)
    assert proposal.confidence_scores == (-100, 0)
    assert proposal.generated_count == 3
    with pytest.raises(TypeError):
        dspark.select_draft_prefix([7, 10], [0.5])
    for scores, count in (([float("nan")], 1), ([float("inf")], 1), ([0.5], 0), ([0.5], 2), ([], 1)):
        with pytest.raises(ValueError):
            dspark.select_draft_prefix([7, 10], scores, verify_count=count)


class MemoryState:
    """Only a test double for an external cache/Markov participant."""

    def __init__(self, tokens, *, fail_rollback=False, fail_restore=False):
        self.tokens = list(tokens)
        self.fail_rollback, self.fail_restore = fail_rollback, fail_restore
        self.restores = 0

    @property
    def position(self):
        return len(self.tokens)

    def snapshot(self):
        return tuple(self.tokens)

    def restore(self, snapshot):
        self.restores += 1
        if self.fail_restore:
            raise ValueError("injected restore failure")
        self.tokens[:] = snapshot

    def rollback(self, position):
        if self.fail_rollback:
            raise ValueError("injected rollback failure")
        del self.tokens[position:]


def proposal():
    return dspark.select_draft_prefix([1, 2, 3, 0], [-1, 0, 1], verify_count=3)


def test_real_engram_participant_rolls_back_with_cache_and_markov():
    layout = engram.EngramLayout(3, (1,), (12,), (((5,), (7,)),), 1, 2)
    hashes = engram.EngramHashState(
        layout,
        (0, 1, 2, 3),
        compressed_vocab_size=4,
        pad_token_id=0,
        multipliers=((3, 5, 7),),
        rollback_window=3,
    )
    hashes.advance([0, 1], start_pos=0)
    cache, markov = MemoryState([0, 1]), MemoryState([0, 1])
    transaction = dspark.DSparkTransaction(
        proposal(), {"engram": hashes, "cache": cache, "markov": markov}, max_draft_tokens=5
    )
    hashes.advance([2, 3, 0], start_pos=2)
    cache.tokens.extend([2, 3, 0])
    markov.tokens.extend([2, 3, 0])
    result = transaction.commit([2, 0, 1, 2])
    assert result.emitted_tokens == (2, 0)
    assert result.pending_target_token == 0
    assert hashes.position == cache.position == markov.position == 3
    assert cache.tokens == markov.tokens == [0, 1, 2]
    assert transaction.status == "committed"
    with pytest.raises(ValueError, match="closed"):
        transaction.abort()


def test_partial_rollback_failure_restores_all_participants():
    first, second = MemoryState([0, 1]), MemoryState([0, 1], fail_rollback=True)
    transaction = dspark.DSparkTransaction(proposal(), {"first": first, "second": second}, max_draft_tokens=3)
    first.tokens.extend([2, 3, 0])
    second.tokens.extend([2, 3, 0])
    with pytest.raises(ValueError, match="injected rollback"):
        transaction.commit([2, 0, 1, 2])
    assert first.tokens == second.tokens == [0, 1]
    assert first.restores == second.restores == 1
    assert transaction.status == "aborted"


def test_interruption_during_commit_restores_every_participant():
    class InterruptedState(MemoryState):
        def rollback(self, position):
            raise KeyboardInterrupt("injected interruption")

    first, interrupted = MemoryState([0, 1]), InterruptedState([0, 1])
    transaction = dspark.DSparkTransaction(
        proposal(), {"first": first, "interrupted": interrupted}, max_draft_tokens=3
    )
    first.tokens.extend([2, 3, 0])
    interrupted.tokens.extend([2, 3, 0])
    with pytest.raises(KeyboardInterrupt):
        transaction.commit([2, 0, 1, 2])
    assert first.tokens == interrupted.tokens == [0, 1]
    assert transaction.status == "aborted"


def test_failure_before_verify_and_bad_target_shape_restore_snapshots():
    for malformed in (False, True):
        participant = MemoryState([0, 1])
        transaction = dspark.DSparkTransaction(proposal(), {"cache": participant}, max_draft_tokens=3)
        participant.tokens.extend([2])
        with pytest.raises(ValueError):
            transaction.commit([2] if malformed else [2, 3, 0, 1])
        assert participant.tokens == [0, 1]
        assert transaction.status == "aborted"


def test_abort_restores_completed_or_partial_speculation():
    participant = MemoryState([0, 1])
    transaction = dspark.DSparkTransaction(proposal(), {"cache": participant}, max_draft_tokens=3)
    participant.tokens.extend([2, 3])
    transaction.abort()
    assert participant.tokens == [0, 1]
    with pytest.raises(ValueError, match="closed"):
        transaction.commit([2, 3, 0, 1])


def test_restore_failure_is_not_hidden_and_other_participants_still_restore():
    bad, good = MemoryState([0, 1], fail_restore=True), MemoryState([0, 1])
    transaction = dspark.DSparkTransaction(proposal(), {"bad": bad, "good": good}, max_draft_tokens=3)
    bad.tokens.extend([2])
    good.tokens.extend([2])
    with pytest.raises(RuntimeError, match="bad"):
        transaction.abort()
    assert good.tokens == [0, 1]
    assert transaction.status == "failed"


def test_request_round_rejects_aliases_misaligned_positions_and_oversized_window():
    shared = MemoryState([0, 1])
    for states, size in (
        ({"a": shared, "b": shared}, 3),
        ({"a": shared, "b": MemoryState([0])}, 3),
        ({"a": shared}, 2),
        ({}, 3),
    ):
        with pytest.raises(ValueError):
            dspark.DSparkTransaction(proposal(), states, max_draft_tokens=size)
