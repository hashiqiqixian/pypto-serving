# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Chunked target taps and speculative cache ownership regressions."""

from types import SimpleNamespace
from threading import RLock

import pytest

torch = pytest.importorskip("torch")

from pypto_serving.model.deepseek_v41.dspark_resources import (
    DSparkResourceLayout,
    allocate_dspark_host_resources,
)
from pypto_serving.model.deepseek_v41.npu_runner import V41ModelRunner, V41ResidentResources
from pypto_serving.model.deepseek_v41.dspark_runtime import DSparkDecodeDispatch
from pypto_serving.model.deepseek_v41.task_args import ForwardStep, RequestLedger, RequestSlice


@pytest.mark.parametrize("length", (1, 80, 81, 128, 161, 257))
def test_chunks_retain_absolute_last_window_and_isolate_partitions(length):
    host = allocate_dspark_host_resources(DSparkResourceLayout(
        dp_size=4, max_seq_len=512, vocab_size=8, max_prefill_tokens=128))
    resident = V41ResidentResources.__new__(V41ResidentResources)
    resident.dspark_host_resources = host
    source = torch.zeros(4, 80, 15360, dtype=torch.bfloat16)
    resident.prefill_host = SimpleNamespace(fields={"dspark_target_hidden": source})
    for start in range(0, length, 80):
        end = min(start + 80, length)
        for rank in (1, 3):
            source[rank].fill_(-42)
            source[rank, :end - start] = torch.arange(start, end)[:, None] + rank * 256
            request = RequestSlice(str(rank), rank, 0, start, (1,) * (end - start), length, {})
            resident._retain_dspark_prefill(ForwardStep("prefill", (request,), start + 1))
    count = min(length, 128)
    taps = host.prefill_workspace["prefill_target_hidden"]
    for rank in (1, 3):
        expected = (torch.arange(length - count, length) + rank * 256).bfloat16()
        assert torch.equal(taps[rank, :count], expected[:, None].expand(-1, 15360))
    assert torch.count_nonzero(taps[0]) == 0
    # Reusing a partition starts a fresh prefix, independent of the prior length.
    source[1].fill_(9)
    fresh = RequestSlice("new", 1, 0, 0, (1, 2), 2, {})
    resident._retain_dspark_prefill(ForwardStep("prefill", (fresh,), 20))
    assert bool((taps[1, :2] == 9).all())
    stale = RequestSlice("1", 1, 0, 2, (3,), 3, {})
    with pytest.raises(ValueError, match="same request"):
        resident._retain_dspark_prefill(ForwardStep("prefill", (stale,), 21))


def _prefilled_ledger():
    ledger = RequestLedger(max_requests=1, max_seq_len=256, partitions=2)
    step = ledger.begin_prefill((("a", 0, 0, (1,) * 127, 127, {"window": (0,)}),))
    ledger.commit(step)
    return ledger


def test_speculative_commit_tracks_rejected_cross_page_writes_and_accepted_length():
    ledger = _prefilled_ledger()
    step = ledger.begin_decode(
        (("a", 0, 127, 7, {"window": (0, 1)}),), drafts={"a": (8, 9, 10, 11, 12)})
    ledger.commit(step, accepted_counts=(1,))
    assert ledger.owners["a"].length == 128
    assert ledger.owners["a"].pages["window"] == (0, 1)
    with pytest.raises(ValueError, match="preserve committed"):
        ledger.begin_decode((("a", 0, 128, 8, {"window": (0,)}),))
    next_step = ledger.begin_decode((("a", 0, 128, 8, {"window": (0, 1)}),))
    ledger.commit(next_step)
    released = []
    ledger.release(("a",), lambda key, owner: released.extend(owner.pages["window"]))
    assert released == [0, 1]
    assert not ledger.owners


def test_speculative_failure_clears_all_touched_pages_and_invalid_count_is_atomic():
    ledger = _prefilled_ledger()
    step = ledger.begin_decode(
        (("a", 0, 127, 7, {"window": (0, 1)}),), drafts={"a": (8, 9, 10, 11, 12)})
    with pytest.raises(ValueError, match="accepted counts"):
        ledger.commit(step, accepted_counts=(7,))
    assert ledger.owners["a"].length == 127
    assert ledger.pending is step
    released = []
    ledger.abort(step, lambda key, owner: released.extend(owner.pages["window"]))
    assert released == [0, 1]
    assert not ledger.owners and ledger.pending is None


def test_speculative_tail_uses_one_row_and_preserves_partition_continuity():
    ledger = _prefilled_ledger()
    with pytest.raises(ValueError, match="DP partition"):
        ledger.begin_decode((("a", 1, 127, 7, {"window": (0,)}),), drafts={"a": ()})
    step = ledger.begin_decode((("a", 0, 127, 7, {"window": (0,)}),), drafts={"a": ()})
    assert step.requests[0].token_ids == (7,)
    ledger.commit(step, accepted_counts=(1,))
    assert ledger.owners["a"].length == 128


@pytest.mark.parametrize("fail", (False, True))
def test_runner_finishes_or_aborts_speculative_ownership_transaction(fail):
    ledger = _prefilled_ledger()
    ledger.begin_decode(
        (("a", 0, 127, 7, {"window": (0, 1)}),), drafts={"a": (8, 9, 10, 11, 12)})
    runner = V41ModelRunner.__new__(V41ModelRunner)
    runner._lock = RLock()
    runner._request_ledger = lambda: ledger
    runner.resources = object()
    events = []
    runner.bindings = SimpleNamespace(wait=lambda resources: events.append("wait"))
    runner._reset_request = lambda key, owner: events.append(tuple(owner.pages["window"]))

    def execute(*args, **kwargs):
        events.append("decode")
        if fail:
            raise RuntimeError("device dispatch failed")
        return SimpleNamespace(accepted_token_ids=[[8, 9]])

    runner.run_dspark_decode = execute
    batch = SimpleNamespace(dspark_stages=DSparkDecodeDispatch(request_ids=("a",), stages=()))
    if fail:
        with pytest.raises(RuntimeError, match="device dispatch failed"):
            runner.run_dspark_decode_batch(batch)
        assert events == ["decode", "wait", (0, 1)]
        assert not ledger.owners
    else:
        result = runner.run_dspark_decode_batch(batch)
        assert result.accepted_token_ids == [[8, 9]]
        assert events == ["decode", "wait"]
        assert ledger.owners["a"].length == 129
        assert ledger.owners["a"].pages["window"] == (0, 1)
    assert ledger.pending is None
