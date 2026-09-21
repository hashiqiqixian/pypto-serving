# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

from pypto_serving.serving.engine.async_engine import (
    ReplicaEngineCore,
    _RequestContext,
)
from pypto_serving.serving.reasoning import OutputParserSpec, create_output_parser
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestOutput,
    RequestStatus,
    SchedulerOutput,
)


def test_incremental_detok_matches_full_decode_and_hides_partial_chars():
    """Incremental detok must equal a full decode and never stream a partial char.

    Guards the O(N^2) -> O(N) detokenization fix: cumulative text produced step
    by step must match tokenizer.decode(all_ids), and an incomplete multi-token
    character (rendered as U+FFFD) must be withheld until it completes.
    """
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)

    class _MultiByteTokenizer:
        # tokens 6+7 together render '★'; 6 alone is an incomplete char.
        _table = {1: "He", 2: "llo", 3: " wor", 4: "ld", 5: "!"}

        def decode(self, ids):
            out, i = [], 0
            while i < len(ids):
                t = ids[i]
                if t == 6:
                    if i + 1 < len(ids) and ids[i + 1] == 7:
                        out.append("★")
                        i += 2
                        continue
                    out.append("�")
                    i += 1
                    continue
                if t == 7:
                    out.append("★")
                    i += 1
                    continue
                out.append(self._table[t])
                i += 1
            return "".join(out)

    core.tokenizer = _MultiByteTokenizer()
    ctx = _RequestContext(request=SimpleNamespace(output_token_ids=[]))

    seq = [1, 2, 3, 4, 5, 6, 7]
    cumulative = ""
    per_step = []
    for k in range(1, len(seq) + 1):
        ctx.request.output_token_ids = seq[:k]
        cumulative = core._detokenize_incrementally(ctx)
        per_step.append(cumulative)

    # No partial char ever leaked, and the final text equals a full decode.
    assert all("�" not in text for text in per_step)
    assert per_step[4] == "Hello world!"  # step 6 (idx 5) withholds partial
    assert per_step[5] == "Hello world!"  # still withheld
    assert cumulative == core.tokenizer.decode(seq) == "Hello world!★"


def test_finalize_detok_flushes_trailing_incomplete_char_at_eos():
    """If generation stops while a multi-token char is incomplete, the finished
    step must flush the authoritative full decode instead of the withheld text.

    Guards the FINAL_ONLY truncation bug: the incremental path withholds a
    trailing U+FFFD forever (no later token completes it once generation ends),
    so _finalize_detokenization must fall back to a full decode.
    """
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)

    class _TrailingByteTokenizer:
        # token 6 alone is an incomplete char (U+FFFD); it is the last token.
        _table = {1: "Hi", 2: "!"}

        def decode(self, ids):
            out = []
            for t in ids:
                out.append("�" if t == 6 else self._table[t])
            return "".join(out)

    core.tokenizer = _TrailingByteTokenizer()
    ctx = _RequestContext(request=SimpleNamespace(output_token_ids=[]))

    # Drive incremental decode up to the trailing incomplete token.
    for k in range(1, 4):
        ctx.request.output_token_ids = [1, 2, 6][:k]
        incremental = core._detokenize_incrementally(ctx)

    # Incremental withholds the trailing U+FFFD (never emits a partial char).
    assert incremental == "Hi!"
    assert "�" not in incremental

    # On finish, the authoritative full decode is flushed (no truncation).
    final = core._finalize_detokenization(ctx)
    assert final == core.tokenizer.decode([1, 2, 6]) == "Hi!�"


def test_process_step_output_splits_reasoning_from_k7_token_burst():
    class _ReasoningTokenizer:
        vocab = {"<think>": 90, "</think>": 91}
        pieces = {1: "先", 2: "思考", 3: "答案"}
        all_special_ids = (90, 91)

        def get_vocab(self):
            return dict(self.vocab)

        def decode(self, ids, *, skip_special_tokens=True):
            parts = []
            for token_id in ids:
                if token_id in self.vocab.values():
                    if not skip_special_tokens:
                        parts.append({90: "<think>", 91: "</think>"}[token_id])
                    continue
                parts.append(self.pieces.get(token_id, ""))
            return "".join(parts)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _ReasoningTokenizer()
    core._pending_free_ids = []
    core.scheduler = _ScriptedScheduler(
        [
            RequestOutput(
                request_id="r",
                new_token_id=3,
                finished=True,
                finish_reason="FINISHED_LENGTH",
            )
        ]
    )
    request = Request(request_id="r", prompt_token_ids=[9], max_new_tokens=4)
    request.output_token_ids.extend([1, 2, 91, 3])
    ctx = _RequestContext(
        request=request,
        stream=True,
        output_parser=create_output_parser(
            OutputParserSpec("deepseek_v4", "reasoning"),
            core.tokenizer,
        ),
    )
    core._request_contexts = {"r": ctx}

    core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})

    output = ctx.queue.get_nowait()
    assert output.reasoning == "先思考"
    assert output.text == "答案"
    assert output.reasoning_delta == "先思考"
    assert output.text_delta == "答案"
    assert output.token_ids == (1, 2, 91, 3)


def test_process_step_output_keeps_incremental_reasoning_state() -> None:
    class _ReasoningTokenizer:
        vocab = {"<think>": 90, "</think>": 91}
        pieces = {1: "先", 2: "思考", 3: "答案"}
        all_special_ids = (90, 91)

        def get_vocab(self):
            return dict(self.vocab)

        def decode(self, ids, *, skip_special_tokens=True):
            parts = []
            for token_id in ids:
                if token_id in self.vocab.values():
                    if not skip_special_tokens:
                        parts.append({90: "<think>", 91: "</think>"}[token_id])
                    continue
                parts.append(self.pieces.get(token_id, ""))
            return "".join(parts)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _ReasoningTokenizer()
    core._pending_free_ids = []
    core.scheduler = _ScriptedScheduler(
        [
            RequestOutput(request_id="r", new_token_id=2),
            RequestOutput(
                request_id="r",
                new_token_id=3,
                finished=True,
                finish_reason="FINISHED_LENGTH",
            ),
        ]
    )
    request = Request(request_id="r", prompt_token_ids=[9], max_new_tokens=4)
    request.output_token_ids.extend([1, 2])
    ctx = _RequestContext(
        request=request,
        stream=True,
        output_parser=create_output_parser(
            OutputParserSpec("deepseek_v4", "reasoning"),
            core.tokenizer,
        ),
    )
    core._request_contexts = {"r": ctx}

    core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})
    first = ctx.queue.get_nowait()
    assert first.reasoning_delta == "先思考"
    assert first.text_delta == ""
    assert first.reasoning == first.text == ""

    request.output_token_ids.extend([91, 3])
    core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})
    final = ctx.queue.get_nowait()
    assert final.reasoning_delta == ""
    assert final.text_delta == "答案"
    assert final.reasoning == "先思考"
    assert final.text == "答案"


def test_process_step_output_contains_parser_failure_to_one_request() -> None:
    class _Tokenizer:
        def decode(self, ids, *, skip_special_tokens=True):
            return "".join(str(token_id) for token_id in ids)

    class _FailingParser:
        def feed(self, delta_text, delta_token_ids):
            return SimpleNamespace(reasoning="", content=delta_text)

        def finish(self):
            raise ValueError("malformed terminal tail")

    class _TwoRequestScheduler:
        def __init__(self):
            self.aborted = []

        def update_from_output(self, scheduler_output, new_tokens):
            return [
                RequestOutput(
                    request_id="bad",
                    new_token_id=1,
                    finished=True,
                    finish_reason="FINISHED_LENGTH",
                ),
                RequestOutput(
                    request_id="good",
                    new_token_id=2,
                    finished=True,
                    finish_reason="FINISHED_LENGTH",
                ),
            ]

        def abort_request(self, request_id):
            self.aborted.append(request_id)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _Tokenizer()
    core._pending_free_ids = []
    core.scheduler = _TwoRequestScheduler()

    bad_request = Request(request_id="bad", prompt_token_ids=[9], max_new_tokens=1)
    bad_request.output_token_ids.append(1)
    bad_ctx = _RequestContext(
        request=bad_request,
        stream=True,
        output_parser=_FailingParser(),
    )
    good_request = Request(request_id="good", prompt_token_ids=[9], max_new_tokens=1)
    good_request.output_token_ids.append(2)
    good_ctx = _RequestContext(request=good_request, stream=True)
    core._request_contexts = {"bad": bad_ctx, "good": good_ctx}

    core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})

    failure = bad_ctx.queue.get_nowait()
    assert isinstance(failure, ValueError)
    assert str(failure) == "malformed terminal tail"
    assert "bad" not in core._request_contexts
    assert core.scheduler.aborted == ["bad"]
    assert core._pending_free_ids.count("bad") == 1

    good_output = good_ctx.queue.get_nowait()
    assert good_output.finished
    assert good_output.text == "2"
    assert core._pending_free_ids.count("good") == 1


def test_parser_detokenization_work_is_linear_in_generated_tokens() -> None:
    class _CountingTokenizer:
        vocab = {"<think>": 90, "</think>": 91}
        all_special_ids = (90, 91)

        def __init__(self):
            self.decoded_token_count = 0

        def get_vocab(self):
            return dict(self.vocab)

        def decode(self, ids, *, skip_special_tokens=True):
            self.decoded_token_count += len(ids)
            parts = []
            for token_id in ids:
                if token_id in self.vocab.values():
                    if not skip_special_tokens:
                        parts.append({90: "<think>", 91: "</think>"}[token_id])
                    continue
                parts.append("x")
            return "".join(parts)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _CountingTokenizer()
    request = Request(request_id="r", prompt_token_ids=[9], max_new_tokens=128)
    ctx = _RequestContext(
        request=request,
        stream=True,
        output_parser=create_output_parser(
            OutputParserSpec("deepseek_v4", "reasoning"),
            core.tokenizer,
        ),
    )

    token_count = 128
    for _ in range(token_count):
        request.output_token_ids.append(1)
        text_delta = core._detokenize_parser_incrementally(ctx)
        new_ids = request.output_token_ids[ctx.parser_token_offset :]
        ctx.parser_token_offset = len(request.output_token_ids)
        ctx.output_parser.feed(text_delta, new_ids)

    tail = core._finalize_parser_detokenization(ctx)
    if tail:
        ctx.output_parser.feed(tail, ())
    ctx.output_parser.finish()

    # The bounded three-token detokenizer context visits at most a constant
    # number of IDs per step, plus one final full decode. A cumulative-prefix
    # parser would visit roughly N*(N+1)/2 IDs and fail this bound.
    assert core.tokenizer.decoded_token_count <= 9 * token_count


def test_stop_string_detection_precedes_reasoning_presentation_split():
    class _ReasoningTokenizer:
        vocab = {"<think>": 90, "</think>": 91}
        pieces = {1: "先", 2: "思考"}
        all_special_ids = (90, 91)

        def get_vocab(self):
            return dict(self.vocab)

        def decode(self, ids, *, skip_special_tokens=True):
            parts = []
            for token_id in ids:
                if token_id in self.vocab.values():
                    if not skip_special_tokens:
                        parts.append({90: "<think>", 91: "</think>"}[token_id])
                    continue
                parts.append(self.pieces.get(token_id, ""))
            return "".join(parts)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _ReasoningTokenizer()
    core._pending_free_ids = []
    scheduler = _ScriptedScheduler(
        [RequestOutput(request_id="r", new_token_id=2)]
    )
    core.scheduler = scheduler
    request = Request(
        request_id="r",
        prompt_token_ids=[9],
        max_new_tokens=4,
        stop_strings=("思考",),
    )
    request.output_token_ids.extend([1, 2])
    ctx = _RequestContext(
        request=request,
        output_parser=create_output_parser(
            OutputParserSpec("deepseek_v4", "reasoning"),
            core.tokenizer,
        ),
    )
    core._request_contexts = {"r": ctx}

    core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})

    assert scheduler.finished == [("r", RequestStatus.FINISHED_STOP)]
    output = ctx.queue.get_nowait()
    assert output.finished
    assert output.finish_reason == "FINISHED_STOP"
    assert output.reasoning == "先思考"
    assert output.text == ""


class _ScriptedScheduler:
    """Stub scheduler: returns one preset RequestOutput per _process_step_output call."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.finished = []

    def update_from_output(self, scheduler_output, new_tokens):
        return [self._outputs.pop(0)]

    def finish_request(self, request_id, status):
        self.finished.append((request_id, status))


class _WordTokenizer:
    _table = {1: "a", 2: "b", 3: "c", 4: "STOP"}

    def decode(self, ids):
        return "".join(self._table[t] for t in ids)


def _drive(core, ctx, token_ids):
    """Append each token and run one _process_step_output step per token."""
    for t in token_ids:
        ctx.request.output_token_ids.append(t)
        core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})


def test_non_streaming_suppresses_intermediate_outputs():
    """A non-streaming request enqueues exactly one (final) TokenOutput; a
    streaming request enqueues one per token."""
    seq = [1, 2, 3]

    # --- non-streaming: stream=False -> only the final token is published ---
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _WordTokenizer()
    core._pending_free_ids = []
    outputs = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=3, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core.scheduler = _ScriptedScheduler(outputs)
    ns_ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=3),
        stream=False,
    )
    core._request_contexts = {"r": ns_ctx}

    _drive(core, ns_ctx, seq)

    assert ns_ctx.queue.qsize() == 1
    final = ns_ctx.queue.get_nowait()
    assert final.finished is True
    assert final.text == "abc"  # full cumulative text on the final output
    assert final.token_ids == (1, 2, 3)
    assert "r" in core._pending_free_ids

    # --- streaming: stream=True -> one TokenOutput per token ---
    core2 = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core2.tokenizer = _WordTokenizer()
    core2._pending_free_ids = []
    outputs2 = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=3, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core2.scheduler = _ScriptedScheduler(outputs2)
    s_ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=3),
        stream=True,
    )
    core2._request_contexts = {"r": s_ctx}

    _drive(core2, s_ctx, seq)

    assert s_ctx.queue.qsize() == 3
    texts = [s_ctx.queue.get_nowait().text for _ in range(3)]
    assert texts == ["a", "ab", "abc"]  # cumulative text grows each step


def test_non_streaming_still_detects_stop_string():
    """Stop-string detection must run every step even when outputs are
    suppressed, so a non-streaming request stops mid-generation and publishes
    exactly one finished output."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _WordTokenizer()
    core._pending_free_ids = []
    # Would run 4 tokens, but token 4 decodes to "STOP" which is a stop string.
    outputs = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=4),  # -> text ends with "STOP"
        RequestOutput(request_id="r", new_token_id=3),  # should never be reached
    ]
    scheduler = _ScriptedScheduler(outputs)
    core.scheduler = scheduler
    ctx = _RequestContext(
        request=Request(
            request_id="r",
            prompt_token_ids=[9],
            max_new_tokens=4,
            stop_strings=("STOP",),
        ),
        stream=False,
    )
    core._request_contexts = {"r": ctx}

    _drive(core, ctx, [1, 2, 4])

    # Stop detected at step 3: scheduler.finish_request called, one final output.
    assert scheduler.finished == [("r", RequestStatus.FINISHED_STOP)]
    assert ctx.queue.qsize() == 1
    final = ctx.queue.get_nowait()
    assert final.finished is True
    assert final.finish_reason == "FINISHED_STOP"
    assert final.text == "abSTOP"


def test_non_streaming_final_output_uses_full_decode_on_incomplete_char():
    """FINAL_ONLY must publish the full-decode text on finish, even when the
    last token leaves a multi-token character incomplete (U+FFFD)."""

    class _TrailingByteTokenizer:
        _table = {1: "a", 2: "b"}

        def decode(self, ids):
            return "".join("�" if t == 6 else self._table[t] for t in ids)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _TrailingByteTokenizer()
    core._pending_free_ids = []
    outputs = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=6, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core.scheduler = _ScriptedScheduler(outputs)
    ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=3),
        stream=False,
    )
    core._request_contexts = {"r": ctx}

    _drive(core, ctx, [1, 2, 6])

    assert ctx.queue.qsize() == 1
    final = ctx.queue.get_nowait()
    assert final.finished is True
    # Not the withheld "ab": the trailing incomplete char is flushed.
    assert final.text == "ab�"


def test_process_step_output_schedules_free_once_on_normal_finish():
    """A normally-finished request is scheduled for worker release exactly once
    by _process_step_output (the add_request finally must not re-add it)."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _WordTokenizer()
    core._pending_free_ids = []
    outputs = [
        RequestOutput(request_id="r", new_token_id=1, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core.scheduler = _ScriptedScheduler(outputs)
    ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=1),
        stream=False,
    )
    core._request_contexts = {"r": ctx}

    _drive(core, ctx, [1])

    # Scheduled once. The engine loop will drain this into the next StepCommand;
    # add_request's finally must not append it again (double-release guard).
    assert core._pending_free_ids == ["r"]
    # _schedule_worker_free is idempotent while the id is still queued.
    core._schedule_worker_free("r")
    assert core._pending_free_ids == ["r"]
