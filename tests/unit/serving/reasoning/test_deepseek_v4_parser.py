# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.serving.reasoning import (
    DeepSeekV4ReasoningParser,
    OutputParserSpec,
)


class _Tokenizer:
    vocab = {"<think>": 90, "</think>": 91, "<eos>": 92}
    pieces = {
        1: "先",
        2: "思考",
        3: "答案",
        4: "</think>",  # ordinary text token, not the control-token ID
        5: "完成",
        6: "secret",
        7: "�",
        8: "好",
    }
    all_special_ids = (90, 91, 92)

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, token_ids, *, skip_special_tokens=True):
        parts = []
        index = 0
        while index < len(token_ids):
            token_id = token_ids[index]
            if tuple(token_ids[index : index + 2]) == (7, 8):
                parts.append("好")
                index += 2
                continue
            if token_id in (90, 91, 92):
                if not skip_special_tokens:
                    parts.append({90: "<think>", 91: "</think>", 92: "<eos>"}[token_id])
            else:
                parts.append(self.pieces[token_id])
            index += 1
        return "".join(parts)


def _parser(initial_state="reasoning", *, include_reasoning=True):
    return DeepSeekV4ReasoningParser(
        _Tokenizer(),
        OutputParserSpec(
            parser_id="deepseek_v4",
            initial_state=initial_state,
            include_reasoning=include_reasoning,
        ),
    )


def _parse(token_ids, initial_state="reasoning", *, include_reasoning=True):
    tokenizer = _Tokenizer()
    parser = DeepSeekV4ReasoningParser(
        tokenizer,
        OutputParserSpec(
            parser_id="deepseek_v4",
            initial_state=initial_state,
            include_reasoning=include_reasoning,
        ),
    )
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return parser.parse_complete(text, token_ids)


def test_prompt_opened_reasoning_splits_k7_burst_at_control_token() -> None:
    parsed = _parse((1, 2, 91, 3, 5, 92))
    assert parsed.reasoning == "先思考"
    assert parsed.content == "答案完成"


def test_literal_marker_text_does_not_trigger_token_terminal() -> None:
    parsed = _parse((1, 4, 2, 91, 3))
    assert parsed.reasoning == "先</think>思考"
    assert parsed.content == "答案"


def test_content_mode_can_enter_reasoning_and_absorbs_duplicate_markers() -> None:
    parsed = _parse((3, 90, 90, 1, 91, 5), initial_state="content")
    assert parsed.reasoning == "先"
    assert parsed.content == "答案完成"


def test_include_reasoning_hides_reasoning_without_leaking_to_content() -> None:
    parsed = _parse((1, 2, 91, 3), include_reasoning=False)
    assert parsed.reasoning == ""
    assert parsed.content == "答案"


def test_missing_reasoning_terminal_fails_closed() -> None:
    tokenizer = _Tokenizer()
    tokenizer.vocab = {"<think>": 90}
    with pytest.raises(ValueError, match="</think>"):
        DeepSeekV4ReasoningParser(
            tokenizer,
            OutputParserSpec("deepseek_v4", "reasoning"),
        )


def test_streaming_parser_keeps_state_and_resolves_deferred_terminal() -> None:
    parser = _parser()

    first = parser.feed("先思考", (1, 2))
    held = parser.feed("", (91,))
    final = parser.feed("</think>答案完成", (3, 5))
    flushed = parser.finish()

    assert first.reasoning == "先思考"
    assert held.reasoning == held.content == ""
    assert final.reasoning == ""
    assert final.content == "答案完成"
    assert flushed.reasoning == flushed.content == ""


def test_deferred_terminal_keeps_token_order_around_literal_lookalike() -> None:
    parser = _parser()

    held = parser.feed("", (4, 6, 91, 7))
    resumed = parser.feed("</think>secret</think>好", (8,))
    flushed = parser.finish()

    assert held.reasoning == held.content == ""
    assert resumed.reasoning == "</think>secret"
    assert resumed.content == "好"
    assert flushed.reasoning == flushed.content == ""


def test_streaming_and_complete_parsing_are_equivalent_for_k7_bursts() -> None:
    token_ids = (1, 2, 91, 3, 5, 92)
    complete = _parse(token_ids)
    parser = _parser()

    deltas = [
        parser.feed("先思考</think>答案", token_ids[:4]),
        parser.feed("完成<eos>", token_ids[4:]),
        parser.finish(),
    ]

    assert "".join(delta.reasoning for delta in deltas) == complete.reasoning
    assert "".join(delta.content for delta in deltas) == complete.content


def test_unreleased_reasoning_terminal_fails_closed_at_finish() -> None:
    parser = _parser()
    parser.feed("", (91,))

    with pytest.raises(ValueError, match="terminal text was not released"):
        parser.finish()
