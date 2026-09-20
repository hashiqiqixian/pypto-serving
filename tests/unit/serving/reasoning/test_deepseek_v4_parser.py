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
    }

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, token_ids, *, skip_special_tokens=True):
        parts = []
        for token_id in token_ids:
            if token_id in (90, 91, 92):
                if not skip_special_tokens:
                    parts.append({90: "<think>", 91: "</think>", 92: "<eos>"}[token_id])
                continue
            parts.append(self.pieces[token_id])
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


def test_prompt_opened_reasoning_splits_k7_burst_at_control_token() -> None:
    parsed = _parser().parse((1, 2, 91, 3, 5, 92))
    assert parsed.reasoning == "先思考"
    assert parsed.content == "答案完成"


def test_literal_marker_text_does_not_trigger_token_terminal() -> None:
    parsed = _parser().parse((1, 4, 2, 91, 3))
    assert parsed.reasoning == "先</think>思考"
    assert parsed.content == "答案"


def test_content_mode_can_enter_reasoning_and_absorbs_duplicate_markers() -> None:
    parsed = _parser("content").parse((3, 90, 90, 1, 91, 5))
    assert parsed.reasoning == "先"
    assert parsed.content == "答案完成"


def test_include_reasoning_hides_reasoning_without_leaking_to_content() -> None:
    parsed = _parser(include_reasoning=False).parse((1, 2, 91, 3))
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
