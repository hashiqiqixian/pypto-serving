# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import pytest

from pypto_serving.model.deepseek.encoding import encode_messages
from pypto_serving.model.tokenizer import (
    DeepSeekV4TokenizerAdapter,
    _load_fast_tokenizer_from_file,
)


def test_fast_tokenizer_load_preserves_checkpoint_chat_template(tmp_path):
    class FakeTokenizer:
        def __init__(self, tokenizer_file, **kwargs):
            self.tokenizer_file = tokenizer_file
            self.kwargs = kwargs

    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": {"content": "<bos>"},
        "chat_template": "{{ messages[0].content }}",
    }))

    tokenizer = _load_fast_tokenizer_from_file(tmp_path, FakeTokenizer)

    assert tokenizer.kwargs["bos_token"] == "<bos>"
    assert tokenizer.kwargs["chat_template"] == "{{ messages[0].content }}"


def test_deepseek_v4_defaults_to_chat_mode():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    prompt = adapter.apply_chat_template([
        {"role": "user", "content": "What is 1+1?"},
    ], tokenize=False, add_generation_prompt=True)

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>What is 1+1?<｜Assistant｜></think>"
    )


def test_deepseek_v4_enables_thinking_with_vllm_compatible_kwarg():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    prompt = adapter.apply_chat_template([
        {"role": "user", "content": "What is 1+1?"},
    ], enable_thinking=True)

    assert prompt.endswith("<｜Assistant｜><think>")


def test_deepseek_v4_reasoning_none_overrides_enable_thinking():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    prompt = adapter.apply_chat_template(
        [{"role": "user", "content": "What is 1+1?"}],
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_multiturn_thinking_only_marks_latest_user_turn():
    prompt = encode_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "And +1?"},
    ], thinking=True)

    assert prompt == (
        "<｜begin▁of▁sentence｜>Be concise."
        "<｜User｜>What is 1+1?<｜Assistant｜></think>"
        "2<｜end▁of▁sentence｜>"
        "<｜User｜>And +1?<｜Assistant｜><think>"
    )


def test_deepseek_v4_rejects_unsupported_message_role():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    with pytest.raises(ValueError, match="does not support chat message role 'tool'"):
        adapter.apply_chat_template([{"role": "tool", "content": "result"}])
