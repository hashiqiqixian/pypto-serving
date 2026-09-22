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


def test_fast_tokenizer_load_registers_backend_special_tokens(tmp_path):
    class FakeTokenizer:
        def __init__(self, tokenizer_file, **kwargs):
            self.tokenizer_file = tokenizer_file
            self.kwargs = kwargs

    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [
            {"id": 0, "content": "<bos>", "special": True},
            {"id": 2, "content": "<internal>", "special": True},
            {"id": 3, "content": "ordinary", "special": False},
        ],
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": {"content": "<bos>"},
    }))

    tokenizer = _load_fast_tokenizer_from_file(tmp_path, FakeTokenizer)

    assert tokenizer.kwargs["bos_token"] == "<bos>"
    assert tokenizer.kwargs["additional_special_tokens"] == ["<internal>"]


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


def test_deepseek_v4_rejects_orphan_tool_result():
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())

    with pytest.raises(ValueError, match="preceding assistant tool_call_id"):
        adapter.apply_chat_template([{"role": "tool", "content": "result"}])


def test_deepseek_tool_prompt_and_history_roundtrip_preserve_input():
    tools = [{"type": "function", "function": {
        "name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Compare cities"},
        {"role": "assistant", "content": None, "reasoning": "Compare both.", "tool_calls": [
            {"id": "first", "type": "function", "function": {"name": "weather", "arguments": '{"city":"杭州"}'}},
            {"id": "second", "type": "function", "function": {"name": "weather", "arguments": '{"city":"北京"}'}},
        ]},
        {"role": "tool", "tool_call_id": "second", "content": "Cold"},
        {"role": "tool", "tool_call_id": "first", "content": "Warm"},
    ]
    original = json.dumps(messages)
    adapter = DeepSeekV4TokenizerAdapter(tokenizer=object())
    prompt = adapter.apply_chat_template(messages, tools=tools, enable_thinking=True)
    assert prompt.startswith("<｜begin▁of▁sentence｜>Be concise.\n\n## Tools")
    assert '"name": "weather"' in prompt
    assert '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>' in prompt
    assert "Compare both.</think>" in prompt
    assert prompt.endswith(
        "<｜User｜><tool_result>Warm</tool_result>\n\n<tool_result>Cold</tool_result><｜Assistant｜><think>"
    )
    assert json.dumps(messages) == original


def test_tool_history_keeps_reasoning_without_reoffering_the_tool():
    prompt = encode_messages([
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": None, "reasoning": "Need data.", "tool_calls": [
            {"id": "one", "function": {"name": "lookup", "arguments": '{"n":2,"ok":true,"v":null}'}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "Data"},
    ], thinking=True)
    assert "Need data.</think>" in prompt
    assert '<｜DSML｜parameter name="n" string="false">2</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="ok" string="false">true</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="v" string="false">null</｜DSML｜parameter>' in prompt
    assert "## Tools" not in prompt
    assert prompt.endswith("<tool_result>Data</tool_result><｜Assistant｜><think>")


@pytest.mark.parametrize("thinking", [False, True])
def test_trailing_system_opens_generation_at_assistant_boundary(thinking):
    prompt = encode_messages([{"role": "system", "content": "Say hello"}], thinking=thinking)
    assert prompt.endswith("<｜Assistant｜>" + ("<think>" if thinking else "</think>"))


def test_tool_result_followed_by_user_text_forms_one_user_message():
    prompt = encode_messages([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "one", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "Data"},
        {"role": "user", "content": "Explain it"},
    ])
    assert "<｜User｜><tool_result>Data</tool_result>\n\nExplain it<｜Assistant｜></think>" in prompt
