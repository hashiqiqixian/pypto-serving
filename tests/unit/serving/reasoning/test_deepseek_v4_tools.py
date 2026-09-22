# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-aligned DSML replay tests, independent of checkpoint and device code."""

import json
import random

import pytest

from pypto_serving.serving.reasoning import OutputParserSpec, create_output_parser
from pypto_serving.serving.reasoning.deepseek_v4_tools import TOOL_END, TOOL_START


class ToolTokenizer:
    specials = (
        "<think>", "</think>", TOOL_START, TOOL_END,
        '<｜DSML｜invoke', '</｜DSML｜invoke>',
        '<｜DSML｜parameter', '</｜DSML｜parameter>', '<eos>',
    )
    vocab = {text: index + 1 for index, text in enumerate(specials)}
    all_special_ids = tuple(vocab.values())

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text):
        ids = []
        offset = 0
        while offset < len(text):
            special = next((part for part in self.specials if text.startswith(part, offset)), None)
            if special:
                ids.append(self.vocab[special])
                offset += len(special)
            else:
                ids.append(1000 + ord(text[offset]))
                offset += 1
        return ids

    def decode(self, ids, *, skip_special_tokens=True):
        out = []
        index = 0
        while index < len(ids):
            token_id = ids[index]
            if token_id == 700:
                if index + 1 < len(ids) and ids[index + 1] == 701:
                    out.append("好")
                    index += 2
                    continue
                out.append("�")
            elif token_id == 701:
                out.append("好")
            elif token_id in self.all_special_ids:
                if not skip_special_tokens:
                    out.append(self.specials[token_id - 1])
            else:
                out.append(chr(token_id - 1000))
            index += 1
        return "".join(out)


def invoke(name="lookup", **parameters):
    parts = [f'<｜DSML｜invoke name="{name}">\n']
    for key, value in parameters.items():
        string = isinstance(value, str)
        body = value if string else json.dumps(value, ensure_ascii=False)
        parts.append(
            f'<｜DSML｜parameter name="{key}" string="{str(string).lower()}">{body}</｜DSML｜parameter>\n'
        )
    return "".join(parts) + "</｜DSML｜invoke>\n"


def parser(*, thinking=False, include_reasoning=True, choice="auto"):
    return create_output_parser(OutputParserSpec(
        "deepseek_v4", "reasoning" if thinking else "content", include_reasoning,
        tool_choice=choice, tool_names=("lookup", "other"),
    ), ToolTokenizer())


def replay(text, chunk_size, **kwargs):
    tokenizer = ToolTokenizer()
    instance = parser(**kwargs)
    ids = tokenizer.encode(text)
    outputs = [
        instance.feed(tokenizer.decode(ids[i:i + chunk_size], skip_special_tokens=False), ids[i:i + chunk_size])
        for i in range(0, len(ids), chunk_size)
    ]
    outputs.append(instance.finish())
    return outputs


def calls_from_deltas(outputs):
    calls = {}
    for output in outputs:
        for delta in output.tool_call_deltas:
            call = calls.setdefault(delta.index, {"id": None, "name": None, "parts": []})
            if delta.id:
                assert call["id"] is None, "ID must only be sent once"
                call["id"] = delta.id
            if delta.name:
                assert call["name"] is None, "name must only be sent once"
                call["name"] = delta.name
            call["parts"].append(delta.arguments)
    return [(call["name"], "".join(call["parts"])) for call in calls.values()]


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 29, 100000])
def test_stream_and_complete_match_across_all_boundaries(chunk_size):
    expected = {"text": '杭州 "quoted" \\ slash\n tab\t nul\0 <xml> </think>',
                "n": 2, "flag": True, "nothing": None, "items": [1, "x"], "options": {"v": False}}
    text = "thinking</think>Before " + TOOL_START + invoke(**expected) + invoke("other") + TOOL_END + " after<eos>"
    outputs = replay(text, chunk_size, thinking=True)
    complete = parser(thinking=True).parse_complete(text, ToolTokenizer().encode(text))
    assert "".join(out.reasoning for out in outputs) == complete.reasoning == "thinking"
    assert "".join(out.content for out in outputs) == complete.content == "Before  after"
    calls = calls_from_deltas(outputs)
    assert calls == [(call.name, call.arguments) for call in complete.tool_calls]
    assert json.loads(calls[0][1]) == expected
    assert json.loads(calls[1][1]) == {}
    assert all(call.complete for call in outputs[-1].tool_calls)
    ids = [call.id for call in outputs[-1].tool_calls]
    assert len(set(ids)) == 2


def test_tool_start_implicitly_closes_hidden_reasoning():
    text = "thinking" + TOOL_START + invoke() + TOOL_END + "answer"
    outputs = replay(text, 7, thinking=True, include_reasoning=False)
    assert not any(out.reasoning for out in outputs)
    assert "".join(out.content for out in outputs) == "answer"
    assert calls_from_deltas(outputs) == [("lookup", "{}")]


def test_none_suppresses_tools_but_keeps_content_and_reasoning():
    text = "reason</think>before" + TOOL_START + invoke() + TOOL_END + "after"
    outputs = replay(text, 7, thinking=True, choice="none")
    assert "".join(out.reasoning for out in outputs) == "reason"
    assert "".join(out.content for out in outputs) == "beforeafter"
    assert not any(out.tool_calls or out.tool_call_deltas for out in outputs)


def test_ordinary_root_marker_lookalike_is_content():
    text = "Example: " + TOOL_START
    result = parser().parse_complete(text, [1000 + ord(char) for char in text])
    assert result.content == text
    assert result.tool_calls == ()


def test_root_token_text_can_be_released_across_multiple_feeds():
    instance = parser()
    tokenizer = ToolTokenizer()
    prefix = TOOL_START[:7]
    first = instance.feed(prefix, [tokenizer.vocab[TOOL_START]])
    suffix = invoke() + TOOL_END
    second = instance.feed(TOOL_START[7:] + suffix, tokenizer.encode(suffix))
    final = instance.finish()
    assert not first.content
    assert calls_from_deltas([first, second, final]) == [("lookup", "{}")]


def test_utf8_holdback_with_literal_and_real_think_end_then_tools():
    instance = parser(thinking=True)
    tokenizer = ToolTokenizer()
    literal = "</think>secret"
    held = instance.feed("", [1000 + ord(char) for char in literal] + [tokenizer.vocab["</think>"], 700])
    tool_text = TOOL_START + invoke() + TOOL_END
    resumed = instance.feed(literal + "</think>好" + tool_text, [701] + tokenizer.encode(tool_text))
    final = instance.finish()
    assert not held.reasoning and not held.tool_call_deltas
    assert resumed.reasoning == literal
    assert resumed.content == "好"
    assert calls_from_deltas([resumed, final]) == [("lookup", "{}")]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e9999", "broken"])
def test_invalid_nonstring_json_is_not_repaired(value):
    text = TOOL_START + f'<｜DSML｜invoke name="lookup"><｜DSML｜parameter name="x" string="false">{value}</｜DSML｜parameter></｜DSML｜invoke>' + TOOL_END
    with pytest.raises(ValueError):
        parser().parse_complete(text, ToolTokenizer().encode(text))


def test_json_nesting_errors_remain_request_local(monkeypatch):
    text = TOOL_START + invoke(value=[1]) + TOOL_END

    def too_deep(value):
        raise RecursionError("nested JSON")

    monkeypatch.setattr(json, "loads", too_deep)
    with pytest.raises(ValueError, match="nesting limit"):
        parser().parse_complete(text, ToolTokenizer().encode(text))


@pytest.mark.parametrize("body", [
    invoke("unknown"),
    '<｜DSML｜invoke name="lookup"><｜DSML｜parameter name="x" string="true">a</｜DSML｜parameter><｜DSML｜parameter name="x" string="true">b</｜DSML｜parameter></｜DSML｜invoke>',
    "not a header",
])
def test_invalid_invocation_is_a_request_error(body):
    text = TOOL_START + body + TOOL_END
    with pytest.raises(ValueError):
        parser().parse_complete(text, ToolTokenizer().encode(text))


def test_tool_end_may_close_last_invoke():
    text = TOOL_START + invoke(value=3).removesuffix("</｜DSML｜invoke>\n") + TOOL_END
    result = parser().parse_complete(text, ToolTokenizer().encode(text))
    assert json.loads(result.tool_calls[0].arguments) == {"value": 3}
    assert result.tool_calls[0].complete


def test_truncation_keeps_the_published_prefix_without_fabricating_json():
    text = TOOL_START + '<｜DSML｜invoke name="lookup"><｜DSML｜parameter name="text" string="true">unfinished'
    tokenizer = ToolTokenizer()
    instance = parser()
    delta = instance.feed(text, tokenizer.encode(text))
    tail = instance.finish(truncated=True)
    complete = parser().parse_complete(text, tokenizer.encode(text), truncated=True)
    assert calls_from_deltas([delta, tail]) == [(call.name, call.arguments) for call in complete.tool_calls]
    assert complete.tool_calls[0].arguments == '{"text":"unfinished'
    assert not complete.tool_calls[0].complete
    with pytest.raises(ValueError, match="incomplete DSML"):
        parser().parse_complete(text, tokenizer.encode(text))


def test_long_string_streams_before_close_without_reprocessing_old_body():
    tokenizer = ToolTokenizer()
    instance = parser()
    processed = []
    original_append = instance._append_value

    def track(text, deltas):
        processed.append(len(text))
        original_append(text, deltas)

    instance._append_value = track
    header = TOOL_START + '<｜DSML｜invoke name="lookup"><｜DSML｜parameter name="text" string="true">'
    outputs = [instance.feed(header, tokenizer.encode(header))]
    body = ('quote=" backslash=\\ unicode=杭州\n' * 1024)
    for offset in range(0, len(body), 32):
        chunk = body[offset:offset + 32]
        result = instance.feed(chunk, tokenizer.encode(chunk))
        assert result.tool_call_deltas, "body must stream before parameter close"
        outputs.append(result)
    assert sum(processed) == len(body)
    end = '</｜DSML｜parameter></｜DSML｜invoke>' + TOOL_END
    outputs += [instance.feed(end, tokenizer.encode(end)), instance.finish()]
    assert json.loads(calls_from_deltas(outputs)[0][1]) == {"text": body}


def test_complete_parse_resets_calls_between_uses():
    instance = parser()
    text = TOOL_START + invoke() + TOOL_END
    first = instance.parse_complete(text, ToolTokenizer().encode(text))
    second = instance.parse_complete(text, ToolTokenizer().encode(text))
    assert len(first.tool_calls) == len(second.tool_calls) == 1
    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_random_partitions_preserve_body_and_delimiter_prefixes():
    values = {"text": 'a< b</ c</｜DSML｜para d\n\\"杭州', "nested": {"items": [1, False, None]}}
    text = TOOL_START + invoke(**values) + invoke("other", text="second") + TOOL_END
    tokenizer = ToolTokenizer()
    ids = tokenizer.encode(text)
    for seed in range(20):
        rng = random.Random(seed)
        instance = parser()
        outputs = []
        offset = 0
        while offset < len(ids):
            end = offset + rng.randint(1, 40)
            chunk = ids[offset:end]
            outputs.append(instance.feed(tokenizer.decode(chunk, skip_special_tokens=False), chunk))
            offset = end
        outputs.append(instance.finish())
        calls = calls_from_deltas(outputs)
        assert json.loads(calls[0][1]) == values
        assert calls[1] == ("other", '{"text":"second"}')
