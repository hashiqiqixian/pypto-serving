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
import re
from collections.abc import Mapping, Sequence


BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
USER_TOKEN = "<\uff5cUser\uff5c>"
ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
LATEST_REMINDER_TOKEN = "<\uff5clatest_reminder\uff5c>"
THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"
DSML = "\uff5cDSML\uff5c"

# Prompt wording follows vLLM's DeepSeek V4 reference encoding (Apache-2.0).
# Copyright contributors to the vLLM project.
_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

_MAX_REASONING_EFFORT_PROMPT = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve "
    "the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and "
    "adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, "
    "considered alternative, and rejected hypothesis to ensure absolutely no assumption is left "
    "unchecked.\n\n"
)


def encode_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    thinking: bool = False,
    reasoning_effort: str | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    drop_thinking: bool = True,
) -> str:
    """Encode text conversations and OpenAI function calls in DeepSeek V4 format."""
    if not messages:
        raise ValueError("DeepSeek V4 chat requests require at least one message")

    if reasoning_effort == "none":
        thinking = False

    messages = _merge_tool_results(messages)
    if tools:
        messages = [dict(message) for message in messages]
        system_index = next((i for i, msg in enumerate(messages) if msg.get("role") == "system"), None)
        if system_index is None:
            messages.insert(0, {"role": "system", "content": ""})
            system_index = 0
        messages[system_index]["tools"] = tools
        # V4 keeps thinking history when tools are available, as in its reference encoding.
        drop_thinking = False

    parts = [BOS_TOKEN]
    if thinking and reasoning_effort in {"max", "xhigh"}:
        parts.append(_MAX_REASONING_EFFORT_PROMPT)

    last_user_index = max(
        (
            index for index, message in enumerate(messages)
            if message.get("role") in {"user", "developer"} and not message.get("_tool_result")
        ),
        default=-1,
    )
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls and content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("DeepSeek V4 chat message content must be a string")

        if role == "system":
            parts.append(content)
            if message.get("tools"):
                schemas = "\n".join(
                    json.dumps(tool["function"], ensure_ascii=False, allow_nan=False)
                    for tool in message["tools"]
                )
                parts.extend(("\n\n", _TOOLS_TEMPLATE.format(dsml=DSML, tool_schemas=schemas)))
        elif role in {"user", "developer"}:
            parts.extend((USER_TOKEN, content))
        elif role == "latest_reminder":
            parts.extend((LATEST_REMINDER_TOKEN, content))
        elif role == "assistant":
            if thinking and (not drop_thinking or index > last_user_index):
                parts.extend((message.get("reasoning") or "", THINKING_END_TOKEN))
            parts.append(content)
            if tool_calls:
                parts.append(_encode_tool_calls(tool_calls))
            parts.append(EOS_TOKEN)
        else:
            raise ValueError(f"DeepSeek V4 does not support chat message role {role!r}")

        next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
        if role in {"user", "developer", "system"} and (next_role == "assistant" or next_role is None):
            parts.append(ASSISTANT_TOKEN)
            parts.append(
                THINKING_START_TOKEN
                if thinking and (not drop_thinking or index >= last_user_index)
                else THINKING_END_TOKEN
            )

    return "".join(parts)


def _encode_tool_calls(tool_calls: Sequence[Mapping[str, object]]) -> str:
    invokes = []
    for call in tool_calls:
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
            raise ValueError("tool function names must be 1-64 ASCII letters, digits, underscores, or hyphens")
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must encode an object")
        parameters = []
        for key, value in arguments.items():
            if '"' in key or "<" in key or ">" in key:
                raise ValueError("tool parameter names cannot contain DSML attribute delimiters")
            is_string = isinstance(value, str)
            encoded = value if is_string else json.dumps(value, ensure_ascii=False, allow_nan=False)
            if f"</{DSML}parameter>" in encoded:
                raise ValueError("tool parameter values cannot contain the DSML parameter terminator")
            parameters.append(
                f'<{DSML}parameter name="{key}" string="{str(is_string).lower()}">'
                f'{encoded}</{DSML}parameter>'
            )
        invokes.append(
            f'<{DSML}invoke name="{name}">\n'
            + "\n".join(parameters) + f"\n</{DSML}invoke>"
        )
    return f"\n\n<{DSML}tool_calls>\n" + "\n".join(invokes) + f"\n</{DSML}tool_calls>"


def _merge_tool_results(messages: Sequence[Mapping[str, object]]) -> list[dict]:
    """Render contiguous tool results in call order without mutating the input."""
    merged: list[dict] = []
    call_order: dict[str, int] = {}
    results: dict[str, str] = {}

    def flush_results() -> None:
        if results:
            content = "\n\n".join(
                f"<tool_result>{results[call_id]}</tool_result>"
                for call_id in sorted(results, key=call_order.__getitem__)
            )
            merged.append({"role": "user", "content": content, "_tool_result": True})
            results.clear()

    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in call_order:
                raise ValueError("tool message must reference a preceding assistant tool_call_id")
            if call_id in results:
                raise ValueError("duplicate tool result for tool_call_id")
            if not isinstance(message.get("content"), str):
                raise ValueError("DeepSeek V4 tool result content must be a string")
            if "</tool_result>" in message["content"]:
                raise ValueError("tool result content cannot contain </tool_result>")
            results[call_id] = message["content"]
            continue
        flush_results()
        call_order = {
            call["id"]: index for index, call in enumerate(message.get("tool_calls") or ())
        }
        if message.get("role") == "user" and merged and merged[-1].get("_tool_result"):
            merged[-1]["content"] += "\n\n" + message["content"]
            merged[-1].pop("_tool_result")
        else:
            merged.append(dict(message))
    flush_results()
    return merged
