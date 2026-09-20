# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-local parsing of generated tokens into public semantic channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


THINK_START = "<think>"
THINK_END = "</think>"


@dataclass(frozen=True)
class OutputParserSpec:
    """Normalized, request-local contract for Serving output parsing."""

    parser_id: str
    initial_state: Literal["content", "reasoning"]
    include_reasoning: bool = True

    def __post_init__(self) -> None:
        if self.parser_id != "deepseek_v4":
            raise ValueError(f"unsupported output parser {self.parser_id!r}")
        if self.initial_state not in ("content", "reasoning"):
            raise ValueError("output parser initial_state must be content or reasoning")


@dataclass(frozen=True)
class ParsedOutput:
    """Cumulative semantic output for one generation request."""

    reasoning: str = ""
    content: str = ""


class OutputParser(Protocol):
    def parse(self, token_ids: Sequence[int]) -> ParsedOutput: ...


class DeepSeekV4ReasoningParser:
    """Split DeepSeek V4 reasoning from visible answer content.

    Transitions use the exact ``<think>`` and ``</think>`` special-token IDs,
    matching vLLM's DeepSeek V4 parser semantics. A literal marker assembled
    from ordinary text tokens therefore cannot change parser state.
    """

    def __init__(self, tokenizer, spec: OutputParserSpec) -> None:
        self._tokenizer = tokenizer
        self._initial_state = spec.initial_state
        self._include_reasoning = spec.include_reasoning
        vocab = tokenizer.get_vocab()
        self._think_start_id = self._require_terminal(vocab, THINK_START)
        self._think_end_id = self._require_terminal(vocab, THINK_END)
        if self._think_start_id == self._think_end_id:
            raise ValueError("DeepSeek V4 reasoning terminals must use distinct token IDs")

    @staticmethod
    def _require_terminal(vocab: dict[str, int], terminal: str) -> int:
        token_id = vocab.get(terminal)
        if type(token_id) is not int or token_id < 0:
            raise ValueError(
                f"DeepSeek V4 tokenizer does not expose reasoning terminal {terminal!r}"
            )
        return token_id

    def parse(self, token_ids: Sequence[int]) -> ParsedOutput:
        """Parse the authoritative cumulative token sequence.

        Replaying the small state machine makes speculative multi-token bursts,
        streaming, and the terminal output share one correctness path.
        """
        state = self._initial_state
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        pending: list[int] = []

        def flush() -> None:
            if not pending:
                return
            text = self._tokenizer.decode(pending, skip_special_tokens=True)
            pending.clear()
            if not text:
                return
            if state == "reasoning":
                reasoning_parts.append(text)
            else:
                content_parts.append(text)

        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id == self._think_start_id:
                flush()
                # Duplicate starts are absorbed, as in vLLM.
                state = "reasoning"
                continue
            if token_id == self._think_end_id:
                flush()
                # A bare end in content is absorbed; reasoning ends otherwise.
                if state == "reasoning":
                    state = "content"
                continue
            pending.append(token_id)
        flush()

        return ParsedOutput(
            reasoning="" if not self._include_reasoning else "".join(reasoning_parts),
            content="".join(content_parts),
        )


def create_output_parser(spec: OutputParserSpec | None, tokenizer) -> OutputParser | None:
    """Create one request-local parser from a normalized Serving contract."""
    if spec is None:
        return None
    if spec.parser_id == "deepseek_v4":
        return DeepSeekV4ReasoningParser(tokenizer, spec)
    raise ValueError(f"unsupported output parser {spec.parser_id!r}")
