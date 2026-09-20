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

_START_TERMINAL = "think_start"
_END_TERMINAL = "think_end"
_DROP_TERMINAL = "drop"


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
class ParsedDelta:
    """New semantic text released by one incremental parser feed."""

    reasoning: str = ""
    content: str = ""


@dataclass(frozen=True)
class ParsedOutput:
    """Complete semantic output for one non-streaming generation request."""

    reasoning: str = ""
    content: str = ""


class OutputParser(Protocol):
    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> ParsedDelta: ...

    def finish(self) -> ParsedDelta: ...

    def parse_complete(
        self,
        text: str,
        token_ids: Sequence[int],
    ) -> ParsedOutput: ...


@dataclass(frozen=True)
class _TextSegment:
    text: str


@dataclass(frozen=True)
class _TerminalSegment:
    kind: str
    text: str


_Segment = _TextSegment | _TerminalSegment


class _TokenTerminalScanner:
    """Align token-ID terminals with context-dependent detokenizer text."""

    def __init__(self, tokenizer, terminals: dict[int, str]) -> None:
        self._tokenizer = tokenizer
        self._terminals = dict(terminals)
        self._terminal_text = {
            token_id: tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in terminals
        }
        self._deferred_terminals: list[_TerminalSegment] = []
        self._deferred_text = ""

    def reset(self) -> None:
        self._deferred_terminals.clear()
        self._deferred_text = ""

    def scan(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[_Segment]:
        prefix, effective_text = self._resolve_deferred(delta_text)
        if not any(int(token_id) in self._terminals for token_id in delta_token_ids):
            if effective_text:
                prefix.append(_TextSegment(effective_text))
            return prefix

        decoded: list[_Segment] = []
        ordinary_ids: list[int] = []

        def flush_ordinary() -> None:
            if not ordinary_ids:
                return
            text = self._tokenizer.decode(
                ordinary_ids,
                skip_special_tokens=False,
            )
            ordinary_ids.clear()
            if text:
                decoded.append(_TextSegment(text))

        for raw_token_id in delta_token_ids:
            token_id = int(raw_token_id)
            terminal = self._terminals.get(token_id)
            if terminal is None:
                ordinary_ids.append(token_id)
                continue
            flush_ordinary()
            terminal_text = self._terminal_text[token_id]
            # A dropped special with no textual representation cannot affect
            # parser state or text alignment.
            if terminal != _DROP_TERMINAL or terminal_text:
                decoded.append(_TerminalSegment(terminal, terminal_text))
        flush_ordinary()

        if not effective_text:
            self._defer_terminals(decoded)
            return prefix

        prefix.extend(self._align_to_text(effective_text, decoded))
        return prefix

    def finish(self) -> list[_Segment]:
        unresolved = [
            terminal
            for terminal in self._deferred_terminals
            if terminal.kind != _DROP_TERMINAL
        ]
        if unresolved:
            names = ", ".join(terminal.kind for terminal in unresolved)
            raise ValueError(f"reasoning terminal text was not released: {names}")
        result = [_TextSegment(self._deferred_text)] if self._deferred_text else []
        self.reset()
        return result

    def _resolve_deferred(self, delta_text: str) -> tuple[list[_Segment], str]:
        if not self._deferred_terminals:
            if self._deferred_text:
                delta_text = self._deferred_text + delta_text
                self._deferred_text = ""
            return [], delta_text

        deferred = self._deferred_terminals
        self._deferred_terminals = []
        remaining = self._deferred_text + delta_text
        self._deferred_text = ""
        result: list[_Segment] = []

        for index, terminal in enumerate(deferred):
            if terminal.kind == _DROP_TERMINAL and not terminal.text:
                continue
            position = remaining.find(terminal.text)
            if position < 0:
                if terminal.kind == _DROP_TERMINAL:
                    continue
                self._deferred_text = remaining
                self._deferred_terminals.extend(deferred[index:])
                return result, ""
            if position:
                result.append(_TextSegment(remaining[:position]))
            result.append(terminal)
            remaining = remaining[position + len(terminal.text) :]
        return result, remaining

    def _align_to_text(
        self,
        delta_text: str,
        decoded: list[_Segment],
    ) -> list[_Segment]:
        reconstructed = "".join(segment.text for segment in decoded)
        if reconstructed:
            position = delta_text.find(reconstructed)
            if position >= 0:
                result: list[_Segment] = []
                if position:
                    result.append(_TextSegment(delta_text[:position]))
                result.extend(decoded)
                suffix_start = position + len(reconstructed)
                if suffix_start < len(delta_text):
                    result.append(_TextSegment(delta_text[suffix_start:]))
                return result

        # Context-dependent decoding can make individually decoded token text
        # differ from the real delta. Rebuild from terminal text anchors,
        # binding right-to-left so a preceding literal lookalike cannot steal
        # a real special-token marker.
        anchors = [
            segment for segment in decoded if isinstance(segment, _TerminalSegment)
        ]
        if not anchors:
            return [_TextSegment(delta_text)]

        positions = [-1] * len(anchors)
        search_end = len(delta_text)
        for index in range(len(anchors) - 1, -1, -1):
            anchor = anchors[index]
            if not anchor.text:
                continue
            position = delta_text.rfind(anchor.text, 0, search_end)
            if position >= 0:
                positions[index] = position
                search_end = position

        result: list[_Segment] = []
        consumed = 0
        for index, anchor in enumerate(anchors):
            position = positions[index]
            if position >= consumed:
                if position > consumed:
                    result.append(_TextSegment(delta_text[consumed:position]))
                result.append(anchor)
                consumed = position + len(anchor.text)
                continue

            if anchor.kind == _DROP_TERMINAL:
                continue
            later_anchor_resolved = any(
                later >= 0 for later in positions[index + 1 :]
            )
            if not later_anchor_resolved:
                self._deferred_text = delta_text[consumed:]
                consumed = len(delta_text)
            self._deferred_terminals.append(anchor)

        if consumed < len(delta_text):
            result.append(_TextSegment(delta_text[consumed:]))
        return result

    def _defer_terminals(self, decoded: Sequence[_Segment]) -> None:
        self._deferred_terminals.extend(
            segment
            for segment in decoded
            if isinstance(segment, _TerminalSegment)
            and (segment.kind != _DROP_TERMINAL or segment.text)
        )


class DeepSeekV4ReasoningParser:
    """Incrementally split DeepSeek V4 reasoning from answer content."""

    def __init__(self, tokenizer, spec: OutputParserSpec) -> None:
        self._tokenizer = tokenizer
        self._initial_state = spec.initial_state
        self._include_reasoning = spec.include_reasoning
        vocab = tokenizer.get_vocab()
        think_start_id = self._require_terminal(vocab, THINK_START)
        think_end_id = self._require_terminal(vocab, THINK_END)
        if think_start_id == think_end_id:
            raise ValueError("DeepSeek V4 reasoning terminals must use distinct token IDs")

        terminals = {
            think_start_id: _START_TERMINAL,
            think_end_id: _END_TERMINAL,
        }
        for raw_token_id in getattr(tokenizer, "all_special_ids", ()):
            token_id = int(raw_token_id)
            terminals.setdefault(token_id, _DROP_TERMINAL)
        self._scanner = _TokenTerminalScanner(tokenizer, terminals)
        self._state = self._initial_state

    @staticmethod
    def _require_terminal(vocab: dict[str, int], terminal: str) -> int:
        token_id = vocab.get(terminal)
        if type(token_id) is not int or token_id < 0:
            raise ValueError(
                f"DeepSeek V4 tokenizer does not expose reasoning terminal {terminal!r}"
            )
        return token_id

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> ParsedDelta:
        return self._consume(self._scanner.scan(delta_text, delta_token_ids))

    def finish(self) -> ParsedDelta:
        return self._consume(self._scanner.finish())

    def parse_complete(
        self,
        text: str,
        token_ids: Sequence[int],
    ) -> ParsedOutput:
        self._reset()
        first = self.feed(text, token_ids)
        final = self.finish()
        return ParsedOutput(
            reasoning=first.reasoning + final.reasoning,
            content=first.content + final.content,
        )

    def _reset(self) -> None:
        self._state = self._initial_state
        self._scanner.reset()

    def _consume(self, segments: Sequence[_Segment]) -> ParsedDelta:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for segment in segments:
            if isinstance(segment, _TextSegment):
                if not segment.text:
                    continue
                if self._state == "reasoning":
                    if self._include_reasoning:
                        reasoning_parts.append(segment.text)
                else:
                    content_parts.append(segment.text)
                continue

            if segment.kind == _START_TERMINAL:
                # Duplicate starts are absorbed, as in vLLM.
                self._state = "reasoning"
            elif segment.kind == _END_TERMINAL:
                # A bare end in content is absorbed; reasoning ends otherwise.
                if self._state == "reasoning":
                    self._state = "content"

        return ParsedDelta(
            reasoning="".join(reasoning_parts),
            content="".join(content_parts),
        )


def create_output_parser(spec: OutputParserSpec | None, tokenizer) -> OutputParser | None:
    """Create one request-local parser from a normalized Serving contract."""
    if spec is None:
        return None
    if spec.parser_id == "deepseek_v4":
        return DeepSeekV4ReasoningParser(tokenizer, spec)
    raise ValueError(f"unsupported output parser {spec.parser_id!r}")
