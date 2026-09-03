# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Enforce the one-way dependency from skills to public documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SKILL_LINK = re.compile(r"(?:\.claude|\.agents)/skills", re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
REFERENCE_LINK = re.compile(r"^[ \t]{0,3}\[[^\]]+\]:[ \t]*(\S+)", re.MULTILINE)
_FENCE = re.compile(r"^(\s*)(```+|~~~+)", re.MULTILINE)


def _non_code_segments(markdown: str):
    """Yield segments of markdown outside properly-paired fenced code blocks.

    Matches opening and closing fence types (`` ``` `` vs `` ~~~ ``) and
    enforces that the closing fence is at least as long as the opening fence,
    per the CommonMark spec.
    """
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    open_fence: str | None = None
    for match in _FENCE.finditer(markdown):
        fence = match.group(2)
        if open_at is None:
            open_at = match.start()
            open_fence = fence
        elif fence[0] == open_fence[0] and len(fence) >= len(open_fence):
            spans.append((open_at, match.end()))
            open_at = None
            open_fence = None
    if open_at is not None:
        spans.append((open_at, len(markdown)))

    pos = 0
    for start, end in spans:
        if pos < start:
            yield markdown[pos:start]
        pos = end
    if pos < len(markdown):
        yield markdown[pos:]


def _public_markdown_files():
    yield ROOT / "README.md"
    yield from sorted((ROOT / "docs").rglob("*.md"))


def _check_public_dependencies() -> list[str]:
    failures = []
    for path in _public_markdown_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if SKILL_LINK.search(line):
                relative = path.relative_to(ROOT)
                failures.append(
                    f"{relative}:{line_number}: public documentation must not depend on skills"
                )
    return failures


def _link_targets(markdown: str):
    for segment in _non_code_segments(markdown):
        yield from MARKDOWN_LINK.findall(segment)
        yield from REFERENCE_LINK.findall(segment)


def _check_public_link_targets() -> list[str]:
    failures = []
    for path in _public_markdown_files():
        for raw_target in _link_targets(path.read_text(encoding="utf-8")):
            target = raw_target.strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            if parsed.path.startswith("/"):
                failures.append(
                    f"{path.relative_to(ROOT)}: absolute documentation link: {raw_target}"
                )
                continue
            if not parsed.path:
                continue

            resolved = (path.parent / unquote(parsed.path)).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"{path.relative_to(ROOT)}: link escapes repository: {raw_target}"
                )
                continue
            if not resolved.exists():
                failures.append(
                    f"{path.relative_to(ROOT)}: missing link target: {raw_target}"
                )
    return failures


def main() -> int:
    failures = (
        _check_public_dependencies()
        + _check_public_link_targets()
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("Validated public boundaries and link targets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())