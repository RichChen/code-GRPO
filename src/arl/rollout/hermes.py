"""Hermes tool-call parsing exactly as vLLM 0.31's chat endpoint does it (``--tool-call-parser hermes``,
``tool_choice="auto"``), so a client that samples raw tokens sees the same assistant messages.

Sources: ``vllm/tool_parsers/hermes_tool_parser.py`` (``Hermes2ProToolParser.extract_tool_calls``) and
``vllm/parser/abstract_parser.py`` (``_extract_tool_calls``, the auto branch):

- no ``<tool_call>`` in the text: no tool calls, content is the whole text;
- otherwise every ``<tool_call>…</tool_call>`` block (or an unclosed last one) must be JSON with ``name`` and
  ``arguments``; any failure means no tool calls and the whole text as content;
- arguments are re-serialised with ``json.dumps(..., ensure_ascii=False)``;
- content is the text before the first ``<tool_call>``, ``None`` when empty or whitespace only.
"""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)


def parse(text: str) -> tuple[str | None, list[tuple[str, str]]]:
    """``(content, [(name, arguments_json), ...])`` as the chat endpoint would return them."""
    if TOOL_CALL_START not in text:
        return text, []
    try:
        raw: list[Any] = [json.loads(a if a else b) for a, b in TOOL_CALL_REGEX.findall(text)]
        calls = [(c["name"], json.dumps(c["arguments"], ensure_ascii=False)) for c in raw]
    except Exception:
        return text, []
    if any(not isinstance(name, str) for name, _ in calls):  # vLLM's FunctionCall(name: str) rejects these
        return text, []
    content: str | None = text[: text.find(TOOL_CALL_START)] or None
    if content is not None and not content.strip():
        content = None
    return content, calls
