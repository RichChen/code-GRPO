"""The five agent tools: read_file, search_code, list_dir, edit_file, run_tests.

There is deliberately no write_file and no shell: a small action space is easier
for small models to learn and leaves less surface for reward hacking.

Every tool returns ``ToolResult(ok, content)``. Errors never raise; ``content``
then holds a one-sentence explanation meant for the model.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

READ_DEFAULT_LIMIT = 200
SEARCH_MAX_RESULTS = 50
SEARCH_LINE_MAX_CHARS = 200
SEARCH_MAX_FILE_BYTES = 1_000_000
TEST_OUTPUT_MAX_CHARS = 4000
DEFAULT_TEST_TARGET = "tests"
HIDDEN_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".git"})


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str


@dataclass(frozen=True)
class RunOutput:
    """What a test runner reports back to ``run_tests``."""

    exit_code: int | None
    output: str
    timed_out: bool = False


TestRunner = Callable[[Path, list[str]], RunOutput]


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function(
        "read_file",
        f"Read a text file in the repository. Lines are numbered. At most {READ_DEFAULT_LIMIT} lines "
        "are returned per call; use offset to continue.",
        {
            "path": {"type": "string", "description": "File path relative to the repository root."},
            "offset": {"type": "integer", "description": "First line to return, starting at 1. Default 1."},
            "limit": {"type": "integer", "description": f"Maximum number of lines to return (max {READ_DEFAULT_LIMIT})."},
        },
        ["path"],
    ),
    _function(
        "search_code",
        f"Search files with a Python regular expression. Returns up to {SEARCH_MAX_RESULTS} lines "
        "formatted as path:line: text.",
        {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "File or directory to search, relative to the root. Default is the root."},
            "glob": {"type": "string", "description": "Only search files whose name matches this glob, e.g. *.py."},
        },
        ["pattern"],
    ),
    _function(
        "list_dir",
        "List one level of a directory. Directory names end with /.",
        {"path": {"type": "string", "description": "Directory relative to the root. Default is the root."}},
        [],
    ),
    _function(
        "edit_file",
        "Replace old_string with new_string in a file. old_string must appear exactly once in the file; "
        "include enough surrounding lines to make it unique.",
        {
            "path": {"type": "string", "description": "File path relative to the repository root."},
            "old_string": {"type": "string", "description": "Exact text to replace, including indentation."},
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        ["path", "old_string", "new_string"],
    ),
    _function(
        "run_tests",
        "Run the visible tests with pytest and return the exit code and output.",
        {"target": {"type": "string", "description": f"Test file, directory or node id. Default {DEFAULT_TEST_TARGET}."}},
        [],
    ),
]

TOOL_NAMES: tuple[str, ...] = tuple(s["function"]["name"] for s in TOOL_SCHEMAS)
_SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}


class ToolArgumentError(ValueError):
    """Raised by ``parse_arguments`` when a tool call is malformed."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def parse_arguments(name: str, arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    """Validate a tool call against its schema. Raises ``ToolArgumentError``.

    ``kind`` is one of ``unknown_tool``, ``invalid_json``, ``invalid_arguments``.
    """
    if name not in _SCHEMA_BY_NAME:
        raise ToolArgumentError("unknown_tool", f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_NAMES)}.")
    if arguments is None or arguments == "":
        args: Any = {}
    elif isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolArgumentError("invalid_json", f"Arguments for {name} are not valid JSON: {exc.msg}.") from None
    else:
        args = arguments
    if not isinstance(args, dict):
        raise ToolArgumentError("invalid_arguments", f"Arguments for {name} must be a JSON object.")

    schema = _SCHEMA_BY_NAME[name]
    props: dict[str, Any] = schema["properties"]
    unexpected = sorted(set(args) - set(props))
    if unexpected:
        raise ToolArgumentError(
            "invalid_arguments",
            f"Unexpected argument(s) for {name}: {', '.join(unexpected)}. Allowed: {', '.join(props)}.",
        )
    missing = [k for k in schema["required"] if k not in args or args[k] is None]
    if missing:
        raise ToolArgumentError("invalid_arguments", f"Missing required argument(s) for {name}: {', '.join(missing)}.")

    clean: dict[str, Any] = {}
    for key, value in args.items():
        if value is None:
            continue
        expected = props[key]["type"]
        if expected == "string":
            if not isinstance(value, str):
                raise ToolArgumentError("invalid_arguments", f"Argument '{key}' of {name} must be a string.")
        elif expected == "integer":
            if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolArgumentError("invalid_arguments", f"Argument '{key}' of {name} must be an integer.")
        clean[key] = value
    return clean


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of ``text`` so that the result has at most ``max_chars`` characters."""
    if len(text) <= max_chars:
        return text
    marker = "\n... [{} characters omitted] ...\n"
    omitted = len(text) - max_chars
    while True:
        m = marker.format(omitted)
        keep = max_chars - len(m)
        if keep <= 0:
            return text[:max_chars]
        new_omitted = len(text) - keep
        if new_omitted == omitted:
            break
        omitted = new_omitted
    head = keep - keep // 2
    tail = keep // 2
    return text[:head] + m + (text[-tail:] if tail else "")


def pytest_argv(target: str) -> list[str]:
    # -B: edits that keep size and mtime-second unchanged would otherwise reuse stale .pyc files.
    return [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no", target]


class Toolbox:
    """Tool implementations bound to one workspace root."""

    def __init__(self, root: str | Path, test_runner: TestRunner | None = None):
        self.root = Path(os.path.realpath(root))
        if not self.root.is_dir():
            raise ValueError(f"workspace root is not a directory: {root}")
        if test_runner is None:
            from arl.env.sandbox import sandbox_runner

            test_runner = sandbox_runner()
        self.test_runner = test_runner

    # -- dispatch -----------------------------------------------------------------

    def call(self, name: str, arguments: str | dict[str, Any] | None) -> ToolResult:
        try:
            args = parse_arguments(name, arguments)
        except ToolArgumentError as exc:
            return ToolResult(False, str(exc))
        return getattr(self, name)(**args)

    # -- path handling --------------------------------------------------------------

    def _resolve(self, raw: str | None) -> tuple[Path | None, str]:
        """Map a model-supplied path to a real path inside the root, or return an error message."""
        raw = (raw or ".").strip() or "."
        if "\x00" in raw:
            return None, "Path contains a NUL byte."
        if os.path.isabs(raw) or raw.startswith("~"):
            return None, f"Path '{raw}' is absolute; use a path relative to the repository root."
        real = Path(os.path.realpath(self.root / raw))
        if real != self.root and not real.is_relative_to(self.root):
            return None, f"Path '{raw}' points outside the repository."
        return real, ""

    def _rel(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return rel or "."

    # -- tools ------------------------------------------------------------------------

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
        real, err = self._resolve(path)
        if real is None:
            return ToolResult(False, err)
        if not real.exists():
            return ToolResult(False, f"File not found: {path}.")
        if real.is_dir():
            return ToolResult(False, f"'{path}' is a directory; use list_dir instead.")
        try:
            text = real.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(False, f"'{path}' is not a UTF-8 text file.")
        except OSError as exc:
            return ToolResult(False, f"Cannot read '{path}': {exc.strerror}.")

        lines = text.splitlines()
        total = len(lines)
        start = 1 if offset is None else offset
        if start < 1:
            return ToolResult(False, "offset must be at least 1.")
        if limit is not None and limit < 1:
            return ToolResult(False, "limit must be at least 1.")
        count = READ_DEFAULT_LIMIT if limit is None else min(limit, READ_DEFAULT_LIMIT)
        if total == 0:
            return ToolResult(True, "(empty file)")
        if start > total:
            return ToolResult(False, f"offset {start} is past the end of the file ({total} lines).")
        end = min(total, start - 1 + count)
        width = len(str(end))
        body = "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(start, end + 1))
        remaining = total - end
        if remaining > 0:
            body += f"\n[{remaining} more lines; call read_file with offset={end + 1} to continue]"
        return ToolResult(True, body)

    def search_code(self, pattern: str, path: str | None = None, glob: str | None = None) -> ToolResult:
        if not pattern:
            return ToolResult(False, "pattern must not be empty.")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(False, f"Invalid regular expression: {exc}.")
        real, err = self._resolve(path)
        if real is None:
            return ToolResult(False, err)
        if not real.exists():
            return ToolResult(False, f"Path not found: {path}.")

        matches: list[str] = []
        total = 0
        for file in self._iter_files(real):
            if glob and not fnmatch.fnmatch(file.name, glob) and not fnmatch.fnmatch(self._rel(file), glob):
                continue
            try:
                if file.stat().st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                text = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = self._rel(file)
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    total += 1
                    if len(matches) < SEARCH_MAX_RESULTS:
                        if len(line) > SEARCH_LINE_MAX_CHARS:
                            line = line[:SEARCH_LINE_MAX_CHARS] + " ..."
                        matches.append(f"{rel}:{number}: {line}")
        if not matches:
            return ToolResult(True, "No matches found.")
        body = "\n".join(matches)
        if total > len(matches):
            body += f"\n[{total - len(matches)} more matches not shown; narrow the pattern or path]"
        return ToolResult(True, body)

    def _iter_files(self, start: Path):
        """Regular files under ``start`` whose real path stays inside the root, in sorted order."""
        if start.is_file():
            yield start
            return
        for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in HIDDEN_DIR_NAMES)
            for name in sorted(filenames):
                file = Path(dirpath) / name
                real = Path(os.path.realpath(file))
                if real.is_relative_to(self.root) and real.is_file():
                    yield file

    def list_dir(self, path: str | None = None) -> ToolResult:
        real, err = self._resolve(path)
        if real is None:
            return ToolResult(False, err)
        if not real.exists():
            return ToolResult(False, f"Directory not found: {path}.")
        if not real.is_dir():
            return ToolResult(False, f"'{path}' is a file, not a directory; use read_file instead.")
        entries = []
        for child in sorted(real.iterdir(), key=lambda p: p.name):
            if child.name in HIDDEN_DIR_NAMES:
                continue
            entries.append(child.name + "/" if child.is_dir() else child.name)
        return ToolResult(True, "\n".join(entries) if entries else "(empty directory)")

    def edit_file(self, path: str, old_string: str, new_string: str) -> ToolResult:
        real, err = self._resolve(path)
        if real is None:
            return ToolResult(False, err)
        if not real.exists():
            return ToolResult(False, f"File not found: {path}. edit_file cannot create files.")
        if real.is_dir():
            return ToolResult(False, f"'{path}' is a directory.")
        if old_string == "":
            return ToolResult(False, "old_string must not be empty.")
        if old_string == new_string:
            return ToolResult(False, "old_string and new_string are identical; nothing to change.")
        try:
            text = real.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(False, f"'{path}' is not a UTF-8 text file.")
        except OSError as exc:
            return ToolResult(False, f"Cannot read '{path}': {exc.strerror}.")
        count = text.count(old_string)
        if count == 0:
            return ToolResult(False, f"old_string was not found in {path}. Read the file and copy the text exactly.")
        if count > 1:
            return ToolResult(
                False, f"old_string appears {count} times in {path}; it must appear exactly once. Include more context."
            )
        idx = text.index(old_string)
        new_text = text[:idx] + new_string + text[idx + len(old_string):]
        try:
            real.write_text(new_text, encoding="utf-8")
        except OSError as exc:
            return ToolResult(False, f"Cannot write '{path}': {exc.strerror}.")
        first = text.count("\n", 0, idx) + 1
        last = first + new_string.count("\n")
        return ToolResult(True, f"Edited {self._rel(real)}: replaced 1 occurrence (now lines {first}-{last}).")

    def run_tests(self, target: str | None = None) -> ToolResult:
        target = (target or DEFAULT_TEST_TARGET).strip() or DEFAULT_TEST_TARGET
        if target.startswith("-"):
            return ToolResult(False, "target must be a path or node id, not an option.")
        file_part = target.split("::", 1)[0]
        real, err = self._resolve(file_part)
        if real is None:
            return ToolResult(False, err)
        if not real.exists():
            return ToolResult(False, f"Test target not found: {file_part}.")
        result = self.test_runner(self.root, pytest_argv(target))
        if result.timed_out:
            header = "Tests timed out and were killed."
        else:
            header = f"exit code: {result.exit_code}"
        body = truncate_middle(result.output, TEST_OUTPUT_MAX_CHARS - len(header) - 1)
        return ToolResult(True, f"{header}\n{body}")
