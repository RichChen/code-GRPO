"""One agent episode: chat with a policy that calls tools, then grade the final workspace.

A policy is any callable ``policy(messages, tools) -> assistant message`` in OpenAI chat
format, so a training framework's sampler can be plugged in directly.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from arl.env.grader import grade
from arl.env.sandbox import SandboxConfig, sandbox_runner
from arl.env.task import Task
from arl.env.tools import TOOL_SCHEMAS, ToolArgumentError, Toolbox, parse_arguments, truncate_middle

Message = dict[str, Any]
Policy = Callable[[list[Message], list[dict[str, Any]]], Message]

TOOL_RESULT_MAX_CHARS = 4000
CONTEXT_MAX_TOKENS = 8000
CHARS_PER_TOKEN = 4
SYSTEM_PROMPT = (Path(__file__).with_name("prompt.md")).read_text(encoding="utf-8").strip()


def user_prompt(task: Task) -> str:
    return f"Fix this issue:\n\n{task.issue}"


def initial_messages(task: Task) -> list[Message]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(task)}]


def estimate_tokens(messages: Iterable[Message], tools: list[dict[str, Any]] | None = TOOL_SCHEMAS) -> int:
    """Rough context size: characters / 4 over message contents, tool calls and tool schemas."""
    chars = len(json.dumps(tools)) if tools else 0
    for m in messages:
        chars += len(m.get("content") or "")
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return chars // CHARS_PER_TOKEN


@dataclass
class ToolCallRecord:
    turn: int
    call_id: str
    name: str
    arguments: str
    ok: bool
    error_kind: str | None
    output_chars: int
    output_truncated: bool


@dataclass
class EpisodeResult:
    task_id: str
    messages: list[Message]
    tool_calls: list[ToolCallRecord]
    reward: int
    grade: dict[str, Any]
    turns: int
    stop_reason: str  # "submitted" | "max_turns" | "context_limit"
    truncated: bool
    context_tokens_est: int
    duration_s: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def format_errors(self) -> int:
        return sum(1 for c in self.tool_calls if c.error_kind in FORMAT_ERROR_KINDS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        data = dict(data)
        data["tool_calls"] = [ToolCallRecord(**c) for c in data["tool_calls"]]
        return cls(**data)


FORMAT_ERROR_KINDS = frozenset({"unknown_tool", "invalid_json", "invalid_arguments", "malformed_call"})


def _normalize_assistant(message: Message, turn: int) -> Message:
    """Keep only the OpenAI fields we rely on; give every tool call an id and string arguments."""
    out: Message = {"role": "assistant", "content": message.get("content")}
    calls = []
    for k, call in enumerate(message.get("tool_calls") or []):
        call = call if isinstance(call, dict) else {}
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        args = fn.get("arguments")
        if args is not None and not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append(
            {
                "id": str(call.get("id") or f"call_{turn}_{k}"),
                "type": "function",
                "function": {"name": fn.get("name") if isinstance(fn.get("name"), str) else "", "arguments": args or ""},
            }
        )
    if calls:
        out["tool_calls"] = calls
    return out


def _execute(toolbox: Toolbox, name: str, arguments: str) -> tuple[bool, str, str | None]:
    if not name:
        return False, "Malformed tool call: missing function name.", "malformed_call"
    try:
        args = parse_arguments(name, arguments)
    except ToolArgumentError as exc:
        return False, str(exc), exc.kind
    result = getattr(toolbox, name)(**args)
    return result.ok, result.content, None if result.ok else "tool_error"


def run_episode(
    task: Task,
    policy: Policy,
    max_turns: int = 8,
    *,
    sandbox: SandboxConfig | None = None,
    context_max_tokens: int = CONTEXT_MAX_TOKENS,
    workspace_dir: str | Path | None = None,
    keep_workspace: bool = False,
) -> EpisodeResult:
    """Run one episode and grade it.

    Each turn is one policy call. The episode ends when the policy answers without tool
    calls (a submission), after ``max_turns`` turns, or when the estimated context
    exceeds ``context_max_tokens`` (``truncated=True``). The workspace is graded in all cases.
    """
    start = time.monotonic()
    workspace = Path(tempfile.mkdtemp(prefix="arl-ep-", dir=workspace_dir))
    try:
        task.materialize(workspace)
        toolbox = Toolbox(workspace, test_runner=sandbox_runner(sandbox))
        messages = initial_messages(task)
        records: list[ToolCallRecord] = []
        stop_reason = "max_turns"
        truncated = False
        turns = 0

        while turns < max_turns:
            if estimate_tokens(messages) > context_max_tokens:
                stop_reason, truncated = "context_limit", True
                break
            reply = _normalize_assistant(policy([dict(m) for m in messages], TOOL_SCHEMAS), turns)
            turns += 1
            messages.append(reply)
            if not reply.get("tool_calls"):
                stop_reason = "submitted"
                break
            for call in reply["tool_calls"]:
                fn = call["function"]
                ok, content, error_kind = _execute(toolbox, fn["name"], fn["arguments"])
                clipped = truncate_middle(content, TOOL_RESULT_MAX_CHARS)
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": fn["name"], "content": clipped})
                records.append(
                    ToolCallRecord(
                        turn=turns,
                        call_id=call["id"],
                        name=fn["name"],
                        arguments=fn["arguments"],
                        ok=ok,
                        error_kind=error_kind,
                        output_chars=len(content),
                        output_truncated=len(clipped) < len(content),
                    )
                )
        if stop_reason == "max_turns" and estimate_tokens(messages) > context_max_tokens:
            truncated = True

        result = grade(task, workspace, sandbox)
        return EpisodeResult(
            task_id=task.task_id,
            messages=messages,
            tool_calls=records,
            reward=result.reward,
            grade=result.to_dict(),
            turns=turns,
            stop_reason=stop_reason,
            truncated=truncated,
            context_tokens_est=estimate_tokens(messages),
            duration_s=round(time.monotonic() - start, 3),
            meta=dict(task.meta),
        )
    finally:
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def write_jsonl(results: Iterable[EpisodeResult], path: str | Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(r.to_json() + "\n")


class ScriptedPolicy:
    """A deterministic policy for tests: replays a fixed list of turns.

    Each turn is either a list of ``(tool_name, arguments)`` pairs, sent as tool calls in
    one assistant message, or a string, sent as a final answer without tool calls. After
    the script runs out the policy submits with an empty answer.
    """

    def __init__(self, turns: list[list[tuple[str, dict[str, Any] | str]] | str]):
        self.turns = list(turns)
        self.calls = 0
        self.seen: list[list[Message]] = []

    def __call__(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        self.seen.append(messages)
        turn = self.turns[self.calls] if self.calls < len(self.turns) else "done"
        self.calls += 1
        if isinstance(turn, str):
            return {"role": "assistant", "content": turn}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{self.calls}_{k}",
                    "type": "function",
                    "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
                }
                for k, (name, args) in enumerate(turn)
            ],
        }
