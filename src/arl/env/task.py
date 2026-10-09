"""Task data model shared by the environment, the grader and task generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

VISIBLE_TEST_PATH = "tests/test_visible.py"


@dataclass(frozen=True)
class HiddenCase:
    """One hidden assertion, split so that the expected value never enters the sandbox.

    ``call`` is evaluated inside the sandbox (after ``Task.hidden_setup``) and its value is
    serialised as plain data. ``check`` is evaluated on the host with ``result`` bound to
    the decoded value (or to a ``Raised`` marker) and ``error`` bound to the exception
    type name or ``None``.
    """

    call: str
    check: str


@dataclass
class Task:
    task_id: str
    issue: str
    files: dict[str, str]
    hidden_setup: str
    hidden_cases: list[HiddenCase]
    solution: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hidden_cases = [c if isinstance(c, HiddenCase) else HiddenCase(**c) for c in self.hidden_cases]
        for path in list(self.files) + list(self.solution):
            _check_relpath(path)
        if VISIBLE_TEST_PATH not in self.files:
            raise ValueError(f"task {self.task_id} has no {VISIBLE_TEST_PATH}")
        if not self.hidden_cases:
            raise ValueError(f"task {self.task_id} has no hidden cases")

    def materialize(self, dest: str | Path, *, solved: bool = False) -> Path:
        """Write the task files (optionally with the reference solution applied) into ``dest``."""
        dest = Path(dest)
        files = dict(self.files)
        if solved:
            files.update(self.solution)
        for rel, content in files.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return dest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(**data)

    @classmethod
    def from_json(cls, line: str) -> "Task":
        return cls.from_dict(json.loads(line))


def _check_relpath(path: str) -> None:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in path:
        raise ValueError(f"task file path must be relative and stay inside the repo: {path!r}")


def load_tasks(path: str | Path) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        return [Task.from_json(line) for line in f if line.strip()]
