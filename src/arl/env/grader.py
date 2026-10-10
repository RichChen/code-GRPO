"""Binary reward from hidden tests the agent never sees.

Grading copies the agent's final workspace, strips every test and test/packaging
config file, restores the task's original config and runs the hidden cases in the
sandbox. Hidden cases are evaluated in two halves so that untrusted code cannot fake
a pass: inside the sandbox a harness only *computes* each case's value and prints it
as plain data; the expected values stay on the host, where the checks run. Exiting
early, hijacking pytest or printing forged records therefore cannot produce a reward
without computing the right answers.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arl.env.sandbox import SandboxConfig, run_in_sandbox
from arl.env.task import VISIBLE_TEST_PATH, Task
from arl.env.tools import pytest_argv

STRIP_FILE_NAMES = frozenset({"conftest.py", "pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml"})
CONFIG_FILE_NAMES = STRIP_FILE_NAMES
IGNORED_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache"})
HARNESS_FILE = "_arl_hidden.py"
SPEC_FILE = "_arl_hidden.json"

# Shared by the grading harness and the task-validation harness: silences the untrusted code's
# stdout/stderr, and defines enc() (value -> tagged plain data), ename() and emit().
HARNESS_PRELUDE = r'''
import json, os, sys

_dumps = json.dumps
_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
_null = os.open(os.devnull, os.O_WRONLY)
os.dup2(_null, 1)
os.dup2(_null, 2)

MAX_ITEMS = 100000
MAX_DEPTH = 100


def enc(v, depth, budget):
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        raise ValueError("value too large")
    d = depth + 1
    if v is None or type(v) is bool:
        return v
    if isinstance(v, int):
        return int.__index__(v)
    if isinstance(v, float):
        return float.__float__(v)
    if isinstance(v, str):
        return str.__str__(v)
    if isinstance(v, list):
        return {"L": [enc(x, d, budget) for x in list.__iter__(v)]}
    if isinstance(v, tuple):
        return {"T": [enc(x, d, budget) for x in tuple.__iter__(v)]}
    if isinstance(v, dict):
        return {"D": [[enc(k, d, budget), enc(x, d, budget)] for k, x in dict.items(v)]}
    if isinstance(v, frozenset):
        return {"F": [enc(x, d, budget) for x in frozenset.__iter__(v)]}
    if isinstance(v, set):
        return {"S": [enc(x, d, budget) for x in set.__iter__(v)]}
    if type(v) is bytes:
        return {"B": v.hex()}
    if type(v) is bytearray:
        return {"BA": v.hex()}
    if type(v) is complex:
        return {"C": [v.real, v.imag]}
    return {"O": type.__dict__["__name__"].__get__(type(v))}


def emit(rec):
    _out.write(_dumps(rec) + "\n")
    _out.flush()


def ename(e):
    try:
        return type.__dict__["__name__"].__get__(type(e))
    except BaseException:
        return "BaseException"
'''

_HARNESS = HARNESS_PRELUDE + r'''

def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    sys.path.insert(0, os.getcwd())
    ns = {"__name__": "__arl_hidden__"}
    try:
        exec(compile(spec["setup"], "<hidden_setup>", "exec"), ns)
    except BaseException as e:
        emit({"setup_error": ename(e)})
        return
    for i, call in enumerate(spec["calls"]):
        try:
            value = eval(compile(call, "<hidden_case>", "eval"), ns)
            rec = {"i": i, "ok": True, "v": enc(value, 0, [MAX_ITEMS])}
        except BaseException as e:
            rec = {"i": i, "ok": False, "error": ename(e)}
        emit(rec)


main()
'''


def harness_argv(*args: str) -> list[str]:
    # -S skips site (no sitecustomize / .pth from anywhere). Not -I: that implies -E, which would
    # drop the sandbox's PYTHONHASHSEED=0 and make set/dict-of-str iteration order vary between runs.
    return [sys.executable, "-S", "-B", *args]


class _Undecidable:
    """A value the host cannot compare. Any comparison or truth test raises, so the check fails."""

    __slots__ = ("label",)

    def __init__(self, label: str):
        self.label = label

    def _refuse(self, *args: Any) -> Any:
        raise TypeError(f"cannot compare {self!r}")

    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __bool__ = __hash__ = __len__ = __iter__ = _refuse

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.label}>"


class Opaque(_Undecidable):
    """The call returned an object of a type the harness does not serialise."""


class Raised(_Undecidable):
    """The call raised; ``label`` is the exception type name."""


def decode(x: Any) -> Any:
    if x is None or type(x) in (bool, int, float, str):
        return x
    if type(x) is dict and len(x) == 1:
        ((tag, val),) = x.items()
        if tag == "L":
            return [decode(i) for i in val]
        if tag == "T":
            return tuple(decode(i) for i in val)
        if tag == "S":
            return {decode(i) for i in val}
        if tag == "F":
            return frozenset(decode(i) for i in val)
        if tag == "D":
            return {decode(k): decode(v) for k, v in val}
        if tag == "B":
            return bytes.fromhex(val)
        if tag == "BA":
            return bytearray.fromhex(val)
        if tag == "C":
            return complex(val[0], val[1])
        if tag == "O":
            return Opaque(str(val)[:100])
    raise ValueError(f"cannot decode {type(x).__name__}")


def evaluate_check(check: str, result: Any, error: str | None) -> bool:
    try:
        return bool(eval(check, {"math": math}, {"result": result, "error": error}))
    except Exception:
        return False


@dataclass
class GradeResult:
    reward: int
    hidden_pass_count: int
    hidden_total: int
    visible_pass: bool | None
    timed_out: bool
    files_changed: list[str]
    isolation: str
    setup_error: str | None = None
    case_passed: list[bool] | None = None
    # analysis only, never part of the reward: {kind: [passed, total]} over task.meta["hidden_case_kinds"]
    hidden_by_kind: dict[str, list[int]] | None = None
    # every non-``generated`` (test + challenge) hidden case passed and the run did not time out
    pass_non_generated: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def kind_breakdown(kinds: list[str] | None, passed: list[bool], timed_out: bool) -> tuple[dict[str, list[int]] | None, bool | None]:
    """Per-kind pass counts and the "test + challenge only" verdict; ``(None, None)`` without kind labels."""
    if not kinds or len(kinds) != len(passed):
        return None, None
    by_kind: dict[str, list[int]] = {}
    for kind, ok in zip(kinds, passed):
        counts = by_kind.setdefault(kind, [0, 0])
        counts[0] += int(ok)
        counts[1] += 1
    non_generated = all(ok for kind, ok in zip(kinds, passed) if kind != "generated")
    return by_kind, non_generated and not timed_out


def _strip_test_files(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        here = Path(dirpath)
        for d in list(dirnames):
            p = here / d
            if d in IGNORED_DIR_NAMES:
                if p.is_symlink():
                    p.unlink()
                else:
                    shutil.rmtree(p)
                dirnames.remove(d)
            elif p.is_symlink() and (d.startswith("test_") and d.endswith(".py") or d in STRIP_FILE_NAMES):
                p.unlink()
                dirnames.remove(d)
        for name in filenames:
            if (name.startswith("test_") and name.endswith(".py")) or name in STRIP_FILE_NAMES or name.endswith(".pyc"):
                (here / name).unlink()


def _write(root: Path, rel: str, content: str) -> None:
    target = root / rel
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _clean_copy(task: Task, workspace: Path, dest: Path) -> Path:
    shutil.copytree(workspace, dest, symlinks=True)
    _strip_test_files(dest)
    for rel, content in task.files.items():
        if Path(rel).name in CONFIG_FILE_NAMES:
            _write(dest, rel, content)
    return dest


def changed_files(task: Task, workspace: str | Path) -> list[str]:
    """Paths whose content differs from the initial task files (added, modified or deleted)."""
    workspace = Path(workspace)
    current: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIR_NAMES]
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            p = Path(dirpath) / name
            current[p.relative_to(workspace).as_posix()] = p
    changed = set(current) ^ set(task.files)
    for rel in set(current) & set(task.files):
        p = current[rel]
        try:
            same = not p.is_symlink() and p.read_text(encoding="utf-8") == task.files[rel]
        except (OSError, UnicodeDecodeError):
            same = False
        if not same:
            changed.add(rel)
    return sorted(changed)


def run_hidden(task: Task, copy: Path, config: SandboxConfig) -> tuple[list[bool], str | None, bool, str]:
    _write(copy, HARNESS_FILE, _HARNESS)
    spec = {"setup": task.hidden_setup, "calls": [c.call for c in task.hidden_cases]}
    _write(copy, SPEC_FILE, json.dumps(spec))
    result = run_in_sandbox(copy, harness_argv(HARNESS_FILE, SPEC_FILE), config)

    n = len(task.hidden_cases)
    records: dict[int, dict[str, Any]] = {}
    duplicated: set[int] = set()
    setup_error = None
    for line in result.output.splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if type(rec) is not dict:
            continue
        if "setup_error" in rec:
            setup_error = str(rec["setup_error"])[:200]
            continue
        i = rec.get("i")
        if type(i) is not int or not 0 <= i < n:
            continue
        if i in records:
            duplicated.add(i)
        records[i] = rec

    passed: list[bool] = []
    for i, case in enumerate(task.hidden_cases):
        rec = records.get(i)
        if rec is None or i in duplicated:
            passed.append(False)
            continue
        try:
            if rec.get("ok") is True:
                value, error = decode(rec["v"]), None
            else:
                error = str(rec.get("error"))[:100]
                value = Raised(error)
        except Exception:
            passed.append(False)
            continue
        passed.append(evaluate_check(case.check, value, error))
    if setup_error is not None:
        passed = [False] * n
    return passed, setup_error, result.timed_out, result.isolation


def run_visible(task: Task, copy: Path, config: SandboxConfig) -> bool:
    _write(copy, VISIBLE_TEST_PATH, task.files[VISIBLE_TEST_PATH])
    result = run_in_sandbox(copy, pytest_argv(VISIBLE_TEST_PATH), config)
    return result.exit_code == 0


def grade(
    task: Task, workspace: str | Path, config: SandboxConfig | None = None, *, check_visible: bool = True
) -> GradeResult:
    """Grade the agent's final ``workspace`` for ``task``. The workspace itself is never modified.

    ``visible_pass`` reruns the task's *original* visible tests against the agent's code; it is
    recorded for analysis only and does not affect the reward.
    """
    cfg = config or SandboxConfig()
    workspace = Path(workspace)
    with tempfile.TemporaryDirectory(prefix="arl-grade-") as tmp:
        hidden_copy = _clean_copy(task, workspace, Path(tmp) / "hidden")
        passed, setup_error, timed_out, isolation = run_hidden(task, hidden_copy, cfg)
        visible = None
        if check_visible:
            visible_copy = _clean_copy(task, workspace, Path(tmp) / "visible")
            visible = run_visible(task, visible_copy, cfg)
    count = sum(passed)
    by_kind, non_generated = kind_breakdown(task.meta.get("hidden_case_kinds"), passed, timed_out)
    return GradeResult(
        reward=int(count == len(task.hidden_cases) and not timed_out),
        hidden_pass_count=count,
        hidden_total=len(task.hidden_cases),
        visible_pass=visible,
        timed_out=timed_out,
        files_changed=changed_files(task, workspace),
        isolation=isolation,
        setup_error=setup_error,
        case_passed=passed,
        hidden_by_kind=by_kind,
        pass_non_generated=non_generated,
    )
