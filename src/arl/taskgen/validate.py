"""Validation of candidate bugs and assembled tasks.

Two stages:

1. ``screen_function`` runs the original source function and all of its mutants against the
   function's cases in one sandbox process (per-case time limit). This cheaply discards
   mutants that change nothing, fail only one case, time out or crash at import.
2. ``validate_task`` grades an assembled task with the real grader, once with the reference
   solution and once as shipped, and checks every condition a task must satisfy.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arl.env.grader import HARNESS_PRELUDE, Raised, decode, evaluate_check, grade, harness_argv
from arl.env.sandbox import SandboxConfig, run_in_sandbox
from arl.env.task import VISIBLE_TEST_PATH, Task
from arl.taskgen.mutate import Mutant
from arl.taskgen.sources import SourceFunction

CASE_TIMEOUT_S = 1.0
SCREEN_CONFIG = SandboxConfig(wall_timeout_s=120, cpu_seconds=120)

_SCREEN_HARNESS = HARNESS_PRELUDE + r'''
import signal


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def run_job(job, limit):
    ns = {"__name__": "__arl_screen__"}
    try:
        signal.setitimer(signal.ITIMER_REAL, limit)
        exec(compile(job["code"], "<candidate>", "exec"), ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except _Timeout:
        emit({"id": job["id"], "import_error": "Timeout"})
        return
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        emit({"id": job["id"], "import_error": ename(e)})
        return
    results = []
    for args in job["calls"]:
        try:
            signal.setitimer(signal.ITIMER_REAL, limit)
            value = eval(compile(job["entry"] + "(" + args + ")", "<case>", "eval"), ns)
            signal.setitimer(signal.ITIMER_REAL, 0)
            results.append({"ok": True, "v": enc(value, 0, [MAX_ITEMS])})
        except _Timeout:
            results.append({"timeout": True})
            break
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            results.append({"ok": False, "error": ename(e)})
    emit({"id": job["id"], "results": results})


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    for job in spec["jobs"]:
        run_job(job, spec["limit"])


main()
'''


@dataclass
class CaseOutcome:
    passed: bool
    actual: str  # repr of the value, or "raises <ExceptionName>"
    raised: str | None = None


@dataclass
class JobResult:
    status: str  # "ok" | "import_error" | "timeout" | "crashed"
    outcomes: list[CaseOutcome] = field(default_factory=list)
    detail: str | None = None


def _decode_outcome(rec: dict[str, Any], check: str) -> CaseOutcome:
    if rec.get("ok") is True:
        value = decode(rec["v"])
        return CaseOutcome(passed=evaluate_check(check, value, None), actual=repr(value))
    error = str(rec.get("error"))
    return CaseOutcome(passed=evaluate_check(check, Raised(error), error), actual=f"raises {error}", raised=error)


def run_jobs(fn: SourceFunction, codes: list[str], config: SandboxConfig = SCREEN_CONFIG) -> list[JobResult]:
    """Run each code variant of ``fn`` against ``fn.cases``. Jobs that hang are isolated and skipped."""
    calls = [c.args for c in fn.cases]
    checks = [c.check() for c in fn.cases]
    results: dict[int, JobResult] = {}
    pending = list(range(len(codes)))
    with tempfile.TemporaryDirectory(prefix="arl-screen-") as tmp:
        ws = Path(tmp)
        (ws / "harness.py").write_text(_SCREEN_HARNESS, encoding="utf-8")
        while pending:
            jobs = [{"id": i, "code": codes[i], "entry": fn.entry, "calls": calls} for i in pending]
            (ws / "spec.json").write_text(json.dumps({"jobs": jobs, "limit": CASE_TIMEOUT_S}), encoding="utf-8")
            run = run_in_sandbox(ws, harness_argv("harness.py", "spec.json"), config)
            for line in run.output.splitlines():
                try:
                    rec = json.loads(line)
                    i = rec["id"]
                except Exception:
                    continue
                if i not in pending or i in results:
                    continue
                if "import_error" in rec:
                    status = "timeout" if rec["import_error"] == "Timeout" else "import_error"
                    results[i] = JobResult(status=status, detail=rec["import_error"])
                    continue
                recs = rec["results"]
                if recs and recs[-1].get("timeout"):
                    results[i] = JobResult(status="timeout")
                    continue
                try:
                    outcomes = [_decode_outcome(r, checks[k]) for k, r in enumerate(recs)]
                except Exception:
                    results[i] = JobResult(status="crashed", detail="undecodable result")
                    continue
                if len(outcomes) != len(calls):
                    results[i] = JobResult(status="crashed", detail="missing results")
                    continue
                results[i] = JobResult(status="ok", outcomes=outcomes)
            remaining = [i for i in pending if i not in results]
            if remaining:
                # The first job without a result is the one that killed or hung the process.
                culprit = remaining[0]
                results[culprit] = JobResult(status="timeout" if run.timed_out else "crashed", detail=run.output[-200:])
                remaining = remaining[1:]
            pending = remaining
    return [results[i] for i in range(len(codes))]


@dataclass
class MutantVerdict:
    mutant: Mutant
    ok: bool
    reason: str
    outcomes: list[CaseOutcome]


@dataclass
class FunctionScreen:
    fn: SourceFunction
    original_ok: bool
    original_reason: str
    verdicts: list[MutantVerdict]


def screen_function(fn: SourceFunction, mutants: list[Mutant], config: SandboxConfig = SCREEN_CONFIG) -> FunctionScreen:
    results = run_jobs(fn, [fn.code] + [m.code for m in mutants], config)
    orig = results[0]
    if orig.status != "ok":
        return FunctionScreen(fn, False, f"original_{orig.status}", [])
    if not all(o.passed for o in orig.outcomes):
        return FunctionScreen(fn, False, "original_fails_cases", [])
    verdicts = []
    for m, r in zip(mutants, results[1:]):
        if r.status != "ok":
            verdicts.append(MutantVerdict(m, False, r.status, []))
            continue
        failing = sum(1 for o in r.outcomes if not o.passed)
        if failing == 0:
            reason = "no_case_fails"
        elif failing == 1:
            reason = "only_one_case_fails"
        else:
            reason = "ok"
        verdicts.append(MutantVerdict(m, reason == "ok", reason, r.outcomes))
    return FunctionScreen(fn, True, "ok", verdicts)


# -- full task validation ---------------------------------------------------------------------


@dataclass
class TaskVerdict:
    ok: bool
    reason: str
    grade_seconds: list[float] = field(default_factory=list)


def check_task_static(task: Task) -> str | None:
    """Checks that need no execution. Returns a discard reason or None."""
    visible = task.files[VISIBLE_TEST_PATH]
    for fn_meta in task.meta["functions"]:
        visible_call = fn_meta["visible_call"]  # "module.func(args)"
        hidden_calls = [task.hidden_cases[i].call for i in fn_meta["hidden_case_indices"]]  # "pkg.module.func(args)"
        if f"pkg.{visible_call}" in hidden_calls:
            return "hidden_call_equals_visible_call"
        if visible_call not in visible:
            return "visible_call_missing"
    if "pkg." in task.issue or ".py" in task.issue:
        return "issue_reveals_location"
    # Function names and values (inside backticks) may legitimately contain words like "swap" or "/".
    prose = re.sub(r"`[^`]*`", "``", task.issue).lower()
    if "/" in prose or "line" in prose or "file" in prose or "module" in prose:
        return "issue_reveals_location"
    for word in ("operator", "off-by-one", "off by one", "branch", "swap", "comparison", "constant", "variable"):
        if word in prose:
            return "issue_reveals_bug_type"
    return None


def validate_task(task: Task, config: SandboxConfig | None = None) -> TaskVerdict:
    reason = check_task_static(task)
    if reason:
        return TaskVerdict(False, reason)
    timings = []
    with tempfile.TemporaryDirectory(prefix="arl-validate-") as tmp:
        solved = task.materialize(Path(tmp) / "solved", solved=True)
        t = time.monotonic()
        g = grade(task, solved, config)
        timings.append(time.monotonic() - t)
        if g.timed_out:
            return TaskVerdict(False, "original_timeout", timings)
        if g.setup_error:
            return TaskVerdict(False, "original_import_error", timings)
        if g.reward != 1:
            return TaskVerdict(False, "original_fails_hidden", timings)
        if not g.visible_pass:
            return TaskVerdict(False, "original_fails_visible", timings)

        buggy = task.materialize(Path(tmp) / "buggy")
        t = time.monotonic()
        g = grade(task, buggy, config)
        timings.append(time.monotonic() - t)
        if g.timed_out:
            return TaskVerdict(False, "buggy_timeout", timings)
        if g.setup_error:
            return TaskVerdict(False, "buggy_import_error", timings)
        if g.visible_pass:
            return TaskVerdict(False, "buggy_passes_visible", timings)
        bug = next(f for f in task.meta["functions"] if f["name"] == task.meta["bug_function"])
        bug_idx = set(bug["hidden_case_indices"])
        if all(g.case_passed[i] for i in bug_idx):
            return TaskVerdict(False, "buggy_passes_hidden", timings)
        if not all(p for i, p in enumerate(g.case_passed) if i not in bug_idx):
            return TaskVerdict(False, "buggy_breaks_other_functions", timings)
        if g.reward != 0:
            return TaskVerdict(False, "buggy_gets_reward", timings)
    return TaskVerdict(True, "ok", timings)


def count_reasons(verdicts: list[MutantVerdict]) -> Counter:
    return Counter(f"{v.mutant.spec.bug_type}:{v.reason}" for v in verdicts)
