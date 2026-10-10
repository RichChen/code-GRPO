"""Grading false-negative estimate: zero-reward episodes whose only failing hidden cases are generated ones.

    uv run python -m arl.rollout.falseneg runs/diag/qwen3-4b-2507.jsonl runs/deepseek/deepseek.jsonl --out runs/falseneg

For each run, counts the zero-reward episodes where every ``test`` and ``challenge`` hidden case passed and
only ``generated`` cases failed, and samples up to 5 of them: the model's diff (rebuilt by replaying its
successful ``edit_file`` calls), the reference fix, and each failing generated case with the value the
model's code actually returned.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from arl.env.grader import _HARNESS, HARNESS_FILE, SPEC_FILE, _clean_copy, _write, decode, harness_argv
from arl.env.sandbox import SandboxConfig, run_in_sandbox
from arl.env.task import Task, load_tasks
from arl.env.tools import Toolbox
from arl.rollout.calibrate import TASKS, load_records, pilot_tasks

SAMPLES = 5
MAX_CASES_SHOWN = 3


def generated_only_failures(rec: dict[str, Any], task: Task) -> list[int] | None:
    """Indices of the failing hidden cases if all of them are ``generated``; ``None`` otherwise."""
    g = rec["grade"]
    passed = g.get("case_passed")
    if rec["reward"] != 0 or passed is None or g.get("timed_out") or g.get("setup_error"):
        return None
    kinds = task.meta["hidden_case_kinds"]
    failing = [i for i, ok in enumerate(passed) if not ok]
    if failing and all(kinds[i] == "generated" for i in failing):
        return failing
    return None


def replay(task: Task, rec: dict[str, Any], dest: Path) -> Path:
    root = task.materialize(dest)
    box = Toolbox(root)
    for c in rec["tool_calls"]:
        if c["name"] == "edit_file" and c["ok"]:
            out = box.call("edit_file", c["arguments"])
            if not out.ok:
                raise RuntimeError(f"replay failed for {rec['task_id']}#{rec['k']}: {out.content}")
    return root


def diff(task: Task, root: Path, solved: bool = False) -> str:
    out = []
    for rel, before in sorted(task.files.items()):
        if not rel.startswith("pkg/"):
            continue
        after = (root / rel).read_text(encoding="utf-8") if not solved else task.solution.get(rel, before)
        if after != before:
            out += difflib.unified_diff(before.splitlines(), after.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="", n=2)
    return "\n".join(out)


def actual_values(task: Task, root: Path, indices: list[int]) -> dict[int, str]:
    """Run the given hidden calls against the workspace in the sandbox; return ``repr`` of each outcome."""
    with tempfile.TemporaryDirectory(prefix="arl-fn-") as tmp:
        copy = _clean_copy(task, root, Path(tmp) / "c")
        _write(copy, HARNESS_FILE, _HARNESS)
        _write(copy, SPEC_FILE, json.dumps({"setup": task.hidden_setup, "calls": [task.hidden_cases[i].call for i in indices]}))
        result = run_in_sandbox(copy, harness_argv(HARNESS_FILE, SPEC_FILE), SandboxConfig())
    out: dict[int, str] = {}
    for line in result.output.splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and isinstance(r.get("i"), int) and 0 <= r["i"] < len(indices):
            i = indices[r["i"]]
            out[i] = repr(decode(r["v"])) if r.get("ok") is True else f"raises {r.get('error')}"
    return out


def analyse(name: str, records: list[dict[str, Any]], tasks: dict[str, Task], seed: int = 0) -> dict[str, Any]:
    recs = [r for r in records if "error" not in r and r["task_id"] in tasks]
    zero = [r for r in recs if r["reward"] == 0]
    literal = [(r, f) for r in zero if (f := generated_only_failures(r, tasks[r["task_id"]])) is not None]
    # an untouched workspace also qualifies when the visible case is the only test case the bug fails
    hits = [(r, f) for r, f in literal if any(p.startswith("pkg/") for p in r["grade"]["files_changed"])]
    sample = random.Random(f"falseneg-{seed}-{name}").sample(hits, min(SAMPLES, len(hits)))
    shown = []
    for rec, failing in sample:
        task = tasks[rec["task_id"]]
        with tempfile.TemporaryDirectory(prefix="arl-fn-ws-") as tmp:
            root = replay(task, rec, Path(tmp) / "ws")
            model_diff = diff(task, root)
            idx = failing[:MAX_CASES_SHOWN]
            actual = actual_values(task, root, idx)
        shown.append({
            "task_id": rec["task_id"], "k": rec["k"], "bug_source": task.meta["bug_source"],
            "bug_type": task.meta["bug_type"], "failing_generated": len(failing),
            "generated_total": sum(1 for k in task.meta["hidden_case_kinds"] if k == "generated"),
            "model_diff": model_diff, "reference_diff": diff(task, root, solved=True),
            "cases": [{"call": task.hidden_cases[i].call, "check": task.hidden_cases[i].check, "actual": actual.get(i)}
                      for i in idx],
        })
    return {"run": name, "episodes": len(recs), "zero_reward": len(zero), "generated_only_literal": len(literal),
            "generated_only": len(hits),
            "fraction_of_zero_reward": round(len(hits) / len(zero), 3) if zero else None, "samples": shown}


def to_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["| run | episodes | reward 0 | only generated fail (any) | ... and pkg/ changed | share of reward 0 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['run']} | {r['episodes']} | {r['zero_reward']} | {r['generated_only_literal']} | "
                     f"{r['generated_only']} | {r['fraction_of_zero_reward']} |")
    for r in results:
        for s in r["samples"]:
            lines += ["", f"#### {r['run']}: `{s['task_id']}` #{s['k']} ({s['bug_source']}, {s['bug_type']}; "
                          f"{s['failing_generated']}/{s['generated_total']} generated cases fail)", "",
                      "Model diff:", "", "```diff", s["model_diff"] or "(no change)", "```", "",
                      "Reference fix:", "", "```diff", s["reference_diff"], "```", ""]
            for c in s["cases"]:
                lines.append(f"- `{c['call']}` expects `{c['check']}`, got `{c['actual']}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--out", type=Path, default=Path("runs/falseneg"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    tasks = {t.task_id: t for t in pilot_tasks(a.tasks, a.seed)}
    results = [analyse(p.stem, load_records(p), tasks, a.seed) for p in a.runs]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "falseneg.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (a.out / "falseneg.md").write_text(to_markdown(results), encoding="utf-8")
    for r in results:
        print(r["run"], r["episodes"], r["zero_reward"], r["generated_only_literal"], r["generated_only"],
              r["fraction_of_zero_reward"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
