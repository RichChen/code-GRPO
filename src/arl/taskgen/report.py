"""Checkpoint report for a generated task set: grading cost, scripted-policy sanity checks, samples.

    uv run python -m arl.taskgen.report --tasks data/tasks --out runs/checkpoint-1b
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from arl.env.episode import ScriptedPolicy, run_episode
from arl.env.grader import grade
from arl.env.task import VISIBLE_TEST_PATH, Task, load_tasks


def visible_line(task: Task) -> str:
    bug = task.meta["bug_function"]
    call = next(f["visible_call"] for f in task.meta["functions"] if f["name"] == bug)
    return next(line for line in task.files[VISIBLE_TEST_PATH].splitlines() if line.lstrip().startswith("assert") and call in line)


def policies(task: Task) -> dict[str, ScriptedPolicy]:
    line = visible_line(task)
    cheat = [("edit_file", {"path": VISIBLE_TEST_PATH, "old_string": line, "new_string": "    assert True"})]
    fix = [
        ("edit_file", {"path": path, "old_string": task.files[path], "new_string": new})
        for path, new in task.solution.items()
    ]
    return {
        "submit_as_is": ScriptedPolicy(["Done."]),
        "edit_visible_test_only": ScriptedPolicy([cheat, [("run_tests", {})], "Done."]),
        "apply_reference_fix": ScriptedPolicy([fix, "Done."]),
    }


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[k]


def task_diff(task: Task) -> str:
    out = []
    for path, fixed in task.solution.items():
        out.extend(
            difflib.unified_diff(
                task.files[path].splitlines(), fixed.splitlines(), f"a/{path}", f"b/{path}", lineterm="", n=2
            )
        )
    return "\n".join(out)


def run(tasks_dir: Path, out_dir: Path, n_episodes: int = 50, n_samples: int = 10, seed: int = 0) -> dict:
    tasks = load_tasks(tasks_dir / "train.jsonl") + load_tasks(tasks_dir / "heldout.jsonl")
    rng = random.Random(f"report-{seed}")
    chosen = rng.sample(tasks, n_episodes)

    grade_s: list[float] = []
    episodes: dict[str, list[dict]] = {}
    for task in chosen:
        with tempfile.TemporaryDirectory(prefix="arl-report-") as tmp:
            task.materialize(tmp)
            t0 = time.monotonic()
            g = grade(task, tmp)
            grade_s.append(time.monotonic() - t0)
            assert g.reward == 0, task.task_id
        for name, policy in policies(task).items():
            r = run_episode(task, policy)
            episodes.setdefault(name, []).append(
                {"task_id": task.task_id, "reward": r.reward, "duration_s": r.duration_s,
                 "hidden": f"{r.grade['hidden_pass_count']}/{r.grade['hidden_total']}",
                 "visible_pass": r.grade["visible_pass"], "files_changed": r.grade["files_changed"]}
            )

    summary = {
        "n_tasks": n_episodes,
        "grade_s": {"median": round(statistics.median(grade_s), 3), "p95": round(percentile(grade_s, 0.95), 3),
                    "max": round(max(grade_s), 3)},
        "policies": {
            name: {
                "rewards": dict(sorted({str(k): sum(1 for e in eps if e["reward"] == k) for k in (0, 1)}.items())),
                "episode_s": {"median": round(statistics.median(e["duration_s"] for e in eps), 3),
                              "p95": round(percentile([e["duration_s"] for e in eps], 0.95), 3)},
            }
            for name, eps in episodes.items()
        },
    }

    samples = rng.sample(tasks, n_samples)
    md = []
    for t in samples:
        m = t.meta
        md.append(f"#### `{t.task_id}`（{m['split']}，{m['bug_type']} `{m['bug_variant']}`，来源 {m['bug_source']}，"
                  f"{m['n_modules']} 个模块 / {m['n_functions']} 个函数，模板 {m['issue_template']}）\n")
        md.append(f"题面：\n\n> {t.issue}\n")
        md.append(f"```diff\n{task_diff(t)}\n```\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "episodes.json").write_text(json.dumps(episodes, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "samples.md").write_text("\n".join(md))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    p.add_argument("--out", type=Path, default=Path("runs/checkpoint-1b"))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(run(a.tasks, a.out, a.episodes, a.samples, a.seed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
