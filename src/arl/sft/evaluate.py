"""Compare the base model and the SFT preview on the dev set (records from ``scripts/eval_dev.sh``).

    uv run python -m arl.sft.evaluate runs/sft-eval/dev-base.jsonl runs/sft-eval/dev-sft.jsonl --out runs/sft-eval

Two solve rates: strict (every hidden case, the training reward) and test + challenge only
(``grade.pass_non_generated``). The paired comparison is per task: SFT solve rate minus base solve rate, with
a 95% interval from bootstrapping over tasks.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from arl.env.task import load_tasks
from arl.rollout.calibrate import K, TASKS, latest, load_records, summarize
from arl.sft.split import read_ids

BOOTSTRAP = 10000


def per_task_rates(recs: list[dict[str, Any]], key: str = "reward") -> dict[str, float]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        v = r["reward"] if key == "reward" else int(bool(r["grade"].get("pass_non_generated")))
        by_task[r["task_id"]].append(v)
    return {t: sum(v) / len(v) for t, v in by_task.items()}


def bootstrap_ci(diffs: list[float], n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def model_report(recs: list[dict[str, Any]], tasks: dict[str, Any], k: int = K) -> dict[str, Any]:
    s = summarize(recs, tasks, k)
    non_gen = [int(bool(r["grade"].get("pass_non_generated"))) for r in recs]
    edit_calls = s["diagnostics"]["edit_calls"]
    return {
        "episodes": s["episodes"],
        "solve_rate_strict": s["solve_rate"],
        "solve_rate_test_challenge": round(sum(non_gen) / len(non_gen), 3) if non_gen else None,
        "solve_count_distribution": s["solve_count_distribution"],
        "tasks_solved_1_to_3": s["tasks_in_range"],
        "fraction_solved_1_to_3": s["fraction_in_range"],
        "format_error_rate_per_tool_call": s["format_error_rate_per_tool_call"],
        "episodes_with_format_error": s["episodes_with_format_error"],
        "nudged": s["diagnostics"]["nudged"],
        "edit_calls": edit_calls,
        "edit_success_rate": s["diagnostics"]["edit_success_rate"],
        "mean_turns": s["mean_turns"],
        "stop_reasons": s["stop_reasons"],
        "by_bug_type": s["by_bug_type"],
        "mean_completion_tokens": s["mean_completion_tokens"],
        "mean_final_context_tokens": s["mean_final_context_tokens"],
        "throughput_episodes_per_hour": s["throughput_episodes_per_hour"],
    }


def compare(base: list[dict[str, Any]], sft: list[dict[str, Any]], tasks: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"base": model_report(base, tasks), "sft": model_report(sft, tasks)}
    for key, name in (("reward", "strict"), ("non_generated", "test_challenge")):
        b, s = per_task_rates(base, key), per_task_rates(sft, key)
        common = sorted(set(b) & set(s))
        diffs = [s[t] - b[t] for t in common]
        lo, hi = bootstrap_ci(diffs)
        out[f"paired_{name}"] = {"tasks": len(common), "mean_diff": round(statistics.fmean(diffs), 4),
                                 "ci95": [round(lo, 4), round(hi, 4)],
                                 "tasks_better": sum(1 for d in diffs if d > 0),
                                 "tasks_worse": sum(1 for d in diffs if d < 0)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("sft", type=Path)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--ids", type=Path, default=TASKS / "dev_ids.txt")
    ap.add_argument("--out", type=Path, default=Path("runs/sft-eval"))
    a = ap.parse_args(argv)
    ids = set(read_ids(a.ids))
    tasks = {t.task_id: t for t in load_tasks(a.tasks / "train.jsonl") if t.task_id in ids}

    def recs(path: Path) -> list[dict[str, Any]]:
        return [r for r in latest(load_records(path)).values() if r["task_id"] in ids]

    result = compare(recs(a.base), recs(a.sft), tasks)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "compare.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
