"""Held-out comparison of the three preregistered models (docs/plan/HELDOUT-PREREG.zh-CN.md).

    PYTHONPATH=src ~/.venvs/vllm/bin/python -m arl.rl.heldout \
        --base runs/heldout/base.jsonl --sft runs/heldout/sft.jsonl --grpo runs/heldout/grpo.jsonl --out runs/heldout

Primary metric: strict solve rate. Paired differences GRPO - SFT and SFT - base, per task (each task's K=4 solve
rate), with a 95% interval from a cluster bootstrap over bug source functions (``meta["bug_source"]``): resample
the clusters with replacement, take every task of each drawn cluster, average the per-task differences over those
tasks; 10000 resamples, seed 0. No paid API, no model calls: this only reads records.
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
from arl.rollout.calibrate import latest, load_records, summarize

BOOTSTRAP = 10000
SEED = 0
K = 4


def per_task_rates(recs: list[dict[str, Any]], key: str = "reward") -> dict[str, float]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        v = r["reward"] if key == "reward" else int(bool(r["grade"].get("pass_non_generated")))
        by_task[r["task_id"]].append(v)
    return {t: sum(v) / len(v) for t, v in by_task.items()}


def cluster_bootstrap(values: dict[str, float], cluster_of: dict[str, str], n: int = BOOTSTRAP,
                      seed: int = SEED) -> tuple[float, float, float]:
    """Mean of ``values`` over tasks and its 95% interval, resampling whole clusters with replacement."""
    clusters: dict[str, list[float]] = defaultdict(list)
    for t, v in values.items():
        clusters[cluster_of[t]].append(v)
    names = sorted(clusters)
    sums = [sum(clusters[c]) for c in names]
    sizes = [len(clusters[c]) for c in names]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        idx = [rng.randrange(len(names)) for _ in names]
        means.append(sum(sums[i] for i in idx) / sum(sizes[i] for i in idx))
    means.sort()
    return statistics.fmean(values.values()), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def model_report(recs: list[dict[str, Any]], tasks: dict[str, Any]) -> dict[str, Any]:
    s = summarize(recs, tasks, K)
    non_gen = [int(bool(r["grade"].get("pass_non_generated"))) for r in recs]
    by_origin: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        by_origin[tasks[r["task_id"]].meta["bug_source"].split("/")[0]].append(r["reward"])
    return {
        "episodes": s["episodes"],
        "tasks": s["tasks_complete"],
        "errors": sum(1 for r in recs if r.get("error")),
        "solve_rate_strict": s["solve_rate"],
        "solve_rate_test_challenge": round(sum(non_gen) / len(non_gen), 4) if non_gen else None,
        "by_bug_type": s["by_bug_type"],
        "by_origin": {o: {"episodes": len(v), "solve_rate": round(sum(v) / len(v), 4)} for o, v in sorted(by_origin.items())},
        "format_error_rate_per_tool_call": s["format_error_rate_per_tool_call"],
        "episodes_with_format_error": s["episodes_with_format_error"],
        "mean_turns": s["mean_turns"],
        "stop_reasons": s["stop_reasons"],
        "solve_count_distribution": s["solve_count_distribution"],
        "tokenize_checked": s.get("tokenize_checked"),
        "tokenize_mismatches": s.get("tokenize_mismatches"),
        "throughput_episodes_per_hour": s["throughput_episodes_per_hour"],
    }


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]], tasks: dict[str, Any], key: str) -> dict[str, Any]:
    """``a`` minus ``b`` per task, cluster-bootstrapped over bug source functions."""
    ra, rb = per_task_rates(a, key), per_task_rates(b, key)
    common = sorted(set(ra) & set(rb))
    diffs = {t: ra[t] - rb[t] for t in common}
    cluster_of = {t: tasks[t].meta["bug_source"] for t in common}
    mean, lo, hi = cluster_bootstrap(diffs, cluster_of)
    return {"tasks": len(common), "clusters": len(set(cluster_of.values())), "mean_diff": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "tasks_better": sum(1 for d in diffs.values() if d > 0),
            "tasks_worse": sum(1 for d in diffs.values() if d < 0), "bootstrap": BOOTSTRAP, "seed": SEED}


def compare(runs: dict[str, list[dict[str, Any]]], tasks: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {name: model_report(recs, tasks) for name, recs in runs.items()}
    for a, b in (("grpo", "sft"), ("sft", "base")):
        if a in runs and b in runs:
            out[f"paired_{a}_minus_{b}"] = {name: paired(runs[a], runs[b], tasks, key)
                                            for key, name in (("reward", "strict"), ("non_generated", "test_challenge"))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--sft", type=Path)
    ap.add_argument("--grpo", type=Path)
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks/heldout.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/heldout"))
    a = ap.parse_args(argv)
    tasks = {t.task_id: t for t in load_tasks(a.tasks)}
    runs = {name: [r for r in latest(load_records(p)).values() if r["task_id"] in tasks]
            for name, p in (("base", a.base), ("sft", a.sft), ("grpo", a.grpo)) if p}
    result = compare(runs, tasks)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "compare.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
