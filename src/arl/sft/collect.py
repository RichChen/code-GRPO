"""Collect SFT trajectories with DeepSeek, on the SFT pool only.

    uv run python -m arl.sft.collect pilot --budget-cny 15   # the first 20 tasks of the fixed order, once each
    uv run python -m arl.sft.collect run --budget-cny 15     # every task once, then once more for the failures
    uv run python -m arl.sft.collect summary

Episodes (with the full conversation) are appended to ``runs/deepseek-sft/episodes.jsonl`` as they finish, so
an interrupted run resumes; attempt ``k=0`` is the first try, ``k=1`` the retry of a failed task. The order of
the SFT pool is fixed (``sftcollect-{seed}``), so a budget stop leaves a random subset. Collection stops at
``MAX_SUCCESSES`` successful trajectories or at the budget, whichever comes first.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from arl.env.task import Task, load_tasks
from arl.rollout.calibrate import CostLedger, diagnostics, latest, load_records, run_calibration
from arl.rollout.client import ClientConfig
from arl.sft.split import read_ids

RUNS = Path("runs/deepseek-sft")
TASKS = Path("data/tasks")
CONFIG = Path("configs/deepseek.json")
PILOT_SIZE = 20
MAX_SUCCESSES = 1500
MAX_ATTEMPTS = 2
PILOT_MAX_CNY_PER_1000 = 12.0


def sft_pool(tasks_dir: Path = TASKS, seed: int = 0) -> list[Task]:
    ids = set(read_ids(tasks_dir / "train_sft_ids.txt"))
    tasks = sorted((t for t in load_tasks(tasks_dir / "train.jsonl") if t.task_id in ids), key=lambda t: t.task_id)
    if len(tasks) != len(ids):
        raise ValueError(f"{len(ids) - len(tasks)} SFT pool ids are not in train.jsonl")
    random.Random(f"sftcollect-{seed}").shuffle(tasks)
    return tasks


def retry_tasks(tasks: list[Task], records: list[dict[str, Any]]) -> list[Task]:
    """Tasks whose first attempt failed, capped so that successes cannot exceed ``MAX_SUCCESSES``."""
    done = latest(records)
    successes = sum(r["reward"] for r in done.values())
    failed = [t for t in tasks if (t.task_id, 0) in done and done[(t.task_id, 0)]["reward"] == 0]
    return failed[:max(0, MAX_SUCCESSES - successes)]


def summary(records: list[dict[str, Any]], tasks: list[Task]) -> dict[str, Any]:
    ids = {t.task_id for t in tasks}
    recs = [r for r in latest(records).values() if r["task_id"] in ids]
    first = [r for r in recs if r["k"] == 0]
    retry = [r for r in recs if r["k"] == 1]
    costs = [r.get("cost_cny", 0.0) for r in records if r.get("task_id") in ids]
    total = sum(costs)
    successes = sum(r["reward"] for r in recs)
    solved_tasks = len({r["task_id"] for r in recs if r["reward"]})
    per_success = total / successes if successes else None
    return {
        "episodes": len(recs),
        "errored_records": sum(1 for r in records if "error" in r),
        "first_attempts": len(first),
        "first_attempt_solve_rate": round(sum(r["reward"] for r in first) / len(first), 3) if first else None,
        "retries": len(retry),
        "retry_solve_rate": round(sum(r["reward"] for r in retry) / len(retry), 3) if retry else None,
        "successful_trajectories": successes,
        "tasks_solved": solved_tasks,
        "tasks_attempted": len({r["task_id"] for r in recs}),
        "cost_cny": round(total, 4),
        "cost_cny_per_episode": round(total / len(costs), 5) if costs else None,
        "cost_cny_per_success": round(per_success, 5) if per_success else None,
        "est_cny_for_1000_successes": round(per_success * 1000, 2) if per_success else None,
        "mean_turns": round(statistics.mean(r["turns"] for r in recs), 2) if recs else None,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in recs)),
        "nudged": sum(1 for r in recs if r.get("nudged")),
        "with_format_errors": sum(1 for r in recs if r["format_errors"]),
        "truncated": sum(1 for r in recs if r["truncated"]),
        "diagnostics": diagnostics(recs) if recs else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "run", "summary"])
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--budget-cny", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    path = a.runs / "episodes.jsonl"
    tasks = sft_pool(a.tasks, a.seed)

    if a.command != "summary":
        if a.budget_cny is None:
            ap.error("--budget-cny is required")
        config = ClientConfig.load(a.config)

        def go(batch: list[Task], k: int) -> None:
            ledger = CostLedger(a.runs / "cost.json", a.budget_cny, config.price_cny_per_mtok, load_records(path))
            run_calibration(batch, path, config, k=k, concurrency=a.concurrency, ledger=ledger, with_messages=True)

        if a.command == "pilot":
            go(tasks[:PILOT_SIZE], 1)
        else:
            go(tasks, 1)
            go(retry_tasks(tasks, load_records(path)), MAX_ATTEMPTS)

    scope = tasks[:PILOT_SIZE] if a.command == "pilot" else tasks
    s = summary(load_records(path), scope)
    name = "pilot-summary.json" if a.command == "pilot" else "summary.json"
    a.runs.mkdir(parents=True, exist_ok=True)
    (a.runs / name).write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=2, ensure_ascii=False))
    if a.command == "pilot" and (s["est_cny_for_1000_successes"] is None
                                 or s["est_cny_for_1000_successes"] > PILOT_MAX_CNY_PER_1000):
        print(f"[collect] STOP: 1000 successful trajectories would cost more than {PILOT_MAX_CNY_PER_1000} CNY")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
