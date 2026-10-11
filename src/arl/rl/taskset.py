"""The GRPO task set: ``rl_train_ids`` tasks solved 1-3 times out of K=4 by SFT-v1 (T=1.0, top_k 0).

    uv run python -m arl.rl.taskset --records runs/grpo-calib/rl-train.jsonl

Writes ``data/tasks/grpo_v1_ids.txt`` and ``data/tasks/grpo_v1.json`` and prints the calibration report.
Exits with status 4 (and writes nothing) when fewer than ``--min-tasks`` tasks are kept, or when the
per-call format-error rate is above ``--max-format-error-rate`` (6%). Exactly 6% is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from arl.env.task import load_tasks
from arl.rollout.calibrate import latest, load_records

MIN_TASKS = 400
MAX_FORMAT_ERROR_RATE = 0.06


def _per_call_format_error_rate(recs: list[dict[str, Any]]) -> float | None:
    calls = sum(r.get("n_tool_calls", 0) for r in recs)
    if not calls:
        return None
    return round(sum(r.get("format_errors", 0) for r in recs) / calls, 4)


def dist(xs: list[int]) -> dict[str, Any]:
    s = sorted(xs)
    q = lambda p: s[max(0, math.ceil(p * len(s)) - 1)]  # noqa: E731
    return {"mean": round(statistics.fmean(s), 1), "median": statistics.median(s), "p90": q(0.9), "p95": q(0.95),
            "p99": q(0.99), "max": s[-1]}


def report(recs: list[dict[str, Any]], ids: list[str], k: int) -> tuple[dict[str, Any], dict[str, int]]:
    """Calibration statistics, and each task's solve count."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        by_task.setdefault(r["task_id"], []).append(r)
    missing = [t for t in ids if len(by_task.get(t, [])) != k]
    if missing:
        raise ValueError(f"{len(missing)} tasks lack {k} finished episodes, e.g. {missing[:3]}")
    errors = [r for r in recs if r.get("error")]
    if errors:
        raise ValueError(f"{len(errors)} episodes ended with an error")
    solved = {t: sum(r["reward"] for r in v) for t, v in by_task.items()}
    non_gen = [int(bool(r["grade"].get("pass_non_generated"))) for r in recs]
    return {
        "episodes": len(recs),
        "solve_rate_strict": round(statistics.fmean(r["reward"] for r in recs), 4),
        "solve_rate_test_challenge": round(statistics.fmean(non_gen), 4),
        "solve_count_distribution": {str(c): sum(1 for v in solved.values() if v == c) for c in range(k + 1)},
        "stop_reasons": dict(Counter(r["stop_reason"] for r in recs)),
        "max_turns_fraction": round(sum(1 for r in recs if r["stop_reason"] == "max_turns") / len(recs), 4),
        "episodes_with_format_error": round(sum(1 for r in recs if r.get("format_errors", 0) > 0) / len(recs), 4),
        "format_error_rate_per_tool_call": _per_call_format_error_rate(recs),
        "turns_mean": round(statistics.fmean(r["turns"] for r in recs), 3),
        "nudged_fraction": round(sum(1 for r in recs if r.get("nudged")) / len(recs), 4),
        "tokens_per_episode": {
            "completion": dist([sum(r["turn_completion_tokens"]) for r in recs]),
            "prompt_summed_over_turns": dist([sum(r["turn_prompt_tokens"]) for r in recs]),
            "trained_sequence_tokens": dist([sum(r["turn_prompt_tokens"]) + sum(r["turn_completion_tokens"])
                                             for r in recs]),
            "final_context": dist([r["turn_prompt_tokens"][-1] + r["turn_completion_tokens"][-1] for r in recs
                                   if r["turn_prompt_tokens"]]),
        },
        "finish_reasons": dict(sum((Counter(r.get("finish_reasons") or {}) for r in recs), Counter())),
        "tokenize_mismatches": sum(r.get("tokenize_mismatches", 0) for r in recs),
    }, solved


def build(records: Path, ids_file: Path, tasks_file: Path, out_ids: Path, out_json: Path, k: int = 4,
          min_tasks: int = MIN_TASKS, max_format_error_rate: float = MAX_FORMAT_ERROR_RATE) -> dict[str, Any]:
    ids = sorted(line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip())
    wanted = set(ids)
    recs = [r for r in latest(load_records(records)).values() if r["task_id"] in wanted]
    rep, solved = report(recs, ids, k)
    kept = sorted(t for t, c in solved.items() if 1 <= c <= k - 1)
    tasks = {t.task_id: t for t in load_tasks(tasks_file)}
    text = "\n".join(kept) + "\n"
    per_fn = Counter(tasks[t].meta["bug_source"] for t in kept)
    summary = {
        "source": {"ids": str(ids_file), "records": str(records), "k": k, "keep": f"solved 1..{k - 1} of {k}",
                   "model": "SFT-v1 (runs/sft-preview/merged, fp16)", "sampling": "T=1.0, top_p 1.0, top_k 0",
                   "env": "phase1-env-v1"},
        "tasks": len(kept),
        "functions": len(per_fn),
        "sha256_ids": hashlib.sha256(text.encode()).hexdigest(),
        "bug_types": dict(sorted(Counter(tasks[t].meta["bug_type"] for t in kept).items())),
        "tasks_per_function": dict(sorted(Counter(per_fn.values()).items())),
        "solve_count_distribution": rep["solve_count_distribution"],
        "calibration": rep,
    }
    reasons = []
    rate = rep["format_error_rate_per_tool_call"]
    if rate is not None and rate > max_format_error_rate:
        reasons.append(f"per-call format-error rate {rate} > {max_format_error_rate}")
    if len(kept) < min_tasks:
        reasons.append(f"only {len(kept)} tasks kept (< {min_tasks})")
    if reasons:
        summary["stop"] = "; ".join(reasons)
        out_ids.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
        return summary
    out_ids.write_text(text, encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=Path("runs/grpo-calib/rl-train.jsonl"))
    ap.add_argument("--ids", type=Path, default=Path("data/tasks/rl_train_ids.txt"))
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks/train.jsonl"))
    ap.add_argument("--out-ids", type=Path, default=Path("data/tasks/grpo_v1_ids.txt"))
    ap.add_argument("--out-json", type=Path, default=Path("data/tasks/grpo_v1.json"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-tasks", type=int, default=MIN_TASKS)
    ap.add_argument("--max-format-error-rate", type=float, default=MAX_FORMAT_ERROR_RATE)
    a = ap.parse_args(argv)
    summary = build(a.records, a.ids, a.tasks, a.out_ids, a.out_json, a.k, a.min_tasks, a.max_format_error_rate)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 4 if "stop" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
