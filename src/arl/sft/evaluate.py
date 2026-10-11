"""Compare the base model and the SFT preview on the dev set (records from ``scripts/eval_dev.sh``).

    uv run python -m arl.sft.evaluate runs/sft-eval/dev-base.jsonl runs/sft-eval/dev-sft.jsonl \
        --reference runs/deepseek-dev/dev-deepseek.jsonl --out runs/sft-eval

Two solve rates: strict (every hidden case, the training reward) and test + challenge only
(``grade.pass_non_generated``). The paired comparison is per task: SFT solve rate minus base solve rate, with
a 95% interval from bootstrapping over tasks. The reference model (DeepSeek) is reported alongside and only
says which dev tasks are solvable at all; it takes part in no choice.

The share of dev tasks the SFT model solves 1-3 times out of 4 is scaled to ``rl_train_ids`` as a rough
estimate of how many GRPO tasks a full calibration would keep.
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
# Audited bootstrap of the fp16 SFT dev eval (runs/sft-eval/dev-sft.jsonl), 10000 resamples over tasks, seed 0.
SFT_DEV_STRICT_RATE = 0.535
SFT_DEV_STRICT_CI = (0.455, 0.6125)


def in_sft_dev_interval(rate: float) -> bool:
    """True when a strict solve rate sits inside the audited 95% interval of 53.5%."""
    lo, hi = SFT_DEV_STRICT_CI
    return lo <= rate <= hi


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


def solve_counts(recs: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in recs:
        out[r["task_id"]] += r["reward"]
    return dict(out)


def calibration_estimate(sft: list[dict[str, Any]], tasks: dict[str, Any], pool_size: int, k: int = K) -> dict[str, Any]:
    """Dev tasks the SFT model solves 1..k-1 times, scaled to a pool of ``pool_size`` tasks (bootstrap 95% CI)."""
    counts = solve_counts(sft)
    in_range = [int(0 < counts.get(t, 0) < k) for t in sorted(tasks)]
    frac = statistics.fmean(in_range)
    lo, hi = bootstrap_ci([float(x) for x in in_range])
    return {"dev_tasks": len(in_range), "dev_tasks_solved_1_to_3": sum(in_range), "fraction": round(frac, 4),
            "fraction_ci95": [round(lo, 4), round(hi, 4)], "pool_tasks": pool_size,
            "estimated_kept": round(frac * pool_size), "estimated_kept_ci95": [round(lo * pool_size), round(hi * pool_size)]}


def headroom(reference: list[dict[str, Any]], others: dict[str, list[dict[str, Any]]],
             tasks: dict[str, Any]) -> dict[str, Any]:
    ref = solve_counts(reference)
    solvable = {t for t in tasks if ref.get(t, 0) > 0}
    out: dict[str, Any] = {"reference_solved_at_least_once": len(solvable), "reference_never_solved": len(tasks) - len(solvable)}
    for name, recs in others.items():
        c = solve_counts(recs)
        out[f"{name}_never_solved_but_reference_did"] = sum(1 for t in solvable if c.get(t, 0) == 0)
        out[f"{name}_solved_but_reference_never"] = sum(1 for t in tasks if t not in solvable and c.get(t, 0) > 0)
    return out


def compare(base: list[dict[str, Any]], sft: list[dict[str, Any]], tasks: dict[str, Any],
            reference: list[dict[str, Any]] | None = None, pool_size: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"base": model_report(base, tasks), "sft": model_report(sft, tasks)}
    if reference is not None:
        out["reference"] = model_report(reference, tasks)
        out["headroom"] = headroom(reference, {"base": base, "sft": sft}, tasks)
    if pool_size is not None:
        out["rl_train_calibration_estimate"] = calibration_estimate(sft, tasks, pool_size)
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
    ap.add_argument("--reference", type=Path, default=None, help="reference model records (DeepSeek on dev)")
    ap.add_argument("--rl-train-ids", type=Path, default=TASKS / "rl_train_ids.txt")
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--ids", type=Path, default=TASKS / "dev_ids.txt")
    ap.add_argument("--out", type=Path, default=Path("runs/sft-eval"))
    a = ap.parse_args(argv)
    ids = set(read_ids(a.ids))
    tasks = {t.task_id: t for t in load_tasks(a.tasks / "train.jsonl") if t.task_id in ids}

    def recs(path: Path) -> list[dict[str, Any]]:
        return [r for r in latest(load_records(path)).values() if r["task_id"] in ids]

    result = compare(recs(a.base), recs(a.sft), tasks, recs(a.reference) if a.reference else None,
                     len(read_ids(a.rl_train_ids)))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "compare.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
