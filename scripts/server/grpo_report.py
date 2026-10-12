"""Per-step table and summary of a GRPO run's ``log.jsonl`` (reward, entropy, format errors, logprob diff, timing).

    python3 scripts/server/grpo_report.py runs/grpo-overfit/log.jsonl [--json out.json] [--every 1]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

COLUMNS = [("step", "step", "{}"), ("reward", "reward_mean", "{:.3f}"), ("mixed", "non_degenerate_fraction", "{:.2f}"),
           ("entropy", "completion_entropy_mean", "{:.3f}"), ("fmt/call", "format_error_rate_per_tool_call", "{:.3f}"),
           ("lpdiff", "logprob_diff_mean", "{:.4f}"), ("lp_p99", "logprob_diff_p99", "{:.3f}"),
           ("kl", "kl_k3_mean", "{:.4f}"), ("gnorm", "grad_norm", "{:.3f}"), ("sync_mad", "sync_logprob_mad", "{:.4f}"),
           ("greedy", "sync_greedy_equal", "{}"), ("turns", "turns_mean", "{:.2f}"), ("sample_s", "sample_s", "{:.0f}"),
           ("train_s", "train_s", "{:.0f}"), ("save_s", "save_s", "{:.0f}"), ("sync_s", "sync_load_check_s", "{:.1f}"),
           ("step_s", "step_s", "{:.0f}"), ("peak_gib", "peak_allocated_gib", "{:.2f}")]


def fmt(v, f: str) -> str:
    if v is None:
        return "-"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--every", type=int, default=1, help="print every n-th step (the summary uses all)")
    a = ap.parse_args(argv)
    recs = [json.loads(line) for line in a.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [r for r in recs if r.get("type") == "step"]
    print("| " + " | ".join(c[0] for c in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in steps:
        if r["step"] % a.every == 0 or r is steps[-1] or r["step"] == 1:
            print("| " + " | ".join(fmt(r.get(k), f) for _, k, f in COLUMNS) + " |")

    def vals(k: str) -> list[float]:
        return [r[k] for r in steps if r.get(k) is not None]

    def p99(xs: list[float]) -> float | None:
        return sorted(xs)[max(0, int(round(0.99 * len(xs))) - 1)] if xs else None

    summary = {
        "steps": len(steps),
        "logprob_diff_mean_over_steps": statistics.fmean(vals("logprob_diff_mean")) if steps else None,
        "logprob_diff_mean_max_step": max(vals("logprob_diff_mean"), default=None),
        "logprob_diff_p99_mean_over_steps": statistics.fmean(vals("logprob_diff_p99")) if steps else None,
        "logprob_diff_p99_of_step_means": p99(vals("logprob_diff_mean")),
        "logprob_diff_max": max(vals("logprob_diff_max"), default=None),
        "sync_logprob_mad_max": max(vals("sync_logprob_mad"), default=None),
        "sync_greedy_all_equal": all(r.get("sync_greedy_equal") for r in steps),
        "skipped_steps": sum(1 for r in steps if r.get("skipped")),
        "peak_allocated_gib_max": max(vals("peak_allocated_gib"), default=None),
        "peak_reserved_gib_max": max(vals("peak_reserved_gib"), default=None),
        "mean_s": {k: round(statistics.fmean(vals(k)), 1) for k in ("sample_s", "train_s", "save_s",
                                                                      "sync_load_check_s", "step_s") if vals(k)},
        "entropy_first5_mean": statistics.fmean(vals("completion_entropy_mean")[:5]) if steps else None,
        "entropy_last": vals("completion_entropy_mean")[-1] if steps else None,
        "measures": {r["tag"]: r["reward_mean"] for r in recs if r.get("type") == "measure"},
        "evals": [{k: r.get(k) for k in ("step", "solve_rate_strict", "solve_rate_test_challenge", "errors")}
                  for r in recs if r.get("type") == "eval"],
        "stop": [r.get("reason") for r in recs if r.get("type") == "stop"],
        "end": [r for r in recs if r.get("type") == "end"],
    }
    print(json.dumps(summary, indent=2))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
