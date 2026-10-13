"""Recompute the GRPO training numbers reported in CHECKPOINT-3 and redo the checkpoint choice (phase 4, step 1).

    uv run python scripts/closeout/verify_training.py [--out runs/closeout/verify-training.json]

Reads ``runs/grpo-v1/log.jsonl``, ``runs/grpo-v1/eval/step-*.jsonl``, ``data/tasks/dev_ids.txt`` and
``configs/grpo/v1.json``. Checks, against CHECKPOINT-3 ("GRPO 训练曲线", "各检查点的 dev 结果和选择", "结论",
"审计修订 3-1 的执行情况") and a few PROGRESS lines:

- the training curve table (mean of every 25 steps) and the run-wide lines under it;
- the dev table, counted again from the per-episode eval records (100 tasks x K=8 = 800 episodes per checkpoint);
- the preregistered choice: highest dev strict solve rate, and among checkpoints less than 1 point below it the
  earliest step. It must come out as step 150;
- the sha256 of ``configs/grpo/v1.json`` against the frozen value, in the work tree and at commit c9984b9.

A number matches when it equals the reported one after rounding to the reported precision; values exactly half a
unit away count as matches and are listed under ``half_unit_ties``. Segment means are plain means of the per-step
values (the per-step reward restored exactly from its 0/1 count). The two values corrected in the erratum at the end
of CHECKPOINT-3 (audit revision 4-1) are checked against the corrected value; the original is kept in the record and
``diagnostics`` gives the other readings. Exit 1 on any mismatch.

Also writes ``training-tables.md`` next to ``--out``: the training curve and dev tables for the final report, from
exact values (fractions) rounded once, half up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

CP3 = "docs/plan/CHECKPOINT-3.md"
CURVE = f"{CP3} GRPO 训练曲线"
DEV = f"{CP3} 各检查点的 dev 结果和选择"
REV = f"{CP3} 审计修订 3-1 的执行情况"
PROGRESS = "docs/plan/PROGRESS.md 阶段 3 · 第 4 步"
FROZEN_SHA256 = "3566ba54767d7f8a1964d785639fb4b8d4b43d2dd91989f27861b0a526f7261c"
FROZEN_COMMIT = "c9984b9"

# CHECKPOINT-3 training curve: (log field, decimals, scale) and one row per 25-step segment.
CURVE_COLUMNS = (("reward_mean", 3, 1), ("non_degenerate_fraction", 2, 1), ("completion_entropy_mean", 3, 1),
                 ("format_error_rate_per_tool_call", 1, 100), ("logprob_diff_mean", 4, 1), ("kl_k3_mean", 4, 1),
                 ("turns_mean", 2, 1), ("step_s", 0, 1))
CURVE_ROWS = {
    (1, 25): (0.474, 0.87, 0.753, 3.9, 0.0018, 0.0007, 7.08, 398),
    (26, 50): (0.534, 0.85, 0.562, 3.1, 0.0015, 0.0033, 7.17, 376),
    (51, 75): (0.591, 0.80, 0.541, 2.4, 0.0015, 0.0056, 6.96, 364),
    (76, 100): (0.595, 0.81, 0.546, 2.1, 0.0015, 0.0089, 6.92, 381),
    (101, 125): (0.653, 0.76, 0.556, 1.2, 0.0016, 0.0122, 6.57, 347),
    (126, 150): (0.668, 0.75, 0.522, 1.1, 0.0015, 0.0146, 6.61, 345),
    (151, 175): (0.701, 0.68, 0.382, 1.1, 0.0012, 0.0173, 6.54, 313),
    (176, 200): (0.724, 0.66, 0.371, 0.85, 0.0012, 0.0217, 6.32, 295),
}
# The 176-200 format error rate is printed with two decimals (0.85%).
CURVE_DECIMALS_OVERRIDE = {((176, 200), "format_error_rate_per_tool_call"): 2}
# Audit revision 4-1: the two values that did not match, corrected in the erratum at the end of CHECKPOINT-3.
ERRATA = {
    "curve 76-100 format_error_rate_per_tool_call": 2.0,
    "mixed-group fraction, mean of the 10 steps up to 195": 0.625,
}
ERRATUM_SOURCE = f"{CP3} 勘误（阶段 4 复核，2026-10-13）"
CURVE_HEADER = ("步", "奖励", "有好有坏的组", "熵", "每次调用格式错误率", "logprob 差均值", "KL(k3)", "平均轮数", "步时 s")
DEV_STEPS = (0, 25, 50, 75, 100, 125, 150, 175, 200)
DEV_SOLVED = (451, 475, 501, 500, 516, 543, 551, 557, 551)
DEV_STRICT_PCT = (56.4, 59.4, 62.6, 62.5, 64.5, 67.9, 68.9, 69.6, 68.9)
DEV_TC_PCT = (61.0, 63.6, 68.1, 68.0, 69.0, 72.1, 72.6, 73.9, 74.1)


def matches(computed: float, reported: float, decimals: int) -> bool:
    """True when ``computed`` rounds to ``reported`` at ``decimals`` places (half-unit tolerance)."""
    return abs(computed - reported) <= 0.5 * 10 ** -decimals + 1e-9


def on_tie(computed: float, reported: float, decimals: int) -> bool:
    """``computed`` sits exactly half a unit from ``reported``: either rounding direction is a valid rounding."""
    return abs(abs(computed - reported) - 0.5 * 10 ** -decimals) < 1e-9


def step_value(s: dict[str, Any], field: str) -> float:
    """A step's value; the reward is a count of 0/1 episodes logged as a 4-decimal mean, so restore it exactly."""
    return float(exact_step_value(s, field))


def exact_step_value(s: dict[str, Any], field: str) -> Fraction:
    if field == "reward_mean":
        return Fraction(round(s["reward_mean"] * s["episodes"]), s["episodes"])
    return Fraction(str(s[field]))


def round_half_up(x: Fraction, decimals: int) -> str:
    """``x`` rounded once, half away from zero for x >= 0, printed with ``decimals`` places."""
    n = math.floor(x * 10 ** decimals + Fraction(1, 2))
    if decimals == 0:
        return str(n)
    return f"{n // 10 ** decimals}.{n % 10 ** decimals:0{decimals}d}"


def curve_table(steps: list[dict[str, Any]]) -> str:
    """The 25-step training curve table, from exact segment means rounded once (audit revision 4-1 item 3)."""
    lines = ["| " + " | ".join(CURVE_HEADER) + " |", "|" + "---|" * len(CURVE_HEADER)]
    for lo, hi in CURVE_ROWS:
        seg = [s for s in steps if lo <= s["step"] <= hi]
        cells = [f"{lo}–{hi}"]
        for field, dec, scale in CURVE_COLUMNS:
            dec = CURVE_DECIMALS_OVERRIDE.get(((lo, hi), field), dec)
            mean = sum(exact_step_value(s, field) for s in seg) / len(seg) * scale
            cells.append(round_half_up(mean, dec) + ("%" if scale == 100 else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def dev_table(rows: list[dict[str, Any]]) -> str:
    """The dev results per checkpoint, rounded once."""
    head = ["步"] + [str(r["step"]) for r in rows]
    solved = ["严格解出数"] + [str(r["solved"]) for r in rows]
    strict = ["严格解出率"] + [round_half_up(Fraction(100 * r["solved"], r["episodes"]), 1) + "%" for r in rows]
    tc = ["test+challenge"] + [round_half_up(Fraction(100 * r["tc_solved"], r["episodes"]), 1) + "%" for r in rows]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in (solved, strict, tc)]
    return "\n".join(out) + "\n"


def select_checkpoint(solved: dict[int, int], episodes: int) -> dict[str, Any]:
    """Preregistered rule: highest strict solve rate; among checkpoints less than 1 point below the best (not
    chained), the earliest step. Integer arithmetic: ``best - n`` solved episodes is under 1 point when
    ``100 * (best - n) < episodes``."""
    best_step = max(solved, key=lambda s: (solved[s], -s))
    best = solved[best_step]
    within = sorted(s for s, n in solved.items() if 100 * (best - n) < episodes)
    return {"best_step": best_step, "best_solved": best, "within_1_point": within, "chosen": within[0]}


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, item: str, computed: Any, reported: Any, source: str, decimals: int | None = None) -> None:
        extra = {}
        if item in ERRATA:
            extra = {"reported_before_erratum": reported, "erratum": ERRATUM_SOURCE}
            reported = ERRATA[item]
        ok = computed == reported if decimals is None else matches(computed, reported, decimals)
        tie = decimals is not None and on_tie(computed, reported, decimals)
        self.items.append({"item": item, "computed": computed, "reported": reported, "decimals": decimals,
                           "source": source, "ok": ok, "half_unit_tie": tie, **extra})

    @property
    def corrected(self) -> list[dict[str, Any]]:
        return [c for c in self.items if "erratum" in c]

    def add_extreme(self, item: str, steps: list[dict[str, Any]], field: str, pick: Any, reported_step: int,
                    reported: float, source: str, decimals: int, scale: float = 1) -> None:
        """The step where ``field`` is extreme (``pick`` is min or max) and its value."""
        vals = [s[field] for s in steps]
        v = pick(vals)
        step = steps[vals.index(v)]["step"]
        self.items.append({"item": item, "computed": [step, scale * v], "reported": [reported_step, reported],
                           "decimals": decimals, "source": source,
                           "ok": step == reported_step and matches(scale * v, reported, decimals),
                           "half_unit_tie": on_tie(scale * v, reported, decimals)})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.items if not c["ok"]]

    @property
    def ties(self) -> list[dict[str, Any]]:
        return [c for c in self.items if c["half_unit_tie"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("runs/grpo-v1"))
    ap.add_argument("--config", type=Path, default=Path("configs/grpo/v1.json"))
    ap.add_argument("--dev-ids", type=Path, default=Path("data/tasks/dev_ids.txt"))
    ap.add_argument("--out", type=Path, default=Path("runs/closeout/verify-training.json"))
    a = ap.parse_args(argv)
    c = Checks()

    raw = a.config.read_bytes()
    c.add("configs/grpo/v1.json sha256 (work tree)", hashlib.sha256(raw).hexdigest(), FROZEN_SHA256,
          f"{CP3} 结论（3566ba54…7261c）；PROGRESS 配置冻结")
    at_commit = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:{a.config.as_posix()}"], capture_output=True,
                               check=True).stdout
    c.add(f"configs/grpo/v1.json sha256 at {FROZEN_COMMIT}", hashlib.sha256(at_commit).hexdigest(), FROZEN_SHA256,
          f"{CP3} 结论（提交 c9984b9）")
    cfg = json.loads(raw)
    client = json.loads(Path(cfg["client"]).read_text(encoding="utf-8"))
    c.add("config n_tasks / group_size / steps / lr", [cfg["n_tasks"], cfg["group_size"], cfg["steps"], cfg["lr"]],
          [16, 8, 200, 1e-5], f"{CP3} 结论")
    c.add("rollout sampling (temperature, top_k) / memory mode",
          [client["temperature"], client["top_k"], cfg["vllm"]["memory_mode"]], [1.0, 0, "colocate"], f"{CP3} 结论")
    c.add("config kl.max_sequences / eval.k", [cfg["kl"]["max_sequences"], cfg["eval"]["k"]], [64, 8], REV)

    recs = [json.loads(line) for line in (a.run / "log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = Counter(r.get("type") for r in recs)
    steps = sorted((r for r in recs if r.get("type") == "step"), key=lambda r: r["step"])
    start = next(r for r in recs if r.get("type") == "start")
    end = [r for r in recs if r.get("type") == "end"]
    c.add("training tasks", start["tasks"], 1018, f"{CP3} 结论")
    c.add("steps logged, 1..200 once each", [s["step"] for s in steps], list(range(1, 201)), f"{CP3} 结论")
    c.add("episodes per step", sorted({s["episodes"] for s in steps}), [128], f"{CP3} 结论（n_tasks 16 × group_size 8）")
    c.add("stop records", kinds.get("stop", 0), 0, f"{CP3} 结论（没有任何 stop 记录）")
    c.add("end status", [e["status"] for e in end], ["done"], f"{CP3} 结论")
    elapsed = end[0]["elapsed_s"]
    c.add("total wall time (minutes, 20 h 57 min)", elapsed / 60, 20 * 60 + 57, f"{CP3} 结论", 0)
    c.add("total wall time (s)", elapsed, 75414, f"{PROGRESS} / 正式训练结束", 0)

    seg_means: dict[str, dict[str, float]] = {}
    for (lo, hi), row in CURVE_ROWS.items():
        seg = [s for s in steps if lo <= s["step"] <= hi]
        means = {}
        for (field, dec, scale), reported in zip(CURVE_COLUMNS, row):
            dec = CURVE_DECIMALS_OVERRIDE.get(((lo, hi), field), dec)
            m = scale * statistics.fmean(step_value(s, field) for s in seg)
            means[field] = m
            c.add(f"curve {lo}-{hi} {field}", m, reported, CURVE, dec)
        seg_means[f"{lo}-{hi}"] = means

    def col(k: str) -> list[float]:
        return [s[k] for s in steps]

    c.add("logprob diff mean over steps", statistics.fmean(col("logprob_diff_mean")), 0.0015, f"{CURVE} 表下第 1 条", 4)
    c.add("logprob diff mean, largest step", max(col("logprob_diff_mean")), 0.0020, f"{CURVE} 表下第 1 条", 4)
    c.add("logprob diff p99, mean over steps", statistics.fmean(col("logprob_diff_p99")), 0.0165,
          f"{CURVE} 表下第 1 条", 4)
    c.add("logprob diff, largest single token", max(col("logprob_diff_max")), 0.573, f"{PROGRESS} / 正式训练结束", 3)
    c.add("TIS capped fraction, all steps", max(col("tis_capped_fraction")), 0.0, f"{CURVE} 表下第 1 条")
    c.add("GradScaler skipped steps", sum(1 for s in steps if s["skipped"]), 0, f"{CURVE} 表下第 2 条")
    c.add("greedy equal after every sync", all(col("sync_greedy_equal")), True, f"{CURVE} 表下第 2 条")
    c.add("sync logprob MAD, largest", max(col("sync_logprob_mad")), 0.00136, f"{CURVE} 表下第 2 条", 5)
    c.add("trainer peak allocated GiB", max(col("peak_allocated_gib")), 5.86, f"{CURVE} 表下第 3 条", 2)
    c.add("trainer peak reserved GiB", max(col("peak_reserved_gib")), 6.04, f"{CURVE} 表下第 3 条", 2)
    c.add("kl_sequences every step", sorted(set(col("kl_sequences"))), [64], REV)
    for k, v, d in (("step_s", 353, 0), ("sample_s", 74, 0), ("train_s", 275, 0), ("save_s", 3, 0),
                    ("sync_load_check_s", 0.5, 1)):
        c.add(f"mean {k}", statistics.fmean(col(k)), v, f"{REV} 第 4 条（整个正式训练平均）", d)
    c.add_extreme("format error rate per call, worst step (step, %)", steps, "format_error_rate_per_tool_call", max,
                  6, 6.7, f"{PROGRESS} / 正式训练结束", 1, 100)
    c.add_extreme("mixed-group fraction, lowest step (step, value)", steps, "non_degenerate_fraction", min,
                  186, 0.375, f"{PROGRESS} / 正式训练结束", 3)

    mixed = col("non_degenerate_fraction")
    ent = col("completion_entropy_mean")
    threshold = cfg["stop"]["entropy_collapse_ratio"] * statistics.fmean(ent[:cfg["stop"]["entropy_baseline_steps"]])
    low = min(range(len(ent)), key=lambda i: ent[i])
    window = mixed[max(0, low - 9):low + 1]
    c.add("entropy threshold (0.3 x first-5 mean)", threshold, 0.248, f"{REV} 第 3 条", 3)
    c.add("lowest-entropy step", steps[low]["step"], 195, f"{REV} 第 3 条")
    c.add("entropy at step 195", ent[low], 0.228, f"{REV} 第 3 条", 3)
    c.add("mixed-group fraction, mean of the 10 steps up to 195", statistics.fmean(window), 0.66, f"{REV} 第 3 条", 2)
    c.add("entropy at step 200 (back up to 0.42)", ent[-1], 0.42, f"{REV} 第 3 条", 2)
    c.add("entropy stop would not fire (needs mixed mean < 0.25)", statistics.fmean(window) >= 0.25, True,
          f"{REV} 第 3 条")

    dev_ids = [line.strip() for line in a.dev_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    logged_eval = {r["step"]: r for r in recs if r.get("type") == "eval"}
    solved: dict[int, int] = {}
    dev_rows = []
    for step, n_rep, s_rep, t_rep in zip(DEV_STEPS, DEV_SOLVED, DEV_STRICT_PCT, DEV_TC_PCT):
        ev = [json.loads(line) for line in (a.run / "eval" / f"step-{step}.jsonl").open(encoding="utf-8")
              if line.strip()]
        per_task: dict[str, list[int]] = defaultdict(list)
        for r in ev:
            per_task[r["task_id"]].append(r["k"])
        n = sum(r["reward"] for r in ev)
        tc = sum(int(bool(r["grade"].get("pass_non_generated"))) for r in ev)
        solved[step] = n
        dev_rows.append({"step": step, "episodes": len(ev), "solved": n, "tc_solved": tc, "strict": n / len(ev),
                         "test_challenge": tc / len(ev)})
        c.add(f"dev step {step} episodes / errors", [len(ev), sum(1 for r in ev if r.get("error"))], [800, 0], DEV)
        c.add(f"dev step {step} tasks are dev_ids, K=8 each",
              sorted(per_task) == sorted(dev_ids) and all(sorted(v) == list(range(8)) for v in per_task.values()),
              True, f"{DEV}（100 题，K=8）")
        c.add(f"dev step {step} strict solved", n, n_rep, DEV)
        c.add(f"dev step {step} strict %", 100 * n / len(ev), s_rep, DEV, 1)
        c.add(f"dev step {step} test+challenge %", 100 * tc / len(ev), t_rep, DEV, 1)
        c.add(f"dev step {step} equals log.jsonl eval record", round(n / len(ev), 3),
              logged_eval[step]["solve_rate_strict"], f"{a.run}/log.jsonl eval")

    choice = select_checkpoint(solved, 800)
    c.add("best dev step", [choice["best_step"], choice["best_solved"]], [175, 557], DEV)
    c.add("steps less than 1 point below the best", choice["within_1_point"], [150, 175, 200], DEV)
    c.add("chosen checkpoint", choice["chosen"], 150, f"{DEV}；{CP3} 结论")
    c.add("step 125 is 1.75 points below the best", 100 * (solved[175] - solved[125]) / 800, 1.75, DEV, 2)
    c.add("step 100 dev beats step 0 (dev_gain_check_step)", solved[100] > solved[0], True,
          f"{CP3} 第 1–5 步各自的结论 第 4 步")
    c.add("dev GRPO step 150 minus step 0 (points)", 100 * (solved[150] - solved[0]) / 800, 12.5,
          f"{CP3} 留出集三方对比 表下第 2 条", 1)

    # Other readings of the two numbers corrected in the erratum, so the auditor can see which one the report used.
    def seg_steps(lo: int, hi: int) -> list[dict[str, Any]]:
        return [s for s in steps if lo <= s["step"] <= hi]

    def fmt_readings(lo: int, hi: int) -> dict[str, float]:
        g = seg_steps(lo, hi)
        mean = statistics.fmean(s["format_error_rate_per_tool_call"] for s in g)
        errors = sum(round(s["format_error_rate_per_tool_call"] * s["tool_calls"]) for s in g)
        return {"mean_of_step_rates_pct": round(100 * mean, 4),
                "mean_rounded_to_4_places_then_pct": round(100 * round(mean, 4), 4),
                "pooled_errors_over_calls_pct": round(100 * errors / sum(s["tool_calls"] for s in g), 4)}

    diagnostics = {
        "format_error_rate_per_call_by_segment": {f"{lo}-{hi}": fmt_readings(lo, hi) for lo, hi in CURVE_ROWS},
        "mixed_group_fraction": {
            "steps_186_195_window_of_step_195": statistics.fmean(window),
            "segment_176_200_mean": statistics.fmean(s["non_degenerate_fraction"] for s in seg_steps(176, 200)),
            "steps_181_190_window_of_step_190": statistics.fmean(s["non_degenerate_fraction"]
                                                                 for s in seg_steps(181, 190)),
        },
    }
    tables = {"curve": curve_table(steps), "dev": dev_table(dev_rows)}
    result = {"checks": len(c.items), "failed": len(c.failed), "failures": c.failed,
              "corrected_by_erratum": [{k: t[k] for k in ("item", "computed", "reported_before_erratum", "reported")}
                                       for t in c.corrected],
              "half_unit_ties": [{k: t[k] for k in ("item", "computed", "reported")} for t in c.ties],
              "diagnostics": diagnostics, "choice": choice, "dev": dev_rows, "curve": seg_means,
              "tables_markdown": tables, "items": c.items}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (a.out.parent / "training-tables.md").write_text(
        "<!-- generated by scripts/closeout/verify_training.py: exact values rounded once -->\n\n"
        + tables["curve"] + "\n" + tables["dev"], encoding="utf-8")
    for it in c.items:
        print(("ok   " if it["ok"] else "FAIL ") + f"{it['item']}: computed {it['computed']!r} reported {it['reported']!r}")
    print(tables["curve"])
    print(tables["dev"])
    print(f"{len(c.items)} checks, {len(c.failed)} failed, {len(c.corrected)} checked against the erratum, "
          f"{len(c.ties)} on a half-unit tie; chosen step {choice['chosen']} -> {a.out}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
