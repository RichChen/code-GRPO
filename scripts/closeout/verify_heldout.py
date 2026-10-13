"""Recompute the held-out numbers reported in CHECKPOINT-3 from the raw records (phase 4, step 1).

    uv run python scripts/closeout/verify_heldout.py [--out runs/closeout/verify-heldout.json]

Reads only ``runs/heldout/{base,sft,grpo}.jsonl``, ``runs/heldout/compare.json`` and ``data/tasks/heldout.jsonl``.
Rates are counted again from the episode records; the paired intervals use ``arl.rl.heldout`` (cluster bootstrap
over the 273 bug source functions, 10000 resamples, seed 0). Every number is compared with ``compare.json`` and with
the value printed in CHECKPOINT-3 ("结论" and "留出集三方对比"), a few with PROGRESS or HELDOUT-PREREG. A number
matches when it equals the reported one after rounding to the reported precision; values exactly half a unit away
(binary float rounding ties such as 3.55 printed as 3.5) count as matches and are listed under ``half_unit_ties``.
Exit 1 on any mismatch. No model calls, no new analysis: only numbers that were already reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from arl.env.task import load_tasks
from arl.rl.heldout import compare, paired
from arl.rollout.calibrate import latest, load_records

MODELS = ("base", "sft", "grpo")
HELDOUT_SHA256 = "d28e0709bb81af8196837efaec77ed3833d952ea25fcc1d457aa87f15d7cc3d4"
CP3 = "docs/plan/CHECKPOINT-3.md"
TABLE = f"{CP3} 留出集三方对比"
PROGRESS = "docs/plan/PROGRESS.md 阶段 3 · 第 5 步 / 三个模型的留出集结果"
PREREG = "docs/plan/HELDOUT-PREREG.zh-CN.md 评测设置"

# Values as printed in CHECKPOINT-3 (percent, one decimal unless noted), in the order base, sft, grpo.
REPORTED_PCT: dict[str, tuple[float, float, float]] = {
    "solve_rate_strict": (13.9, 47.1, 60.3),
    "solve_rate_test_challenge": (21.5, 53.0, 65.6),
    "bug_type/arith_swap": (16.7, 55.7, 69.1),
    "bug_type/bool_swap": (4.0, 31.2, 51.3),
    "bug_type/compare_swap": (8.5, 38.0, 49.2),
    "bug_type/const_off_by_one": (18.7, 47.8, 63.2),
    "bug_type/remove_branch": (2.5, 33.2, 44.1),
    "bug_type/var_swap": (18.6, 55.5, 67.7),
    "origin/mbpp": (14.8, 48.1, 61.0),
    "origin/humaneval": (8.3, 41.2, 56.1),
    "format_error_rate_per_tool_call": (2.7, 1.9, 0.7),
    "episodes_with_format_error": (3.5, 7.3, 2.1),
}
REPORTED_TURNS = (5.24, 6.84, 6.77)
REPORTED_ALL_SOLVED = (109, 320, 581)
REPORTED_NONE_SOLVED = (1157, 413, 299)
REPORTED_TOKENIZE_CHECKED = (31470, 41073, 40609)
REPORTED_STOPS = ({"submitted": 5360, "max_turns": 627, "context_limit": 13},
                  {"submitted": 4229, "max_turns": 1766, "context_limit": 5},
                  {"submitted": 4323, "max_turns": 1677, "context_limit": 0})
# (strict mean, lo, hi, test+challenge mean, lo, hi) in points; tasks better / worse (strict).
REPORTED_PAIRED = {
    ("grpo", "sft"): ((13.2, 11.7, 14.7, 12.6, 11.1, 14.1), (667, 178), f"{CP3} 结论；{TABLE}"),
    ("sft", "base"): ((33.2, 30.4, 36.1, 31.5, 28.6, 34.4), (926, 82), f"{TABLE}；变好/变差见 {PROGRESS}"),
}
# Per-type and per-origin GRPO - SFT gains quoted under the table: range 10.9-20.1, bool_swap +20.1, HumanEval +14.9.
REPORTED_GAIN_RANGE = (10.9, 20.1)
REPORTED_GAINS = {"bug_type/bool_swap": 20.1, "origin/humaneval": 14.9}
PREREG_COUNTS = {"tasks": 1500, "bug_sources": 273, "origin/mbpp": 1290, "origin/humaneval": 210,
                 "bug_type/const_off_by_one": 368, "bug_type/var_swap": 345, "bug_type/arith_swap": 313,
                 "bug_type/remove_branch": 236, "bug_type/compare_swap": 182, "bug_type/bool_swap": 56}


def matches(computed: float, reported: float, decimals: int) -> bool:
    """True when ``computed`` rounds to ``reported`` at ``decimals`` places (half-unit tolerance)."""
    return abs(computed - reported) <= 0.5 * 10 ** -decimals + 1e-9


def on_tie(computed: float, reported: float, decimals: int) -> bool:
    """``computed`` sits exactly half a unit from ``reported``: either rounding direction is a valid rounding."""
    return abs(abs(computed - reported) - 0.5 * 10 ** -decimals) < 1e-9


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, item: str, computed: Any, reported: Any, source: str, decimals: int | None = None) -> None:
        tie = False
        if decimals is None:
            ok = computed == reported
        else:
            ok = matches(computed, reported, decimals)
            tie = on_tie(computed, reported, decimals)
        self.items.append({"item": item, "computed": computed, "reported": reported, "decimals": decimals,
                           "source": source, "ok": ok, "half_unit_tie": tie})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.items if not c["ok"]]

    @property
    def ties(self) -> list[dict[str, Any]]:
        return [c for c in self.items if c["half_unit_tie"]]


def recount(recs: list[dict[str, Any]], tasks: dict[str, Any]) -> dict[str, Any]:
    """The per-model numbers, counted directly from the episode records."""
    def rate(xs: list[int]) -> float:
        return sum(xs) / len(xs)

    groups: dict[str, list[int]] = defaultdict(list)
    per_task: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        meta = tasks[r["task_id"]].meta
        groups[f"bug_type/{meta['bug_type']}"].append(r["reward"])
        groups[f"origin/{meta['bug_source'].split('/')[0]}"].append(r["reward"])
        per_task[r["task_id"]].append(r["reward"])
    out: dict[str, Any] = {
        "episodes": len(recs),
        "tasks": len(per_task),
        "episodes_per_task": sorted(Counter(len(v) for v in per_task.values()).items()),
        "errors": sum(1 for r in recs if r.get("error")),
        "solve_rate_strict": rate([r["reward"] for r in recs]),
        "solve_rate_test_challenge": rate([int(bool(r["grade"].get("pass_non_generated"))) for r in recs]),
        "format_error_rate_per_tool_call": sum(r["format_errors"] for r in recs) / sum(r["n_tool_calls"] for r in recs),
        "episodes_with_format_error": rate([1 if r["format_errors"] else 0 for r in recs]),
        "mean_turns": statistics.fmean(r["turns"] for r in recs),
        "all_solved": sum(1 for v in per_task.values() if sum(v) == len(v)),
        "none_solved": sum(1 for v in per_task.values() if sum(v) == 0),
        "tokenize_checked": sum(r["tokenize_checked"] for r in recs),
        "tokenize_mismatches": sum(r["tokenize_mismatches"] for r in recs),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in recs)),
    }
    out.update({k: rate(v) for k, v in groups.items()})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs/heldout"))
    ap.add_argument("--tasks", type=Path, default=Path("data/tasks/heldout.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/closeout/verify-heldout.json"))
    a = ap.parse_args(argv)

    c = Checks()
    c.add("heldout.jsonl sha256", hashlib.sha256(a.tasks.read_bytes()).hexdigest(), HELDOUT_SHA256, PREREG)
    tasks = {t.task_id: t for t in load_tasks(a.tasks)}
    metas = [t.meta for t in tasks.values()]
    composition = {"tasks": len(tasks), "bug_sources": len({m["bug_source"] for m in metas})}
    composition.update({f"origin/{o}": n for o, n in Counter(m["bug_source"].split("/")[0] for m in metas).items()})
    composition.update({f"bug_type/{b}": n for b, n in Counter(m["bug_type"] for m in metas).items()})
    for k, v in PREREG_COUNTS.items():
        c.add(f"heldout composition {k}", composition.get(k), v, PREREG)

    runs = {m: [r for r in latest(load_records(a.runs / f"{m}.jsonl")).values() if r["task_id"] in tasks]
            for m in MODELS}
    raw_lines = {m: sum(1 for line in (a.runs / f"{m}.jsonl").open(encoding="utf-8") if line.strip()) for m in MODELS}
    stored = json.loads((a.runs / "compare.json").read_text(encoding="utf-8"))
    counted = {m: recount(runs[m], tasks) for m in MODELS}

    for i, m in enumerate(MODELS):
        n = counted[m]
        src = f"{CP3} 结论（每个模型 6000 个 episode，1500 题，每题 4 次）"
        c.add(f"{m} raw lines", raw_lines[m], 6000, src)
        c.add(f"{m} episodes", n["episodes"], 6000, src)
        c.add(f"{m} tasks", n["tasks"], 1500, src)
        c.add(f"{m} episodes per task", n["episodes_per_task"], [(4, 1500)], src)
        c.add(f"{m} errors", n["errors"], 0, f"{CP3} 第 5 步（报错 0）")
        c.add(f"{m} tokenize mismatches", n["tokenize_mismatches"], 0, f"{CP3} 第 5 步（/tokenize 不一致 0）")
        c.add(f"{m} tokenize checked", n["tokenize_checked"], REPORTED_TOKENIZE_CHECKED[i], PROGRESS)
        c.add(f"{m} stop reasons", {k: n["stop_reasons"].get(k, 0) for k in REPORTED_STOPS[i]}, REPORTED_STOPS[i],
              PROGRESS)
        for key, vals in REPORTED_PCT.items():
            c.add(f"{m} {key} (CHECKPOINT-3, %)", 100 * n[key], vals[i], TABLE, 1)
        c.add(f"{m} mean_turns (CHECKPOINT-3)", n["mean_turns"], REPORTED_TURNS[i], TABLE, 2)
        c.add(f"{m} tasks solved 4/4", n["all_solved"], REPORTED_ALL_SOLVED[i], TABLE)
        c.add(f"{m} tasks solved 0/4", n["none_solved"], REPORTED_NONE_SOLVED[i], TABLE)

        s = stored[m]
        cj = f"{a.runs}/compare.json {m}"
        for key, val in (("solve_rate_strict", n["solve_rate_strict"]),
                         ("solve_rate_test_challenge", n["solve_rate_test_challenge"]),
                         ("format_error_rate_per_tool_call", n["format_error_rate_per_tool_call"]),
                         ("episodes_with_format_error", n["episodes_with_format_error"])):
            c.add(f"{m} {key} (compare.json)", val, s[key], cj, len(str(s[key]).split(".")[1]))
        for bt, v in s["by_bug_type"].items():
            c.add(f"{m} bug_type/{bt} (compare.json)", n[f"bug_type/{bt}"], v["solve_rate"], cj, 3)
        for o, v in s["by_origin"].items():
            c.add(f"{m} origin/{o} (compare.json)", n[f"origin/{o}"], v["solve_rate"], cj, 4)
        c.add(f"{m} mean_turns (compare.json)", n["mean_turns"], s["mean_turns"], cj, 2)
        c.add(f"{m} solved 4/4 and 0/4 (compare.json)", [n["all_solved"], n["none_solved"]],
              [s["solve_count_distribution"]["4"], s["solve_count_distribution"]["0"]], cj)

    pairs: dict[str, Any] = {}
    for (x, y), (pts, (better, worse), src) in REPORTED_PAIRED.items():
        name = f"paired_{x}_minus_{y}"
        strict = paired(runs[x], runs[y], tasks, "reward")
        tc = paired(runs[x], runs[y], tasks, "non_generated")
        pairs[name] = {"strict": strict, "test_challenge": tc}
        c.add(f"{name} clusters / bootstrap / seed", [strict["clusters"], strict["bootstrap"], strict["seed"]],
              [273, 10000, 0], src)
        got = (strict["mean_diff"], *strict["ci95"], tc["mean_diff"], *tc["ci95"])
        labels = ("strict mean", "strict lo", "strict hi", "test+challenge mean", "test+challenge lo",
                  "test+challenge hi")
        for label, g, r in zip(labels, got, pts):
            c.add(f"{name} {label} (points)", round(100 * g, 4), r, src, 1)
        c.add(f"{name} tasks better / worse", [strict["tasks_better"], strict["tasks_worse"]], [better, worse], src)
        # Independent mean of per-task differences, to check the bootstrap's point estimate.
        rx = defaultdict(list)
        ry = defaultdict(list)
        for r in runs[x]:
            rx[r["task_id"]].append(r["reward"])
        for r in runs[y]:
            ry[r["task_id"]].append(r["reward"])
        mean_diff = statistics.fmean(sum(rx[t]) / len(rx[t]) - sum(ry[t]) / len(ry[t]) for t in rx)
        c.add(f"{name} strict mean (recounted, points)", 100 * mean_diff, pts[0], src, 1)
        for kind in ("strict", "test_challenge"):
            c.add(f"{name} {kind} equals compare.json", pairs[name][kind], stored[name][kind],
                  f"{a.runs}/compare.json {name}")

    gains = {k: 100 * (counted["grpo"][k] - counted["sft"][k]) for k in REPORTED_PCT
             if k.startswith(("bug_type/", "origin/"))}
    c.add("GRPO - SFT gain, smallest over types and origins (points)", min(gains.values()), REPORTED_GAIN_RANGE[0],
          f"{TABLE} 表下第 1 条", 1)
    c.add("GRPO - SFT gain, largest over types and origins (points)", max(gains.values()), REPORTED_GAIN_RANGE[1],
          f"{TABLE} 表下第 1 条", 1)
    c.add("GRPO > SFT on every type and origin", all(g > 0 for g in gains.values()), True, f"{TABLE} 表下第 1 条")
    for k, v in REPORTED_GAINS.items():
        c.add(f"GRPO - SFT gain {k} (points)", gains[k], v, f"{TABLE} 表下第 1 条", 1)

    regenerated = json.loads(json.dumps(compare(runs, tasks)))
    c.add("compare() on the raw records reproduces compare.json exactly", regenerated == stored, True,
          f"{a.runs}/compare.json")

    result = {"checks": len(c.items), "failed": len(c.failed), "failures": c.failed,
              "half_unit_ties": [{k: t[k] for k in ("item", "computed", "reported")} for t in c.ties],
              "summary": {m: {k: counted[m][k] for k in ("episodes", "tasks", "solve_rate_strict",
                                                          "solve_rate_test_challenge", "all_solved", "none_solved")}
                          for m in MODELS},
              "paired": pairs, "gains_grpo_minus_sft_points": {k: round(v, 2) for k, v in gains.items()},
              "items": c.items}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for it in c.items:
        print(("ok   " if it["ok"] else "FAIL ") + f"{it['item']}: computed {it['computed']!r} reported {it['reported']!r}")
    print(f"{len(c.items)} checks, {len(c.failed)} failed, {len(c.ties)} on a half-unit tie -> {a.out}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
