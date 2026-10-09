"""Solve-rate calibration: run each task K times against a served model and keep the informative ones.

    # pilot: 50 train + 50 heldout tasks, K=4
    uv run python -m arl.rollout.calibrate pilot --config configs/rollout.json
    # full: every heldout task, a source-stratified sample of train tasks
    uv run python -m arl.rollout.calibrate run --config configs/rollout.json --split heldout
    uv run python -m arl.rollout.calibrate run --config configs/rollout.json --split train --sample 1200
    # summary, selection (solved 1..K-1 times) and the reference-solution reproducibility check
    uv run python -m arl.rollout.calibrate select

Every finished episode is appended to ``runs/calibration/<name>.jsonl`` as soon as it is graded, so an
interrupted run resumes where it stopped (episodes that errored are retried).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from arl.env.episode import FORMAT_ERROR_KINDS, EpisodeResult, run_episode
from arl.env.grader import grade
from arl.env.task import Task, load_tasks
from arl.rollout.client import ClientConfig, OpenAIPolicy
from arl.taskgen.sources import sha256_file

K = 4
CONCURRENCY = 8
MAX_TURNS = 8
RUNS = Path("runs/calibration")
TASKS = Path("data/tasks")
PILOT_PER_SPLIT = 50
TRAIN_MIN_SAMPLE = 1200


def request_seed(task_id: str, k: int) -> int:
    return int(hashlib.sha256(f"{task_id}#{k}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def episode_record(task: Task, k: int, result: EpisodeResult, policy: OpenAIPolicy, session: str,
                   started: float) -> dict[str, Any]:
    u = policy.usage
    return {
        "task_id": task.task_id,
        "k": k,
        "session": session,
        "started_at": round(started, 3),
        "finished_at": round(time.time(), 3),
        "reward": result.reward,
        "turns": result.turns,
        "stop_reason": result.stop_reason,
        "truncated": result.truncated,
        "duration_s": result.duration_s,
        "n_tool_calls": len(result.tool_calls),
        "format_errors": result.format_errors,
        "format_error_kinds": dict(Counter(c.error_kind for c in result.tool_calls if c.error_kind in FORMAT_ERROR_KINDS)),
        "tool_errors": sum(1 for c in result.tool_calls if c.error_kind == "tool_error"),
        "tool_names": dict(Counter(c.name for c in result.tool_calls)),
        "unparsed_tool_calls": u.unparsed_tool_calls,
        "model_calls": u.calls,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "final_context_tokens": u.final_context_tokens,
        "context_tokens_est": result.context_tokens_est,
        "finish_reasons": u.finish_reasons,
        "grade": result.grade,
        "tool_calls": [c.__dict__ for c in result.tool_calls],
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off by an interrupted write
    return out


def latest(records: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Latest successful record per (task_id, k)."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in records:
        if "error" not in r:
            out[(r["task_id"], r["k"])] = r
    return out


def run_calibration(tasks: list[Task], out_path: Path, config: ClientConfig, k: int = K, concurrency: int = CONCURRENCY,
                    max_turns: int = MAX_TURNS, policy_factory: Callable[[Task, int], Any] | None = None,
                    max_consecutive_errors: int = 20, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Run every (task, k) episode not yet recorded in ``out_path``; return all records for these tasks."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = latest(load_records(out_path))
    pending = [(t, i) for i in range(k) for t in tasks if (t.task_id, i) not in done]
    session = uuid.uuid4().hex[:8]
    log(f"[calibrate] {len(tasks)} tasks x {k}: {len(done)} done, {len(pending)} to run (session {session})")
    lock = threading.Lock()
    errors = {"consecutive": 0, "total": 0}
    stop = threading.Event()
    factory = policy_factory or (lambda task, i: OpenAIPolicy(config, seed=request_seed(task.task_id, i)))

    def one(task: Task, i: int) -> dict[str, Any] | None:
        if stop.is_set():
            return None
        started = time.time()
        try:
            policy = factory(task, i)
            result = run_episode(task, policy, max_turns=max_turns)
            rec = episode_record(task, i, result, policy, session, started)
        except Exception as exc:  # server errors etc.: record, retry on resume
            rec = {"task_id": task.task_id, "k": i, "session": session, "error": f"{type(exc).__name__}: {exc}"[:500]}
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if "error" in rec:
                errors["consecutive"] += 1
                errors["total"] += 1
                if errors["consecutive"] >= max_consecutive_errors:
                    stop.set()
            else:
                errors["consecutive"] = 0
        return rec

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(concurrency) as ex:
        futures = [ex.submit(one, t, i) for t, i in pending]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is None:
                continue
            n += 1
            if n % 50 == 0 or n == len(pending):
                rate = n / max(time.time() - t0, 1e-9) * 3600
                log(f"[calibrate] {n}/{len(pending)} episodes, {rate:.0f}/h, errors {errors['total']}")
    if stop.is_set():
        raise RuntimeError(f"stopped after {max_consecutive_errors} consecutive episode errors; see {out_path}")
    ids = {t.task_id for t in tasks}
    return [r for r in latest(load_records(out_path)).values() if r["task_id"] in ids]


# -- summaries ---------------------------------------------------------------------------------------


def solve_counts(records: Iterable[dict[str, Any]], k: int = K) -> dict[str, int]:
    """Solved count per task, only for tasks with all ``k`` attempts recorded."""
    by_task: dict[str, dict[int, int]] = defaultdict(dict)
    for r in records:
        if "error" not in r and r["k"] < k:
            by_task[r["task_id"]][r["k"]] = r["reward"]
    return {tid: sum(v.values()) for tid, v in by_task.items() if len(v) == k}


def throughput_per_hour(records: list[dict[str, Any]]) -> float | None:
    """Episodes per wall-clock hour, measured within each session and pooled."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if "started_at" in r:
            by_session[r["session"]].append(r)
    n, seconds = 0, 0.0
    for recs in by_session.values():
        if len(recs) < 2:
            continue
        n += len(recs)
        seconds += max(r["finished_at"] for r in recs) - min(r["started_at"] for r in recs)
    return round(n / seconds * 3600, 1) if seconds else None


def _rate(xs: list[int]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(records: list[dict[str, Any]], tasks: dict[str, Task], k: int = K) -> dict[str, Any]:
    recs = [r for r in records if "error" not in r and r["task_id"] in tasks]
    counts = solve_counts(recs, k)
    n_tasks = len(counts)
    tool_calls = sum(r["n_tool_calls"] for r in recs)
    fmt = sum(r["format_errors"] for r in recs)

    def group(key: Callable[[Task], str]) -> dict[str, Any]:
        g: dict[str, list[int]] = defaultdict(list)
        for r in recs:
            g[key(tasks[r["task_id"]])].append(r["reward"])
        return {name: {"episodes": len(v), "solve_rate": _rate(v)} for name, v in sorted(g.items())}

    return {
        "episodes": len(recs),
        "tasks_complete": n_tasks,
        "solve_rate": _rate([r["reward"] for r in recs]),
        "solve_count_distribution": {str(i): sum(1 for c in counts.values() if c == i) for i in range(k + 1)},
        "tasks_in_range": sum(1 for c in counts.values() if 1 <= c <= k - 1),
        "fraction_in_range": round(sum(1 for c in counts.values() if 1 <= c <= k - 1) / n_tasks, 3) if n_tasks else None,
        "format_error_rate_per_tool_call": round(fmt / tool_calls, 3) if tool_calls else None,
        "episodes_with_format_error": _rate([1 if r["format_errors"] else 0 for r in recs]),
        "format_error_kinds": dict(sum((Counter(r["format_error_kinds"]) for r in recs), Counter())),
        "unparsed_tool_calls": sum(r["unparsed_tool_calls"] for r in recs),
        "tool_calls": tool_calls,
        "tool_usage": dict(sum((Counter(r["tool_names"]) for r in recs), Counter())),
        "mean_turns": round(statistics.mean(r["turns"] for r in recs), 2) if recs else None,
        "mean_final_context_tokens": round(statistics.mean(r["final_context_tokens"] for r in recs), 1) if recs else None,
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in recs), 1) if recs else None,
        "mean_prompt_tokens_summed": round(statistics.mean(r["prompt_tokens"] for r in recs), 1) if recs else None,
        "stop_reasons": dict(Counter(r["stop_reason"] for r in recs)),
        "finish_reasons": dict(sum((Counter(r["finish_reasons"]) for r in recs), Counter())),
        "mean_episode_s": round(statistics.mean(r["duration_s"] for r in recs), 2) if recs else None,
        "throughput_episodes_per_hour": throughput_per_hour(recs),
        "by_split": group(lambda t: t.meta["split"]),
        "by_bug_type": group(lambda t: t.meta["bug_type"]),
        "by_issue_template": group(lambda t: t.meta["issue_template"]),
        "unbound_local_error": group(lambda t: "UnboundLocalError" if "UnboundLocalError" in t.issue else "other"),
    }


# -- task selection ----------------------------------------------------------------------------------


def stratified_sample(tasks: list[Task], n: int, seed: int) -> list[Task]:
    """Round-robin over bug source functions (shuffled), one task per function per round."""
    by_fn: dict[str, list[Task]] = defaultdict(list)
    for t in sorted(tasks, key=lambda t: t.task_id):
        by_fn[t.meta["bug_source"]].append(t)
    rng = random.Random(f"stratify-{seed}")
    order = sorted(by_fn)
    rng.shuffle(order)
    for fn in order:
        rng.shuffle(by_fn[fn])
    out: list[Task] = []
    while len(out) < n and any(by_fn.values()):
        for fn in order:
            if by_fn[fn] and len(out) < n:
                out.append(by_fn[fn].pop())
    return out


def pilot_tasks(tasks_dir: Path, seed: int = 0, per_split: int = PILOT_PER_SPLIT) -> list[Task]:
    out = []
    for split in ("train", "heldout"):
        tasks = sorted(load_tasks(tasks_dir / f"{split}.jsonl"), key=lambda t: t.task_id)
        out += random.Random(f"pilot-{seed}-{split}").sample(tasks, per_split)
    return out


def reference_check(tasks: list[Task], workers: int = 8) -> dict[str, Any]:
    """Grade every task's reference solution; all rewards must be 1."""
    def one(t: Task) -> tuple[str, int]:
        with tempfile.TemporaryDirectory(prefix="arl-ref-") as tmp:
            return t.task_id, grade(t, t.materialize(tmp, solved=True)).reward

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, tasks))
    failed = [tid for tid, r in results if r != 1]
    return {"checked": len(results), "reward_1": len(results) - len(failed), "failed": failed}


def select(tasks_dir: Path, runs: Path, out_dir: Path, k: int = K) -> dict[str, Any]:
    records = []
    for path in sorted(runs.glob("*.jsonl")):
        if path.name.startswith("pilot"):
            continue
        records += load_records(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"k": k, "keep_solved": [1, k - 1], "splits": {}}
    hashes = {}
    for split in ("train", "heldout"):
        tasks = load_tasks(tasks_dir / f"{split}.jsonl")
        by_id = {t.task_id: t for t in tasks}
        recs = [r for r in records if r["task_id"] in by_id]
        counts = solve_counts(recs, k)
        kept = sorted((by_id[tid] for tid, c in counts.items() if 1 <= c <= k - 1), key=lambda t: t.task_id)
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for t in kept:
                f.write(t.to_json() + "\n")
        hashes[path.name] = sha256_file(path)
        summary["splits"][split] = {
            "candidates": len(tasks),
            "calibrated": len(counts),
            "kept": len(kept),
            "distinct_bug_functions": len({t.meta["bug_source"] for t in kept}),
            "bug_types": dict(sorted(Counter(t.meta["bug_type"] for t in kept).items())),
            "calibration": summarize(recs, by_id, k),
            "reference_check": reference_check(kept),
        }
    summary["files"] = hashes
    (out_dir / "calibration.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "run", "select", "summary"])
    ap.add_argument("--config", type=Path, default=Path("configs/rollout.json"))
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=TASKS / "calibrated")
    ap.add_argument("--split", choices=["train", "heldout"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.command == "select":
        print(json.dumps(select(a.tasks, a.runs, a.out, a.k), indent=2, ensure_ascii=False))
        return 0
    if a.command == "pilot":
        tasks, name = pilot_tasks(a.tasks, a.seed), a.name or "pilot"
    else:
        if not a.split:
            ap.error("--split is required")
        tasks = load_tasks(a.tasks / f"{a.split}.jsonl")
        if a.sample and a.sample < len(tasks):
            tasks = stratified_sample(tasks, a.sample, a.seed)
        name = a.name or a.split
    path = a.runs / f"{name}.jsonl"
    by_id = {t.task_id: t for t in tasks}
    if a.command == "summary":
        recs = [r for r in load_records(path) if r["task_id"] in by_id]
    else:
        config = ClientConfig.load(a.config)
        recs = run_calibration(tasks, path, config, k=a.k, concurrency=a.concurrency)
    summary = summarize(recs, by_id, a.k)
    (a.runs / f"{name}-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
