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
import math
import random
import re
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

from arl.env.episode import FORMAT_ERROR_KINDS, MAX_TURNS, NUDGE_MESSAGE, EpisodeResult, run_episode
from arl.env.grader import grade
from arl.env.task import Task, load_tasks
from arl.rollout.client import ClientConfig, OpenAIPolicy
from arl.taskgen.sources import sha256_file

K = 4
CONCURRENCY = 8

RUNS = Path("runs/calibration")
TASKS = Path("data/tasks")
PILOT_PER_SPLIT = 50
TRAIN_MIN_SAMPLE = 1200


def request_seed(task_id: str, k: int) -> int:
    return int(hashlib.sha256(f"{task_id}#{k}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def policy_cost(policy: Any) -> float:
    config = getattr(policy, "config", None)
    return round(config.cost_cny(policy.usage), 6) if config is not None and hasattr(policy, "usage") else 0.0


def behaviour(result: EpisodeResult) -> dict[str, Any]:
    """Diagnostics for the model-selection scan: nudges, edits, visible tests passing during the episode."""
    nudge_turn = None
    assistant_turns = 0
    run_tests_passed = 0
    for m in result.messages:
        if m["role"] == "assistant":
            assistant_turns += 1
        elif m["role"] == "user" and m.get("content") == NUDGE_MESSAGE:
            nudge_turn = assistant_turns
        elif m["role"] == "tool" and m.get("name") == "run_tests" and (m.get("content") or "").startswith("exit code: 0"):
            run_tests_passed += 1
    edits = [c for c in result.tool_calls if c.name == "edit_file"]
    return {
        "nudged": result.nudged,
        "nudge_turn": nudge_turn,
        "first_turn_tool_call": any(c.turn == 1 for c in result.tool_calls),
        "tool_call_after_nudge": nudge_turn is not None and any(c.turn > nudge_turn for c in result.tool_calls),
        "edit_calls": len(edits),
        "edit_ok": sum(1 for c in edits if c.ok),
        "run_tests_passed": run_tests_passed,
    }


def make_policy(config: ClientConfig, seed: int | None) -> Any:
    """``OpenAIPolicy`` for the chat endpoint, ``TokenPolicy`` for ``endpoint: "tokens"``."""
    if config.endpoint == "tokens":
        from arl.rollout.tokenclient import TokenPolicy

        return TokenPolicy(config, seed=seed)
    if config.endpoint != "chat":
        raise ValueError(f"unknown endpoint {config.endpoint!r}")
    return OpenAIPolicy(config, seed=seed)


def token_stats(policy: Any, with_tokens: bool = False) -> dict[str, Any]:
    """Per-turn token data of a ``TokenPolicy`` episode (lengths always, the token ids on request)."""
    if not hasattr(policy, "tokenize_checked"):
        return {}
    turns = policy.turns
    out: dict[str, Any] = {
        "turn_prompt_tokens": [len(t["prompt_ids"]) for t in turns],
        "turn_completion_tokens": [len(t["completion_ids"]) for t in turns],
        "tokenize_checked": policy.tokenize_checked,
        "tokenize_mismatches": policy.tokenize_mismatches,
    }
    if with_tokens:
        out["turn_tokens"] = turns
    return out


def episode_record(task: Task, k: int, result: EpisodeResult, policy: Any, session: str,
                   started: float, with_messages: bool = False, with_tokens: bool = False) -> dict[str, Any]:
    u = policy.usage
    extra = {"messages": result.messages} if with_messages else {}
    extra.update(token_stats(policy, with_tokens))
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
        "cache_hit_tokens": u.cache_hit_tokens,
        "cost_cny": policy_cost(policy),
        "final_context_tokens": u.final_context_tokens,
        "context_tokens_est": result.context_tokens_est,
        "finish_reasons": u.finish_reasons,
        **behaviour(result),
        "grade": result.grade,
        "tool_calls": [c.__dict__ for c in result.tool_calls],
        **extra,
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


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Drop Authorization headers / bearer tokens and any known secret from an error message."""
    text = re.sub(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^'\",}\s]+(\s+[^'\",}\s]+)?", r"\1[redacted]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[redacted]")
    return text


class CostLedger:
    """Cumulative paid-API spend, persisted to ``path`` after every episode; resumes from the records."""

    def __init__(self, path: Path, budget_cny: float, prices: dict[str, float] | None, records: list[dict[str, Any]]):
        self.path, self.budget, self.prices = path, budget_cny, prices
        prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.preflight = prior.get("preflight_cny", 0.0)
        self.episodes_cny = sum(r.get("cost_cny", 0.0) for r in records)
        self.episodes = sum(1 for r in records if "cost_cny" in r)
        self.max_episode = max((r.get("cost_cny", 0.0) for r in records), default=0.0)
        self.save()

    @property
    def spent(self) -> float:
        return self.preflight + self.episodes_cny

    def add(self, cost: float) -> None:
        self.episodes_cny += cost
        self.episodes += 1
        self.max_episode = max(self.max_episode, cost)
        self.save()

    def exhausted(self, in_flight: int) -> bool:
        """True once the spend, plus a worst-case allowance for the episodes still running, reaches the budget."""
        return self.spent + in_flight * self.max_episode >= self.budget

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "budget_cny": self.budget, "spent_cny": round(self.spent, 4), "preflight_cny": round(self.preflight, 6),
            "episodes_cny": round(self.episodes_cny, 4), "episodes": self.episodes,
            "mean_episode_cny": round(self.episodes_cny / self.episodes, 5) if self.episodes else None,
            "max_episode_cny": round(self.max_episode, 5), "price_cny_per_mtok": self.prices,
            "updated_at": round(time.time(), 1),
        }, indent=2) + "\n", encoding="utf-8")


def run_calibration(tasks: list[Task], out_path: Path, config: ClientConfig, k: int = K, concurrency: int = CONCURRENCY,
                    max_turns: int = MAX_TURNS, policy_factory: Callable[[Task, int], Any] | None = None,
                    max_consecutive_errors: int = 20, log: Callable[[str], None] = print,
                    ledger: CostLedger | None = None, with_messages: bool = False,
                    with_tokens: bool = False) -> list[dict[str, Any]]:
    """Run every (task, k) episode not yet recorded in ``out_path``; return all records for these tasks.

    With a ``ledger``, no new episode starts once the budget would be reached (see ``CostLedger.exhausted``).
    ``with_messages`` stores the full conversation in each record (SFT trajectories); ``with_tokens`` the
    token-level client's per-turn token ids and logprobs.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = latest(load_records(out_path))
    pending = [(t, i) for i in range(k) for t in tasks if (t.task_id, i) not in done]
    session = uuid.uuid4().hex[:8]
    log(f"[calibrate] {len(tasks)} tasks x {k}: {len(done)} done, {len(pending)} to run (session {session})")
    lock = threading.Lock()
    errors = {"consecutive": 0, "total": 0}
    stop = threading.Event()
    over_budget = threading.Event()
    in_flight = {"n": 0}
    factory = policy_factory or (lambda task, i: make_policy(config, request_seed(task.task_id, i)))

    def one(task: Task, i: int) -> dict[str, Any] | None:
        with lock:
            if stop.is_set() or over_budget.is_set():
                return None
            if ledger is not None and ledger.exhausted(in_flight["n"]):
                over_budget.set()
                return None
            in_flight["n"] += 1
        started = time.time()
        policy = None
        try:
            policy = factory(task, i)
            result = run_episode(task, policy, max_turns=max_turns)
            rec = episode_record(task, i, result, policy, session, started, with_messages, with_tokens)
        except Exception as exc:  # server errors etc.: record, retry on resume
            rec = {"task_id": task.task_id, "k": i, "session": session,
                   "error": redact(f"{type(exc).__name__}: {exc}", [config.api_key])[:500]}
            if policy is not None:
                rec["cost_cny"] = policy_cost(policy)
        with lock:
            in_flight["n"] -= 1
            if ledger is not None:
                ledger.add(rec.get("cost_cny", 0.0))
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
    if over_budget.is_set():
        log(f"[calibrate] budget reached: spent {ledger.spent:.4f} of {ledger.budget} CNY; stopped launching episodes")
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


def _quantile(xs: list[int], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, math.ceil(q * len(xs)) - 1)]  # nearest rank


def diagnostics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Behaviour metrics for the model-selection scan (records without them are counted from tool calls)."""
    def first_turn_tool(r: dict[str, Any]) -> bool:
        return r.get("first_turn_tool_call", any(c["turn"] == 1 for c in r["tool_calls"]))

    edit_calls = sum(r.get("edit_calls", r["tool_names"].get("edit_file", 0)) for r in recs)
    edit_ok = sum(r.get("edit_ok", sum(1 for c in r["tool_calls"] if c["name"] == "edit_file" and c["ok"])) for r in recs)
    nudged = [r for r in recs if r.get("nudged")]
    solved_turns = [r["turns"] for r in recs if r["reward"] == 1]
    costs = [r["cost_cny"] for r in recs if "cost_cny" in r]
    return {
        "first_turn_no_tool_call": _rate([0 if first_turn_tool(r) else 1 for r in recs]),
        "nudged": _rate([1 if r.get("nudged") else 0 for r in recs]),
        "tool_call_after_nudge_of_nudged": _rate([1 if r.get("tool_call_after_nudge") else 0 for r in nudged]),
        "edit_calls": edit_calls,
        "edit_success_rate": round(edit_ok / edit_calls, 3) if edit_calls else None,
        "episodes_with_successful_edit": _rate([1 if any(c["name"] == "edit_file" and c["ok"] for c in r["tool_calls"]) else 0
                                                for r in recs]),
        "episodes_visible_pass_final": _rate([1 if r["grade"]["visible_pass"] else 0 for r in recs]),
        "episodes_run_tests_passed": _rate([1 if r.get("run_tests_passed") else 0 for r in recs]),
        "episodes_visible_pass_any": _rate([1 if r["grade"]["visible_pass"] or r.get("run_tests_passed") else 0 for r in recs]),
        "turns_distribution": dict(sorted(Counter(r["turns"] for r in recs).items())),
        "solved_turns_median": statistics.median(solved_turns) if solved_turns else None,
        "solved_turns_p90": _quantile(solved_turns, 0.9),
        "solved_turns_distribution": dict(sorted(Counter(solved_turns).items())),
        "cost_cny_total": round(sum(costs), 4) if costs else None,
        "cost_cny_mean_per_episode": round(statistics.mean(costs), 5) if costs else None,
    }


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
        **({"tokenize_checked": sum(r["tokenize_checked"] for r in recs),
            "tokenize_mismatches": sum(r["tokenize_mismatches"] for r in recs)}
           if recs and all("tokenize_checked" in r for r in recs) else {}),
        "diagnostics": diagnostics(recs),
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


def preflight(config: ClientConfig, cost_file: Path, budget_cny: float | None) -> int:
    """One minimal request to check the key and the model name before any episode is run."""
    policy = OpenAIPolicy(config)
    try:
        reply = policy([{"role": "user", "content": "Reply with ok."}], [])
    except Exception as exc:
        print(f"[preflight] FAILED: {redact(f'{type(exc).__name__}: {exc}', [config.api_key])[:500]}")
        return 3
    cost = policy_cost(policy)
    if config.price_cny_per_mtok:
        prior = json.loads(cost_file.read_text(encoding="utf-8")) if cost_file.exists() else {}
        cost_file.parent.mkdir(parents=True, exist_ok=True)
        cost_file.write_text(json.dumps({**prior, "budget_cny": budget_cny,
                                         "preflight_cny": round(prior.get("preflight_cny", 0.0) + cost, 6),
                                         "spent_cny": round(prior.get("spent_cny", 0.0) + cost, 4),
                                         "price_cny_per_mtok": config.price_cny_per_mtok}, indent=2) + "\n",
                             encoding="utf-8")
    print(f"[preflight] ok: model={config.model} reply={reply.get('content')!r} "
          f"usage={policy.usage.prompt_tokens}+{policy.usage.completion_tokens} tokens "
          f"(cache hit {policy.usage.cache_hit_tokens}) cost={cost:.6f} CNY")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "run", "select", "summary", "preflight"])
    ap.add_argument("--config", type=Path, default=Path("configs/rollout.json"))
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=TASKS / "calibrated")
    ap.add_argument("--split", choices=["train", "heldout"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--ids", type=Path, default=None, help="only the task ids listed in this file (with --split)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--budget-cny", type=float, default=None, help="paid APIs: stop starting episodes at this spend")
    ap.add_argument("--cost-file", type=Path, default=None, help="cumulative spend (default <runs>/cost.json)")
    ap.add_argument("--with-messages", action="store_true", help="store the full conversation in each record")
    ap.add_argument("--with-tokens", action="store_true", help="tokens endpoint: store each turn's token ids")
    a = ap.parse_args(argv)
    cost_file = a.cost_file or a.runs / "cost.json"

    if a.command == "select":
        print(json.dumps(select(a.tasks, a.runs, a.out, a.k), indent=2, ensure_ascii=False))
        return 0
    if a.command == "preflight":
        return preflight(ClientConfig.load(a.config), cost_file, a.budget_cny)
    if a.command == "pilot" or (a.command == "summary" and not a.split):
        tasks, name = pilot_tasks(a.tasks, a.seed), a.name or "pilot"
    else:
        if not a.split:
            ap.error("--split is required")
        tasks = load_tasks(a.tasks / f"{a.split}.jsonl")
        if a.ids:
            wanted = {line.strip() for line in a.ids.read_text(encoding="utf-8").splitlines() if line.strip()}
            tasks = [t for t in tasks if t.task_id in wanted]
            if len(tasks) != len(wanted):
                ap.error(f"{len(wanted) - len(tasks)} ids in {a.ids} are not in the {a.split} split")
        if a.sample and a.sample < len(tasks):
            tasks = stratified_sample(tasks, a.sample, a.seed)
        name = a.name or a.split
    path = a.runs / f"{name}.jsonl"
    by_id = {t.task_id: t for t in tasks}
    if a.command == "summary":
        recs = [r for r in load_records(path) if r["task_id"] in by_id]
    else:
        config = ClientConfig.load(a.config)
        ledger = None
        if config.price_cny_per_mtok:
            if a.budget_cny is None:
                ap.error("--budget-cny is required for a priced (paid) API")
            ledger = CostLedger(cost_file, a.budget_cny, config.price_cny_per_mtok, load_records(path))
        recs = run_calibration(tasks, path, config, k=a.k, concurrency=a.concurrency, max_turns=a.max_turns,
                               ledger=ledger, with_messages=a.with_messages, with_tokens=a.with_tokens)
    summary = summarize(recs, by_id, a.k)
    (a.runs / f"{name}-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
