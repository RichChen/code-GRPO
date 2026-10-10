"""Phase 2C step 1: checks that the token client (``/v1/completions``) behaves like the chat client.

* ``ids``: a seeded sample of dev task ids for the greedy comparison.
* ``greedy``: two temperature-0 runs (chat vs tokens, ``--with-messages``), episode by episode: the tool call
  sequence (name and arguments) and the reward must be identical.
* ``replay``: every assistant turn of a chat run, re-asked through the token client with the run's exact history.
* ``stats``: the token client's strict solve rate on dev must fall inside the bootstrap 95% CI (over tasks) of the
  chat client's reference run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from arl.rollout.calibrate import latest, load_records
from arl.sft.evaluate import bootstrap_ci, per_task_rates


def sample_ids(ids_file: Path, n: int, seed: int = 0) -> list[str]:
    ids = sorted(line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return sorted(random.Random(f"greedy-{seed}").sample(ids, n))


def tool_sequence(rec: dict[str, Any]) -> list[tuple[str, str]]:
    return [(c["function"]["name"], c["function"]["arguments"])
            for m in rec["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]


def first_difference(a: list[Any], b: list[Any]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def greedy(chat: list[dict[str, Any]], tokens: list[dict[str, Any]]) -> dict[str, Any]:
    a, b = latest(chat), latest(tokens)
    if set(a) != set(b):
        raise ValueError(f"the runs cover different episodes: {sorted(set(a) ^ set(b))[:5]}")
    episodes = []
    for key in sorted(a):
        ra, rb = a[key], b[key]
        sa, sb = tool_sequence(ra), tool_sequence(rb)
        same = sa == sb and ra["reward"] == rb["reward"]
        ep = {"task_id": key[0], "k": key[1], "identical": same, "reward": [ra["reward"], rb["reward"]],
              "tool_calls": [len(sa), len(sb)], "stop_reason": [ra["stop_reason"], rb["stop_reason"]]}
        if not same:
            i = first_difference(sa, sb)
            ep["first_difference"] = {"call": i, "chat": sa[i] if i is not None and i < len(sa) else None,
                                      "tokens": sb[i] if i is not None and i < len(sb) else None}
        episodes.append(ep)
    return {"episodes": len(episodes), "identical": sum(e["identical"] for e in episodes),
            "tokenize_mismatches": sum(r.get("tokenize_mismatches", 0) for r in b.values()),
            "tokenize_checked": sum(r.get("tokenize_checked", 0) for r in b.values()), "detail": episodes}


def assistant_view(message: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
    return (message.get("content") or "").strip(), calls


def replay(records: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    """Feed every assistant turn's exact history from a chat run to the token client (greedy) and compare the
    reply with the chat run's message. Separates client differences from differences in the environment's output."""
    from arl.env.tools import TOOL_SCHEMAS
    from arl.rollout.tokenclient import TokenPolicy

    turns, diffs = 0, []
    for key, rec in sorted(latest(records).items()):
        msgs = rec["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            policy = TokenPolicy(config, seed=0)
            got = assistant_view(policy(msgs[:i], TOOL_SCHEMAS))
            turns += 1
            if got != assistant_view(m):
                t = policy.turns[-1]
                diffs.append({"task_id": key[0], "message": i, "chat": assistant_view(m), "tokens": got,
                              "min_logprob": min(t["completion_logprobs"]) if t["completion_logprobs"] else None})
    return {"turns": turns, "identical": turns - len(diffs), "detail": diffs}


def stats(reference: list[dict[str, Any]], new: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    ref, cur = list(latest(reference).values()), list(latest(new).values())
    ref_rates = per_task_rates(ref)
    if set(ref_rates) != set(per_task_rates(cur)):
        raise ValueError("the runs cover different tasks")
    lo, hi = bootstrap_ci(list(ref_rates.values()), seed=seed)
    rate = sum(r["reward"] for r in cur) / len(cur)
    ref_rate = sum(r["reward"] for r in ref) / len(ref)
    return {"reference_rate": round(ref_rate, 4), "reference_ci95": [round(lo, 4), round(hi, 4)],
            "new_rate": round(rate, 4), "new_episodes": len(cur), "inside": lo <= rate <= hi,
            "tokenize_checked": sum(r.get("tokenize_checked", 0) for r in cur),
            "tokenize_mismatches": sum(r.get("tokenize_mismatches", 0) for r in cur)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ids", "greedy", "replay", "stats"])
    ap.add_argument("--ids", type=Path, default=Path("data/tasks/dev_ids.txt"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a", type=Path, help="chat run (greedy) or reference run (stats)")
    ap.add_argument("--b", type=Path, help="token client run")
    ap.add_argument("--config", type=Path, help="token client config (replay)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.command == "ids":
        text = "\n".join(sample_ids(a.ids, a.n, a.seed)) + "\n"
        if a.out:
            a.out.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    if a.command == "replay":
        from arl.rollout.client import ClientConfig

        report = replay(load_records(a.a), ClientConfig.load(a.config))
    else:
        fn = greedy if a.command == "greedy" else stats
        report = fn(load_records(a.a), load_records(a.b))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
