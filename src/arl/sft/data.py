"""Build the SFT dataset from the DeepSeek trajectories.

    PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/vllm/bin/python -m arl.sft.data build
    # optional, with the base model served by vLLM: compare every prefix with the server's own tokenization
    PYTHONPATH=src ~/.venvs/vllm/bin/python -m arl.sft.data crosscheck --url http://127.0.0.1:8000

Rendering needs the Qwen3 tokenizer and vLLM's hermes tool parser, so it runs in the vLLM venv (the same
transformers that renders prompts at inference); the filter is plain Python. ``Renderer`` itself only needs
transformers and is also the prompt renderer of the token-level rollout client (``arl.rollout.tokenclient``).

One training sample per assistant turn. Qwen3's template drops the empty think block from earlier assistant
turns, while the generation prompt with ``enable_thinking=False`` ends in ``<think>\\n\\n</think>\\n\\n``. A
whole trajectory rendered as one sequence therefore never matches what the model saw when it generated a
non-final turn. Each sample is instead exactly the inference input for that turn (all previous messages plus
the generation prompt, loss masked) followed by the turn's output and ``<|im_end|>`` (loss on).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from arl.env.task import Task, load_tasks
from arl.env.tools import TOOL_SCHEMAS

MODEL = "Qwen/Qwen3-1.7B"
EPISODES = Path("runs/deepseek-sft/episodes.jsonl")
TASKS = Path("data/tasks")
OUT = Path("data/sft/v2")
MAX_TURNS = 8
MAX_PER_FUNCTION = 5  # audit revision 2A-1; v1 used 3
VAL_FRACTION = 0.05
MAX_INCONSISTENT = 0.02
SEED = 0
IM_END = "<|im_end|>"

# first failing check wins; the order is the order of the plan
DISCARD_REASONS = ("not_solved", "format_error", "nudged", "truncated", "too_many_turns", "changed_outside_pkg",
                   "duplicate_task", "function_cap", "inconsistent")


# -- filter -----------------------------------------------------------------------------------------


def discard_reason(rec: dict[str, Any]) -> str | None:
    if "error" in rec or rec["reward"] != 1:
        return "not_solved"
    if rec["format_errors"] or rec.get("unparsed_tool_calls"):
        return "format_error"
    if rec.get("nudged"):
        return "nudged"
    if rec["truncated"] or rec["stop_reason"] == "context_limit":
        return "truncated"
    if rec["turns"] > MAX_TURNS:
        return "too_many_turns"
    changed = rec["grade"]["files_changed"]
    if not changed or not all(p.startswith("pkg/") for p in changed):
        return "changed_outside_pkg"
    return None


def select(records: Iterable[dict[str, Any]], tasks: dict[str, Task], seed: int = SEED,
           max_per_function: int = MAX_PER_FUNCTION) -> tuple[list[dict[str, Any]], Counter]:
    """Trajectories passing every filter: at most one per task, at most ``max_per_function`` per bug source.

    Within a function, candidates are taken in a seeded random order, preferring bug types not yet taken.
    """
    discards: Counter = Counter()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        reason = discard_reason(r)
        if reason:
            discards[reason] += 1
        else:
            by_task[r["task_id"]].append(r)
    by_fn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tid, recs in by_task.items():
        recs.sort(key=lambda r: r["k"])
        discards["duplicate_task"] += len(recs) - 1
        by_fn[tasks[tid].meta["bug_source"]].append(recs[0])
    rng = random.Random(f"sftdata-{seed}")
    kept = []
    for fn in sorted(by_fn):
        cands = sorted(by_fn[fn], key=lambda r: r["task_id"])
        rng.shuffle(cands)
        taken: list[dict[str, Any]] = []
        for prefer_new_type in (True, False):
            for r in cands:
                if len(taken) == max_per_function or r in taken:
                    continue
                types = {tasks[t["task_id"]].meta["bug_type"] for t in taken}
                if not prefer_new_type or tasks[r["task_id"]].meta["bug_type"] not in types:
                    taken.append(r)
        discards["function_cap"] += len(cands) - len(taken)
        kept += taken
    return sorted(kept, key=lambda r: (r["task_id"], r["k"])), discards


def split_val(trajectories: list[dict[str, Any]], tasks: dict[str, Task], seed: int = SEED,
              fraction: float = VAL_FRACTION) -> set[str]:
    """Bug source functions whose trajectories form the validation set."""
    fns = sorted({tasks[r["task_id"]].meta["bug_source"] for r in trajectories})
    random.Random(f"sftval-{seed}").shuffle(fns)
    return set(fns[:math.ceil(fraction * len(fns))])


# -- rendering ---------------------------------------------------------------------------------------


def template_arguments(arguments: Any) -> dict[str, Any]:
    """Tool call arguments as vLLM 0.31 hands them to the chat template (``chat_utils._postprocess_messages``):
    a JSON object string is parsed; empty, malformed or non-object arguments become ``{}``."""
    if not arguments:
        return {}
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The episode's OpenAI-format messages as the chat template sees them (tool arguments parsed to dicts,
    as vLLM does before rendering)."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"type": "function", "function": {
                    "name": c["function"]["name"], "arguments": template_arguments(c["function"]["arguments"])}}
                    for c in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m.get("content") or "", "tool_call_id": m.get("tool_call_id"),
                        "name": m.get("name")})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def assistant_text(message: dict[str, Any]) -> str:
    """What the model generates for this turn in non-thinking mode (hermes ``<tool_call>`` blocks)."""
    content = message.get("content") or ""
    parts = [content]
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call["function"]
        args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
        sep = "\n" if (i == 0 and content) or i > 0 else ""
        parts.append(f'{sep}<tool_call>\n{{"name": "{fn["name"]}", "arguments": '
                     f'{json.dumps(args, ensure_ascii=False)}}}\n</tool_call>')
    return "".join(parts)


class Renderer:
    def __init__(self, model: str = MODEL):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.im_end = self.tok.convert_tokens_to_ids(IM_END)
        self._parser: Any = None

    @property
    def parser(self) -> Any:
        if self._parser is None:  # vLLM is only installed in its own venv
            from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

            self._parser = Hermes2ProToolParser(self.tok)
        return self._parser

    def render(self, messages: list[dict[str, Any]], generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(messages, tools=TOOL_SCHEMAS, add_generation_prompt=generation_prompt,
                                            enable_thinking=False, tokenize=False)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def parse_matches(self, text: str, message: dict[str, Any]) -> bool:
        info = self.parser.extract_tool_calls(text, None)
        calls = message.get("tool_calls") or []
        if bool(info.tools_called) != bool(calls) or len(info.tool_calls) != len(calls):
            return False
        for got, want in zip(info.tool_calls, calls):
            if got.function.name != want["function"]["name"]:
                return False
            if json.loads(got.function.arguments) != want["function"]["arguments"]:
                return False
        return (info.content or "").strip() == (message.get("content") or "").strip()

    def turn_samples(self, rec: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """One sample per assistant turn, or ``([], kind)`` naming the first consistency check that failed."""
        msgs = template_messages(rec["messages"])
        out = []
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prefix = self.render(msgs[:j], True)
            text = assistant_text(m)
            if self.render(msgs[: j + 1], False) != prefix + text + IM_END + "\n":
                return [], "template_text"
            prefix_ids = self.encode(prefix)
            ids = prefix_ids + self.encode(text) + [self.im_end]
            if self.encode(prefix + text + IM_END + "\n") != ids + self.encode("\n"):
                return [], "token_ids"
            if not self.parse_matches(text, m):
                return [], "hermes_parse"
            out.append({"traj_id": f"{rec['task_id']}#{rec['k']}", "task_id": rec["task_id"], "turn": len(out) + 1,
                        "input_ids": ids, "loss_start": len(prefix_ids)})
        return out, None


# -- build -------------------------------------------------------------------------------------------


def _dist(xs: list[int]) -> dict[str, Any]:
    if not xs:
        return {}
    s = sorted(xs)
    return {"median": statistics.median(s), "p95": s[max(0, math.ceil(0.95 * len(s)) - 1)], "max": s[-1],
            "mean": round(statistics.mean(s), 1)}


def build(episodes: Path, tasks_dir: Path, out: Path, model: str = MODEL, seed: int = SEED,
          max_per_function: int = MAX_PER_FUNCTION) -> dict[str, Any]:
    from arl.rollout.calibrate import latest, load_records

    tasks = {t.task_id: t for t in load_tasks(tasks_dir / "train.jsonl")}
    records = list(latest(load_records(episodes)).values())
    chosen, discards = select(records, tasks, seed, max_per_function)
    renderer = Renderer(model)
    kept: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    inconsistent: Counter = Counter()
    for rec in chosen:
        samples, failure = renderer.turn_samples(rec)
        if failure:
            inconsistent[failure] += 1
        else:
            kept.append((rec, samples))
    discards["inconsistent"] = sum(inconsistent.values())
    val_fns = split_val([r for r, _ in kept], tasks, seed)
    out.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {"train": [], "val": []}
    for rec, samples in kept:
        per_split["val" if tasks[rec["task_id"]].meta["bug_source"] in val_fns else "train"].append((rec, samples))
    for name, items in per_split.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for _, samples in items:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

    def split_stats(items: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> dict[str, Any]:
        samples = [s for _, ss in items for s in ss]
        lengths = [len(s["input_ids"]) for s in samples]
        loss_tokens = sum(len(s["input_ids"]) - s["loss_start"] for s in samples)
        final = [len(ss[-1]["input_ids"]) for _, ss in items]
        return {
            "trajectories": len(items),
            "samples": len(samples),
            "bug_source_functions": len({tasks[r["task_id"]].meta["bug_source"] for r, _ in items}),
            "sample_tokens": _dist(lengths),
            "trajectory_tokens_final_turn": _dist(final),
            "turns_per_trajectory": dict(sorted(Counter(len(ss) for _, ss in items).items())),
            "tokens_total": sum(lengths),
            "assistant_tokens": loss_tokens,
            "assistant_token_fraction": round(loss_tokens / sum(lengths), 4) if lengths else None,
            "over_8192_tokens": sum(1 for n in lengths if n > 8192),
            "bug_types": dict(sorted(Counter(tasks[r["task_id"]].meta["bug_type"] for r, _ in items).items())),
        }

    checked = len(chosen)
    stats = {
        "model": model,
        "episodes_file": str(episodes),
        "episodes": len(records),
        "selected_before_rendering": checked,
        "max_per_function": max_per_function,
        "kept_per_function": dict(sorted(Counter(Counter(tasks[r["task_id"]].meta["bug_source"] for r, _ in kept).values()).items())),
        "discards": {k: discards.get(k, 0) for k in DISCARD_REASONS},
        "inconsistent_by_check": dict(inconsistent),
        "inconsistent_fraction": round(sum(inconsistent.values()) / checked, 4) if checked else None,
        "val_functions": len(val_fns),
        "all": split_stats(kept),
        "train": split_stats(per_split["train"]),
        "val": split_stats(per_split["val"]),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def crosscheck(episodes: Path, tasks_dir: Path, url: str, model: str = MODEL, seed: int = SEED,
               max_per_function: int = MAX_PER_FUNCTION) -> dict[str, Any]:
    """Compare each kept turn's prefix tokens with what the running vLLM server renders for the same request."""
    import requests

    from arl.rollout.calibrate import latest, load_records

    tasks = {t.task_id: t for t in load_tasks(tasks_dir / "train.jsonl")}
    chosen, _ = select(list(latest(load_records(episodes)).values()), tasks, seed, max_per_function)
    renderer = Renderer(model)
    checked, mismatched = 0, []
    for rec in chosen:
        msgs = rec["messages"]
        tmsgs = template_messages(msgs)
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            body = {"model": model, "messages": msgs[:j], "tools": TOOL_SCHEMAS, "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            server = requests.post(f"{url}/tokenize", json=body, timeout=60).json()["tokens"]
            checked += 1
            if server != renderer.encode(renderer.render(tmsgs[:j], True)):
                mismatched.append(f"{rec['task_id']}#{rec['k']} turn@{j}")
    return {"prefixes_checked": checked, "mismatched": len(mismatched), "examples": mismatched[:10]}


def show(path: Path, n: int, seed: int, model: str = MODEL, max_chars: int = 600) -> str:
    """``n`` random samples decoded, with the loss boundary marked. The system block and long tool results
    are shortened for display only."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    samples = [json.loads(line) for line in open(path, encoding="utf-8")]
    out = []
    for s in random.Random(f"show-{seed}").sample(samples, n):
        prefix = tok.decode(s["input_ids"][: s["loss_start"]])
        target = tok.decode(s["input_ids"][s["loss_start"]:])
        head, sep, rest = prefix.partition("<|im_end|>\n")
        system = f"{head[:120]}…[system prompt + tool schemas, {len(tok.encode(head))} tokens, shortened here]{sep}"
        parts = rest.split("<tool_response>\n")
        for i in range(1, len(parts)):
            body, end, tail = parts[i].partition("\n</tool_response>")
            if len(body) > max_chars:
                body = body[: max_chars // 2] + f"\n…[{len(body) - max_chars} chars shortened here]…\n" + body[-max_chars // 2:]
            parts[i] = body + end + tail
        header = (f"### {s['traj_id']} turn {s['turn']} ({len(s['input_ids'])} tokens, "
                  f"loss on {len(s['input_ids']) - s['loss_start']})")
        body = system + "<tool_response>\n".join(parts) + "⟦LOSS START⟧" + target + "⟦LOSS END⟧"
        out.append(f"{header}\n\n```text\n{body}\n```\n")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "crosscheck", "show"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=Path, default=EPISODES)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-per-function", type=int, default=MAX_PER_FUNCTION)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    a = ap.parse_args(argv)
    if a.command == "show":
        print(show(a.out / "train.jsonl", a.n, a.seed, a.model))
        return 0
    if a.command == "crosscheck":
        r = crosscheck(a.episodes, a.tasks, a.url, a.model, max_per_function=a.max_per_function)
        print(json.dumps(r, indent=2))
        return 0 if r["mismatched"] == 0 else 1
    stats = build(a.episodes, a.tasks, a.out, a.model, max_per_function=a.max_per_function)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats["inconsistent_fraction"] and stats["inconsistent_fraction"] > MAX_INCONSISTENT:
        print(f"[sft-data] STOP: {stats['inconsistent_fraction']:.2%} of trajectories are inconsistent (> 2%)")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
