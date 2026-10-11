"""Advantages and training sequences of a GRPO batch (no torch).

Each turn of an episode is one training sequence: ``prompt_ids + completion_ids`` exactly as sampled, with the
loss on the completion only. Turns are never re-rendered or concatenated: Qwen3's template renders earlier
assistant turns without the empty think block the generation prompt had, so turn ``t+1``'s prompt does not
extend turn ``t``'s sequence.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None = None


@dataclass
class Episode:
    task_id: str
    group: int
    index: int
    reward: float
    turns: list[Turn]
    stop_reason: str = ""
    format_errors: int = 0
    nudged: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sequence:
    input_ids: list[int]
    start: int  # first completion position
    vllm_logprobs: list[float]
    advantage: float
    episode: int
    turn: int

    @property
    def n_completion(self) -> int:
        return len(self.input_ids) - self.start


def group_advantages(rewards: list[float]) -> list[float] | None:
    """Dr. GRPO: ``r_i - mean(r)``, no division by the std. ``None`` for a group with identical rewards."""
    if len(set(rewards)) <= 1:
        return None
    mean = statistics.fmean(rewards)
    return [r - mean for r in rewards]


def turn_from_record(rec: dict[str, Any]) -> Turn:
    turn = Turn(list(rec["prompt_ids"]), list(rec["completion_ids"]), list(rec["completion_logprobs"]),
                rec.get("finish_reason"))
    if not turn.completion_ids or len(turn.logprobs) != len(turn.completion_ids):
        raise ValueError("a turn needs completion ids with one logprob each")
    return turn


def sequences(episodes: list[Episode]) -> tuple[list[Sequence], dict[str, Any]]:
    """Training sequences of the non-degenerate groups, and the batch's group statistics."""
    by_group: dict[int, list[int]] = {}
    for i, ep in enumerate(episodes):
        by_group.setdefault(ep.group, []).append(i)
    out: list[Sequence] = []
    skipped = 0
    for members in by_group.values():
        adv = group_advantages([episodes[i].reward for i in members])
        if adv is None:
            skipped += 1
            continue
        for i, a in zip(members, adv):
            for t, turn in enumerate(episodes[i].turns):
                out.append(Sequence(turn.prompt_ids + turn.completion_ids, len(turn.prompt_ids), turn.logprobs, a, i, t))
    groups = len(by_group)
    return out, {"groups": groups, "degenerate_groups": skipped,
                 "non_degenerate_fraction": round((groups - skipped) / groups, 4) if groups else None}


def t_norm(n_tasks: int, group_size: int, max_completion_tokens_per_episode: int) -> int:
    """The fixed loss normalizer: not the batch's actual token count, so long trajectories are not diluted."""
    return n_tasks * group_size * max_completion_tokens_per_episode


def batch_stats(episodes: list[Episode]) -> dict[str, Any]:
    rewards = [e.reward for e in episodes]
    comp = [sum(len(t.completion_ids) for t in e.turns) for e in episodes]
    return {
        "episodes": len(episodes),
        "reward_mean": round(statistics.fmean(rewards), 4) if rewards else None,
        "turns_mean": round(statistics.fmean(len(e.turns) for e in episodes), 3) if episodes else None,
        "completion_tokens_mean": round(statistics.fmean(comp), 1) if comp else None,
        "completion_tokens_max": max(comp) if comp else None,
        "sequence_tokens": sum(len(t.prompt_ids) + len(t.completion_ids) for e in episodes for t in e.turns),
        "stop_reasons": dict(Counter(e.stop_reason for e in episodes)),
        "finish_reasons": dict(Counter(t.finish_reason for e in episodes for t in e.turns)),
        "episodes_with_format_error": sum(1 for e in episodes if e.format_errors),
        "nudged": sum(1 for e in episodes if e.nudged),
    }
