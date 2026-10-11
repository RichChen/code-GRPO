"""GRPO rollouts: ``group_size`` episodes per task with the token client, against a given LoRA adapter name."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable

from arl.env.episode import run_episode
from arl.env.task import Task
from arl.rl.batch import Episode, turn_from_record
from arl.rollout.client import ClientConfig


def episode_seed(seed: int, step: int, task_id: str, index: int) -> int:
    return int(hashlib.sha256(f"grpo-{seed}-{step}-{task_id}#{index}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def make_token_policy(config: ClientConfig, seed: int, model: str) -> Any:
    from arl.rollout.tokenclient import TokenPolicy

    return TokenPolicy(config, seed=seed, model=model)


def collect(tasks: list[Task], group_size: int, config: ClientConfig, model: str, step: int, seed: int,
            concurrency: int, max_turns: int,
            policy_factory: Callable[[ClientConfig, int, str], Any] = make_token_policy) -> list[Episode]:
    """Episodes in (task, index) order; ``Episode.group`` is the task's position in ``tasks``."""
    config = replace(config, model=model)
    jobs = [(gi, task, k) for gi, task in enumerate(tasks) for k in range(group_size)]

    def one(job: tuple[int, Task, int]) -> Episode:
        gi, task, k = job
        policy = policy_factory(config, episode_seed(seed, step, task.task_id, k), model)
        result = run_episode(task, policy, max_turns)
        return Episode(task_id=task.task_id, group=gi, index=k, reward=float(result.reward),
                       turns=[turn_from_record(t) for t in policy.turns], stop_reason=result.stop_reason,
                       format_errors=result.format_errors, tool_calls=len(result.tool_calls), nudged=result.nudged,
                       meta={"duration_s": round(result.duration_s, 2), "pass_non_generated":
                             bool(result.grade.get("pass_non_generated"))})

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, jobs))
