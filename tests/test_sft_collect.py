from arl.env.episode import ScriptedPolicy
from arl.env.task import Task
from arl.rollout.calibrate import load_records, run_calibration
from arl.rollout.client import ClientConfig, Usage
from arl.sft import collect
from conftest import make_task

FIX = ("edit_file", {"path": "pkg/nums.py", "old_string": "return a - b", "new_string": "return a + b"})


def toy(i: int) -> Task:
    t = make_task()
    meta = dict(t.meta, split="train", bug_source=f"mbpp/{i}", issue_template="returns_a")
    return Task.from_dict({**t.to_dict(), "task_id": f"toy-{i}", "meta": meta})


def solves_odd_tasks_and_retries(task: Task, k: int):
    solve = int(task.task_id.split("-")[1]) % 2 == 1 or k == 1
    policy = ScriptedPolicy([[FIX], "done"] if solve else ["give up", "still nothing"])
    policy.usage = Usage()
    return policy


def test_first_pass_then_retry_of_failures_with_messages(tmp_path):
    tasks = [toy(i) for i in range(4)]
    path = tmp_path / "episodes.jsonl"
    run = dict(policy_factory=solves_odd_tasks_and_retries, log=lambda *_: None, with_messages=True)
    run_calibration(tasks, path, ClientConfig(), k=1, **run)
    retry = collect.retry_tasks(tasks, load_records(path))
    assert [t.task_id for t in retry] == ["toy-0", "toy-2"]
    run_calibration(retry, path, ClientConfig(), k=2, **run)

    recs = load_records(path)
    assert len(recs) == 6 and all(r["messages"][0]["role"] == "system" for r in recs)
    assert recs[0]["messages"][-1]["role"] == "assistant"
    s = collect.summary(recs, tasks)
    assert s["first_attempts"] == 4 and s["first_attempt_solve_rate"] == 0.5
    assert s["retries"] == 2 and s["retry_solve_rate"] == 1.0
    assert s["successful_trajectories"] == 4 and s["tasks_solved"] == 4
    assert s["nudged"] == 2  # the two failed first attempts were nudged once


def test_retry_is_capped_by_max_successes(monkeypatch):
    tasks = [toy(i) for i in range(4)]
    recs = [{"task_id": t.task_id, "k": 0, "reward": int(i < 2)} for i, t in enumerate(tasks)]
    monkeypatch.setattr(collect, "MAX_SUCCESSES", 3)
    assert [t.task_id for t in collect.retry_tasks(tasks, recs)] == ["toy-2"]
    monkeypatch.setattr(collect, "MAX_SUCCESSES", 2)
    assert collect.retry_tasks(tasks, recs) == []
