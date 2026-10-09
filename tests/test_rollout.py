import json
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from arl.env.episode import ScriptedPolicy, replay_workspace, run_episode
from arl.env.grader import grade
from arl.env.task import Task
from arl.rollout.calibrate import (
    load_records,
    request_seed,
    run_calibration,
    solve_counts,
    stratified_sample,
    summarize,
)
from arl.rollout.client import ClientConfig, OpenAIPolicy, Usage
from conftest import BUGGY_NUMS, FIXED_NUMS, make_task


def reply(content=None, calls=(), finish="stop", prompt=100, completion=20):
    tool_calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


FIX = ("edit_file", {"path": "pkg/nums.py", "old_string": "return a - b", "new_string": "return a + b"})


def test_policy_runs_an_episode_and_tracks_usage():
    client = FakeClient([reply(calls=[FIX], prompt=500, completion=40), reply("Fixed.", prompt=700, completion=5)])
    policy = OpenAIPolicy(ClientConfig(), client=client, seed=7)
    result = run_episode(make_task(), policy)
    assert result.reward == 1 and result.stop_reason == "submitted"
    assert policy.usage.calls == 2 and policy.usage.completion_tokens == 45 and policy.usage.final_context_tokens == 705
    req = client.requests[0]
    assert req["model"] == "Qwen/Qwen2.5-1.5B-Instruct" and req["temperature"] == 0.8 and req["tool_choice"] == "auto"
    assert req["seed"] == 7 and client.requests[1]["seed"] == 8
    assert {t["function"]["name"] for t in req["tools"]} >= {"read_file", "edit_file", "run_tests"}


def test_unparsed_tool_call_text_is_a_format_error_not_a_submission():
    text = '<tool_call>\n{"name": "edit_file", "arguments": {"path": "pkg/nums.py", oops}}\n</tool_call>'
    client = FakeClient([reply(text), reply(calls=[FIX]), reply("done")])
    policy = OpenAIPolicy(ClientConfig(), client=client)
    result = run_episode(make_task(), policy)
    assert result.format_errors == 1 and result.tool_calls[0].error_kind == "malformed_call"
    assert result.reward == 1 and result.turns == 3 and policy.usage.unparsed_tool_calls == 1


def test_context_length_error_ends_the_episode_as_truncated():
    err = openai.BadRequestError(
        "This model's maximum context length is 8192 tokens.",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat/completions")), body=None,
    )
    client = FakeClient([reply(calls=[FIX]), err])
    result = run_episode(make_task(), OpenAIPolicy(ClientConfig(), client=client))
    assert result.stop_reason == "context_limit" and result.truncated and result.reward == 1


def test_client_config_reads_file_and_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"base_url": "http://h:1/v1", "model": "m"}))
    c = ClientConfig.load(p, temperature=0.5)
    assert (c.base_url, c.model, c.temperature) == ("http://h:1/v1", "m", 0.5)
    p.write_text(json.dumps({"modle": "m"}))
    with pytest.raises(ValueError, match="modle"):
        ClientConfig.load(p)


def test_replay_workspace_reproduces_the_graded_code(tmp_path):
    task = make_task()
    noise = ("edit_file", {"path": "pkg/nums.py", "old_string": "missing", "new_string": "x"})
    result = run_episode(task, ScriptedPolicy([[noise, FIX], "done"]))
    ws = replay_workspace(task, result, tmp_path / "ws")
    assert (ws / "pkg/nums.py").read_text() == FIXED_NUMS
    assert grade(task, ws).reward == result.reward == 1


# -- calibration ------------------------------------------------------------------------------------


def toy(i: int, fn: str = "f") -> Task:
    t = make_task()
    meta = dict(t.meta, split="train", bug_source=f"mbpp/{fn}", issue_template="returns_a")
    return Task.from_dict({**t.to_dict(), "task_id": f"toy-{i}", "meta": meta})


def scripted(task: Task, k: int):
    policy = ScriptedPolicy([[FIX], "done"] if k % 2 == 0 else ["done"])
    policy.usage = Usage()
    return policy


def test_calibration_runs_k_attempts_resumes_and_summarizes(tmp_path):
    tasks = [toy(0), toy(1)]
    path = tmp_path / "cal.jsonl"
    recs = run_calibration(tasks, path, ClientConfig(), k=4, concurrency=4, policy_factory=scripted, log=lambda *_: None)
    assert len(recs) == 8
    assert solve_counts(recs) == {"toy-0": 2, "toy-1": 2}
    n_lines = len(load_records(path))
    run_calibration(tasks, path, ClientConfig(), k=4, policy_factory=scripted, log=lambda *_: None)
    assert len(load_records(path)) == n_lines  # nothing rerun
    s = summarize(recs, {t.task_id: t for t in tasks})
    assert s["solve_count_distribution"]["2"] == 2 and s["fraction_in_range"] == 1.0
    assert s["solve_rate"] == 0.5 and s["stop_reasons"] == {"submitted": 8}


def test_calibration_retries_errored_episodes_on_resume(tmp_path):
    path = tmp_path / "cal.jsonl"
    calls = {"n": 0}

    def flaky(task, k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("server down")
        return scripted(task, k)

    tasks = [toy(0)]
    recs = run_calibration(tasks, path, ClientConfig(), k=2, concurrency=1, policy_factory=flaky, log=lambda *_: None)
    assert len(recs) == 1 and any("error" in r for r in load_records(path))
    recs = run_calibration(tasks, path, ClientConfig(), k=2, concurrency=1, policy_factory=scripted, log=lambda *_: None)
    assert len(recs) == 2


def test_stratified_sample_covers_functions_before_repeating():
    tasks = [toy(i, fn=f"f{i % 5}") for i in range(20)]
    sample = stratified_sample(tasks, 7, seed=0)
    fns = [t.meta["bug_source"] for t in sample]
    assert len(set(fns[:5])) == 5 and len(sample) == 7
    assert stratified_sample(tasks, 7, seed=0) == sample


def test_request_seed_is_stable():
    assert request_seed("a", 1) == request_seed("a", 1) != request_seed("a", 2)
