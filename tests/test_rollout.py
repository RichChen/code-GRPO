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


# -- paid API config, cost and diagnostics -----------------------------------------------------------

PRICES = {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0}


def test_dotenv_is_read_and_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nX_BASE_URL=https://api.example/v1\nX_MODEL='m-flash'\nX_API_KEY=k-from-file\n\n")
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"env_prefix": "X", "extra_body": {"thinking": {"type": "disabled"}}}))
    monkeypatch.delenv("X_API_KEY", raising=False)
    monkeypatch.setenv("X_MODEL", "m-env")
    c = ClientConfig.load(cfg, dotenv=env)
    assert (c.base_url, c.model, c.api_key) == ("https://api.example/v1", "m-env", "k-from-file")
    assert "api_key" not in c.to_dict()
    env.write_text("X_BASE_URL=u\n")
    monkeypatch.delenv("X_MODEL")
    with pytest.raises(ValueError, match="X_MODEL"):
        ClientConfig.load(cfg, dotenv=env)


def test_extra_body_is_sent_and_cache_hits_are_priced():
    r1 = reply(calls=[FIX], prompt=1000, completion=100)
    r1.usage.prompt_cache_hit_tokens = 600
    client = FakeClient([r1, reply("done", prompt=1200, completion=10)])  # no cache field: all misses
    cfg = ClientConfig(extra_body={"chat_template_kwargs": {"enable_thinking": False}}, price_cny_per_mtok=PRICES)
    policy = OpenAIPolicy(cfg, client=client)
    run_episode(make_task(), policy)
    assert client.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert policy.usage.cache_hit_tokens == 600
    expected = (600 * 0.04 + (2200 - 600) * 2.0 + 110 * 8.0) / 1e6
    assert cfg.cost_cny(policy.usage) == pytest.approx(expected)


def test_request_without_tools_omits_tool_choice():
    client = FakeClient([reply("ok")])
    OpenAIPolicy(ClientConfig(), client=client)([{"role": "user", "content": "Reply with ok."}], [])
    assert "tools" not in client.requests[0] and "tool_choice" not in client.requests[0]


def test_redact_removes_keys_and_authorization():
    from arl.rollout.calibrate import redact
    text = "401 headers={'Authorization': 'Bearer fake-secret-123456'} key fake-secret-123456 Bearer xyz"
    out = redact(text, ["fake-secret-123456"])
    assert "fake-secret-123456" not in out and "xyz" not in out and "[redacted]" in out


def test_budget_stops_new_episodes(tmp_path):
    from arl.rollout.calibrate import CostLedger

    class Priced(ScriptedPolicy):
        def __init__(self, turns):
            super().__init__(turns)
            self.config = ClientConfig(price_cny_per_mtok=PRICES)
            self.usage = Usage(prompt_tokens=100_000)  # 0.2 CNY per episode

    tasks = [toy(i) for i in range(10)]
    ledger = CostLedger(tmp_path / "cost.json", 1.0, PRICES, [])
    recs = run_calibration(tasks, tmp_path / "cal.jsonl", ClientConfig(), k=1, concurrency=1,
                           policy_factory=lambda t, k: Priced([[FIX], "done"]), log=lambda *_: None, ledger=ledger)
    assert len(recs) == 5 and ledger.spent == pytest.approx(1.0)
    saved = json.loads((tmp_path / "cost.json").read_text())
    assert saved["spent_cny"] == pytest.approx(1.0) and saved["episodes"] == 5
    resumed = CostLedger(tmp_path / "cost.json", 1.0, PRICES, load_records(tmp_path / "cal.jsonl"))
    assert resumed.spent == pytest.approx(1.0) and resumed.exhausted(0)


def test_behaviour_metrics_in_records_and_summary(tmp_path):
    from arl.rollout.calibrate import diagnostics

    def factory(task, k):
        turns = {0: ["thinking", [FIX], [("run_tests", {})], "done"],  # nudged, then fixes
                 1: [[FIX], "done"],                                    # fixes straight away
                 2: ["no", "still no"]}[k]
        p = ScriptedPolicy(turns)
        p.usage = Usage()
        return p

    recs = run_calibration([toy(0)], tmp_path / "cal.jsonl", ClientConfig(), k=3, concurrency=1,
                           policy_factory=factory, log=lambda *_: None)
    by_k = {r["k"]: r for r in recs}
    assert by_k[0]["nudged"] and by_k[0]["nudge_turn"] == 1 and by_k[0]["tool_call_after_nudge"]
    assert by_k[0]["run_tests_passed"] == 1 and by_k[0]["edit_ok"] == 1 and not by_k[0]["first_turn_tool_call"]
    assert not by_k[1]["nudged"] and by_k[1]["first_turn_tool_call"]
    assert by_k[2]["nudged"] and not by_k[2]["tool_call_after_nudge"] and by_k[2]["reward"] == 0
    d = diagnostics(recs)
    assert d["nudged"] == 0.667 and d["tool_call_after_nudge_of_nudged"] == 0.5
    assert d["first_turn_no_tool_call"] == 0.667 and d["edit_success_rate"] == 1.0
    assert d["episodes_with_successful_edit"] == 0.667 and d["episodes_visible_pass_any"] == 0.667
    assert d["solved_turns_median"] == 3 and d["solved_turns_p90"] == 4


def test_regrade_matches_recorded_rewards(tmp_path):
    from arl.rollout.regrade import regrade

    tasks = [toy(0), toy(1)]
    for t in tasks:
        t.meta["hidden_case_kinds"] = ["test", "challenge", "generated"]
    path = tmp_path / "cal.jsonl"
    recs = run_calibration(tasks, path, ClientConfig(), k=2, concurrency=2, policy_factory=scripted, log=lambda *_: None)
    r = regrade(recs, {t.task_id: t for t in tasks}, workers=2)
    assert r["episodes"] == 4 and r["reward_mismatches"] == 0 and r["field_mismatches"] == {}
    assert r["reward_1"] == 2 and r["pass_non_generated"] == 2

    forged = [dict(recs[0], reward=1 - recs[0]["reward"])]
    assert regrade(forged, {t.task_id: t for t in tasks}, workers=1)["reward_mismatches"] == 1


def test_generated_only_failures():
    from arl.rollout.falseneg import generated_only_failures

    task = SimpleNamespace(meta={"hidden_case_kinds": ["test", "challenge", "generated", "generated"]})

    def rec(passed, reward=0, **grade):
        return {"reward": reward, "grade": {"case_passed": passed, "timed_out": False, "setup_error": None, **grade}}

    assert generated_only_failures(rec([True, True, False, True]), task) == [2]
    assert generated_only_failures(rec([False, True, False, True]), task) is None   # a test case fails too
    assert generated_only_failures(rec([True, False, True, True]), task) is None    # a challenge case fails
    assert generated_only_failures(rec([True] * 4, reward=1), task) is None
    assert generated_only_failures(rec([True, True, False, False], timed_out=True), task) is None
