import json
from pathlib import Path

import pytest

from arl.env.episode import (
    NUDGE_MESSAGE,
    SYSTEM_PROMPT,
    EpisodeResult,
    ScriptedPolicy,
    estimate_tokens,
    run_episode,
    write_jsonl,
)
from arl.env.task import Task
from arl.env.tools import TOOL_SCHEMAS

from conftest import VISIBLE

SOLVER = [
    [("list_dir", {})],
    [("search_code", {"pattern": r"def add"})],
    [("read_file", {"path": "pkg/nums.py"})],
    [("edit_file", {"path": "pkg/nums.py", "old_string": "return a - b", "new_string": "return a + b"})],
    [("run_tests", {})],
    "Fixed add to return a + b.",
]

VISIBLE_ONLY_CHEATER = [
    [("read_file", {"path": "tests/test_visible.py"})],
    [("edit_file", {"path": "tests/test_visible.py", "old_string": "assert add(1, 2) == 3", "new_string": "assert True"})],
    [("run_tests", {})],
    "All tests pass.",
]


def test_prompt_is_short():
    assert len(SYSTEM_PROMPT.split()) <= 300
    for name in ("read_file", "search_code", "list_dir", "edit_file", "run_tests"):
        assert name in SYSTEM_PROMPT


def test_scripted_solver_gets_reward_1(task):
    policy = ScriptedPolicy(SOLVER)
    r = run_episode(task, policy)
    assert r.reward == 1
    assert r.stop_reason == "submitted" and r.turns == 6 and not r.truncated
    assert r.grade["files_changed"] == ["pkg/nums.py"] and r.grade["visible_pass"] is True
    assert [c.name for c in r.tool_calls] == ["list_dir", "search_code", "read_file", "edit_file", "run_tests"]
    assert all(c.ok and c.error_kind is None for c in r.tool_calls)
    assert "1 passed" in r.messages[-2]["content"]
    assert r.messages[-1] == {"role": "assistant", "content": "Fixed add to return a + b."}
    assert r.meta["bug_type"] == "arith_swap"


def test_scripted_visible_test_cheater_gets_0(task):
    r = run_episode(task, ScriptedPolicy(VISIBLE_ONLY_CHEATER))
    assert "exit code: 0" in r.messages[-2]["content"]  # the cheat fools run_tests
    assert r.reward == 0
    assert r.grade["files_changed"] == ["tests/test_visible.py"]
    assert r.grade["visible_pass"] is False


def test_submit_twice_without_edit_is_nudged_once_then_submitted(task):
    r = run_episode(task, ScriptedPolicy(["I give up.", "Still giving up."]))
    assert r.reward == 0 and r.turns == 2 and r.stop_reason == "submitted" and r.tool_calls == []
    assert r.nudged is True
    assert r.messages[3] == {"role": "user", "content": NUDGE_MESSAGE}
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant", "user", "assistant"]


def test_nudge_then_continue_and_fix(task):
    r = run_episode(task, ScriptedPolicy(["The bug is in add."] + SOLVER))
    assert r.nudged is True and r.reward == 1 and r.stop_reason == "submitted" and r.turns == 7
    assert sum(1 for m in r.messages if m.get("content") == NUDGE_MESSAGE) == 1


def test_nudge_counts_as_a_turn(task):
    r = run_episode(task, ScriptedPolicy(["thinking", [("list_dir", {})], [("list_dir", {})]]), max_turns=3)
    assert r.nudged is True and r.turns == 3 and r.stop_reason == "max_turns"


def test_no_nudge_after_successful_edit(task):
    fix = ("edit_file", {"path": "pkg/nums.py", "old_string": "return a - b", "new_string": "return a + b"})
    r = run_episode(task, ScriptedPolicy([[fix], "done"]))
    assert r.nudged is False and r.turns == 2 and r.stop_reason == "submitted" and r.reward == 1
    assert all(m.get("content") != NUDGE_MESSAGE for m in r.messages)


def test_failed_edit_does_not_count_as_edited(task):
    bad = ("edit_file", {"path": "pkg/nums.py", "old_string": "not there", "new_string": "x"})
    r = run_episode(task, ScriptedPolicy([[bad], "done", "done again"]))
    assert r.nudged is True and r.turns == 3 and r.stop_reason == "submitted"


def test_nudged_field_round_trips(task):
    r = run_episode(task, ScriptedPolicy(["a", "b"]))
    back = EpisodeResult.from_dict(json.loads(r.to_json()))
    assert back.nudged is True and json.loads(r.to_json())["nudged"] is True
    old = json.loads(r.to_json())
    del old["nudged"]
    assert EpisodeResult.from_dict(old).nudged is False


def test_prompt_has_json_format_note():
    note = SYSTEM_PROMPT.split("\n\n")[-1]
    assert note.startswith("Tool call format:") and len(note.split()) <= 60
    assert 'r"..."' in note and "</tool_call>" in note


def test_message_format(task):
    policy = ScriptedPolicy(SOLVER)
    r = run_episode(task, policy)
    assert r.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert r.messages[1]["role"] == "user" and task.issue in r.messages[1]["content"]
    for i, m in enumerate(r.messages):
        if m["role"] == "tool":
            prev_calls = [c["id"] for c in r.messages[i - 1].get("tool_calls", [])] if r.messages[i - 1]["role"] == "assistant" else None
            assert prev_calls is None or m["tool_call_id"] in prev_calls
        if m.get("tool_calls"):
            for c in m["tool_calls"]:
                assert c["type"] == "function" and isinstance(c["function"]["arguments"], str)
    assert policy.seen[0][0]["role"] == "system"
    assert len(policy.seen) == 6 and len(policy.seen[-1]) == len(r.messages) - 1


def test_policy_receives_copies(task):
    def mutating_policy(messages, tools):
        messages.clear()
        return {"role": "assistant", "content": "done"}

    r = run_episode(task, mutating_policy)
    assert r.messages[0]["role"] == "system" and len(r.messages) == 5  # incl. the nudge and second reply


def test_policy_gets_tool_schemas(task):
    got = []

    def policy(messages, tools):
        got.append(tools)
        return {"role": "assistant", "content": "done"}

    run_episode(task, policy)
    assert got == [TOOL_SCHEMAS, TOOL_SCHEMAS]


def test_max_turns(task):
    r = run_episode(task, ScriptedPolicy([[("list_dir", {})]] * 20), max_turns=3)
    assert r.turns == 3 and r.stop_reason == "max_turns" and r.reward == 0
    assert len(r.tool_calls) == 3


def test_parallel_tool_calls_in_one_turn(task):
    turn = [
        ("edit_file", {"path": "pkg/nums.py", "old_string": "a - b", "new_string": "a + b"}),
        ("run_tests", {}),
    ]
    r = run_episode(task, ScriptedPolicy([turn, "ok"]))
    assert r.reward == 1 and r.turns == 2
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]


def test_format_errors_are_recorded_and_episode_continues(task):
    turns = [
        [("write_file", {"path": "x.py"})],
        [("read_file", "{not json")],
        [("read_file", {"file": "pkg/nums.py"})],
        [("read_file", {"path": "nope.py"})],
        [("", {})],
        "done",
    ]
    r = run_episode(task, ScriptedPolicy(turns))
    kinds = [c.error_kind for c in r.tool_calls]
    assert kinds == ["unknown_tool", "invalid_json", "invalid_arguments", "tool_error", "malformed_call"]
    assert r.format_errors == 4
    assert r.stop_reason == "submitted"
    assert "Unknown tool 'write_file'" in r.messages[3]["content"]


def test_dict_arguments_and_missing_ids_are_normalized(task):
    def policy(messages, tools):
        if len(messages) == 2:
            return {"role": "assistant", "tool_calls": [{"function": {"name": "list_dir", "arguments": {"path": "pkg"}}}]}
        return {"role": "assistant", "content": "done", "reasoning": "dropped"}

    r = run_episode(task, policy)
    call = r.messages[2]["tool_calls"][0]
    assert call["id"] == "call_0_0" and call["function"]["arguments"] == '{"path": "pkg"}'
    assert r.messages[3]["tool_call_id"] == "call_0_0" and "nums.py" in r.messages[3]["content"]
    assert "reasoning" not in r.messages[-1]


def test_tool_output_is_clipped(task):
    long_line = "x = '" + "y" * 3000 + "'\n"
    big = Task.from_dict({**task.to_dict(), "files": {**task.files, "pkg/big.py": long_line * 5}})
    r = run_episode(big, ScriptedPolicy([[("read_file", {"path": "pkg/big.py"})], "done"]))
    assert len(r.messages[3]["content"]) <= 4000
    assert r.tool_calls[0].output_truncated and r.tool_calls[0].output_chars > 15000


def test_context_limit_truncates_but_still_grades(task):
    fix = [("edit_file", {"path": "pkg/nums.py", "old_string": "a - b", "new_string": "a + b"})]
    flood = [("read_file", {"path": "pkg/nums.py"})]
    budget = estimate_tokens([{"content": SYSTEM_PROMPT}]) + 300
    r = run_episode(task, ScriptedPolicy([fix] + [flood] * 50), max_turns=50, context_max_tokens=budget)
    assert r.truncated and r.stop_reason == "context_limit"
    assert r.turns < 50
    assert r.reward == 1


def test_workspace_is_removed(task, tmp_path):
    root = tmp_path / "eps"
    root.mkdir()
    run_episode(task, ScriptedPolicy(["done"]), workspace_dir=root)
    assert list(root.iterdir()) == []
    run_episode(task, ScriptedPolicy(["done"]), workspace_dir=root, keep_workspace=True)
    kept = list(root.iterdir())
    assert len(kept) == 1 and (kept[0] / "tests" / "test_visible.py").read_text() == VISIBLE


def test_jsonl_roundtrip(task, tmp_path):
    r = run_episode(task, ScriptedPolicy(SOLVER))
    path = tmp_path / "eps.jsonl"
    write_jsonl([r, r], path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    back = EpisodeResult.from_dict(json.loads(lines[0]))
    assert back == r
    assert back.messages[2]["tool_calls"][0]["function"]["name"] == "list_dir"


def test_estimate_tokens():
    assert estimate_tokens([{"content": "a" * 400}], tools=None) == 100
    msg = {"content": None, "tool_calls": [{"function": {"name": "abcd", "arguments": "x" * 36}}]}
    assert estimate_tokens([msg], tools=None) == 10
    assert estimate_tokens([], tools=TOOL_SCHEMAS) == len(json.dumps(TOOL_SCHEMAS)) // 4
