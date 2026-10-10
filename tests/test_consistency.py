import pytest

from arl.rollout.consistency import first_difference, greedy, replay, sample_ids, stats, tool_sequence


def rec(task, reward, calls=(), k=0, **extra):
    msgs = [{"role": "user", "content": "x"}]
    for name, args in calls:
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c", "type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "content": "ok"})
    return {"task_id": task, "k": k, "reward": reward, "stop_reason": "final", "messages": msgs, **extra}


def test_sample_ids_is_seeded_and_sorted(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("\n".join(f"t{i}" for i in range(50)) + "\n")
    a = sample_ids(f, 10)
    assert a == sample_ids(f, 10) and a == sorted(a) and len(set(a)) == 10
    assert a != sample_ids(f, 10, seed=1)


def test_tool_sequence_and_first_difference():
    r = rec("t", 1, [("read_file", '{"path": "a"}'), ("run_tests", "{}")])
    assert tool_sequence(r) == [("read_file", '{"path": "a"}'), ("run_tests", "{}")]
    assert first_difference([1, 2], [1, 2]) is None
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1], [1, 2]) == 1


def test_greedy_counts_identical_episodes_and_locates_differences():
    chat = [rec("a", 1, [("read_file", "{}")]), rec("b", 0, [("read_file", "{}")]), rec("c", 1)]
    tok = [rec("a", 1, [("read_file", "{}")], tokenize_checked=2, tokenize_mismatches=0),
           rec("b", 0, [("list_dir", "{}")], tokenize_checked=2, tokenize_mismatches=0),
           rec("c", 0, tokenize_checked=1, tokenize_mismatches=1)]
    g = greedy(chat, tok)
    assert g["episodes"] == 3 and g["identical"] == 1
    assert g["tokenize_checked"] == 5 and g["tokenize_mismatches"] == 1
    diff = {e["task_id"]: e for e in g["detail"]}
    assert diff["b"]["first_difference"] == {"call": 0, "chat": ("read_file", "{}"), "tokens": ("list_dir", "{}")}
    assert diff["c"]["reward"] == [1, 0] and "first_difference" in diff["c"]
    with pytest.raises(ValueError):
        greedy(chat, tok[:2])


def test_replay_reasks_each_assistant_turn_with_the_exact_history(monkeypatch):
    seen = []

    class FakePolicy:
        def __init__(self, config, seed=None):
            self.turns = [{"completion_logprobs": [-0.5, -0.1]}]

        def __call__(self, messages, tools):
            seen.append(len(messages))
            if len(messages) == 3:
                return {"role": "assistant", "content": "other", "tool_calls": None}
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}

    monkeypatch.setattr("arl.rollout.tokenclient.TokenPolicy", FakePolicy)
    r = rec("a", 1, [("read_file", "{}"), ("read_file", "{}")])
    report = replay([r], config=None)
    assert seen == [1, 3] and report["turns"] == 2 and report["identical"] == 1
    assert report["detail"][0]["message"] == 3 and report["detail"][0]["min_logprob"] == -0.5


def test_stats_uses_the_reference_ci():
    ref = [rec(f"t{i}", int(i % 2 == 0), k=k) for i in range(40) for k in range(4)]
    inside = [rec(f"t{i}", int(i % 2 == 1), k=k) for i in range(40) for k in range(4)]
    outside = [rec(f"t{i}", 1, k=k) for i in range(40) for k in range(4)]
    s = stats(ref, inside)
    assert s["reference_rate"] == 0.5 and s["inside"] and s["reference_ci95"][0] < 0.5 < s["reference_ci95"][1]
    assert not stats(ref, outside)["inside"]
    with pytest.raises(ValueError):
        stats(ref, inside[:4])
