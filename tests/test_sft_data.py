import json
from types import SimpleNamespace

from arl.sft.data import assistant_text, discard_reason, select, split_val, template_messages


def rec(task_id="t0", k=0, reward=1, **kw):
    r = {"task_id": task_id, "k": k, "reward": reward, "format_errors": 0, "unparsed_tool_calls": 0, "nudged": False,
         "truncated": False, "stop_reason": "submitted", "turns": 5, "grade": {"files_changed": ["pkg/a.py"]}}
    r.update(kw)
    return r


def test_discard_reasons_in_plan_order():
    assert discard_reason(rec()) is None
    assert discard_reason(rec(reward=0)) == "not_solved"
    assert discard_reason({"task_id": "t0", "k": 0, "error": "boom"}) == "not_solved"
    assert discard_reason(rec(format_errors=1)) == "format_error"
    assert discard_reason(rec(unparsed_tool_calls=1)) == "format_error"
    assert discard_reason(rec(nudged=True)) == "nudged"
    assert discard_reason(rec(truncated=True)) == "truncated"
    assert discard_reason(rec(stop_reason="context_limit")) == "truncated"
    assert discard_reason(rec(turns=9)) == "too_many_turns"
    assert discard_reason(rec(turns=8)) is None
    assert discard_reason(rec(grade={"files_changed": ["pkg/a.py", "tests/test_visible.py"]})) == "changed_outside_pkg"
    assert discard_reason(rec(grade={"files_changed": []})) == "changed_outside_pkg"
    assert discard_reason(rec(reward=0, nudged=True)) == "not_solved"


def tasks_for(spec):
    return {tid: SimpleNamespace(meta={"bug_source": fn, "bug_type": bt}) for tid, (fn, bt) in spec.items()}


def test_select_one_per_task_and_three_per_function_preferring_new_bug_types():
    spec = {f"t{i}": ("mbpp/1", "arith_swap" if i < 3 else f"type{i}") for i in range(6)}
    spec["u0"] = ("mbpp/2", "var_swap")
    tasks = tasks_for(spec)
    records = [rec(t) for t in spec] + [rec("t0", k=1), rec("u0", k=1, reward=0)]
    kept, discards = select(records, tasks, seed=0)
    by_fn = {}
    for r in kept:
        by_fn.setdefault(tasks[r["task_id"]].meta["bug_source"], []).append(r)
    assert len(by_fn["mbpp/1"]) == 3 and len(by_fn["mbpp/2"]) == 1
    assert len({r["task_id"] for r in kept}) == len(kept)
    types = [tasks[r["task_id"]].meta["bug_type"] for r in by_fn["mbpp/1"]]
    assert types.count("arith_swap") <= 1  # four distinct types available, so no repeats
    assert next(r for r in kept if r["task_id"] == "u0")["k"] == 0
    assert discards == {"duplicate_task": 1, "function_cap": 3, "not_solved": 1}
    assert select(records, tasks, seed=0)[0] == kept


def test_split_val_takes_whole_functions():
    tasks = tasks_for({f"t{i}": (f"mbpp/{i % 40}", "x") for i in range(80)})
    trajs = [rec(t) for t in tasks]
    val = split_val(trajs, tasks)
    assert len(val) == 2  # ceil(5% of 40)
    assert split_val(trajs, tasks) == val


def test_assistant_text_matches_hermes_layout():
    m = {"role": "assistant", "content": "Let me look.", "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "pkg/é.py"})}},
        {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]}
    assert assistant_text(m) == ('Let me look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "pkg/é.py"}}\n'
                                 '</tool_call>\n<tool_call>\n{"name": "list_dir", "arguments": {}}\n</tool_call>')
    assert assistant_text({"role": "assistant", "content": "", "tool_calls": m["tool_calls"][1:]}).startswith("<tool_call>")
    assert assistant_text({"role": "assistant", "content": "Done."}) == "Done."


def test_template_messages_parse_arguments_and_keep_tool_results():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "pkg"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "list_dir", "content": "a.py"}]
    out = template_messages(msgs)
    assert out[2] == {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "list_dir", "arguments": {"path": "pkg"}}}]}
    assert out[3]["content"] == "a.py" and out[3]["role"] == "tool"
