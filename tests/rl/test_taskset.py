import json

from arl.env.task import Task
from arl.rl.taskset import build


def write(tmp_path, base_task, solved_counts):
    tasks, recs = [], []
    for i, (tid, solved) in enumerate(solved_counts.items()):
        meta = dict(base_task.meta, bug_source=f"mbpp/f{i % 2}", bug_type=["arith_swap", "var_swap"][i % 2])
        tasks.append(Task.from_dict({**base_task.to_dict(), "task_id": tid, "meta": meta}))
        for k in range(4):
            recs.append({"task_id": tid, "k": k, "reward": int(k < solved), "grade": {"pass_non_generated": k < solved},
                         "stop_reason": "submitted" if k < 3 else "max_turns", "turns": 3, "format_errors": 0,
                         "nudged": False, "turn_prompt_tokens": [100, 200, 300], "turn_completion_tokens": [10, 20, 30],
                         "finish_reasons": {"stop": 3}})
    recs.append({"task_id": "a", "k": 0, "error": "boom"})  # an errored attempt that was retried earlier
    (tmp_path / "tasks.jsonl").write_text("".join(t.to_json() + "\n" for t in tasks))
    (tmp_path / "ids.txt").write_text("\n".join(solved_counts) + "\n")
    (tmp_path / "recs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [recs[-1]] + recs[:-1]))


def run(tmp_path, min_tasks):
    return build(tmp_path / "recs.jsonl", tmp_path / "ids.txt", tmp_path / "tasks.jsonl", tmp_path / "out.txt",
                 tmp_path / "out.json", min_tasks=min_tasks)


def test_keeps_tasks_solved_one_to_three_times_and_reports(tmp_path, task):
    write(tmp_path, task, {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 2})
    s = run(tmp_path, min_tasks=2)
    assert (tmp_path / "out.txt").read_text() == "b\nc\nd\nf\n"
    assert s["tasks"] == 4 and s["functions"] == 2 and s["tasks_per_function"] == {1: 1, 3: 1}
    assert s["solve_count_distribution"] == {"0": 1, "1": 1, "2": 2, "3": 1, "4": 1}
    assert s["bug_types"] == {"arith_swap": 1, "var_swap": 3}
    c = s["calibration"]
    assert c["episodes"] == 24 and c["solve_rate_strict"] == round(12 / 24, 4) and c["max_turns_fraction"] == 0.25
    assert c["tokens_per_episode"]["trained_sequence_tokens"]["max"] == 660
    assert json.loads((tmp_path / "out.json").read_text())["sha256_ids"] == s["sha256_ids"]


def test_stops_below_the_minimum_without_writing(tmp_path, task):
    write(tmp_path, task, {"a": 0, "b": 1, "c": 4})
    s = run(tmp_path, min_tasks=400)
    assert "stop" in s and not (tmp_path / "out.txt").exists() and not (tmp_path / "out.json").exists()
