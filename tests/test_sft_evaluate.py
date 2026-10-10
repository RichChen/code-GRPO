from arl.sft.evaluate import bootstrap_ci, compare, per_task_rates
from test_rollout import toy


def rec(task_id, k, reward, non_gen):
    return {"task_id": task_id, "k": k, "reward": reward, "turns": 4, "stop_reason": "submitted", "truncated": False,
            "duration_s": 1.0, "n_tool_calls": 2, "format_errors": 0, "format_error_kinds": {}, "tool_errors": 0,
            "tool_names": {"edit_file": 1, "run_tests": 1}, "unparsed_tool_calls": 0, "prompt_tokens": 10,
            "completion_tokens": 5, "final_context_tokens": 15, "finish_reasons": {}, "nudged": False,
            "grade": {"visible_pass": True, "pass_non_generated": non_gen},
            "tool_calls": [{"turn": 1, "name": "edit_file", "ok": True}]}


def test_per_task_rates_use_strict_or_test_challenge_verdicts():
    recs = [rec("a", 0, 1, True), rec("a", 1, 0, True), rec("b", 0, 0, False), rec("b", 1, 0, True)]
    assert per_task_rates(recs) == {"a": 0.5, "b": 0.0}
    assert per_task_rates(recs, "non_generated") == {"a": 1.0, "b": 0.5}


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    diffs = [0.25] * 30 + [0.0] * 60 + [-0.25] * 10
    lo, hi = bootstrap_ci(diffs, n=2000, seed=0)
    assert lo < 0.05 < hi and (lo, hi) == bootstrap_ci(diffs, n=2000, seed=0)
    assert bootstrap_ci([0.5] * 10, n=100) == (0.5, 0.5)


def test_compare_reports_both_models_and_paired_differences():
    tasks = {f"toy-{i}": toy(i) for i in range(3)}
    base = [rec(t, k, int(k == 0 and t == "toy-0"), k == 0) for t in tasks for k in range(4)]
    sft = [rec(t, k, int(k < 2), True) for t in tasks for k in range(4)]
    out = compare(base, sft, tasks)
    assert out["base"]["solve_rate_strict"] == round(1 / 12, 3) and out["sft"]["solve_rate_strict"] == 0.5
    assert out["base"]["solve_count_distribution"] == {"0": 2, "1": 1, "2": 0, "3": 0, "4": 0}
    assert out["sft"]["tasks_solved_1_to_3"] == 3
    assert out["paired_strict"]["mean_diff"] == round((0.25 + 0.5 + 0.5) / 3, 4)
    assert out["paired_strict"]["tasks_better"] == 3
    assert out["paired_test_challenge"]["mean_diff"] == 0.75
