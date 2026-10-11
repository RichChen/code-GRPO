from arl.sft.evaluate import SFT_DEV_STRICT_CI, bootstrap_ci, calibration_estimate, compare, in_sft_dev_interval, per_task_rates
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


def test_audited_sft_dev_rate_is_inside_its_interval_and_the_edges_count():
    lo, hi = SFT_DEV_STRICT_CI
    assert in_sft_dev_interval(0.535) and in_sft_dev_interval(lo) and in_sft_dev_interval(hi)
    assert not in_sft_dev_interval(lo - 0.001) and not in_sft_dev_interval(hi + 0.001)


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


def test_reference_column_headroom_and_calibration_estimate():
    tasks = {f"toy-{i}": toy(i) for i in range(4)}
    base = [rec(t, k, int(t == "toy-0"), True) for t in tasks for k in range(4)]  # toy-0 4/4
    sft = [rec(t, k, int(t in ("toy-0", "toy-1") and k < 2), True) for t in tasks for k in range(4)]  # 2/4, 2/4
    ref = [rec(t, k, int(t != "toy-3"), True) for t in tasks for k in range(4)]  # toy-3 never
    out = compare(base, sft, tasks, ref, pool_size=1000)
    assert out["reference"]["solve_rate_strict"] == 0.75
    assert out["headroom"] == {"reference_solved_at_least_once": 3, "reference_never_solved": 1,
                               "base_never_solved_but_reference_did": 2, "base_solved_but_reference_never": 0,
                               "sft_never_solved_but_reference_did": 1, "sft_solved_but_reference_never": 0}
    est = out["rl_train_calibration_estimate"]
    assert est["dev_tasks_solved_1_to_3"] == 2 and est["fraction"] == 0.5 and est["estimated_kept"] == 500
    lo, hi = est["estimated_kept_ci95"]
    assert lo <= 500 <= hi
    assert "reference" not in compare(base, sft, tasks)
    assert calibration_estimate(sft, tasks, 10)["estimated_kept"] == 5
