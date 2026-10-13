from types import SimpleNamespace

from arl.rl.heldout import cluster_bootstrap, paired


def rec(task, reward, non_gen=None):
    return {"task_id": task, "reward": reward, "grade": {"pass_non_generated": bool(reward if non_gen is None else non_gen)}}


def test_cluster_bootstrap_keeps_whole_clusters():
    # Two clusters with constant values: every resample mean is a mix of 0 and 1 weighted by task count.
    values = {"a1": 1.0, "a2": 1.0, "a3": 1.0, "b1": 0.0}
    cluster_of = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    mean, lo, hi = cluster_bootstrap(values, cluster_of, n=2000, seed=0)
    assert mean == 0.75
    # Possible resample means: {A,A} -> 1, {A,B} -> 3/4, {B,B} -> 0. Resampling single tasks would give others.
    assert lo == 0.0 and hi == 1.0


def test_cluster_bootstrap_is_seeded_and_degenerate_when_constant():
    values = {f"t{i}": 0.5 for i in range(10)}
    cluster_of = {f"t{i}": f"c{i % 3}" for i in range(10)}
    assert cluster_bootstrap(values, cluster_of, n=500, seed=0) == (0.5, 0.5, 0.5)
    v2 = {f"t{i}": float(i % 2) for i in range(10)}
    assert cluster_bootstrap(v2, cluster_of, n=500, seed=0) == cluster_bootstrap(v2, cluster_of, n=500, seed=0)


def test_paired_differences_per_task_over_k():
    tasks = {"x": SimpleNamespace(meta={"bug_source": "mbpp/1"}), "y": SimpleNamespace(meta={"bug_source": "mbpp/2"})}
    a = [rec("x", 1), rec("x", 1), rec("x", 0), rec("x", 0), rec("y", 1), rec("y", 1), rec("y", 1), rec("y", 1)]
    b = [rec("x", 0)] * 4 + [rec("y", 1)] * 4
    out = paired(a, b, tasks, "reward")
    assert out["tasks"] == 2 and out["clusters"] == 2
    assert out["mean_diff"] == 0.25 and out["tasks_better"] == 1 and out["tasks_worse"] == 0
