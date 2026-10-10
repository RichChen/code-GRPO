from pathlib import Path
from types import SimpleNamespace

import pytest

from arl.sft.split import POOLS, pool_of_functions, read_ids, split_tasks

TASKS = Path("data/tasks")


def test_pool_of_functions_keeps_clusters_whole_and_reaches_the_fraction():
    cluster_list = [["a"], ["b", "c"], ["d"], ["e", "f", "g"], ["h"], ["x", "y"]]
    bug_sources = {"a", "b", "c", "d", "e", "f", "g", "h"}  # x, y: heldout side, never a train bug source
    pool_of = pool_of_functions(cluster_list, bug_sources, seed=0, fraction=0.4)
    assert "x" not in pool_of and "y" not in pool_of
    for c in cluster_list[:-1]:
        assert len({pool_of[s] for s in c}) == 1
    n_sft = sum(1 for s in bug_sources if pool_of[s] == "sft")
    assert 0.4 * len(bug_sources) <= n_sft < 0.4 * len(bug_sources) + 3  # overshoot < one cluster
    assert pool_of_functions(cluster_list, bug_sources, seed=0, fraction=0.4) == pool_of


def test_split_tasks_follows_the_bug_source():
    tasks = [SimpleNamespace(task_id=f"t{i}", meta={"bug_source": s}) for i, s in enumerate("aabc")]
    ids = split_tasks(tasks, {"a": "sft", "b": "rl", "c": "rl"})
    assert ids == {"sft": ["t0", "t1"], "rl": ["t2", "t3"]}


def test_dev_sample_takes_one_task_per_function_first():
    from arl.sft.split import dev_sample

    tasks = [SimpleNamespace(task_id=f"t{i}", meta={"bug_source": f"f{i % 7}"}) for i in range(30)]
    ids = dev_sample(tasks, n=5, seed=0)
    fns = {f"f{int(i[1:]) % 7}" for i in ids}
    assert len(ids) == 5 and len(fns) == 5 and ids == sorted(ids)
    assert dev_sample(tasks, n=5, seed=0) == ids
    assert len(set(dev_sample(tasks, n=10, seed=0))) == 10


needs_tasks = pytest.mark.skipif(not (TASKS / "train_sft_ids.txt").exists() or not (TASKS / "train.jsonl").exists(),
                                 reason="task set or split files not present")


@needs_tasks
def test_real_split_is_a_partition_of_train_by_cluster():
    import json

    from arl.env.task import load_tasks
    from arl.taskgen.dedup import clusters
    from arl.taskgen.sources import load_sources, sha256_file

    ids = {p: read_ids(TASKS / f"train_{p}_ids.txt") for p in POOLS}
    sft, rl = set(ids["sft"]), set(ids["rl"])
    train = {t.task_id: t for t in load_tasks(TASKS / "train.jsonl")}
    heldout = {t.task_id for t in load_tasks(TASKS / "heldout.jsonl")}
    assert len(sft) == len(ids["sft"]) and len(rl) == len(ids["rl"])
    assert not sft & rl
    assert sft | rl == set(train)
    assert not (sft | rl) & heldout

    funcs, _ = load_sources(Path("data/raw"))
    for c in clusters(funcs):
        pools = {("sft" if tid in sft else "rl") for tid, t in train.items() if t.meta["bug_source"] in c}
        assert len(pools) <= 1, c

    manifest = json.loads((TASKS / "phase2-split.json").read_text())
    for p in POOLS:
        assert manifest["files"][f"train_{p}_ids.txt"] == sha256_file(TASKS / f"train_{p}_ids.txt")

    dev_path = TASKS / "dev_ids.txt"
    if dev_path.exists():
        dev = read_ids(dev_path)
        assert len(dev) == len(set(dev)) == 100
        assert set(dev) <= rl
        assert len({train[i].meta["bug_source"] for i in dev}) == 100


def test_rl_train_ids_are_rl_pool_minus_dev():
    import json

    from arl.env.task import load_tasks
    from arl.taskgen.sources import sha256_file

    rl = set(read_ids(TASKS / "train_rl_ids.txt"))
    dev = set(read_ids(TASKS / "dev_ids.txt"))
    rl_train_list = read_ids(TASKS / "rl_train_ids.txt")
    rl_train = set(rl_train_list)
    assert len(rl_train) == len(rl_train_list)
    assert not rl_train & dev
    assert rl_train == rl - dev
    fn_of = {t.task_id: t.meta["bug_source"] for t in load_tasks(TASKS / "train.jsonl")}
    entry = json.loads((TASKS / "phase2-split.json").read_text())["rl_train_ids"]
    assert entry["tasks"] == len(rl_train)
    assert entry["bug_source_functions"] == len({fn_of[i] for i in rl_train})
    assert entry["sha256"] == sha256_file(TASKS / "rl_train_ids.txt")
