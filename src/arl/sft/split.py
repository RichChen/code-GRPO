"""Split the training tasks into an SFT pool and an RL pool, whole near-duplicate clusters at a time.

    uv run python -m arl.sft.split        # SFT / RL pools
    uv run python -m arl.sft.split dev    # dev set: 100 RL-pool tasks, one per bug source function
    uv run python -m arl.sft.split rl-train   # GRPO training ids: RL pool minus the dev set

Clusters are the ones the task build uses (``arl.taskgen.dedup.clusters`` over every source function).
Training-side clusters are shuffled with ``sftsplit-{seed}`` and added to the SFT pool until it holds at
least ``SFT_FRACTION`` of the training set's bug source functions; the rest go to the RL pool. A task
belongs to the pool of its bug source function. The task jsonl files are not touched: the split is two id
lists plus ``phase2-split.json`` with their sha256. The dev set is for model selection and diagnosis only;
GRPO trains on ``rl_train_ids.txt`` (RL pool minus dev), which is recorded in the same manifest.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from arl.env.task import Task, load_tasks
from arl.taskgen.dedup import clusters
from arl.taskgen.sources import load_sources, sha256_file

SFT_FRACTION = 0.4
DEV_SIZE = 100
TASKS = Path("data/tasks")
RAW = Path("data/raw")
POOLS = ("sft", "rl")


def pool_of_functions(cluster_list: list[list[str]], bug_sources: set[str], seed: int = 0,
                      fraction: float = SFT_FRACTION) -> dict[str, str]:
    """Map every source id in a cluster that holds a training bug source to "sft" or "rl"."""
    train_clusters = sorted((c for c in cluster_list if any(s in bug_sources for s in c)), key=lambda c: c[0])
    random.Random(f"sftsplit-{seed}").shuffle(train_clusters)
    target = fraction * len(bug_sources)
    out: dict[str, str] = {}
    in_sft = 0
    for c in train_clusters:
        pool = "sft" if in_sft < target else "rl"
        if pool == "sft":
            in_sft += sum(1 for s in c if s in bug_sources)
        for sid in c:
            out[sid] = pool
    return out


def split_tasks(tasks: list[Task], pool_of: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {p: [] for p in POOLS}
    for t in tasks:
        out[pool_of[t.meta["bug_source"]]].append(t.task_id)
    return {p: sorted(ids) for p, ids in out.items()}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pool_stats(tasks: list[Task]) -> dict[str, Any]:
    per_fn = Counter(t.meta["bug_source"] for t in tasks)
    return {
        "tasks": len(tasks),
        "bug_source_functions": len(per_fn),
        "bug_types": dict(sorted(Counter(t.meta["bug_type"] for t in tasks).items())),
        "origins": dict(sorted(Counter(t.meta["bug_source"].split("/")[0] for t in tasks).items())),
        "tasks_per_function": dict(sorted(Counter(per_fn.values()).items())),
        # upper bound on trajectories kept by the SFT filter (one per task, at most 3 per source function)
        "max_kept_at_3_per_function": sum(min(3, n) for n in per_fn.values()),
    }


def make_split(tasks_dir: Path = TASKS, raw_dir: Path = RAW, seed: int = 0) -> dict[str, Any]:
    funcs, _ = load_sources(raw_dir)
    cluster_list = clusters(funcs)
    train = load_tasks(tasks_dir / "train.jsonl")
    bug_sources = {t.meta["bug_source"] for t in train}
    pool_of = pool_of_functions(cluster_list, bug_sources, seed)
    ids = split_tasks(train, pool_of)
    files = {}
    for pool in POOLS:
        path = tasks_dir / f"train_{pool}_ids.txt"
        path.write_text("".join(f"{i}\n" for i in ids[pool]), encoding="utf-8")
        files[path.name] = sha256_file(path)
    by_id = {t.task_id: t for t in train}
    cluster_of = {s: i for i, c in enumerate(cluster_list) for s in c}
    pools = {}
    for pool in POOLS:
        ts = [by_id[i] for i in ids[pool]]
        pools[pool] = pool_stats(ts) | {"clusters": len({cluster_of[t.meta["bug_source"]] for t in ts})}
    sft_fns = {by_id[i].meta["bug_source"] for i in ids["sft"]}
    distractor_overlap = sum(1 for i in ids["rl"] if any(s in sft_fns for s in by_id[i].meta["source_ids"]
                                                         if s != by_id[i].meta["bug_source"]))
    manifest = {
        "seed": f"sftsplit-{seed}",
        "sft_fraction_of_bug_functions": SFT_FRACTION,
        "train_bug_source_functions": len(bug_sources),
        "train_tasks": len(train),
        "train_jsonl_sha256": sha256_file(tasks_dir / "train.jsonl"),
        "clusters_total": len(cluster_list),
        "pools": pools,
        "rl_tasks_with_an_sft_function_as_distractor": distractor_overlap,
        "files": files,
    }
    (tasks_dir / "phase2-split.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if (tasks_dir / "dev_ids.txt").exists():
        manifest = make_rl_train(tasks_dir)
    return manifest


def dev_sample(tasks: list[Task], n: int = DEV_SIZE, seed: int = 0) -> list[str]:
    """Source-stratified sample: shuffled functions round-robin, one task per function per round (``dev-{seed}``)."""
    by_fn: dict[str, list[Task]] = {}
    for t in sorted(tasks, key=lambda t: t.task_id):
        by_fn.setdefault(t.meta["bug_source"], []).append(t)
    rng = random.Random(f"dev-{seed}")
    order = sorted(by_fn)
    rng.shuffle(order)
    for fn in order:
        rng.shuffle(by_fn[fn])
    out: list[str] = []
    while len(out) < n and any(by_fn.values()):
        for fn in order:
            if by_fn[fn] and len(out) < n:
                out.append(by_fn[fn].pop().task_id)
    return sorted(out)


def make_dev(tasks_dir: Path = TASKS, seed: int = 0) -> dict[str, Any]:
    rl = set(read_ids(tasks_dir / "train_rl_ids.txt"))
    tasks = [t for t in load_tasks(tasks_dir / "train.jsonl") if t.task_id in rl]
    ids = dev_sample(tasks, DEV_SIZE, seed)
    path = tasks_dir / "dev_ids.txt"
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    by_id = {t.task_id: t for t in tasks}
    result = {"seed": f"dev-{seed}", "tasks": len(ids), "sha256": sha256_file(path),
              **{k: v for k, v in pool_stats([by_id[i] for i in ids]).items() if k != "max_kept_at_3_per_function"}}
    make_rl_train(tasks_dir)
    return result


def make_rl_train(tasks_dir: Path = TASKS) -> dict[str, Any]:
    """Write ``rl_train_ids.txt`` (RL pool minus dev) and record it, with the dev set, in ``phase2-split.json``."""
    rl = set(read_ids(tasks_dir / "train_rl_ids.txt"))
    dev = set(read_ids(tasks_dir / "dev_ids.txt"))
    assert dev <= rl, "dev set must come from the RL pool"
    ids = sorted(rl - dev)
    path = tasks_dir / "rl_train_ids.txt"
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    fn_of = {t.task_id: t.meta["bug_source"] for t in load_tasks(tasks_dir / "train.jsonl")}
    manifest_path = tasks_dir / "phase2-split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dev_ids"] = {"tasks": len(dev), "bug_source_functions": len({fn_of[i] for i in dev}),
                           "sha256": sha256_file(tasks_dir / "dev_ids.txt")}
    manifest["rl_train_ids"] = {"definition": "train_rl_ids minus dev_ids", "tasks": len(ids),
                                "bug_source_functions": len({fn_of[i] for i in ids}), "sha256": sha256_file(path)}
    manifest["files"] = manifest["files"] | {"dev_ids.txt": manifest["dev_ids"]["sha256"],
                                             path.name: manifest["rl_train_ids"]["sha256"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["pools", "dev", "rl-train"], default="pools")
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.command == "dev":
        result = make_dev(a.tasks, a.seed)
    elif a.command == "rl-train":
        result = {k: v for k, v in make_rl_train(a.tasks).items() if k in ("dev_ids", "rl_train_ids")}
    else:
        result = make_split(a.tasks, a.raw, a.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
