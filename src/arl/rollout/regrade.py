"""Re-grade recorded episodes with the current grader and check that nothing that feeds the reward changed.

    uv run python -m arl.rollout.regrade runs/diag/qwen3-1.7b.jsonl --out runs/regrade

Each episode's final workspace is rebuilt by replaying its successful ``edit_file`` calls (see
``arl.rollout.falseneg.replay``) and graded again. The reward, the per-case results and the visible-test
verdict must match the recorded ones exactly; the new per-kind fields are summarised.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from arl.env.grader import grade
from arl.env.task import Task, load_tasks
from arl.rollout.calibrate import TASKS, load_records
from arl.rollout.falseneg import replay

COMPARED = ("reward", "hidden_pass_count", "hidden_total", "case_passed", "visible_pass", "timed_out", "setup_error")


def regrade_one(task: Task, rec: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="arl-regrade-") as tmp:
        new = grade(task, replay(task, rec, Path(tmp) / "ws")).to_dict()
    old = rec["grade"]
    diffs = [k for k in COMPARED if old.get(k) != new.get(k)]
    return {"task_id": rec["task_id"], "k": rec["k"], "old_reward": rec["reward"], "new_reward": new["reward"],
            "diffs": diffs, "hidden_by_kind": new["hidden_by_kind"], "pass_non_generated": new["pass_non_generated"]}


def regrade(records: list[dict[str, Any]], tasks: dict[str, Task], workers: int = 8) -> dict[str, Any]:
    recs = [r for r in records if "error" not in r]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda r: regrade_one(tasks[r["task_id"]], r), recs))
    reward_mismatch = [r for r in rows if r["old_reward"] != r["new_reward"]]
    any_mismatch = [r for r in rows if r["diffs"]]
    return {
        "episodes": len(rows),
        "reward_mismatches": len(reward_mismatch),
        "field_mismatches": dict(Counter(k for r in any_mismatch for k in r["diffs"])),
        "mismatched": [{k: r[k] for k in ("task_id", "k", "old_reward", "new_reward", "diffs")} for r in any_mismatch],
        "reward_1": sum(r["new_reward"] for r in rows),
        "pass_non_generated": sum(1 for r in rows if r["pass_non_generated"]),
        "kinds_seen": dict(Counter(k for r in rows for k in (r["hidden_by_kind"] or {}))),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--out", type=Path, default=Path("runs/regrade"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    tasks = {t.task_id: t for split in ("train", "heldout") for t in load_tasks(a.tasks / f"{split}.jsonl")}
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in a.runs:
        results[path.stem] = regrade(load_records(path), tasks, a.workers)
        r = results[path.stem]
        print(f"{path.stem}: {r['episodes']} episodes, reward mismatches {r['reward_mismatches']}, "
              f"field mismatches {r['field_mismatches']}, reward 1: {r['reward_1']}, "
              f"pass_non_generated: {r['pass_non_generated']}")
    (a.out / "regrade.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if any(r["reward_mismatches"] for r in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
