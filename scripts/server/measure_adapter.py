"""Measure a saved GRPO adapter on a run's training tasks with ``arl.rl.grpo.measure`` (the before/after function).

    PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/train/bin/python scripts/server/measure_adapter.py \
        --config configs/grpo/overfit-local.json --adapter runs/grpo-overfit/adapters/step-13 --tag after-step13

Starts vLLM alone (no trainer), hot-loads the adapter under its directory name, samples K episodes per task with
the run's GRPO client (T=1.0, top_p 1.0, top_k 0) and the same seeds as the run's ``before`` measurement, and
appends a ``measure`` record to ``<out_dir>/log.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from arl.env.task import load_tasks
from arl.rl.config import client_config, load, validate
from arl.rl.grpo import TASKS, measure, training_ids
from arl.rl.vllm_server import VLLMServer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    cfg = validate(load(a.config))
    k = cfg["measure"]["k"]
    out = Path(cfg["out_dir"])
    by_id = {t.task_id: t for t in load_tasks(TASKS)}
    tasks = [by_id[i] for i in training_ids(cfg)]
    server = VLLMServer(cfg, out / "serve.log")
    name = a.adapter.name
    try:
        server.start()
        server.load_adapter(name, a.adapter)
        t0 = time.time()
        rec = {"type": "measure", **measure(tasks, k, client_config(cfg), name, cfg, a.tag),
               "adapter_path": str(a.adapter), "measure_s": round(time.time() - t0, 1)}
    finally:
        server.stop()
    with open(out / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
