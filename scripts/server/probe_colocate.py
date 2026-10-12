"""Can the trainer and vLLM stay resident together? Worst case of one GRPO training sequence, with vLLM up.

    PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/train/bin/python scripts/server/probe_colocate.py \
        --config configs/grpo/v1.json --out runs/server-bench/colocate-probe

Builds the trainer under ``cuda_memory_limit_gib``, starts vLLM next to it with ``gpu_memory_utilization``, then
runs one training step on a synthetic sequence of ``max_model_len`` tokens whose last ``--completion`` tokens are
the completion (the trainer handles one sequence at a time, so this is the largest activation footprint a step can
have). Card usage is sampled with nvidia-smi throughout. Afterwards vLLM must still answer a greedy request.
The trainer's update is thrown away; nothing is saved.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from pathlib import Path

from arl.rl.batch import Sequence
from arl.rl.config import load
from arl.rl.trainer import PolicyTrainer
from arl.rl.vllm_server import VLLMServer, gpu_used_mib


class CardMonitor(threading.Thread):
    def __init__(self, every_s: float = 0.5):
        super().__init__(daemon=True)
        self.every_s, self.peak, self.running = every_s, 0, True

    def run(self) -> None:
        while self.running:
            used = gpu_used_mib()
            if used is not None:
                self.peak = max(self.peak, used)
            time.sleep(self.every_s)


def kv_cache_tokens(log: Path) -> int | None:
    found = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log.read_text(encoding="utf-8", errors="replace"))
    return int(found[-1].replace(",", "")) if found else None


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--completion", type=int, default=1024, help="completion tokens of the synthetic sequence")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    cfg["vllm"]["memory_mode"] = "colocate"
    a.out.mkdir(parents=True, exist_ok=True)
    total = cfg["vllm"]["max_model_len"]
    rng = random.Random(0)
    ids = [rng.randrange(0, 150000) for _ in range(total)]
    seq = Sequence(ids, total - a.completion, [0.0] * a.completion, 1.0, 0, 0)
    report: dict = {"cuda_memory_limit_gib": cfg["cuda_memory_limit_gib"],
                    "gpu_memory_utilization": cfg["vllm"]["gpu_memory_utilization"],
                    "sequence_tokens": total, "completion_tokens": a.completion, "card_mib_before": gpu_used_mib()}
    trainer = PolicyTrainer(cfg)
    server = VLLMServer(cfg, a.out / "serve.log")
    mon = CardMonitor()
    mon.start()
    try:
        report["trainer_build_s"] = round(trainer.build(), 1)
        report["trainer_allocated_gib_after_build"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        report["card_mib_trainer_only"] = gpu_used_mib()
        report["vllm_start_s"] = round(server.start(), 1)
        report["vllm_kv_cache_tokens"] = kv_cache_tokens(a.out / "serve.log")
        report["card_mib_both_idle"] = gpu_used_mib()
        server.load_adapter("probe", cfg["init_adapter"])
        prompt = ids[:64]
        before = server.greedy("probe", prompt, 16)
        t0 = time.time()
        stats = trainer.step([seq], 1000.0)
        report["train_step_s"] = round(time.time() - t0, 1)
        report["trainer_peak_allocated_gib"] = stats.get("peak_allocated_gib")
        report["trainer_peak_reserved_gib"] = stats.get("peak_reserved_gib")
        report["train_step_skipped"] = stats.get("skipped")
        report["card_mib_after_step"] = gpu_used_mib()
        report["vllm_greedy_same_after_step"] = server.greedy("probe", prompt, 16) == before
        report["ok"] = True
    except Exception as exc:  # the report is the point: keep the error
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        mon.running = False
        server.stop()
        report["card_mib_peak_sampled"] = mon.peak
    (a.out / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
