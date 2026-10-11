"""One forward and backward on the longest completion seen for the overfit tasks.

The calibration records for those 8 tasks do not store token ids, only per-turn lengths. The longest
completion is 1024 tokens with a 2553-token prompt (train-mbpp-837-var_swap-41). Memory follows the
shapes, so the ids are a fixed in-vocab sequence of that length. This script uses the current
``completion_logits`` path (full-vocab fp32 logits, then entropy, then backward).
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from arl.rl.batch import Sequence
from arl.rl.config import load
from arl.rl.trainer import PolicyTrainer

PROMPT = 2553
COMPLETION = 1024


def gib(n: int) -> float:
    return round(n / 2**30, 3)


def allocated() -> float:
    import torch
    torch.cuda.synchronize()
    return gib(torch.cuda.memory_allocated())


def main() -> int:
    import torch
    import torch.nn.functional as F

    cfg = load("configs/grpo/overfit-local.json")
    # Measure the production memory cap and the real LoRA, not the overfit learning rate.
    cfg["lr"] = load("configs/grpo/v1.json")["lr"]
    trainer = PolicyTrainer(cfg)
    trainer.build()
    trainer.model.train()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report: dict = {"prompt_tokens": PROMPT, "completion_tokens": COMPLETION,
                    "source": "longest completion among the 8 overfit tasks in runs/grpo-calib/rl-train.jsonl",
                    "weights_gib": allocated()}
    ids = [1] * PROMPT + [2] * COMPLETION
    seq = Sequence(ids, PROMPT, [-1.0] * COMPLETION, 1.0, 0, 0)
    device = trainer.device
    stage = "forward"
    try:
        ids_t = torch.tensor([seq.input_ids], device=device)
        n = COMPLETION
        base = trainer.model.get_base_model() if hasattr(trainer.model, "get_base_model") else trainer.model
        with torch.autocast("cuda", dtype=trainer.amp_dtype()):
            hidden = base.model(input_ids=ids_t, use_cache=False).last_hidden_state
        report["after_forward_gib"] = allocated()
        report["after_forward_peak_gib"] = gib(torch.cuda.max_memory_allocated())
        stage = "logits"
        # The pre-change step materializes every completion logit in fp32 before the backward.
        rows = hidden[0, -(n + 1):-1]
        logits = base.lm_head(rows).float()
        targets = ids_t[0, seq.start:]
        report["after_logits_gib"] = allocated()
        report["logits_shape"] = list(logits.shape)
        report["logits_gib"] = gib(logits.numel() * logits.element_size())
        stage = "cross_entropy"
        with torch.autocast("cuda", dtype=trainer.amp_dtype()):
            logp = -F.cross_entropy(logits, targets, reduction="none")
        stage = "entropy"
        with torch.no_grad():
            detached = logits.detach()
            logp_full = detached - torch.logsumexp(detached, dim=-1, keepdim=True)
            ent = -(logp_full.exp() * logp_full).sum(dim=-1)
        report["after_entropy_gib"] = allocated()
        report["entropy_mean"] = float(ent.mean())
        stage = "backward"
        loss = -(seq.advantage * logp).sum() / float(
            cfg["n_tasks"] * cfg["group_size"] * cfg["max_completion_tokens_per_episode"])
        trainer.scaler.scale(loss).backward()
        report["after_backward_gib"] = allocated()
        report["backward_peak_gib"] = gib(torch.cuda.max_memory_allocated())
        report["loss"] = float(loss.detach())
        report["status"] = "ok"
    except RuntimeError as exc:
        report["status"] = "oom"
        report["oom_stage"] = stage
        report["error"] = str(exc).splitlines()[0][:500]
        try:
            report["allocated_at_error_gib"] = allocated()
        except RuntimeError:
            pass
        report["peak_gib"] = gib(torch.cuda.max_memory_allocated())
    print(json.dumps(report, indent=2))
    out = Path("runs/grpo-mem/before.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
