"""Local SFT preview: LoRA on fp16 Qwen3-1.7B, fp16 autocast with a GradScaler. Runs in ``~/.venvs/train``.

    PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/train/bin/python -m arl.sft.train --config configs/sft/preview.json

Samples come from ``arl.sft.data`` (token ids plus ``loss_start``). The loss of an optimizer step is the mean
cross entropy over all assistant tokens of its micro-batches. Logs go to ``<out_dir>/log.jsonl``: train
records every ``log_every`` steps, val records at step 0, every ``eval_every`` steps and after each epoch.

Every step is checked. Any stop condition of the plan (non-finite loss, loss scale below ``min_scale`` or
dropping for ``max_consecutive_skipped_steps`` steps in a row, more than ``max_skipped_fraction`` of all
optimizer steps skipped by the GradScaler, OOM, estimated time over ``max_hours``, val loss after epoch 1 not
below the initial one) writes a ``stop`` record and exits with status 4; nothing is changed and retried.
``cuda_memory_limit_gib`` caps PyTorch's share of the card so that running out of memory is an OOM, not a
silent spill into system memory. After training the adapter is merged into the base and saved as fp16 in
``<out_dir>/merged`` together with the base model's own tokenizer files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Under WSL an allocation past the card's memory spills into shared system memory instead of failing, and
# varying sequence lengths fragment the default allocator until it does: training then runs ~100x slower.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")


class Stop(Exception):
    """A stop condition of the plan: report it, do not work around it."""


def load_samples(path: str | Path, max_len: int) -> tuple[list[dict[str, Any]], int]:
    """Samples no longer than ``max_len`` tokens (longer ones are dropped, never truncated) and the drop count."""
    out, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if len(s["input_ids"]) > max_len:
                dropped += 1
            else:
                out.append(s)
    return out, dropped


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warmup over ``warmup`` steps, then cosine decay to 0 at ``total`` (``step`` is 0-based)."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def epoch_batches(samples: list[dict[str, Any]], accum: int, seed: int, epoch: int) -> list[list[dict[str, Any]]]:
    order = list(range(len(samples)))
    random.Random(f"sft-{seed}-{epoch}").shuffle(order)
    return [[samples[i] for i in order[k:k + accum]] for k in range(0, len(order), accum)]


def loss_tokens(sample: dict[str, Any]) -> int:
    return len(sample["input_ids"]) - sample["loss_start"]


def sample_loss(model: Any, sample: dict[str, Any], device: str) -> tuple[torch.Tensor, int]:
    """Summed cross entropy over the sample's assistant tokens; logits are computed for those positions only."""
    ids = torch.tensor([sample["input_ids"]], device=device)
    n = loss_tokens(sample)
    logits = model(input_ids=ids, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float()
    return F.cross_entropy(logits, ids[0, sample["loss_start"]:], reduction="sum"), n


def autocast(cfg: dict[str, Any], device: str):
    dtype = getattr(torch, cfg["autocast_dtype"])
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda")


def evaluate(model: Any, samples: list[dict[str, Any]], cfg: dict[str, Any], device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in samples:
            with autocast(cfg, device):
                loss, k = sample_loss(model, s, device)
            total += loss.item()
            n += k
    model.train()
    return total / n


def apply_lora(model: Any, cfg: dict[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    adapter_dtype = getattr(torch, cfg["adapter_dtype"])
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(adapter_dtype)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(p.dtype != adapter_dtype for p in trainable):
        raise RuntimeError("LoRA parameters are missing or not in the adapter dtype")
    return model


def load_base(cfg: dict[str, Any], device: str) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg.get("base_revision"),
                                                 dtype=getattr(torch, cfg["weights_dtype"]), attn_implementation="sdpa")
    return model.to(device)


def gib(n: int) -> float:
    return round(n / 2**30, 2)


def train(cfg: dict[str, Any], model: Any = None, device: str = "cuda",
          log: Callable[[str], None] = print) -> dict[str, Any]:
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    log_path.unlink(missing_ok=True)
    t0 = time.time()

    def record(rec: dict[str, Any]) -> None:
        rec["elapsed_s"] = round(time.time() - t0, 1)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log(json.dumps(rec))

    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    stop_cfg = cfg["stop"]
    train_s, dropped_train = load_samples(cfg["train_data"], cfg["max_seq_len"])
    val_s, dropped_val = load_samples(cfg["val_data"], cfg["max_seq_len"])
    accum = cfg["grad_accum"] * cfg["micro_batch_size"]
    steps_per_epoch = math.ceil(len(train_s) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, round(cfg["warmup_ratio"] * total_steps))
    tokens_per_epoch = sum(len(s["input_ids"]) for s in train_s)
    summary: dict[str, Any] = {
        "train_samples": len(train_s), "val_samples": len(val_s), "dropped_over_max_len": dropped_train + dropped_val,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup,
        "train_tokens_per_epoch": tokens_per_epoch,
        "train_loss_tokens_per_epoch": sum(loss_tokens(s) for s in train_s),
    }
    record({"type": "start", **summary})

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        if cfg.get("cuda_memory_limit_gib"):
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["cuda_memory_limit_gib"] * 2**30 / total))
    if model is None:
        model = load_base(cfg, device)
    model = apply_lora(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    summary["trainable_params"] = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["adam_betas"]), weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device, enabled=cfg["grad_scaler"] and device == "cuda")
    model.train()

    val0 = evaluate(model, val_s, cfg, device)
    record({"type": "val", "step": 0, "epoch": 0, "val_loss": round(val0, 5)})
    vals = [(0, val0)]
    step, skipped_total, consecutive_skips = 0, 0, 0
    window: dict[str, float] = {"loss": 0.0, "tokens": 0, "seq_tokens": 0, "skipped": 0, "gnorm_max": 0.0}
    gnorms: list[float] = []
    train_time, train_seq_tokens = 0.0, 0
    try:
        for epoch in range(cfg["epochs"]):
            for batch in epoch_batches(train_s, accum, seed, epoch):
                ts = time.time()
                lr = lr_at(step, total_steps, warmup, cfg["lr"])
                for g in opt.param_groups:
                    g["lr"] = lr
                n_tokens = sum(loss_tokens(s) for s in batch)
                step_loss = 0.0
                for s in batch:
                    with autocast(cfg, device):
                        loss_sum, _ = sample_loss(model, s, device)
                    if not torch.isfinite(loss_sum):
                        raise Stop(f"non-finite loss at step {step + 1}: {loss_sum.item()}")
                    scaler.scale(loss_sum / n_tokens).backward()
                    step_loss += loss_sum.item()
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scale = scaler.get_scale()
                skipped = scaler.is_enabled() and scale < scale_before
                step += 1
                train_time += time.time() - ts
                seq_tokens = sum(len(s["input_ids"]) for s in batch)
                train_seq_tokens += seq_tokens
                if skipped:
                    skipped_total += 1
                    consecutive_skips += 1
                else:
                    consecutive_skips = 0
                    gnorms.append(gnorm)
                if scaler.is_enabled() and scale < stop_cfg["min_scale"]:
                    raise Stop(f"loss scale fell to {scale} at step {step}")
                if consecutive_skips >= stop_cfg["max_consecutive_skipped_steps"]:
                    raise Stop(f"loss scale dropped on {consecutive_skips} consecutive steps (now {scale}) at step {step}")
                if skipped_total > stop_cfg["max_skipped_fraction"] * total_steps:
                    raise Stop(f"{skipped_total} skipped steps exceed {stop_cfg['max_skipped_fraction']:.0%} of "
                               f"{total_steps} total steps at step {step}")
                window["loss"] += step_loss
                window["tokens"] += n_tokens
                window["seq_tokens"] += seq_tokens
                window["skipped"] += int(skipped)
                if math.isfinite(gnorm):
                    window["gnorm_max"] = max(window["gnorm_max"], gnorm)
                if step == stop_cfg["estimate_after_steps"]:
                    rate = train_seq_tokens / train_time
                    est_h = (tokens_per_epoch * cfg["epochs"] / rate) / 3600
                    summary["estimated_train_hours"] = round(est_h, 2)
                    record({"type": "estimate", "step": step, "tokens_per_s": round(rate, 1),
                            "estimated_train_hours": round(est_h, 2)})
                    if est_h > stop_cfg["max_hours"]:
                        raise Stop(f"estimated training time {est_h:.1f} h exceeds {stop_cfg['max_hours']} h")
                if step % cfg["log_every"] == 0 or step == total_steps:
                    rec = {"type": "train", "step": step, "epoch": epoch + 1, "loss": round(window["loss"] / window["tokens"], 5),
                           "lr": lr, "grad_norm": round(gnorm, 4) if math.isfinite(gnorm) else str(gnorm),
                           "grad_norm_max": round(window["gnorm_max"], 4), "scale": scale,
                           "skipped_steps": int(window["skipped"]), "skipped_total": skipped_total,
                           "tokens_per_s": round(train_seq_tokens / train_time, 1)}
                    if device == "cuda":
                        rec["peak_mem_gib"] = gib(torch.cuda.max_memory_allocated())
                        rec["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
                    record(rec)
                    window = {"loss": 0.0, "tokens": 0, "seq_tokens": 0, "skipped": 0, "gnorm_max": 0.0}
                if step % cfg["eval_every"] == 0 and step % steps_per_epoch != 0:
                    v = evaluate(model, val_s, cfg, device)
                    vals.append((step, v))
                    record({"type": "val", "step": step, "epoch": epoch + 1, "val_loss": round(v, 5)})
            v = evaluate(model, val_s, cfg, device)
            vals.append((step, v))
            record({"type": "val", "step": step, "epoch": epoch + 1, "val_loss": round(v, 5), "end_of_epoch": True})
            model.save_pretrained(out / f"adapter-epoch{epoch + 1}")
            if epoch == 0 and not v < val0:
                raise Stop(f"val loss after epoch 1 ({v:.5f}) is not below the initial {val0:.5f}")
    except torch.OutOfMemoryError as exc:
        summary["stop"] = f"OOM at step {step + 1}: {str(exc)[:300]}"
    except Stop as exc:
        summary["stop"] = str(exc)
    summary.update({
        "steps_done": step, "skipped_steps": skipped_total,
        "skipped_fraction": round(skipped_total / total_steps, 4), "final_scale": scaler.get_scale() if scaler.is_enabled() else None,
        "val_losses": [[s, round(v, 5)] for s, v in vals], "train_hours": round(train_time / 3600, 3),
        "wall_hours": round((time.time() - t0) / 3600, 3),
        "grad_norm_median": round(sorted(gnorms)[len(gnorms) // 2], 4) if gnorms else None,
    })
    if device == "cuda":
        summary["peak_mem_gib"] = gib(torch.cuda.max_memory_allocated())
        summary["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
    if "stop" in summary:
        record({"type": "stop", "reason": summary["stop"]})
    summary["model"] = model
    return summary


def merge_and_save(model: Any, cfg: dict[str, Any], dest: Path) -> None:
    from huggingface_hub import snapshot_download

    merged = model.merge_and_unload()
    merged = merged.to(getattr(torch, cfg["weights_dtype"]))
    merged.config.use_cache = True
    if dest.exists():
        shutil.rmtree(dest)
    merged.save_pretrained(dest, safe_serialization=True)
    snap = Path(snapshot_download(cfg["base_model"], revision=cfg.get("base_revision"), local_files_only=True))
    for name in TOKENIZER_FILES:
        if (snap / name).exists():
            shutil.copy(snap / name, dest / name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sft/preview.json"))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    summary = train(cfg)
    model = summary.pop("model")
    out = Path(cfg["out_dir"])
    if "stop" not in summary:
        t = time.time()
        merge_and_save(model, cfg, out / "merged")
        summary["merge_s"] = round(time.time() - t, 1)
    summary["config"] = cfg
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    return 4 if "stop" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
