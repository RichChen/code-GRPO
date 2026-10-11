"""GRPO run configuration: a JSON file, optionally ``{"base": <path>, "overrides": {...}}`` on top of another one."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arl.rollout.client import ClientConfig

# Sampling must be the model's own distribution: the TIS ratio compares the trainer's logp with vLLM's raw logprob.
SAMPLING = (1.0, 1.0, 0)
MEMORY_MODES = ("sleep", "restart", "colocate")
REQUIRED = ("base_model", "base_revision", "init_adapter", "weights_dtype", "autocast_dtype", "grad_scaler", "lora",
            "tasks", "n_tasks", "group_size", "steps", "seed", "max_turns", "rollout_concurrency", "client", "lr",
            "adam_betas", "weight_decay", "max_grad_norm", "max_completion_tokens_per_episode", "tis", "kl", "eval",
            "sync_check", "stop", "vllm", "out_dir")


class Stop(Exception):
    """A stop condition of the plan: report it, do not work around it."""


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "base" in data:
        data = merge(load(data["base"]), data.get("overrides", {}))
    return data


def client_config(cfg: dict[str, Any]) -> ClientConfig:
    return ClientConfig.load(cfg["client"])


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"missing GRPO config keys: {missing}")
    c = client_config(cfg)
    if c.endpoint != "tokens":
        raise ValueError("GRPO rollouts use the token client (endpoint 'tokens')")
    if (c.temperature, c.top_p, c.top_k) != SAMPLING:
        raise ValueError(f"GRPO sampling must be (temperature, top_p, top_k) == {SAMPLING}, "
                         f"got {(c.temperature, c.top_p, c.top_k)}")
    if cfg["lora"].get("dropout", 0.0) != 0.0:
        raise ValueError("LoRA dropout must be 0: the trainer's logp has to match the sampler's")
    if cfg["vllm"]["memory_mode"] not in MEMORY_MODES:
        raise ValueError(f"vllm.memory_mode must be one of {MEMORY_MODES}")
    if cfg["kl"].get("beta", 0.0) != 0.0:
        raise ValueError("beta must be 0 (the KL to SFT-v1 is only logged)")
    if cfg["group_size"] < 2 or cfg["n_tasks"] < 1:
        raise ValueError("need n_tasks >= 1 and group_size >= 2")
    return cfg
