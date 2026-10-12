"""The GRPO trainer: the SFT-v1 LoRA (dropout 0) on the frozen base, trained with ``arl.rl.loss.train_step``.

A second, frozen adapter ``ref`` holds SFT-v1 for the logged KL. With ``memory_mode`` sleep or restart the
trainer only occupies the GPU while it trains: ``build()`` loads the base and restores the adapter, optimizer and
GradScaler state kept on the CPU, ``release()`` stores them back and frees the card for vLLM.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from arl.rl.batch import Sequence, kl_sample  # noqa: E402
from arl.rl.loss import completion_logps, train_step  # noqa: E402

ADAPTER_FILE = "adapter_model.safetensors"


def to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    return obj


class PolicyTrainer:
    def __init__(self, cfg: dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.model: Any = None
        self.opt: torch.optim.Optimizer | None = None
        self.scaler: Any = None
        self.params: list[torch.nn.Parameter] = []
        self.lora_state: dict[str, torch.Tensor] | None = None
        self.opt_state: dict[str, Any] | None = None
        self.scaler_state: dict[str, Any] | None = None
        self.limited = False

    @property
    def resident(self) -> bool:
        return self.model is not None

    def lora_config(self) -> Any:
        from peft import LoraConfig

        lora = self.cfg["lora"]
        return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                          target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM")

    def init_state(self) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        return load_file(str(Path(self.cfg["init_adapter"]) / ADAPTER_FILE))

    def load_adapter_state(self, state: dict[str, torch.Tensor], name: str) -> None:
        from peft import set_peft_model_state_dict

        result = set_peft_model_state_dict(self.model, state, adapter_name=name)
        missing = [k for k in result.missing_keys if "lora_" in k and f".{name}." in k]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"adapter {name}: {len(missing)} missing, {len(result.unexpected_keys)} unexpected keys")

    def build(self) -> float:
        if self.resident:
            return 0.0
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM

        t0 = time.time()
        if self.device == "cuda" and not self.limited and self.cfg.get("cuda_memory_limit_gib"):
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, self.cfg["cuda_memory_limit_gib"] * 2**30 / total))
            self.limited = True
        base = AutoModelForCausalLM.from_pretrained(self.cfg["base_model"], revision=self.cfg["base_revision"],
                                                    dtype=getattr(torch, self.cfg["weights_dtype"]),
                                                    attn_implementation="sdpa").to(self.device)
        base.config.use_cache = False
        if self.cfg.get("gradient_checkpointing"):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = get_peft_model(base, self.lora_config())
        init = self.init_state()
        self.load_adapter_state(self.lora_state if self.lora_state is not None else init, "default")
        self.model.add_adapter("ref", self.lora_config())
        self.load_adapter_state(init, "ref")
        self.model.set_adapter("default")
        adapter_dtype = getattr(torch, self.cfg.get("adapter_dtype", "float32"))
        for n, p in self.model.named_parameters():
            if ".lora_" in n:
                p.data = p.data.to(adapter_dtype)
        self.params = self.trainable()
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg["lr"], betas=tuple(self.cfg["adam_betas"]),
                                     weight_decay=self.cfg["weight_decay"])
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)
        cuda = self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda" if cuda else "cpu", enabled=bool(self.cfg["grad_scaler"]) and cuda)
        if self.scaler_state is not None:
            self.scaler.load_state_dict(self.scaler_state)
        return time.time() - t0

    def trainable(self) -> list[torch.nn.Parameter]:
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not named or any(".default." not in n for n, _ in named):
            raise RuntimeError("only the 'default' adapter may be trainable")
        return [p for _, p in named]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        return to_cpu(get_peft_model_state_dict(self.model, adapter_name="default"))

    def release(self) -> float:
        if not self.resident:
            return 0.0
        t0 = time.time()
        self.lora_state = self.adapter_state()
        self.opt_state = to_cpu(self.opt.state_dict())
        self.scaler_state = self.scaler.state_dict()
        self.model = self.opt = self.scaler = None
        self.params = []
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return time.time() - t0

    def amp_dtype(self) -> torch.dtype | None:
        return getattr(torch, self.cfg["autocast_dtype"]) if self.device == "cuda" else None

    def ref_logps(self, s: Sequence) -> torch.Tensor:
        self.model.set_adapter("ref")
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=self.amp_dtype(), enabled=self.device == "cuda"):
                return completion_logps(self.model, s.input_ids, s.start, self.device)
        finally:
            self.model.set_adapter("default")
            if self.trainable() != self.params:
                raise RuntimeError("switching adapters changed the trainable parameters")

    def step(self, seqs: list[Sequence], t_norm: float, step: int | None = None) -> dict[str, Any]:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tis = self.cfg["tis"]
        kl_indices = kl_sample(len(seqs), self.cfg["kl"].get("max_sequences"), self.cfg.get("seed", 0), step)
        stats = train_step(self.model, self.params, self.opt, self.scaler, seqs, t_norm,
                           tis_cap=tis["cap"] if tis["enabled"] else None, max_grad_norm=self.cfg["max_grad_norm"],
                           device=self.device, amp_dtype=self.amp_dtype(), ref_logps=self.ref_logps,
                           kl_indices=kl_indices)
        stats["lr"] = self.opt.param_groups[0]["lr"]
        if self.device == "cuda":
            stats["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            stats["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        return stats

    def save_adapter(self, dest: Path) -> None:
        self.model.save_pretrained(str(dest), selected_adapters=["default"])

    def greedy(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        self.model.eval()
        ids = torch.tensor([prompt_ids], device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=self.amp_dtype(), enabled=self.device == "cuda"):
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True)
        return out[0, len(prompt_ids):].tolist()

    def logps(self, input_ids: list[int], start: int) -> list[float]:
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=self.amp_dtype(), enabled=self.device == "cuda"):
            return completion_logps(self.model, input_ids, start, self.device).tolist()

    def save_state(self, path: Path) -> None:
        resident = self.resident
        state = {"lora": self.adapter_state() if resident else self.lora_state,
                 "opt": to_cpu(self.opt.state_dict()) if resident else self.opt_state,
                 "scaler": self.scaler.state_dict() if resident else self.scaler_state}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    def load_state(self, path: Path) -> None:
        if self.resident:
            raise RuntimeError("load_state before build()")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.lora_state, self.opt_state, self.scaler_state = state["lora"], state["opt"], state["scaler"]
