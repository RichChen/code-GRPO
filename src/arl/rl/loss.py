"""The GRPO policy-gradient step (torch). Shared by the LoRA trainer and the toy test.

``loss = -sum_i A_i * sum_{t in i} w_t * logp_t / T_norm`` over completion tokens only, with
``w_t = min(exp(logp_train - logp_vllm), cap)`` (truncated importance sampling, detached) or 1 without TIS.
One optimizer update per batch: the batch was sampled by the current weights, so there is no PPO ratio.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from arl.rl.batch import Sequence
from arl.rl.config import Stop


# Rows of the vocab projection handled at once. A full fp32 logit matrix is n × vocab; chunking keeps
# that matrix, its cross-entropy gradient, and the entropy intermediates to one chunk.
CHUNK_ROWS = 256


def _body_and_head(model: Any) -> tuple[Any, Any]:
    """Transformer body and ``lm_head``. LoRA, when present, already lives inside the body modules."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def _completion_hidden(model: Any, input_ids: list[int], start: int, device: str | torch.device):
    """Hidden states that predict ``input_ids[start:]``, and those target ids."""
    ids = torch.tensor([input_ids], device=device)
    n = len(input_ids) - start
    body, head = _body_and_head(model)
    hidden = body(input_ids=ids, use_cache=False).last_hidden_state[0, -(n + 1):-1]
    return head, hidden, ids[0, start:]


def completion_logits(model: Any, input_ids: list[int], start: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Full fp32 logits at completion positions. The training step does not use this; tests do."""
    ids = torch.tensor([input_ids], device=device)
    n = len(input_ids) - start
    logits = model(input_ids=ids, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float()
    return logits, ids[0, start:]


def _chunked_logps(head: Any, hidden: torch.Tensor, targets: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    parts = []
    for row in range(0, hidden.shape[0], chunk_rows):
        logits = head(hidden[row:row + chunk_rows]).float()
        parts.append(-F.cross_entropy(logits, targets[row:row + chunk_rows], reduction="none"))
        del logits
    return torch.cat(parts)


def completion_logps(model: Any, input_ids: list[int], start: int, device: str | torch.device,
                     chunk_rows: int = CHUNK_ROWS) -> torch.Tensor:
    """Log-probabilities of ``input_ids[start:]``. The vocab projection is applied in row chunks."""
    head, hidden, targets = _completion_hidden(model, input_ids, start, device)
    return _chunked_logps(head, hidden, targets, chunk_rows)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of each row. Callers that must keep it out of the graph pass detached logits."""
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -(logp.exp() * logp).sum(dim=-1)


def policy_loss(logp: torch.Tensor, vllm_logp: torch.Tensor, advantage: float, t_norm: float,
                tis_cap: float | None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The sequence's loss term, ``logp - vllm_logp`` (detached) and the number of TIS-capped tokens."""
    with torch.no_grad():
        diff = logp - vllm_logp
        if tis_cap is None:
            weight, capped = torch.ones_like(diff), 0
        else:
            ratio = diff.exp()
            weight, capped = ratio.clamp(max=tis_cap), int((ratio > tis_cap).sum())
    return -(advantage * weight * logp).sum() / t_norm, diff, capped


def k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimate of KL(policy || ref) on tokens sampled from the policy."""
    d = ref_logp - logp
    return d.exp() - 1.0 - d


def quantile(xs: torch.Tensor, q: float) -> float:
    s = xs.sort().values
    return float(s[max(0, math.ceil(q * len(s)) - 1)])


def train_step(model: Any, params: list[torch.nn.Parameter], opt: torch.optim.Optimizer, scaler: Any,
               seqs: list[Sequence], t_norm: float, *, tis_cap: float | None, max_grad_norm: float,
               device: str | torch.device, amp_dtype: torch.dtype | None = None,
               ref_logps: Callable[[Sequence], torch.Tensor] | None = None,
               kl_max_sequences: int | None = None, chunk_rows: int = CHUNK_ROWS) -> dict[str, Any]:
    """One optimizer update on all ``seqs`` (gradients accumulated sequence by sequence).

    Log-probabilities come from a chunked vocab projection. Each chunk is backwarded before the next is
    built, and the entropy of that chunk is computed only after that backward, on detached logits.
    """
    amp = (torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None and str(device).startswith("cuda")
           else contextlib.nullcontext())
    model.train()
    opt.zero_grad(set_to_none=True)
    diffs, kept, ents, loss_sum, capped, tokens = [], [], [], 0.0, 0, 0
    kl_n = len(seqs) if kl_max_sequences is None else min(len(seqs), kl_max_sequences)
    for i, s in enumerate(seqs):
        with amp:
            head, hidden, targets = _completion_hidden(model, s.input_ids, s.start, device)
        vllm_logp = torch.tensor(s.vllm_logprobs, device=hidden.device)
        seq_logp = []
        n = hidden.shape[0]
        for row in range(0, n, chunk_rows):
            with amp:
                logits = head(hidden[row:row + chunk_rows]).float()
                logp = -F.cross_entropy(logits, targets[row:row + chunk_rows], reduction="none")
            piece, diff, c = policy_loss(logp, vllm_logp[row:row + chunk_rows].to(dtype=logp.dtype),
                                         s.advantage, t_norm, tis_cap)
            if not torch.isfinite(piece):
                raise Stop(f"non-finite loss on sequence {i} (episode {s.episode}, turn {s.turn})")
            last = row + chunk_rows >= n
            scaler.scale(piece).backward(retain_graph=not last)
            with torch.no_grad():
                ents.append(token_entropy(logits.detach()))
            seq_logp.append(logp.detach())
            loss_sum += float(piece.detach())
            diffs.append(diff)
            capped += c
            tokens += len(diff)
            del logits, logp, piece
        if i < kl_n and seq_logp:
            kept.append(torch.cat(seq_logp))
    stats: dict[str, Any] = {"sequences": len(seqs), "loss_tokens": tokens, "loss": loss_sum}
    if diffs:
        d = torch.cat(diffs).abs()
        stats.update(logprob_diff_mean=float(d.mean()), logprob_diff_p99=quantile(d, 0.99),
                     logprob_diff_max=float(d.max()), tis_capped_fraction=capped / tokens if tis_cap else None)
    if ents:
        stats["completion_entropy_mean"] = float(torch.cat(ents).mean())
    if ref_logps is not None and kept:
        with torch.no_grad():
            kl = torch.cat([k3(lp, ref_logps(s).to(lp)) for lp, s in zip(kept, seqs)])
        stats.update(kl_k3_mean=float(kl.mean()), kl_sequences=len(kept))
    scaler.unscale_(opt)
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
    if not math.isfinite(gnorm) and not scaler.is_enabled():
        raise Stop("non-finite gradient norm")
    scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    stats.update(grad_norm=gnorm, grad_scale=scale, skipped=bool(scaler.get_scale() < scale))
    return stats
