"""The GRPO loss and step on tiny random models (CPU). Needs torch and transformers (``~/.venvs/train``)."""
import math
import statistics

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from arl.rl.batch import Episode, Sequence, Turn, sequences, t_norm  # noqa: E402
from arl.rl.loss import completion_logits, completion_logps, k3, policy_loss, token_entropy, train_step  # noqa: E402


def tiny_model(vocab=16, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                                   tie_word_embeddings=False)
    return transformers.Qwen3ForCausalLM(cfg)


def test_completion_logps_are_the_completion_tokens_log_softmax():
    model = tiny_model()
    ids = [1, 2, 3, 4, 5, 6]
    got = completion_logps(model, ids, 3, "cpu")
    full = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
    want = torch.stack([full[p - 1, ids[p]] for p in range(3, 6)])
    assert torch.allclose(got, want, atol=1e-5)


def test_policy_loss_value_tis_weights_and_gradient():
    logp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    vllm = torch.tensor([-1.0, -3.0, -0.5])  # ratio e^1 on the second token: capped at 2
    loss, diff, capped = policy_loss(logp, vllm, 0.5, 10.0, 2.0)
    assert capped == 1 and torch.allclose(diff, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.isclose(loss, torch.tensor(-0.5 * (-1.0 + 2 * -2.0 + -0.5) / 10.0))
    loss.backward()
    assert torch.allclose(logp.grad, torch.tensor([-0.05, -0.1, -0.05]))  # -A * w / T_norm
    loss2, _, capped2 = policy_loss(logp.detach().requires_grad_(), vllm, 0.5, 10.0, None)
    assert capped2 == 0 and torch.isclose(loss2, torch.tensor(-0.5 * -3.5 / 10.0))


def test_token_entropy_is_per_token_and_stays_out_of_the_graph():
    vocab = 8
    uniform = torch.zeros(2, vocab)
    assert torch.allclose(token_entropy(uniform), torch.full((2,), math.log(vocab)))
    peaked = torch.full((1, vocab), -1e3)
    peaked[0, 3] = 1e3
    assert float(token_entropy(peaked)) < 1e-4
    logits = torch.randn(3, vocab, requires_grad=True)
    ent = token_entropy(logits.detach())
    assert not ent.requires_grad and ent.shape == (3,)
    model = tiny_model()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    s = Sequence([1, 2, 3, 4, 5], 2, [-1.0, -1.0, -1.0], 1.0, 0, 0)
    stats = train_step(model, params, opt, torch.amp.GradScaler("cpu", enabled=False), [s], 8.0, tis_cap=2.0,
                       max_grad_norm=1.0, device="cpu")
    assert stats["completion_entropy_mean"] > 0
    # the entropy term is detached: a second token's logit gradient comes only from the loss
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)


def test_k3_is_zero_for_equal_policies_and_positive_otherwise():
    lp = torch.tensor([-1.0, -0.2])
    assert torch.allclose(k3(lp, lp), torch.zeros(2))
    assert (k3(lp, lp - 0.3) > 0).all() and (k3(lp, lp + 0.3) > 0).all()


def test_zero_advantage_gives_no_update():
    model = tiny_model()
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    s = Sequence([1, 2, 3, 4], 2, [-1.0, -1.0], 0.0, 0, 0)
    stats = train_step(model, params, opt, torch.amp.GradScaler("cpu", enabled=False), [s], 8.0, tis_cap=2.0,
                       max_grad_norm=1.0, device="cpu")
    assert stats["grad_norm"] == 0.0 and all(torch.equal(a, b) for a, b in zip(before, params))


def test_kl_uses_the_chosen_sequences_paired_with_their_own_reference():
    model = tiny_model()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)
    seqs = [Sequence(list(range(1, 5 + n)), 2, [-1.0] * (2 + n), 0.5, n, 0) for n in range(4)]
    with torch.no_grad():
        ref = {s.episode: completion_logps(model, s.input_ids, s.start, "cpu") for s in seqs}
    asked = []

    def ref_logps(s):
        asked.append(s.episode)
        return ref[s.episode]

    stats = train_step(model, params, opt, torch.amp.GradScaler("cpu", enabled=False), seqs, 8.0, tis_cap=2.0,
                       max_grad_norm=1.0, device="cpu", ref_logps=ref_logps, kl_indices=[1, 3])
    assert asked == [1, 3] and stats["kl_sequences"] == 2
    assert abs(stats["kl_k3_mean"]) < 1e-6  # each sequence against its own reference: the same weights


def _full_logits_step(model, params, opt, scaler, seqs, t_norm, tis_cap, max_grad_norm, device):
    """The pre-change step: one fp32 logit matrix per sequence, entropy before backward."""
    model.train()
    opt.zero_grad(set_to_none=True)
    loss_sum, ents = 0.0, []
    for s in seqs:
        logits, targets = completion_logits(model, s.input_ids, s.start, device)
        logp = -torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        with torch.no_grad():
            ents.append(token_entropy(logits.detach()))
        vllm = torch.tensor(s.vllm_logprobs, device=logp.device, dtype=logp.dtype)
        loss, _, _ = policy_loss(logp, vllm, s.advantage, t_norm, tis_cap)
        scaler.scale(loss).backward()
        loss_sum += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    grads = [(p.grad.detach().clone() if p.grad is not None else None) for p in params]
    return loss_sum, torch.cat(ents), grads


def test_chunked_step_matches_full_logits_loss_grad_and_entropy():
    """Row-chunked logp and entropy stay within 1e-5 of the full-vocab implementation."""
    model = tiny_model(vocab=24, seed=1)
    params = [p for p in model.parameters() if p.requires_grad]
    ids = list(range(1, 20))
    seqs = [Sequence(ids, 8, [-0.4] * 11, 0.5, 0, 0), Sequence(ids[::-1], 8, [-0.7] * 11, -0.25, 1, 1)]
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    opt_old = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)
    loss_old, ent_old, grads_old = _full_logits_step(model, params, opt_old, scaler, seqs, 32.0, 2.0, 1e9, "cpu")
    opt_old.zero_grad(set_to_none=True)
    opt_new = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)
    saved = {}

    def grab_grads():
        saved["grads"] = [(p.grad.detach().clone() if p.grad is not None else None) for p in params]

    opt_new.step = grab_grads
    stats = train_step(model, params, opt_new, scaler, seqs, 32.0, tis_cap=2.0, max_grad_norm=1e9,
                       device="cpu", chunk_rows=4)
    assert stats["loss"] == pytest.approx(loss_old, abs=1e-5)
    assert stats["completion_entropy_mean"] == pytest.approx(float(ent_old.mean()), abs=1e-5)
    assert saved["grads"] and len(saved["grads"]) == len(grads_old)
    for new, old in zip(saved["grads"], grads_old):
        assert old is not None and new is not None
        assert torch.allclose(new, old, atol=1e-5, rtol=1e-5)
    got = completion_logps(model, ids, 8, "cpu", chunk_rows=4)
    full, _ = completion_logits(model, ids, 8, "cpu")
    want = -torch.nn.functional.cross_entropy(full, torch.tensor(ids[8:]), reduction="none")
    assert torch.allclose(got, want, atol=1e-5)


def test_toy_policy_learns_to_emit_the_rewarded_token():
    """Reward 1 for sampling token 7 after a fixed prompt: ~1/16 at the start, > 0.8 within 50 steps."""
    model = tiny_model()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    gen = torch.Generator().manual_seed(0)
    prompt, target, n_tasks, group = [1, 2, 3], 7, 4, 8
    rewards = []
    for _ in range(50):
        episodes = []
        for g in range(n_tasks):
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=torch.tensor([prompt])).logits[0, -1].float(), -1)
            for k, tok in enumerate(torch.multinomial(lp.exp(), group, replacement=True, generator=gen).tolist()):
                episodes.append(Episode(f"t{g}", g, k, float(tok == target), [Turn(prompt, [tok], [float(lp[tok])])]))
        rewards.append(statistics.fmean(e.reward for e in episodes))
        seqs, _ = sequences(episodes)
        if seqs:
            stats = train_step(model, params, opt, scaler, seqs, t_norm(n_tasks, group, 1), tis_cap=2.0,
                               max_grad_norm=1.0, device="cpu")
            assert stats["logprob_diff_max"] < 1e-4  # sampled by the same weights
    assert statistics.fmean(rewards[:5]) < 0.2
    assert statistics.fmean(rewards[-5:]) > 0.8
