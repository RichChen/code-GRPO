"""The GRPO loss and step on tiny random models (CPU). Needs torch and transformers (``~/.venvs/train``)."""
import statistics

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from arl.rl.batch import Episode, Sequence, Turn, sequences, t_norm  # noqa: E402
from arl.rl.loss import completion_logps, k3, policy_loss, train_step  # noqa: E402


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
