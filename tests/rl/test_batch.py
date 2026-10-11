"""GRPO batch logic without torch: advantages, training sequences, config checks, task schedule, rollouts."""
import json

import pytest

from arl.rl import config as rlconfig
from arl.rl.batch import Episode, Turn, batch_stats, group_advantages, sequences, t_norm, turn_from_record
from arl.rl.grpo import entropy_collapse_stop, format_rate_stop, step_task_ids, training_ids
from arl.rl.rollout import collect, episode_seed
from arl.rollout.client import ClientConfig


def ep(group, index, reward, turns):
    return Episode(f"t{group}", group, index, reward, turns, stop_reason="submitted")


def turn(prompt, completion):
    return Turn(list(prompt), list(completion), [-0.5] * len(completion), "stop")


def test_advantages_all_correct_all_wrong_and_mixed():
    assert group_advantages([1, 1, 1, 1]) is None
    assert group_advantages([0, 0, 0, 0]) is None
    assert group_advantages([1, 0, 0, 1]) == [0.5, -0.5, -0.5, 0.5]
    adv = group_advantages([1, 0, 0, 0])
    assert adv == [0.75, -0.25, -0.25, -0.25] and abs(sum(adv)) < 1e-12  # no division by the std


def test_degenerate_groups_are_skipped_and_counted():
    eps = [ep(0, k, 1, [turn([1], [2])]) for k in range(4)] + [ep(1, k, k % 2, [turn([1], [2])]) for k in range(4)]
    seqs, stats = sequences(eps)
    assert stats == {"groups": 2, "degenerate_groups": 1, "non_degenerate_fraction": 0.5}
    assert {s.episode for s in seqs} == {4, 5, 6, 7}
    assert [s.advantage for s in seqs] == [-0.5, 0.5, -0.5, 0.5]


def test_loss_mask_covers_only_each_turns_completion_in_a_multi_turn_trajectory():
    # turn 2's prompt holds turn 1's completion and a tool result; only turn 2's own completion is trained there
    t1 = turn([10, 11, 12], [20, 21, 99])
    t2 = turn([10, 11, 12, 20, 21, 99, 30, 31], [40, 99])
    seqs, _ = sequences([ep(0, 0, 1, [t1, t2]), ep(0, 1, 0, [turn([10, 11, 12], [50, 99])])])
    a, b, c = seqs
    assert (a.turn, b.turn, c.turn) == (0, 1, 0)
    for s, prompt, completion in ((a, t1.prompt_ids, t1.completion_ids), (b, t2.prompt_ids, t2.completion_ids)):
        mask = [i >= s.start for i in range(len(s.input_ids))]
        assert [x for x, m in zip(s.input_ids, mask) if m] == completion
        assert [x for x, m in zip(s.input_ids, mask) if not m] == prompt
        assert s.n_completion == len(s.vllm_logprobs) == len(completion)
    assert b.input_ids[b.start - 2:b.start] == [30, 31]  # the tool result stays in the masked prompt


def test_training_sequence_is_the_sampled_tokens_without_re_rendering(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("training sequences must not be re-rendered")

    monkeypatch.setattr("arl.sft.data.Renderer.render", boom)
    monkeypatch.setattr("arl.sft.data.Renderer.encode", boom)
    # ids no tokenizer would produce for any text: they are used as they are
    rec = {"prompt_ids": [151644, 7, 7, 7], "completion_ids": [3, 3, 151645], "completion_logprobs": [-1.0, -2.0, -0.1],
           "finish_reason": "stop"}
    t = turn_from_record(rec)
    seqs, _ = sequences([ep(0, 0, 1, [t]), ep(0, 1, 0, [t])])
    assert seqs[0].input_ids == rec["prompt_ids"] + rec["completion_ids"]
    assert seqs[0].vllm_logprobs == rec["completion_logprobs"] and seqs[0].start == 4


def test_turn_records_need_one_logprob_per_token():
    with pytest.raises(ValueError):
        turn_from_record({"prompt_ids": [1], "completion_ids": [2, 3], "completion_logprobs": [-1.0]})
    with pytest.raises(ValueError):
        turn_from_record({"prompt_ids": [1], "completion_ids": [], "completion_logprobs": []})


def test_t_norm_is_a_fixed_constant_and_batch_stats():
    assert t_norm(16, 8, 4096) == 16 * 8 * 4096
    s = batch_stats([ep(0, 0, 1, [turn([1, 2], [3]), turn([1, 2, 3], [4, 5])]), ep(0, 1, 0, [turn([1], [2])])])
    assert s["reward_mean"] == 0.5 and s["turns_mean"] == 1.5 and s["completion_tokens_max"] == 3
    assert s["sequence_tokens"] == 3 + 5 + 2
    bad = ep(0, 0, 0, [turn([1], [2])])
    bad.format_errors, bad.tool_calls = 1, 4
    ok = ep(0, 1, 1, [turn([1], [2])])
    ok.tool_calls = 6
    s = batch_stats([bad, ok])
    assert s["format_error_rate_per_tool_call"] == 0.1
    assert s["episodes_with_format_error"] == 1 and s["episodes_with_format_error_fraction"] == 0.5
    assert batch_stats([ok]).get("format_error_rate_per_tool_call") == 0.0


def test_format_error_stop_is_the_mean_per_call_rate_over_ten_steps():
    assert format_rate_stop([0.2] * 9) is None
    assert format_rate_stop([0.2] * 10) is not None
    assert format_rate_stop([0.0] * 5 + [0.2] * 5) is None  # mean 0.10, the limit is exclusive
    assert format_rate_stop([0.0] * 5 + [0.2001] * 5) is not None
    assert format_rate_stop([0.5] * 10 + [0.01] * 10) is None  # only the last 10 steps count
    assert format_rate_stop([0.2] * 9 + [None]) is None  # a step with no tool calls breaks the run


def test_entropy_stop_compares_with_the_first_five_steps():
    assert entropy_collapse_stop([1.0] * 5) is None
    assert entropy_collapse_stop([1.0] * 5 + [0.30]) is None  # 30% of the baseline is the floor, not a breach
    assert entropy_collapse_stop([1.0] * 5 + [0.29]) is not None
    # a slow slide still trips the original baseline; the mean of the previous 5 (0.7) would not
    slow = [1.0] * 5 + [0.9, 0.8, 0.7, 0.6, 0.5, 0.29]
    assert entropy_collapse_stop(slow) is not None
    assert entropy_collapse_stop([2.0, 2.0, 2.0, 2.0, 2.0, 0.7]) is None  # 0.7 > 0.3 * 2.0


def write_cfg(tmp_path, **client):
    c = {"endpoint": "tokens", "temperature": 1.0, "top_p": 1.0, "top_k": 0, **client}
    (tmp_path / "client.json").write_text(json.dumps(c))
    base = json.loads(open("configs/grpo/v1.json").read())
    base["client"] = str(tmp_path / "client.json")
    return base


@pytest.mark.parametrize("client", [{"temperature": 0.8}, {"top_p": 0.95}, {"top_k": 20}, {"top_k": None},
                                    {"endpoint": "chat"}])
def test_grpo_refuses_to_start_unless_sampling_is_the_full_distribution(tmp_path, client):
    with pytest.raises(ValueError):
        rlconfig.validate(write_cfg(tmp_path, **client))


def test_config_checks_and_the_shipped_config(tmp_path):
    assert rlconfig.validate(write_cfg(tmp_path))
    rlconfig.validate(rlconfig.load("configs/grpo/v1.json"))
    for key, value in (("lora", {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": []}),
                       ("kl", {"beta": 0.1})):
        cfg = write_cfg(tmp_path)
        cfg[key] = value
        with pytest.raises(ValueError):
            rlconfig.validate(cfg)
    cfg = write_cfg(tmp_path)
    cfg["vllm"] = dict(cfg["vllm"], memory_mode="swap")
    with pytest.raises(ValueError):
        rlconfig.validate(cfg)


def test_learning_rates_follow_revision_2c3():
    """Production LoRA rate is a constant 1e-5; the overfit run uses 5e-5 and does not add warmup."""
    v1 = json.loads(open("configs/grpo/v1.json", encoding="utf-8").read())
    over = json.loads(open("configs/grpo/overfit-local.json", encoding="utf-8").read())
    assert v1["lr"] == 1e-5 and "warmup" not in v1
    assert "1e-5" in v1["lr_note"] and "1e-4" in v1["lr_note"]
    assert over["overrides"]["lr"] == 5e-5 and "5e-5" in over["lr_note"]
    assert rlconfig.load("configs/grpo/v1.json")["lr"] == 1e-5
    loaded = rlconfig.load("configs/grpo/overfit-local.json")
    assert loaded["lr"] == 5e-5 and "warmup" not in loaded


def test_config_overrides(tmp_path):
    (tmp_path / "o.json").write_text(json.dumps({"base": "configs/grpo/v1.json",
                                                 "overrides": {"steps": 3, "vllm": {"memory_mode": "restart"}}}))
    cfg = rlconfig.load(tmp_path / "o.json")
    assert cfg["steps"] == 3 and cfg["vllm"]["memory_mode"] == "restart" and cfg["vllm"]["port"] == 8000


def test_task_schedule_covers_every_task_once_per_epoch(tmp_path):
    ids = [f"t{i:02d}" for i in range(10)]
    first = [x for s in range(5) for x in step_task_ids(ids, 4, s, 0)]
    assert sorted(first[:8]) == sorted(set(first[:8])) and len(first) == 20
    assert sorted(first[:10]) == ids and sorted(first[10:20]) == ids
    assert step_task_ids(ids, 4, 3, 0) == step_task_ids(ids, 4, 3, 0) != step_task_ids(ids, 4, 3, 1)
    f = tmp_path / "ids.txt"
    f.write_text("\n".join(ids) + "\n")
    cfg = {"tasks": str(f), "task_sample": 3, "task_sample_seed": 0}
    assert training_ids(cfg) == training_ids(cfg) and len(training_ids(cfg)) == 3
    assert training_ids({"tasks": str(f), "task_sample": None}) == ids


class FakeTokenPolicy:
    """Fixes the toy task in one edit, then submits; records token turns like ``TokenPolicy``."""

    def __init__(self, config, seed, model):
        self.seed, self.model, self.turns = seed, model, []

    def __call__(self, messages, tools):
        n = len(self.turns)
        self.turns.append({"prompt_ids": [n, self.seed % 97], "completion_ids": [n + 1, 151645],
                           "completion_logprobs": [-0.1, -0.2], "finish_reason": "stop"})
        if n == 0:
            args = {"path": "pkg/nums.py", "old_string": "return a - b", "new_string": "return a + b"}
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "edit_file", "arguments": json.dumps(args)}}]}
        return {"role": "assistant", "content": "Fixed.", "tool_calls": None}


def test_collect_runs_groups_and_keeps_the_token_turns(task):
    made = []

    def factory(config, seed, model):
        made.append((config.model, seed, model))
        return FakeTokenPolicy(config, seed, model)

    eps = collect([task], 3, ClientConfig(endpoint="tokens"), "step-4", step=4, seed=0, concurrency=2, max_turns=5,
                  policy_factory=factory)
    assert [(e.group, e.index, e.reward) for e in eps] == [(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)]
    assert all(len(e.turns) == 2 and e.turns[1].completion_ids == [2, 151645] for e in eps)
    assert {m for m, _, _ in made} == {"step-4"} and len({s for _, s, _ in made}) == 3
    assert episode_seed(0, 4, task.task_id, 0) != episode_seed(0, 5, task.task_id, 0)
