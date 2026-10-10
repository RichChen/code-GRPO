"""Training-loop checks on a tiny random Qwen3 model. Run in ``~/.venvs/train``:

    PYTHONPATH=src ~/.venvs/train/bin/python -m pytest tests/train/test_sft_train.py

Skipped when torch / transformers / peft are missing (e.g. in the main project venv).
"""

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from arl.sft.train import epoch_batches, load_samples, lr_at, sample_loss, train  # noqa: E402

TARGET = [5, 6, 7, 8, 9, 3]


def tiny_model(dtype=torch.float32):
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).to(dtype)


def write_samples(path, n, seed):
    g = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for _ in range(n):
            prefix = torch.randint(10, 64, (int(torch.randint(4, 12, (1,), generator=g)),), generator=g).tolist()
            f.write(json.dumps({"input_ids": prefix + TARGET, "loss_start": len(prefix)}) + "\n")


def config(tmp_path, **kw):
    write_samples(tmp_path / "train.jsonl", 64, 0)
    write_samples(tmp_path / "val.jsonl", 8, 1)
    cfg = {"lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj", "up_proj", "down_proj"]},
           "weights_dtype": "float32", "adapter_dtype": "float32", "autocast_dtype": "float16", "grad_scaler": True,
           "gradient_checkpointing": True, "max_seq_len": 32, "micro_batch_size": 1, "grad_accum": 4, "lr": 3e-2,
           "warmup_ratio": 0.03, "adam_betas": [0.9, 0.999], "weight_decay": 0.0, "max_grad_norm": 1.0, "epochs": 2, "seed": 0, "log_every": 2,
           "eval_every": 5, "train_data": str(tmp_path / "train.jsonl"), "val_data": str(tmp_path / "val.jsonl"),
           "out_dir": str(tmp_path / "out"),
           "stop": {"max_hours": 12, "min_scale": 1.0, "max_consecutive_skipped_steps": 10,
                    "max_skipped_fraction": 0.05, "estimate_after_steps": 3}}
    cfg.update(kw)
    return cfg


def test_lr_schedule_warms_up_then_decays_to_zero():
    lrs = [lr_at(s, 100, 3, 1e-4) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-4 / 3) and lrs[2] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:]))
    assert lr_at(100, 100, 3, 1e-4) == pytest.approx(0.0)


def test_batches_cover_every_sample_once_per_epoch():
    samples = [{"i": i} for i in range(37)]
    b0, b1 = epoch_batches(samples, 16, 0, 0), epoch_batches(samples, 16, 0, 1)
    assert [len(b) for b in b0] == [16, 16, 5]
    assert sorted(s["i"] for b in b0 for s in b) == list(range(37))
    assert b0 != b1 and epoch_batches(samples, 16, 0, 0) == b0


def test_long_samples_are_dropped_not_truncated(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"input_ids": [1] * 10, "loss_start": 8}) + "\n" + json.dumps({"input_ids": [1] * 5, "loss_start": 2}) + "\n")
    samples, dropped = load_samples(p, 8)
    assert dropped == 1 and samples == [{"input_ids": [1] * 5, "loss_start": 2}]


def test_sample_loss_counts_only_assistant_tokens():
    model = tiny_model().eval()
    sample = {"input_ids": [11, 12, 13, 14] + TARGET, "loss_start": 4}
    loss, n = sample_loss(model, sample, "cpu")
    ids = torch.tensor([sample["input_ids"]])
    logits = model(input_ids=ids).logits[0]
    want = torch.nn.functional.cross_entropy(logits[3:-1], ids[0, 4:], reduction="sum")
    assert n == len(TARGET) and torch.allclose(loss, want, atol=1e-5)


def run(cfg, dtype, device):
    summary = train(cfg, model=tiny_model(dtype).to(device), device=device, log=lambda *_: None)
    return summary, [json.loads(line) for line in open(f"{cfg['out_dir']}/log.jsonl")]


def test_cpu_training_lowers_val_loss_and_trains_only_fp32_lora(tmp_path):
    summary, log = run(config(tmp_path), torch.float32, "cpu")
    model = summary["model"]
    assert "stop" not in summary
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and p.dtype == torch.float32 for n, p in trainable)
    vals = [r["val_loss"] for r in log if r["type"] == "val"]
    assert vals[-1] < vals[0]
    assert summary["dropped_over_max_len"] == 0 and summary["total_steps"] == 32
    assert {r["type"] for r in log} >= {"start", "val", "train", "estimate"}
    assert (tmp_path / "out" / "adapter-epoch1").is_dir()


def test_val_loss_not_lower_after_epoch_one_stops(tmp_path):
    summary, log = run(config(tmp_path, lr=0.0), torch.float32, "cpu")
    assert summary["stop"].startswith("val loss after epoch 1")
    assert log[-1]["type"] == "stop" and summary["steps_done"] == 16


def test_time_estimate_over_the_limit_stops(tmp_path):
    cfg = config(tmp_path)
    cfg["stop"]["max_hours"] = 1e-9
    summary, _ = run(cfg, torch.float32, "cpu")
    assert summary["stop"].startswith("estimated training time") and summary["steps_done"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fp16_weights_with_autocast_and_grad_scaler_on_gpu(tmp_path):
    summary, log = run(config(tmp_path), torch.float16, "cuda")
    assert "stop" not in summary, summary.get("stop")
    model = summary["model"]
    frozen = {p.dtype for p in model.parameters() if not p.requires_grad}
    assert frozen == {torch.float16}
    train_recs = [r for r in log if r["type"] == "train"]
    assert all(math.isfinite(r["loss"]) and r["scale"] >= 1 for r in train_recs)
    assert "peak_mem_gib" in train_recs[0]
    vals = [r["val_loss"] for r in log if r["type"] == "val"]
    assert vals[-1] < vals[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_too_many_skipped_steps_stops(tmp_path, monkeypatch):
    real = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **kw: real(*a, init_scale=2.0 ** 60, **kw))
    summary, log = run(config(tmp_path), torch.float16, "cuda")
    # 32 total steps: the second skipped step is above 5%
    assert summary["stop"].startswith("2 skipped steps exceed 5% of 32"), summary.get("stop")
    assert summary["skipped_steps"] == 2 and summary["steps_done"] == 2
    assert log[-1]["type"] == "stop"
