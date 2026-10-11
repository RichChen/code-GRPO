"""``PolicyTrainer`` on a tiny random base and adapter (CPU): adapters, release/build, saving. Needs torch and peft."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from arl.rl.batch import Sequence  # noqa: E402
from arl.rl.trainer import ADAPTER_FILE, PolicyTrainer  # noqa: E402
from test_loss import tiny_model  # noqa: E402

LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}


@pytest.fixture
def cfg(tmp_path):
    tiny_model().save_pretrained(tmp_path / "base")
    init = get_peft_model(tiny_model(), LoraConfig(r=4, lora_alpha=8, target_modules=LORA["target_modules"],
                                                    init_lora_weights=False))  # B != 0: the adapter matters
    init.save_pretrained(tmp_path / "adapter")
    return {"base_model": str(tmp_path / "base"), "base_revision": None, "init_adapter": str(tmp_path / "adapter"),
            "weights_dtype": "float32", "autocast_dtype": "float16", "grad_scaler": False, "lora": LORA,
            "adapter_dtype": "float32", "gradient_checkpointing": False, "lr": 1e-2, "adam_betas": [0.9, 0.999],
            "weight_decay": 0.0, "max_grad_norm": 1.0, "tis": {"enabled": True, "cap": 2.0},
            "kl": {"beta": 0.0, "max_sequences": None}}


def seq(trainer, adv=1.0):
    ids = [1, 2, 3, 4, 5, 6]
    return Sequence(ids, 3, trainer.logps(ids, 3), adv, 0, 0)


def test_build_loads_sft_v1_into_both_adapters_and_trains_only_default(cfg):
    t = PolicyTrainer(cfg, device="cpu")
    t.build()
    init = load_file(f"{cfg['init_adapter']}/{ADAPTER_FILE}")
    assert {k: v for k, v in t.adapter_state().items()}.keys() == init.keys()
    assert all(torch.equal(t.adapter_state()[k], init[k]) for k in init)
    assert t.params and all(p.dtype == torch.float32 for p in t.params)
    s = seq(t)
    assert torch.allclose(t.ref_logps(s), torch.tensor(s.vllm_logprobs), atol=1e-6)
    stats = t.step([s], 8.0)
    assert stats["kl_k3_mean"] == pytest.approx(0.0, abs=1e-6) and stats["logprob_diff_max"] < 1e-5
    assert not all(torch.equal(t.adapter_state()[k], init[k]) for k in init)  # default moved
    ref = {n: p for n, p in t.model.named_parameters() if ".ref." in n}
    assert ref and not any(p.requires_grad for p in ref.values())
    assert torch.allclose(t.ref_logps(s), torch.tensor(s.vllm_logprobs), atol=1e-6)  # ref did not


def test_release_and_build_restore_adapter_optimizer_and_continue_identically(cfg, tmp_path):
    a, b = PolicyTrainer(cfg, device="cpu"), PolicyTrainer(cfg, device="cpu")
    a.build()
    b.build()
    for _ in range(2):
        s = seq(a)
        a.step([s], 8.0)
        b.step([Sequence(s.input_ids, s.start, s.vllm_logprobs, s.advantage, 0, 0)], 8.0)
        a.release()
        assert not a.resident
        a.build()
    assert all(torch.equal(x, y) for x, y in zip(a.adapter_state().values(), b.adapter_state().values()))
    a.save_state(tmp_path / "state.pt")
    c = PolicyTrainer(cfg, device="cpu")
    c.load_state(tmp_path / "state.pt")
    c.build()
    assert all(torch.equal(x, y) for x, y in zip(a.adapter_state().values(), c.adapter_state().values()))
    assert c.opt.state_dict()["state"].keys() == a.opt.state_dict()["state"].keys()


def test_saved_adapter_is_the_default_one_in_the_sft_v1_format(cfg, tmp_path):
    t = PolicyTrainer(cfg, device="cpu")
    t.build()
    t.step([seq(t)], 8.0)
    t.save_adapter(tmp_path / "out")
    saved = load_file(str(tmp_path / "out" / ADAPTER_FILE))
    assert saved.keys() == load_file(f"{cfg['init_adapter']}/{ADAPTER_FILE}").keys()
    assert all(torch.equal(saved[k], v) for k, v in t.adapter_state().items())
    conf = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert conf["r"] == 4 and conf["lora_dropout"] == 0.0 and sorted(conf["target_modules"]) == ["q_proj", "v_proj"]
    assert not (tmp_path / "out" / "ref").exists()


def test_greedy_and_logps_agree(cfg):
    t = PolicyTrainer(cfg, device="cpu")
    t.build()
    out = t.greedy([1, 2, 3], 5)
    lp = t.logps([1, 2, 3] + out, 3)
    assert len(out) == len(lp) == 5
    # each greedy token is the argmax, so its logprob is the row maximum
    full = torch.log_softmax(t.model(input_ids=torch.tensor([[1, 2, 3] + out])).logits[0].float(), -1)
    assert all(abs(full[2 + i].max().item() - lp[i]) < 1e-5 for i in range(5))
