"""Sampling parameters of the phase 2C client configs (audit revision 2C-1, item 3)."""
import json
from pathlib import Path

import pytest

from arl.rollout.client import ClientConfig

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def test_eval_config_matches_the_earlier_dev_evaluations():
    c = ClientConfig.load(ROOT / "configs/sft/v1-eval.json")
    assert (c.endpoint, c.temperature, c.top_p, c.top_k) == ("tokens", 0.8, 1.0, 20)


def test_explicit_eval_top_k_is_the_server_default_used_before():
    path = SNAPSHOT / "generation_config.json"
    if not path.exists():
        pytest.skip("Qwen3-1.7B snapshot not downloaded")
    assert json.loads(path.read_text())["top_k"] == ClientConfig.load(ROOT / "configs/sft/v1-eval.json").top_k
    merged = ROOT / "runs/sft-preview/merged/generation_config.json"
    if merged.exists():
        assert json.loads(merged.read_text())["top_k"] == 20


def test_calibration_samples_the_full_distribution():
    c = ClientConfig.load(ROOT / "configs/grpo/calibrate-v1.json")
    assert (c.endpoint, c.temperature, c.top_p, c.top_k) == ("tokens", 1.0, 1.0, 0)
