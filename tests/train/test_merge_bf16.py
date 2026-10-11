"""The bf16 merge refuses the local fp16 weights. Run in ``~/.venvs/train``."""
import pytest

pytest.importorskip("torch")

from arl.sft.merge_bf16 import merge_bf16  # noqa: E402


def test_merge_refuses_the_local_fp16_weights(tmp_path):
    with pytest.raises(ValueError):
        merge_bf16("runs/sft-preview/merged", "70d244cc86cc", tmp_path, tmp_path / "out")
