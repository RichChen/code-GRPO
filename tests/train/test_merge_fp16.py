"""The fp16 merge refuses the local preview weights. Run in ``~/.venvs/train``."""
import pytest

pytest.importorskip("torch")

from arl.sft.merge_fp16 import merge_fp16  # noqa: E402


def test_merge_refuses_the_local_preview_weights(tmp_path):
    with pytest.raises(ValueError):
        merge_fp16("runs/sft-preview/merged", "70d244cc86cc", tmp_path, tmp_path / "out")
