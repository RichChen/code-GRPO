import json
from pathlib import Path

import pytest

from arl.taskgen.sources import sha256_file

CONFIG = json.loads(Path("configs/sft/v1.json").read_text())


def test_sft_v1_config_records_every_adapter_and_data_file():
    assert set(CONFIG["adapter_sha256"]) == {"adapter_config.json", "adapter_model.safetensors", "README.md"}
    assert set(CONFIG["train_data_sha256"]) == {"train.jsonl", "val.jsonl", "stats.json"}
    assert CONFIG["base_revision"].startswith("70d244cc86cc")
    assert sha256_file(Path(CONFIG["train_data"]) / "stats.json") == CONFIG["train_data_sha256"]["stats.json"]


@pytest.mark.skipif(not Path(CONFIG["adapter_dir"]).is_dir(), reason="adapter is not in git")
def test_sft_v1_adapter_files_match_the_recorded_sha256():
    for name, digest in CONFIG["adapter_sha256"].items():
        assert sha256_file(Path(CONFIG["adapter_dir"]) / name) == digest, name
    for name, digest in CONFIG["train_data_sha256"].items():
        path = Path(CONFIG["train_data"]) / name
        if path.exists():
            assert sha256_file(path) == digest, name
