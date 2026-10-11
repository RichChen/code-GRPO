"""Merge the SFT-v1 LoRA into the original bf16 base.

    PYTHONPATH=src ~/.venvs/train/bin/python -m arl.sft.merge_bf16 \
        --base Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
        --adapter runs/sft-v1/adapter --out runs/sft-v1/merged-bf16

The local fp16 ``runs/sft-preview/merged`` must not be the base: casting those weights to bf16 drops mantissa bits.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from arl.sft.train import TOKENIZER_FILES  # noqa: E402

FORBIDDEN = "sft-preview/merged"


def merge_bf16(base: str, revision: str, adapter: Path, dest: Path) -> None:
    if FORBIDDEN in base.replace("\\", "/"):
        raise ValueError(f"refusing to start from the local fp16 merge ({base})")
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(adapter)).to(device)
    merged = model.merge_and_unload()
    if dest.exists():
        shutil.rmtree(dest)
    merged.save_pretrained(dest, safe_serialization=True)
    snap = Path(snapshot_download(base, revision=revision, local_files_only=True))
    for name in TOKENIZER_FILES:
        if (snap / name).exists():
            shutil.copy(snap / name, dest / name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--adapter", type=Path, default=Path("runs/sft-v1/adapter"))
    ap.add_argument("--out", type=Path, default=Path("runs/sft-v1/merged-bf16"))
    a = ap.parse_args(argv)
    merge_bf16(a.base, a.revision, a.adapter, a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
