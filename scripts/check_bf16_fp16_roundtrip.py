"""Cast the base checkpoint bf16 -> fp16 -> bf16 and report how many elements change.

One tensor at a time, so the check does not need the whole model in memory. Result is also written as JSON
when --out is set. Does not touch the held-out set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def roundtrip(directory: Path) -> dict:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {directory}")
    total = changed = other = 0
    max_abs = 0.0
    dtypes: dict[str, int] = {}
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                dtypes[str(tensor.dtype)] = dtypes.get(str(tensor.dtype), 0) + tensor.numel()
                if tensor.dtype != torch.bfloat16:
                    other += tensor.numel()
                    continue
                back = tensor.to(torch.float16).to(torch.bfloat16)
                diff = back != tensor
                n_changed = int(diff.sum())
                if n_changed:
                    err = (back.float() - tensor.float()).abs().max().item()
                    max_abs = max(max_abs, float(err))
                changed += n_changed
                total += tensor.numel()
                del tensor, back, diff
    return {"directory": str(directory), "bf16_elements": total, "changed_elements": changed,
            "max_abs_error": max_abs, "non_bf16_elements": other, "dtypes": dtypes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    report = roundtrip(a.dir)
    print(json.dumps(report, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
