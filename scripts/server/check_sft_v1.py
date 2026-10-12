"""Check runs/sft-v1/adapter against configs/sft/v1.json, and every adapter tensor for NaN or inf.

    PYTHONPATH=src ~/.venvs/train/bin/python scripts/server/check_sft_v1.py --out runs/phase3/sft-v1-check.json

Exits 1 on a sha256 mismatch and 2 on a non-finite value. Never rewrites the json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import torch
    from safetensors.torch import load_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sft/v1.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    adapter = Path(cfg["adapter_dir"])
    files = []
    for name, expected in sorted(cfg["adapter_sha256"].items()):
        p = adapter / name
        got = sha256(p) if p.exists() else None
        files.append({"file": name, "bytes": p.stat().st_size if p.exists() else None, "expected": expected,
                      "actual": got, "match": got == expected})
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    bad = []
    elements = 0
    dtypes: dict[str, int] = {}
    for name, t in sorted(tensors.items()):
        elements += t.numel()
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
        nan, inf = int(torch.isnan(t).sum()), int(torch.isinf(t).sum())
        if nan or inf:
            bad.append({"tensor": name, "nan": nan, "inf": inf})
    absmax = max(float(t.abs().max()) for t in tensors.values())
    report = {"files": files, "all_sha256_match": all(f["match"] for f in files), "tensors": len(tensors),
              "elements": elements, "dtypes": dtypes, "abs_max": absmax, "non_finite_tensors": bad}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_sha256_match"]:
        return 1
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
