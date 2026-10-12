"""The GRPO sampler: a vLLM 0.31 server with runtime LoRA loading, managed from the training process.

Interfaces (vLLM 0.31 source, see PROGRESS):
* runtime LoRA: env ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``, ``POST /v1/load_lora_adapter {lora_name, lora_path}``,
  ``POST /v1/unload_lora_adapter {lora_name}``; requests name the adapter as ``model``. Every step loads a new
  name and path (reloading a path in place would keep stale prefix-cache entries).
* sleep mode: ``--enable-sleep-mode``, routes only with ``VLLM_SERVER_DEV_MODE=1``: ``POST /sleep?level=N``,
  ``POST /wake_up?tags=...``. Level 2 discards the weights; after ``wake_up?tags=weights`` they are reloaded from
  disk with ``POST /collective_rpc {"method": "reload_weights"}``, then ``wake_up?tags=kv_cache``.
* ``/v1/completions`` with ``prompt_logprobs: 0`` returns each prompt token's logprob (first position ``None``).

    PYTHONPATH=src ~/.venvs/train/bin/python -m arl.rl.vllm_server probe --config configs/grpo/v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

VLLM = Path.home() / ".venvs/vllm/bin/vllm"


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class VLLMServer:
    def __init__(self, cfg: dict[str, Any], log_path: Path):
        self.cfg = cfg["vllm"]
        self.model = cfg["base_model"]
        self.revision = cfg["base_revision"]
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.sleeping = False
        self.url = f"http://127.0.0.1:{self.cfg['port']}"
        import httpx

        self.http = httpx.Client(timeout=self.cfg.get("request_timeout_s", 600), trust_env=False)

    def command(self) -> list[str]:
        c = self.cfg
        cmd = [str(VLLM), "serve", self.model, "--revision", self.revision, "--dtype", c["dtype"],
               "--max-model-len", str(c["max_model_len"]), "--host", "127.0.0.1", "--port", str(c["port"]),
               "--gpu-memory-utilization", str(c["gpu_memory_utilization"]), "--seed", str(c["seed"]),
               "--enable-lora", "--max-lora-rank", str(c["max_lora_rank"]), "--max-loras", str(c["max_loras"])]
        if c["memory_mode"] == "sleep":
            cmd.append("--enable-sleep-mode")
        return cmd + list(c.get("extra_args", []))

    def env(self) -> dict[str, str]:
        env = dict(os.environ, VLLM_ALLOW_RUNTIME_LORA_UPDATING="1", VLLM_SERVER_DEV_MODE="1", HF_HUB_OFFLINE="1")
        # FlashInfer JIT-builds its sampling kernels with `ninja` from the vLLM venv.
        env["PATH"] = os.pathsep.join([str(VLLM.parent), env.get("PATH", "")])
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # the trainer's allocator setting is not vLLM's
        return env

    def start(self) -> float:
        t0 = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "a", encoding="utf-8")
        log.write(f"\n# {time.strftime('%F %T')} {' '.join(self.command())}\n")
        log.flush()
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, env=self.env(),
                                     start_new_session=True)
        deadline = t0 + self.cfg.get("startup_timeout_s", 900)
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if self.http.get(f"{self.url}/v1/models", timeout=2).status_code == 200:
                    self.sleeping = False
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"vLLM did not come up in time; see {self.log_path}")

    def stop(self) -> float:
        t0 = time.time()
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        self.proc = None
        subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], check=False)
        return time.time() - t0

    def post(self, path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
        r = self.http.post(f"{self.url}{path}", json=body, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]}")
        return r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else None

    def sleep(self) -> float:
        t0 = time.time()
        self.post("/sleep", params={"level": self.cfg.get("sleep_level", 1)})
        self.sleeping = True
        return time.time() - t0

    def wake_up(self) -> float:
        t0 = time.time()
        if self.cfg.get("sleep_level", 1) == 2:
            self.post("/wake_up", params={"tags": "weights"})
            self.post("/collective_rpc", {"method": "reload_weights"})
            self.post("/wake_up", params={"tags": "kv_cache"})
        else:
            self.post("/wake_up")
        self.sleeping = False
        return time.time() - t0

    def load_adapter(self, name: str, path: str | Path) -> None:
        self.post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(Path(path).resolve())})

    def unload_adapter(self, name: str) -> None:
        self.post("/v1/unload_lora_adapter", {"lora_name": name})

    def complete(self, model: str, prompt: list[int], **kw: Any) -> dict[str, Any]:
        body = {"model": model, "prompt": prompt, "return_token_ids": True, "skip_special_tokens": False, **kw}
        return self.post("/v1/completions", body)["choices"][0]

    def greedy(self, model: str, prompt: list[int], max_tokens: int) -> list[int]:
        return self.complete(model, prompt, max_tokens=max_tokens, temperature=0.0)["token_ids"]

    def prompt_logprobs(self, model: str, ids: list[int]) -> list[float]:
        """Logprob of ``ids[1:]`` given the preceding tokens."""
        choice = self.complete(model, ids, max_tokens=1, temperature=0.0, prompt_logprobs=0)
        out = []
        for tok, entry in zip(ids[1:], choice["prompt_logprobs"][1:]):
            out.append(float(entry[str(tok)]["logprob"]))
        return out


def probe(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Can this machine use sleep mode with runtime LoRA? Times each transition and checks outputs survive it."""
    from arl.sft.data import Renderer

    r = Renderer(cfg["base_model"])
    prompt = r.encode(r.render([{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "List the files in the repository root."}], True))
    server = VLLMServer(cfg, out / "probe-serve.log")
    report: dict[str, Any] = {"command": " ".join(server.command()), "gpu_mib_before": gpu_used_mib()}
    try:
        report["startup_s"] = round(server.start(), 1)
        report["gpu_mib_awake"] = gpu_used_mib()
        server.load_adapter("probe-a", cfg["init_adapter"])
        first = server.greedy("probe-a", prompt, 32)
        lp = server.prompt_logprobs("probe-a", prompt + first)
        report["greedy_tokens"] = len(first)
        report["prompt_logprobs_tail"] = [round(x, 4) for x in lp[-3:]]
        if cfg["vllm"]["memory_mode"] == "sleep":
            report["sleep_s"] = round(server.sleep(), 2)
            report["gpu_mib_sleeping"] = gpu_used_mib()
            report["wake_s"] = round(server.wake_up(), 2)
            report["gpu_mib_woken"] = gpu_used_mib()
            server.load_adapter("probe-b", cfg["init_adapter"])
            report["same_after_wake_new_name"] = server.greedy("probe-b", prompt, 32) == first
            report["same_after_wake_old_name"] = server.greedy("probe-a", prompt, 32) == first
        report["ok"] = True
    except Exception as exc:  # the report is the point: keep the error
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        report["stop_s"] = round(server.stop(), 1)
    (out / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    from arl.rl.config import load

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probe"])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--memory-mode", choices=["sleep", "restart", "colocate"])
    ap.add_argument("--sleep-level", type=int)
    ap.add_argument("--out", type=Path, default=Path("runs/grpo-probe"))
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.memory_mode:
        cfg["vllm"]["memory_mode"] = a.memory_mode
    if a.sleep_level:
        cfg["vllm"]["sleep_level"] = a.sleep_level
    a.out.mkdir(parents=True, exist_ok=True)
    report = probe(cfg, a.out)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
