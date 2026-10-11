"""GRPO training loop. Runs in ``~/.venvs/train`` (torch, peft, httpx, transformers); vLLM is a subprocess.

    PYTHONPATH=src HF_HUB_OFFLINE=1 ~/.venvs/train/bin/python -m arl.rl.grpo --config configs/grpo/v1.json

Each step: sample ``n_tasks x group_size`` episodes with the current adapter (token client, T=1.0, top_p 1.0,
top_k 0); free the card for the trainer (``memory_mode``: sleep -> vLLM sleeps, restart -> vLLM stops, colocate ->
both stay); one optimizer update; save the adapter under a new name; give the card back to vLLM and hot-load the
adapter; check that vLLM and the trainer agree on it (greedy tokens of a fixed prompt, and the mean absolute
logprob difference on one of this step's sequences). Logs go to ``<out_dir>/log.jsonl``. Any stop condition
writes a ``stop`` record and exits with status 4.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from arl.env.episode import initial_messages
from arl.env.task import Task, load_tasks
from arl.rl.batch import Episode, Sequence, batch_stats, sequences, t_norm
from arl.rl.config import Stop, client_config, load, validate
from arl.rl.rollout import collect
from arl.rollout.client import ClientConfig

TASKS = Path("data/tasks/train.jsonl")


def read_ids(path: str | Path) -> list[str]:
    return sorted(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def training_ids(cfg: dict[str, Any]) -> list[str]:
    ids = read_ids(cfg["tasks"])
    if cfg.get("task_sample"):
        ids = sorted(random.Random(f"grpo-tasks-{cfg['task_sample_seed']}").sample(ids, cfg["task_sample"]))
    return ids


def step_task_ids(ids: list[str], n: int, step: int, seed: int) -> list[str]:
    """Tasks of a step: epochs of seeded shuffles of ``ids``, consumed ``n`` at a time."""
    out = []
    pos = step * n
    while len(out) < n:
        epoch, i = divmod(pos, len(ids))
        order = list(ids)
        random.Random(f"grpo-{seed}-epoch-{epoch}").shuffle(order)
        out.append(order[i])
        pos += 1
    return out


def format_rate_stop(rates: list[float | None], window: int = 10, limit: float = 0.10) -> str | None:
    """Stop when the mean per-call format-error rate over the last ``window`` steps exceeds ``limit``.

    The rate is errors per tool call, not the fraction of episodes that erred once. A step with no tool
    calls (rate ``None``) breaks the run of consecutive steps.
    """
    if len(rates) < window:
        return None
    recent = rates[-window:]
    if any(r is None for r in recent):
        return None
    mean = statistics.fmean(recent)
    if mean > limit:
        return f"mean per-call format-error rate over {window} steps is {mean:.4f} > {limit}"
    return None


def entropy_collapse_stop(entropies: list[float], baseline_steps: int = 5, ratio: float = 0.30) -> str | None:
    """Stop when the latest entropy is below ``ratio`` of the mean of the first ``baseline_steps``.

    The baseline is the start of training, not a rolling window: a slow collapse still crosses it.
    The check begins on the first step after the baseline. Entropy is the mean over completion tokens.
    """
    if len(entropies) <= baseline_steps:
        return None
    base = statistics.fmean(entropies[:baseline_steps])
    current = entropies[-1]
    if current < ratio * base:
        return (f"completion entropy {current:.4f} < {ratio} of the first {baseline_steps} steps' "
                f"mean {base:.4f}")
    return None


def logged_monitors(path: Path) -> tuple[list[float | None], list[float]]:
    """Per-step format-error rates and entropies already written to ``log.jsonl`` (for resume)."""
    rates: list[float | None] = []
    entropies: list[float] = []
    if not path.exists():
        return rates, entropies
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") != "step":
            continue
        rates.append(rec.get("format_error_rate_per_tool_call"))
        if rec.get("completion_entropy_mean") is not None:
            entropies.append(rec["completion_entropy_mean"])
    return rates, entropies


def check_prompt(task: Task, renderer: Any) -> list[int]:
    return renderer.encode(renderer.render(initial_messages(task), True))


def measure(tasks: list[Task], k: int, client: ClientConfig, model: str, cfg: dict[str, Any], tag: str) -> dict[str, Any]:
    eps = collect(tasks, k, client, model, step=-1, seed=cfg["seed"], concurrency=cfg["rollout_concurrency"],
                  max_turns=cfg["max_turns"])
    per_task: dict[str, list[float]] = {}
    for e in eps:
        per_task.setdefault(e.task_id, []).append(e.reward)
    return {"tag": tag, "model": model, "k": k, "reward_mean": round(statistics.fmean(e.reward for e in eps), 4),
            "per_task": {t: round(statistics.fmean(v), 3) for t, v in sorted(per_task.items())}, **batch_stats(eps)}


def evaluate(cfg: dict[str, Any], model: str, out: Path, step: int) -> dict[str, Any]:
    from arl.rollout.calibrate import run_calibration, summarize

    ev = cfg["eval"]
    config = replace(ClientConfig.load(ev["client"]), model=model)
    wanted = set(read_ids(ev["ids"]))
    tasks = [t for t in load_tasks(TASKS) if t.task_id in wanted]
    path = out / "eval" / f"step-{step}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = run_calibration(tasks, path, config, k=ev["k"], concurrency=ev.get("concurrency", 8),
                           max_turns=cfg["max_turns"], log=lambda *_: None)
    s = summarize(recs, {t.task_id: t for t in tasks}, ev["k"])
    non_gen = [int(bool(r["grade"].get("pass_non_generated"))) for r in recs]
    return {"step": step, "episodes": s["episodes"], "solve_rate_strict": s["solve_rate"],
            "solve_rate_test_challenge": round(sum(non_gen) / len(non_gen), 4) if non_gen else None,
            "errors": sum(1 for r in recs if r.get("error"))}


class Run:
    def __init__(self, cfg: dict[str, Any], resume: bool = False):
        from arl.rl.trainer import PolicyTrainer
        from arl.rl.vllm_server import VLLMServer
        from arl.sft.data import Renderer

        self.cfg = validate(cfg)
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "log.jsonl"
        self.client = client_config(cfg)
        self.mode = cfg["vllm"]["memory_mode"]
        all_tasks = {t.task_id: t for t in load_tasks(TASKS)}
        self.ids = training_ids(cfg)
        self.tasks = all_tasks
        self.renderer = Renderer(cfg["base_model"])
        self.prompt = check_prompt(all_tasks[self.ids[0]], self.renderer)
        self.server = VLLMServer(cfg, self.out / "serve.log")
        self.trainer = PolicyTrainer(cfg)
        self.t_norm = t_norm(cfg["n_tasks"], cfg["group_size"], cfg["max_completion_tokens_per_episode"])
        self.step = 0
        self.adapter = ("step-0", Path(cfg["init_adapter"]))
        self.skipped_in_row = 0
        self.format_rates: list[float | None] = []
        self.entropies: list[float] = []
        state = self.out / "state.json"
        if resume and state.exists():
            s = json.loads(state.read_text())
            self.step, self.adapter = s["step"], (s["adapter_name"], Path(s["adapter_path"]))
            self.trainer.load_state(self.out / "trainer.pt")
            self.format_rates, self.entropies = logged_monitors(self.log_path)
        elif not resume:
            self.log_path.unlink(missing_ok=True)
        self.t0 = time.time()

    def record(self, rec: dict[str, Any]) -> None:
        rec["elapsed_s"] = round(time.time() - self.t0, 1)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    # -- card handover ---------------------------------------------------------------------------------

    def free_for_trainer(self) -> dict[str, float]:
        if self.mode == "sleep":
            return {"vllm_sleep_s": round(self.server.sleep(), 2)}
        if self.mode == "restart":
            return {"vllm_stop_s": round(self.server.stop(), 2)}
        return {}

    def give_back_to_vllm(self) -> dict[str, float]:
        t: dict[str, float] = {}
        if self.mode != "colocate":
            t["trainer_release_s"] = round(self.trainer.release(), 2)
        if self.mode == "sleep" and self.server.sleeping:
            t["vllm_wake_s"] = round(self.server.wake_up(), 2)
        elif self.mode == "restart" or self.server.proc is None:
            t["vllm_start_s"] = round(self.server.start(), 2)
        return t

    def sync(self, name: str, path: Path, previous: str | None, ref_greedy: list[int], check: Sequence,
             ref_lp: list[float]) -> dict[str, Any]:
        t0 = time.time()
        self.server.load_adapter(name, path)
        if previous and previous != name:
            try:
                self.server.unload_adapter(previous)
            except RuntimeError:
                pass  # not loaded (e.g. after a restart)
        got = self.server.greedy(name, self.prompt, self.cfg["sync_check"]["greedy_tokens"])
        lp = self.server.prompt_logprobs(name, check.input_ids)[check.start - 1:]
        mad = statistics.fmean(abs(a - b) for a, b in zip(ref_lp, lp))
        first = next((i for i, (a, b) in enumerate(zip(got, ref_greedy)) if a != b), None)
        if first is None and len(got) != len(ref_greedy):
            first = min(len(got), len(ref_greedy))
        return {"sync_load_check_s": round(time.time() - t0, 2), "sync_greedy_equal": first is None,
                "sync_greedy_first_diff": first, "sync_logprob_mad": round(mad, 5), "sync_check_tokens": len(lp)}

    # -- loop ------------------------------------------------------------------------------------------

    def start(self) -> None:
        """Trainer reference for the starting adapter, then vLLM up with it, then the same checks as every step."""
        self.trainer.build()
        greedy = self.trainer.greedy(self.prompt, self.cfg["sync_check"]["greedy_tokens"])
        check = Sequence(self.prompt + greedy, len(self.prompt), [], 0.0, -1, -1)
        lp = self.trainer.logps(check.input_ids, check.start)
        t = self.give_back_to_vllm()
        rec = self.sync(*self.adapter, None, greedy, check, lp)
        self.record({"type": "start", "step": self.step, "adapter": self.adapter[0], "mode": self.mode,
                     "tasks": len(self.ids), "t_norm": self.t_norm, **t, **rec})
        self.check_sync(rec)

    def check_sync(self, rec: dict[str, Any]) -> None:
        if not rec["sync_greedy_equal"]:
            raise Stop(f"greedy output of vLLM and the trainer differ at token {rec['sync_greedy_first_diff']}")
        if rec["sync_logprob_mad"] > self.cfg["sync_check"]["max_logprob_mad"]:
            raise Stop(f"vLLM/trainer logprob mean absolute difference {rec['sync_logprob_mad']} after sync")

    def one_step(self) -> None:
        cfg, step = self.cfg, self.step
        ids = step_task_ids(self.ids, cfg["n_tasks"], step, cfg["seed"])
        t0 = time.time()
        episodes = collect([self.tasks[i] for i in ids], cfg["group_size"], self.client, self.adapter[0], step,
                           cfg["seed"], cfg["rollout_concurrency"], cfg["max_turns"])
        rec: dict[str, Any] = {"type": "step", "step": step + 1, "sample_s": round(time.time() - t0, 1)}
        seqs, groups = sequences(episodes)
        rec.update(batch_stats(episodes), **groups)
        rec.update(self.free_for_trainer())
        rec["trainer_build_s"] = round(self.trainer.build(), 2)
        t1 = time.time()
        if seqs:
            stats = self.trainer.step(seqs, self.t_norm)
            rec.update(stats)
            self.skipped_in_row = self.skipped_in_row + 1 if stats["skipped"] else 0
        else:
            rec["no_signal"] = True
        rec["train_s"] = round(time.time() - t1, 1)
        name = f"step-{step + 1}"
        path = self.out / "adapters" / name
        t2 = time.time()
        self.trainer.save_adapter(path)
        ref_greedy = self.trainer.greedy(self.prompt, cfg["sync_check"]["greedy_tokens"])
        check = seqs[0] if seqs else Sequence(self.prompt + ref_greedy, len(self.prompt), [], 0.0, -1, -1)
        ref_lp = self.trainer.logps(check.input_ids, check.start)
        self.trainer.save_state(self.out / "trainer.pt")
        rec["save_s"] = round(time.time() - t2, 1)
        rec.update(self.give_back_to_vllm())
        previous = self.adapter[0]
        sync = self.sync(name, path, previous, ref_greedy, check, ref_lp)
        rec.update(sync)
        self.drop_adapter(self.adapter[1])
        self.adapter = (name, path)
        self.step = step + 1
        (self.out / "state.json").write_text(json.dumps({"step": self.step, "adapter_name": name,
                                                         "adapter_path": str(path)}) + "\n")
        rec["step_s"] = round(time.time() - t0, 1)
        self.record(rec)
        self.check_step(rec)
        self.check_sync(sync)

    def drop_adapter(self, path: Path) -> None:
        """Adapters are kept at eval steps and the last step only."""
        if path.parent == self.out / "adapters" and not (self.out / "eval" / f"{path.name}.jsonl").exists():
            shutil.rmtree(path, ignore_errors=True)

    def check_step(self, rec: dict[str, Any]) -> None:
        stop = self.cfg["stop"]
        if rec.get("logprob_diff_mean") is not None and rec["logprob_diff_mean"] > stop["max_logprob_diff_mean"]:
            raise Stop(f"trainer/vLLM logprob difference {rec['logprob_diff_mean']:.4f} > {stop['max_logprob_diff_mean']}")
        if self.skipped_in_row > stop["max_skipped_steps"]:
            raise Stop(f"{self.skipped_in_row} optimizer steps in a row skipped by the GradScaler")
        self.format_rates.append(rec.get("format_error_rate_per_tool_call"))
        reason = format_rate_stop(self.format_rates, stop.get("format_error_window", 10),
                                   stop.get("format_error_rate_per_call", 0.10))
        if reason:
            raise Stop(reason)
        if rec.get("completion_entropy_mean") is not None:
            self.entropies.append(rec["completion_entropy_mean"])
            reason = entropy_collapse_stop(self.entropies, stop.get("entropy_baseline_steps", 5),
                                           stop.get("entropy_collapse_ratio", 0.30))
            if reason:
                raise Stop(reason)

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        m = cfg.get("measure")
        result: dict[str, Any] = {}
        try:
            self.start()
            if m and self.step == 0:
                result["before"] = measure([self.tasks[i] for i in self.ids], m["k"], self.client, self.adapter[0],
                                           cfg, "before")
                self.record({"type": "measure", **result["before"]})
            while self.step < cfg["steps"]:
                self.one_step()
                ev = cfg["eval"]
                if ev.get("every") and self.step % ev["every"] == 0:
                    rec = evaluate(cfg, self.adapter[0], self.out, self.step)
                    self.record({"type": "eval", **rec})
            if m:
                result["after"] = measure([self.tasks[i] for i in self.ids], m["k"], self.client, self.adapter[0],
                                          cfg, "after")
                self.record({"type": "measure", **result["after"]})
                result["reward_gain"] = round(result["after"]["reward_mean"] - result["before"]["reward_mean"], 4)
            result["status"] = "done"
        except Stop as exc:
            result.update(status="stop", reason=str(exc))
            self.record({"type": "stop", "step": self.step, "reason": str(exc)})
        finally:
            self.server.stop()
        self.record({"type": "end", **{k: v for k, v in result.items() if k not in ("before", "after")}})
        return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, help="override the number of steps (bench)")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    if a.steps is not None:
        cfg["steps"] = a.steps
    result = Run(cfg, resume=a.resume).run()
    return 4 if result["status"] == "stop" else 0


if __name__ == "__main__":
    sys.exit(main())
