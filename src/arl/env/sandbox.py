"""Run untrusted code (agent edits, tests, grading) with isolation and resource limits.

Preferred isolation is bubblewrap: the workspace is the only writable host path
(mounted at ``/workspace``), system and Python directories are read-only, there is
no network, a private PID namespace, and a whitelisted environment. Where bwrap is
unusable the sandbox degrades to rlimits plus a wall-clock timeout and reports
``isolation="rlimit_only"``.

rlimits are applied by a small launcher that runs *inside* the sandbox, right
before exec'ing the target. ``RLIMIT_NPROC`` counts every task of the user, so
setting it outside bwrap would count the host's processes and threads; inside a
fresh user namespace it only counts the sandbox's own tasks.
"""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from arl.env.tools import RunOutput, TestRunner

BWRAP = "/usr/bin/bwrap"
SANDBOX_WORKDIR = "/workspace"
ENV_WHITELIST = ("LANG", "LC_ALL", "LC_CTYPE", "TZ")
BASE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONHASHSEED": "0",
}

# Sets rlimits on itself and execs the target. Runs with `python -I -S` so nothing from the
# workspace can be imported before the limits are in place.
_LAUNCHER = r"""
import os, resource, sys
limits = {k: int(v) for k, v in (a.split("=") for a in sys.argv[1].split(","))}
for name, value in limits.items():
    resource.setrlimit(getattr(resource, name), (value, value))
os.execv(sys.argv[2], sys.argv[2:])
"""


@dataclass(frozen=True)
class SandboxConfig:
    wall_timeout_s: float = 20.0
    cpu_seconds: int = 20
    address_space_mb: int = 1024
    file_size_mb: int = 16
    max_processes: int = 64
    tmpfs_mb: int = 64
    max_output_bytes: int = 1_000_000
    isolation: str = "auto"  # "auto" | "bwrap" | "rlimit_only"


@dataclass
class SandboxResult:
    exit_code: int | None
    output: str
    timed_out: bool
    isolation: str
    duration_s: float
    output_truncated: bool = False

    def to_run_output(self) -> RunOutput:
        return RunOutput(exit_code=self.exit_code, output=self.output, timed_out=self.timed_out)


class SandboxUnavailable(RuntimeError):
    pass


def _python_mounts() -> list[str]:
    """bwrap args that make the current interpreter (venv + base install) usable read-only."""
    args: list[str] = []
    seen: set[str] = set()

    def add_prefix(prefix: str) -> None:
        prefix = os.path.normpath(prefix)
        real = os.path.realpath(prefix)
        if real not in seen:
            seen.add(real)
            args.extend(["--ro-bind", real, real])
        if prefix != real and prefix not in seen:
            seen.add(prefix)
            args.extend(["--symlink", real, prefix])

    add_prefix(sys.prefix)
    add_prefix(sys.base_prefix)
    exe = sys.executable
    for _ in range(16):
        if not os.path.islink(exe):
            break
        target = os.path.normpath(os.path.join(os.path.dirname(exe), os.readlink(exe)))
        add_prefix(os.path.dirname(os.path.dirname(target)))
        exe = target
    return args


def _bwrap_prefix(workspace: Path, cfg: SandboxConfig) -> list[str]:
    args = [
        BWRAP,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--ro-bind", "/usr", "/usr",
    ]
    for name in ("bin", "lib", "lib64", "lib32", "sbin"):
        host = Path("/") / name
        if host.is_symlink():
            args += ["--symlink", os.readlink(host), str(host)]
        elif host.is_dir():
            args += ["--ro-bind", str(host), str(host)]
    args += _python_mounts()
    args += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--size", str(cfg.tmpfs_mb * 1024 * 1024), "--tmpfs", "/tmp",
        "--bind", str(workspace), SANDBOX_WORKDIR,
        "--remount-ro", "/",
        "--chdir", SANDBOX_WORKDIR,
        "--clearenv",
    ]
    for key, value in _sandbox_env("/tmp").items():
        args += ["--setenv", key, value]
    return args


def _sandbox_env(tmpdir: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in ENV_WHITELIST if k in os.environ}
    env.update(BASE_ENV)
    env.update({"HOME": tmpdir, "TMPDIR": tmpdir})
    return env


def _count_user_tasks() -> int:
    """Number of tasks (threads included) owned by the current real uid."""
    uid = os.getuid()
    total = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            total += len(os.listdir(f"/proc/{entry.name}/task"))
        except OSError:
            continue
    return total


def _limits(cfg: SandboxConfig, nproc: int) -> str:
    mb = 1024 * 1024
    return ",".join(
        [
            f"RLIMIT_CPU={cfg.cpu_seconds}",
            f"RLIMIT_AS={cfg.address_space_mb * mb}",
            f"RLIMIT_FSIZE={cfg.file_size_mb * mb}",
            f"RLIMIT_NPROC={nproc}",
            "RLIMIT_CORE=0",
        ]
    )


@functools.lru_cache(maxsize=1)
def bwrap_available() -> bool:
    if not os.path.exists(BWRAP):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        argv = _bwrap_prefix(Path(tmp), SandboxConfig()) + [sys.executable, "-I", "-S", "-c", "pass"]
        try:
            proc = subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
    return proc.returncode == 0


def resolve_isolation(requested: str = "auto") -> str:
    if requested == "rlimit_only":
        return "rlimit_only"
    if requested not in ("auto", "bwrap"):
        raise ValueError(f"unknown isolation mode: {requested}")
    if bwrap_available():
        return "bwrap"
    if requested == "bwrap":
        raise SandboxUnavailable("bwrap was requested but cannot create a sandbox on this machine")
    return "rlimit_only"


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_in_sandbox(workspace: str | Path, argv: list[str], config: SandboxConfig | None = None) -> SandboxResult:
    """Run ``argv`` with the workspace as working directory. ``argv[0]`` must be an absolute path."""
    cfg = config or SandboxConfig()
    workspace = Path(os.path.realpath(workspace))
    if not workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")
    if not os.path.isabs(argv[0]):
        raise ValueError("argv[0] must be an absolute path")
    isolation = resolve_isolation(cfg.isolation)

    with tempfile.TemporaryDirectory(prefix="arl-sbx-") as scratch:
        out_path = Path(scratch) / "output"
        if isolation == "bwrap":
            nproc = cfg.max_processes
            prefix = _bwrap_prefix(workspace, cfg)
            env = None
            cwd = None
        else:
            nproc = _count_user_tasks() + cfg.max_processes
            prefix = []
            tmpdir = Path(scratch) / "tmp"
            tmpdir.mkdir()
            env = _sandbox_env(str(tmpdir))
            cwd = workspace
        full = prefix + [sys.executable, "-I", "-S", "-c", _LAUNCHER, _limits(cfg, nproc), *argv]

        start = time.monotonic()
        timed_out = False
        with open(out_path, "wb") as out:
            proc = subprocess.Popen(
                full,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=cfg.wall_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                _kill_group(proc)
                proc.wait()
        duration = time.monotonic() - start

        with open(out_path, "rb") as f:
            raw = f.read(cfg.max_output_bytes + 1)
    truncated = len(raw) > cfg.max_output_bytes
    output = raw[: cfg.max_output_bytes].decode("utf-8", "replace")
    if isolation == "bwrap":
        output = output.replace(str(workspace), SANDBOX_WORKDIR)
    return SandboxResult(
        exit_code=None if timed_out else proc.returncode,
        output=output,
        timed_out=timed_out,
        isolation=isolation,
        duration_s=duration,
        output_truncated=truncated,
    )


def sandbox_runner(config: SandboxConfig | None = None) -> TestRunner:
    """A ``TestRunner`` for ``Toolbox`` that executes inside the sandbox."""

    def run(root: Path, argv: list[str]) -> RunOutput:
        return run_in_sandbox(root, argv, config).to_run_output()

    return run
