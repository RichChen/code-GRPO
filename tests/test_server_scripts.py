"""Server scripts: syntax, and the preflight checks at the top of scripts/server/setup.sh.

Only the failing branches are exercised. Each run puts fake ``id``, ``python3.12`` and ``bwrap`` first on PATH, plus
fakes for every command that comes after the checks; those leave a marker file, so a test fails if setup.sh got past
a check that should have stopped it. Nothing is installed and no system setting is touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "server" / "setup.sh"
APPARMOR = ROOT / "scripts" / "server" / "bwrap-userns.apparmor"
LATER_COMMANDS = ("nvidia-smi", "uv", "curl", "apt-get", "apparmor_parser")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def shell_scripts() -> list[Path]:
    return sorted((ROOT / "scripts").rglob("*.sh"))


@pytest.mark.parametrize("script", shell_scripts(), ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_scripts_parse(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


def run_setup(tmp_path: Path, *, uid: int, include_dir: Path | None, bwrap_ok: bool | None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "got-past-checks"
    fake(bin_dir, "id", f'if [ "$1" = "-u" ]; then echo {uid}; else echo tester; fi')
    if include_dir is not None:
        fake(bin_dir, "python3.12", f"echo {include_dir}")
    if bwrap_ok is not None:
        fake(bin_dir, "bwrap", "exit 0" if bwrap_ok else "exit 1")
    for name in LATER_COMMANDS:
        fake(bin_dir, name, f"touch {marker}; exit 1")
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"}
    proc = subprocess.run(["bash", str(SETUP)], env=env, capture_output=True, text=True, timeout=60)
    return proc, marker.exists()


def test_setup_refuses_root(tmp_path: Path) -> None:
    proc, past = run_setup(tmp_path, uid=0, include_dir=None, bwrap_ok=True)
    assert proc.returncode != 0
    assert "running as root" in proc.stderr and "adduser" in proc.stderr
    assert not past


def test_setup_stops_without_python_headers(tmp_path: Path) -> None:
    empty = tmp_path / "include"
    empty.mkdir()
    proc, past = run_setup(tmp_path, uid=1000, include_dir=empty, bwrap_ok=True)
    assert proc.returncode != 0
    assert "apt-get install -y python3.12-dev" in proc.stderr
    assert not past


def test_setup_stops_when_bwrap_cannot_create_a_namespace(tmp_path: Path) -> None:
    headers = tmp_path / "include"
    headers.mkdir()
    (headers / "Python.h").write_text("", encoding="utf-8")
    proc, past = run_setup(tmp_path, uid=1000, include_dir=headers, bwrap_ok=False)
    assert proc.returncode != 0
    assert "python3.12 headers ok" in proc.stdout
    assert "scripts/server/bwrap-userns.apparmor /etc/apparmor.d/bwrap-userns" in proc.stderr
    assert "apparmor_parser -r" in proc.stderr
    assert not past


def test_apparmor_rule_allows_userns_for_bwrap_only() -> None:
    text = APPARMOR.read_text(encoding="utf-8")
    rules = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert "profile bwrap /usr/bin/bwrap flags=(unconfined) {" in rules
    assert "userns," in rules
    assert rules.count("}") == 1
