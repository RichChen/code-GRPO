import os
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import arl
from arl.env.sandbox import SandboxConfig, bwrap_available, resolve_isolation, run_in_sandbox, sandbox_runner
from arl.env.tools import Toolbox

needs_bwrap = pytest.mark.skipif(not bwrap_available(), reason="bwrap cannot create a sandbox here")
BOTH = pytest.mark.parametrize("isolation", [pytest.param("bwrap", marks=needs_bwrap), "rlimit_only"])
PY = sys.executable


def run(ws: Path, code: str, isolation: str = "bwrap", **cfg) -> object:
    (ws / "prog.py").write_text(textwrap.dedent(code))
    return run_in_sandbox(ws, [PY, "prog.py"], SandboxConfig(isolation=isolation, **cfg))


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    d = tmp_path / "ws"
    d.mkdir()
    return d


def _pids_with_marker(marker: str) -> list[int]:
    pids = []
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                cmd = Path(f"/proc/{entry.name}/cmdline").read_bytes()
            except OSError:
                continue
            if marker.encode() in cmd:
                pids.append(int(entry.name))
    return pids


def test_auto_isolation_uses_bwrap_when_available():
    assert resolve_isolation("auto") == ("bwrap" if bwrap_available() else "rlimit_only")
    assert resolve_isolation("rlimit_only") == "rlimit_only"
    with pytest.raises(ValueError):
        resolve_isolation("docker")


@BOTH
def test_normal_pytest_passes(ws, isolation):
    (ws / "pkg").mkdir()
    (ws / "pkg" / "__init__.py").write_text("def inc(x):\n    return x + 1\n")
    (ws / "tests").mkdir()
    (ws / "tests" / "test_a.py").write_text("from pkg import inc\n\ndef test_inc():\n    assert inc(1) == 2\n")
    r = run_in_sandbox(ws, [PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], SandboxConfig(isolation=isolation))
    assert r.exit_code == 0, r.output
    assert "1 passed" in r.output
    assert r.isolation == isolation and not r.timed_out


@BOTH
def test_infinite_loop_times_out(ws, isolation):
    start = time.monotonic()
    r = run(ws, "while True:\n    pass\n", isolation, wall_timeout_s=2)
    assert r.timed_out and r.exit_code is None
    assert time.monotonic() - start < 6


@BOTH
def test_timeout_kills_whole_process_tree(ws, isolation):
    marker = f"arl-marker-{os.getpid()}-{isolation}"
    code = f"""
    import os, subprocess, sys, time
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "{marker}"])
    if os.fork() == 0:
        time.sleep(60)
    time.sleep(60)
    """
    r = run(ws, code, isolation, wall_timeout_s=2)
    assert r.timed_out
    deadline = time.monotonic() + 3
    while _pids_with_marker(marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _pids_with_marker(marker) == []


@BOTH
def test_cpu_limit(ws, isolation):
    r = run(ws, "while True:\n    pass\n", isolation, cpu_seconds=1, wall_timeout_s=10)
    assert not r.timed_out
    assert r.exit_code != 0


@BOTH
def test_memory_limit(ws, isolation):
    r = run(ws, "x = bytearray(2 * 1024 ** 3)\nprint('allocated')\n", isolation, address_space_mb=512)
    assert r.exit_code != 0
    assert "MemoryError" in r.output and "allocated" not in r.output


@BOTH
def test_file_size_limit(ws, isolation):
    r = run(ws, "open('big.bin', 'wb').write(b'x' * (3 * 1024 * 1024))\nprint('written')\n", isolation, file_size_mb=1)
    assert r.exit_code != 0 and "written" not in r.output
    assert (ws / "big.bin").stat().st_size <= 1024 * 1024


@BOTH
def test_fork_bomb_is_blocked_by_process_limit(ws, isolation):
    code = """
    import os, time
    n = 0
    try:
        while True:
            if os.fork() == 0:
                time.sleep(30)
                os._exit(0)
            n += 1
    except OSError as e:
        print("blocked", n, e.errno)
    """
    r = run(ws, code, isolation, max_processes=32, wall_timeout_s=10)
    assert not r.timed_out
    assert "blocked" in r.output, r.output
    forked = int(r.output.split()[1])
    assert forked < 32


@BOTH
def test_output_is_capped(ws, isolation):
    r = run(ws, "import sys\nsys.stdout.write('x' * 500000)\n", isolation, max_output_bytes=1000)
    assert r.output_truncated and len(r.output) == 1000


@BOTH
def test_environment_is_whitelisted(ws, isolation, monkeypatch):
    monkeypatch.setenv("ARL_SECRET_TOKEN", "hunter2")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    r = run(ws, "import os\nprint(sorted(os.environ))\nprint(os.environ.get('ARL_SECRET_TOKEN'))\n", isolation)
    assert "hunter2" not in r.output and "ARL_SECRET_TOKEN" not in r.output
    assert "PYTHONPATH" not in r.output


@needs_bwrap
def test_network_is_unreachable(ws):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = []
    threading.Thread(target=lambda: accepted.append(server.accept()), daemon=True).start()
    code = f"""
    import socket
    for addr in [("127.0.0.1", {port}), ("1.1.1.1", 53)]:
        s = socket.socket()
        s.settimeout(3)
        print(addr[0], s.connect_ex(addr))
    """
    r = run(ws, code)
    server.close()
    assert r.exit_code == 0, r.output
    lines = dict(line.split() for line in r.output.splitlines())
    assert lines["127.0.0.1"] != "0" and lines["1.1.1.1"] != "0"
    assert accepted == []


@needs_bwrap
def test_writes_outside_workspace_fail(ws, tmp_path):
    target = tmp_path / "escape.txt"
    home_target = Path.home() / f"arl-escape-{os.getpid()}.txt"
    code = f"""
    for p in ["/workspace/inside.txt", "{target}", "{home_target}", "/usr/escape.txt", "/escape.txt", "../escape.txt"]:
        try:
            open(p, "w").write("x")
            print("WROTE", p)
        except OSError as e:
            print("DENIED", p)
    """
    r = run(ws, code)
    assert r.exit_code == 0, r.output
    assert r.output.count("WROTE") == 1 and "WROTE /workspace/inside.txt" in r.output
    assert (ws / "inside.txt").exists()
    assert not target.exists() and not home_target.exists() and not (tmp_path / "escape.txt").exists()


@needs_bwrap
def test_project_files_are_not_visible(ws):
    project_root = Path(arl.__file__).resolve().parents[2]
    r = run(ws, f"import os\nprint(os.path.exists({str(project_root / 'src')!r}), os.path.exists({str(project_root / 'docs')!r}))\n")
    assert r.output.strip() == "False False"


@needs_bwrap
def test_bwrap_paths_are_normalised(ws):
    r = run(ws, "import os\nprint(os.getcwd())\nraise SystemExit(3)\n")
    assert r.exit_code == 3
    assert r.output.splitlines()[0] == "/workspace"
    assert str(ws) not in r.output


def test_toolbox_runs_tests_in_sandbox_by_default(ws):
    (ws / "tests").mkdir()
    (ws / "tests" / "test_env.py").write_text(
        "import os, socket\n\ndef test_isolated():\n    assert os.getcwd() == '/workspace'\n"
    )
    r = Toolbox(ws).run_tests()
    expected = "exit code: 0" if bwrap_available() else "exit code: 1"
    assert r.content.startswith(expected), r.content


def test_sandbox_runner_reports_timeout(ws):
    (ws / "tests").mkdir()
    (ws / "tests" / "test_slow.py").write_text("def test_slow():\n    while True:\n        pass\n")
    tb = Toolbox(ws, test_runner=sandbox_runner(SandboxConfig(wall_timeout_s=2)))
    r = tb.run_tests()
    assert r.ok and r.content.startswith("Tests timed out")
