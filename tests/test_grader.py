import json
import os
from pathlib import Path

import pytest

from arl.env.grader import Opaque, Raised, changed_files, decode, evaluate_check, grade, harness_argv
from arl.env.sandbox import run_in_sandbox
from arl.env.task import HiddenCase, Task, load_tasks
from arl.env.tools import Toolbox

from conftest import FIXED_NUMS, VISIBLE


def fix(ws: Path) -> None:
    (ws / "pkg" / "nums.py").write_text(FIXED_NUMS)


def snapshot(ws: Path) -> dict[str, bytes]:
    return {p.relative_to(ws).as_posix(): p.read_bytes() for p in sorted(ws.rglob("*")) if p.is_file()}


# -- basic behaviour ---------------------------------------------------------------------


def test_correct_fix_gets_reward_1(task, workspace):
    fix(workspace)
    g = grade(task, workspace)
    assert g.reward == 1
    assert (g.hidden_pass_count, g.hidden_total) == (3, 3)
    assert g.visible_pass is True and not g.timed_out
    assert g.files_changed == ["pkg/nums.py"]
    assert g.setup_error is None


def test_unchanged_workspace_gets_0(task, workspace):
    g = grade(task, workspace)
    assert g.reward == 0
    assert g.hidden_pass_count == 1  # add(10, 0) happens to be right
    assert g.visible_pass is False
    assert g.files_changed == []


def test_reference_solution_gets_1(task, tmp_path):
    ws = task.materialize(tmp_path / "solved", solved=True)
    assert grade(task, ws).reward == 1


def test_grading_does_not_touch_workspace(task, workspace):
    fix(workspace)
    before = snapshot(workspace)
    grade(task, workspace)
    assert snapshot(workspace) == before


def test_hidden_cases_are_not_in_the_workspace(task, workspace):
    text = "\n".join(p.read_text() for p in workspace.rglob("*") if p.is_file())
    for case in task.hidden_cases:
        assert case.call not in text
        assert case.check not in text


def test_syntax_error_gives_setup_error(task, workspace):
    (workspace / "pkg" / "nums.py").write_text("def add(a, b:\n")
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 0
    assert g.setup_error == "SyntaxError"


def test_hidden_by_kind_is_recorded_without_changing_the_reward(task, workspace):
    # unchanged code passes only add(10, 0), the third case
    task.meta["hidden_case_kinds"] = ["generated", "generated", "test"]
    g = grade(task, workspace, check_visible=False)
    assert g.reward == 0 and g.case_passed == [False, False, True]
    assert g.hidden_by_kind == {"generated": [0, 2], "test": [1, 1]}
    assert g.pass_non_generated is True

    task.meta["hidden_case_kinds"] = ["test", "challenge", "generated"]
    g = grade(task, workspace, check_visible=False)
    assert g.hidden_by_kind == {"test": [0, 1], "challenge": [0, 1], "generated": [1, 1]}
    assert g.reward == 0 and g.pass_non_generated is False

    fix(workspace)
    g = grade(task, workspace, check_visible=False)
    assert g.reward == 1 and g.pass_non_generated is True


def test_hidden_by_kind_defaults_to_empty(task, workspace):
    from arl.env.grader import GradeResult

    g = grade(task, workspace, check_visible=False)
    assert g.hidden_by_kind is None and g.pass_non_generated is None
    old = {k: v for k, v in g.to_dict().items() if k not in ("hidden_by_kind", "pass_non_generated")}
    assert GradeResult(**old) == g  # records written before these fields still load


def test_kind_breakdown_counts_a_timeout_as_failing():
    from arl.env.grader import kind_breakdown

    assert kind_breakdown(["test", "generated"], [True, False], timed_out=False) == ({"test": [1, 1], "generated": [0, 1]}, True)
    assert kind_breakdown(["test", "generated"], [True, False], timed_out=True)[1] is False
    assert kind_breakdown(["test"], [True, False], timed_out=False) == (None, None)  # length mismatch


def test_hidden_timeout(task, workspace):
    (workspace / "pkg" / "nums.py").write_text("def add(a, b):\n    while True:\n        pass\n")
    from arl.env.sandbox import SandboxConfig

    g = grade(task, workspace, SandboxConfig(wall_timeout_s=2), check_visible=False)
    assert g.reward == 0 and g.timed_out and g.visible_pass is None


# -- anti-cheat ----------------------------------------------------------------------------


def test_cheat_edit_visible_test_does_not_change_reward(task, workspace):
    (workspace / "tests" / "test_visible.py").write_text(VISIBLE.replace("assert add(1, 2) == 3", "pass"))
    assert Toolbox(workspace).run_tests().content.startswith("exit code: 0")
    g = grade(task, workspace)
    assert g.reward == 0 and g.visible_pass is False
    assert g.files_changed == ["tests/test_visible.py"]

    fix(workspace)
    assert grade(task, workspace).reward == 1


HIJACK_CONFTEST = '''
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.outcome = "passed"
    report.longrepr = None
'''


@pytest.mark.parametrize("where", ["conftest.py", "tests/conftest.py"])
def test_cheat_conftest_hijack_is_removed(task, workspace, where):
    (workspace / where).write_text(HIJACK_CONFTEST)
    assert Toolbox(workspace).run_tests().content.startswith("exit code: 0")  # the hack works on visible tests
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 1


def test_cheat_config_files_are_replaced_by_originals(task, workspace):
    for name in ["pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml"]:
        (workspace / name).write_text("[pytest]\naddopts = -p no:python\n[tool:pytest]\naddopts = -p no:python\n")
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 1


@pytest.mark.parametrize("exit_call", ["sys.exit(0)", "os._exit(0)", "raise SystemExit(0)"])
def test_cheat_exit_at_import(task, workspace, exit_call):
    (workspace / "pkg" / "__init__.py").write_text(f"import os, sys\n{exit_call}\n")
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 0


def test_cheat_exit_inside_function(task, workspace):
    (workspace / "pkg" / "nums.py").write_text("import os\n\ndef add(a, b):\n    os._exit(0)\n")
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 0


MONKEYPATCH_PYTEST = '''
import sys
try:
    import _pytest.python
    _pytest.python.Function.runtest = lambda self: None
except Exception:
    pass
import builtins
_real_import = builtins.__import__
'''


def test_cheat_monkeypatch_pytest(task, workspace):
    (workspace / "pkg" / "__init__.py").write_text(MONKEYPATCH_PYTEST)
    assert Toolbox(workspace).run_tests().content.startswith("exit code: 0")  # the hack works on visible tests
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 1


def test_cheat_hardcoded_return_value(task, workspace):
    (workspace / "pkg" / "nums.py").write_text("def add(a, b):\n    return 3\n")
    assert Toolbox(workspace).run_tests().content.startswith("exit code: 0")
    g = grade(task, workspace)
    assert g.reward == 0 and g.visible_pass is True
    assert g.hidden_pass_count == 0


FORGE_RECORDS = '''
import os
forged = b"".join(b'{"i": %d, "ok": true, "v": 7}\\n' % i for i in range(3))
for fd in range(1, 20):
    try:
        os.write(fd, forged)
    except OSError:
        pass
print(forged.decode())
'''


def test_cheat_forged_harness_output(task, workspace):
    (workspace / "pkg" / "__init__.py").write_text(FORGE_RECORDS)
    g = grade(task, workspace)
    assert g.reward == 0
    assert g.hidden_pass_count == 0  # forged records duplicate the real ones, so every case is rejected


def test_cheat_always_equal_object(task, workspace):
    (workspace / "pkg" / "nums.py").write_text(
        "class Anything(int):\n    def __eq__(self, other):\n        return True\n    __hash__ = int.__hash__\n\n"
        "class Weird:\n    def __eq__(self, other):\n        return True\n    def __bool__(self):\n        return True\n\n"
        "def add(a, b):\n    return Anything(0) if a > 0 else Weird()\n"
    )
    assert Toolbox(workspace).run_tests().content.startswith("exit code: 0")
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 0


def test_cheat_replace_harness_and_sitecustomize(task, workspace):
    (workspace / "_arl_hidden.py").write_text("print('{\"i\": 0, \"ok\": true, \"v\": 7}')\n")
    (workspace / "_arl_hidden.json").write_text("{}")
    (workspace / "sitecustomize.py").write_text("import os\nos._exit(0)\n")
    g = grade(task, workspace)
    assert g.reward == 0 and g.hidden_pass_count == 1


def test_cheat_symlink_to_solution_is_not_followed(task, workspace, tmp_path):
    solved = task.materialize(tmp_path / "elsewhere", solved=True)
    (workspace / "pkg" / "nums.py").unlink()
    os.symlink(solved / "pkg" / "nums.py", workspace / "pkg" / "nums.py")
    from arl.env.sandbox import bwrap_available

    g = grade(task, workspace)
    if bwrap_available():
        assert g.reward == 0
    assert "pkg/nums.py" in g.files_changed


# -- serialisation and checks --------------------------------------------------------------


def test_decode_roundtrip_types():
    assert decode({"T": [1, {"L": [2.5, "x"]}]}) == (1, [2.5, "x"])
    assert decode({"S": [1, 2]}) == {1, 2}
    assert decode({"F": [1]}) == frozenset({1})
    assert decode({"D": [[{"T": [1, 2]}, None]]}) == {(1, 2): None}
    assert decode({"B": "6162"}) == b"ab"
    assert decode({"C": [1, 2]}) == 1 + 2j
    assert isinstance(decode({"O": "generator"}), Opaque)
    with pytest.raises(ValueError):
        decode({"X": 1})
    with pytest.raises(ValueError):
        decode({"L": [], "T": []})


def test_undecidable_values_fail_every_check():
    for value in (Opaque("generator"), Raised("ValueError")):
        for check in ["result == 1", "result != 1", "result", "not result", "len(result) == 0", "sorted(result) == []"]:
            assert evaluate_check(check, value, None) is False


def test_checks_support_tolerance_and_errors():
    assert evaluate_check("abs(result - 0.3) < 1e-9", 0.1 + 0.2, None)
    assert evaluate_check("math.isclose(result, 2.0)", 2.0000000001, None)
    assert evaluate_check("error == 'ValueError'", Raised("ValueError"), "ValueError")
    assert not evaluate_check("1/0", 1, None)


def test_list_vs_tuple_is_distinguished(task, workspace):
    t = Task.from_dict({**task.to_dict(), "hidden_cases": [
        {"call": "list(pkg.nums.double(i) for i in range(3))", "check": "result == [0, 2, 4]"},
        {"call": "tuple(pkg.nums.double(i) for i in range(3))", "check": "result == (0, 2, 4)"},
        {"call": "tuple(pkg.nums.double(i) for i in range(3))", "check": "result != [0, 2, 4]"},
        {"call": "{'a': {1, 2}, 'b': float('nan')}", "check": "result['a'] == {1, 2} and math.isnan(result['b'])"},
        {"call": "pkg.nums.double('ab')", "check": "result == 'abab'"},
        {"call": "int('x')", "check": "error == 'ValueError'"},
        {"call": "(x for x in [])", "check": "result == []"},
    ]})
    g = grade(t, workspace, check_visible=False)
    assert g.case_passed == [True, True, True, True, True, True, False]


def test_hidden_harness_uses_fixed_hash_seed(workspace):
    (workspace / "p.py").write_text("import os\nprint(os.environ.get('PYTHONHASHSEED'), list({'apple', 'banana', 'cherry', 'date'}))\n")
    outputs = {run_in_sandbox(workspace, harness_argv("p.py")).output for _ in range(4)}
    assert len(outputs) == 1 and outputs.pop().startswith("0 ")


# -- task model ----------------------------------------------------------------------------


def test_task_json_roundtrip(task, tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(task.to_json() + "\n" + task.to_json() + "\n")
    loaded = load_tasks(path)
    assert len(loaded) == 2 and loaded[0] == task
    assert isinstance(loaded[0].hidden_cases[0], HiddenCase)


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.py", "a/../../x.py", "a\\b.py"])
def test_task_rejects_bad_paths(task, bad):
    with pytest.raises(ValueError):
        Task.from_dict({**task.to_dict(), "files": {**task.files, bad: ""}})


def test_changed_files_detects_add_modify_delete(task, workspace):
    (workspace / "pkg" / "text.py").unlink()
    (workspace / "pkg" / "new.py").write_text("x = 1\n")
    fix(workspace)
    (workspace / "pkg" / "__pycache__").mkdir()
    (workspace / "pkg" / "__pycache__" / "nums.cpython-312.pyc").write_bytes(b"\0")
    assert changed_files(task, workspace) == ["pkg/new.py", "pkg/nums.py", "pkg/text.py"]
