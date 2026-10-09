from pathlib import Path

import pytest

from arl.env.task import HiddenCase, Task

BUGGY_NUMS = '''def add(a, b):
    """Return a + b."""
    return a - b


def double(x):
    return x * 2
'''

FIXED_NUMS = BUGGY_NUMS.replace("return a - b", "return a + b")

TEXT = '''def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")
'''

VISIBLE = '''from pkg.nums import add


def test_add():
    assert add(1, 2) == 3
'''


def make_task() -> Task:
    return Task(
        task_id="toy-add",
        issue="`pkg.nums.add(1, 2)` returns -1, expected 3.",
        files={
            "pkg/__init__.py": "",
            "pkg/nums.py": BUGGY_NUMS,
            "pkg/text.py": TEXT,
            "tests/test_visible.py": VISIBLE,
            "pytest.ini": "[pytest]\naddopts = -p no:cacheprovider\n",
        },
        hidden_setup="import pkg.nums",
        hidden_cases=[
            HiddenCase(call="pkg.nums.add(2, 5)", check="result == 7"),
            HiddenCase(call="pkg.nums.add(-4, 4)", check="result == 0"),
            HiddenCase(call="pkg.nums.add(10, 0)", check="result == 10"),
        ],
        solution={"pkg/nums.py": FIXED_NUMS},
        meta={"bug_file": "pkg/nums.py", "bug_function": "add", "bug_type": "arith_swap"},
    )


@pytest.fixture
def task() -> Task:
    return make_task()


@pytest.fixture
def workspace(task: Task, tmp_path: Path) -> Path:
    return task.materialize(tmp_path / "ws")
