import json
import os
from pathlib import Path

import pytest

from arl.env.tools import (
    TOOL_NAMES,
    TOOL_SCHEMAS,
    RunOutput,
    ToolArgumentError,
    Toolbox,
    parse_arguments,
    truncate_middle,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "text.py").write_text(
        "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n\n\ndef shout(s):\n    return s.upper()\n"
    )
    (root / "pkg" / "nums.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "tests" / "test_visible.py").write_text(
        "from pkg.nums import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    return root


@pytest.fixture
def tb(repo: Path) -> Toolbox:
    return Toolbox(repo)


def test_schemas_are_openai_function_format():
    assert TOOL_NAMES == ("read_file", "search_code", "list_dir", "edit_file", "run_tests")
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])
    json.dumps(TOOL_SCHEMAS)
    assert "write_file" not in TOOL_NAMES


# -- read_file -----------------------------------------------------------------------


def test_read_file_numbers_lines(tb):
    r = tb.read_file("pkg/nums.py")
    assert r.ok
    assert r.content.splitlines() == ["1| def add(a, b):", "2|     return a - b"]


def test_read_file_default_limit_and_continuation_hint(tb, repo):
    (repo / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(1, 451)) + "\n")
    r = tb.read_file("big.py")
    assert r.ok
    lines = r.content.splitlines()
    assert len(lines) == 201
    assert lines[0].endswith("x1 = 1") and lines[199].endswith("x200 = 200")
    assert lines[-1] == "[250 more lines; call read_file with offset=201 to continue]"
    r2 = tb.read_file("big.py", offset=401)
    assert r2.ok and r2.content.splitlines()[-1].endswith("x450 = 450")
    assert "more lines" not in r2.content


def test_read_file_offset_and_limit(tb):
    r = tb.read_file("pkg/text.py", offset=5, limit=1)
    assert r.ok
    assert r.content.splitlines()[0] == "5| def shout(s):"
    assert "[1 more lines; call read_file with offset=6 to continue]" in r.content


def test_read_file_limit_is_capped(tb, repo):
    (repo / "big.py").write_text("\n".join(str(i) for i in range(500)))
    r = tb.read_file("big.py", limit=1000)
    assert len(r.content.splitlines()) == 201


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"path": "missing.py"}, "File not found"),
        ({"path": "pkg"}, "is a directory"),
        ({"path": "pkg/nums.py", "offset": 0}, "offset must be at least 1"),
        ({"path": "pkg/nums.py", "offset": 99}, "past the end"),
        ({"path": "pkg/nums.py", "limit": 0}, "limit must be at least 1"),
    ],
)
def test_read_file_errors(tb, kwargs, fragment):
    r = tb.read_file(**kwargs)
    assert not r.ok and fragment in r.content


def test_read_file_binary(tb, repo):
    (repo / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    r = tb.read_file("blob.bin")
    assert not r.ok and "UTF-8" in r.content


def test_read_file_empty(tb, repo):
    (repo / "empty.py").write_text("")
    assert tb.read_file("empty.py").content == "(empty file)"


# -- search_code ---------------------------------------------------------------------


def test_search_code_finds_matches(tb):
    r = tb.search_code(r"def \w+")
    assert r.ok
    assert r.content.splitlines() == [
        "pkg/nums.py:1: def add(a, b):",
        "pkg/text.py:1: def count_vowels(s):",
        "pkg/text.py:5: def shout(s):",
        "tests/test_visible.py:4: def test_add():",
    ]


def test_search_code_path_and_glob(tb):
    r = tb.search_code("add", path="tests")
    assert r.ok and all(line.startswith("tests/") for line in r.content.splitlines())
    r = tb.search_code("def", glob="nums.py")
    assert r.content == "pkg/nums.py:1: def add(a, b):"
    r = tb.search_code("def", path="pkg/text.py")
    assert len(r.content.splitlines()) == 2


def test_search_code_caps_results(tb, repo):
    (repo / "many.py").write_text("\n".join("hit = 1" for _ in range(80)))
    r = tb.search_code("hit")
    lines = r.content.splitlines()
    assert len(lines) == 51
    assert lines[-1] == "[30 more matches not shown; narrow the pattern or path]"


def test_search_code_no_match_and_errors(tb):
    assert tb.search_code("zzz_nothing").content == "No matches found."
    r = tb.search_code("(unclosed")
    assert not r.ok and "Invalid regular expression" in r.content
    r = tb.search_code("x", path="nope")
    assert not r.ok and "not found" in r.content
    assert not tb.search_code("").ok


def test_search_code_skips_pycache(tb, repo):
    (repo / "pkg" / "__pycache__").mkdir()
    (repo / "pkg" / "__pycache__" / "junk.py").write_text("def add(): pass\n")
    assert "__pycache__" not in tb.search_code("def add").content


# -- list_dir ------------------------------------------------------------------------


def test_list_dir_root_and_subdir(tb):
    assert tb.list_dir().content.splitlines() == ["pkg/", "tests/"]
    assert tb.list_dir("pkg").content.splitlines() == ["__init__.py", "nums.py", "text.py"]
    assert tb.list_dir(".").ok


def test_list_dir_errors(tb, repo):
    assert "not found" in tb.list_dir("nope").content
    r = tb.list_dir("pkg/nums.py")
    assert not r.ok and "is a file" in r.content
    (repo / "empty").mkdir()
    assert tb.list_dir("empty").content == "(empty directory)"


# -- edit_file -----------------------------------------------------------------------


def test_edit_file_replaces_unique_occurrence(tb, repo):
    r = tb.edit_file("pkg/nums.py", "return a - b", "return a + b")
    assert r.ok and "replaced 1 occurrence" in r.content
    assert (repo / "pkg" / "nums.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_edit_file_reports_occurrence_count(tb, repo):
    (repo / "dup.py").write_text("x = 1\nx = 1\nx = 1\n")
    r = tb.edit_file("dup.py", "x = 1", "x = 2")
    assert not r.ok and "appears 3 times" in r.content
    assert (repo / "dup.py").read_text() == "x = 1\nx = 1\nx = 1\n"


def test_edit_file_errors(tb):
    assert "not found in" in tb.edit_file("pkg/nums.py", "nonexistent", "y").content
    assert "cannot create files" in tb.edit_file("new.py", "a", "b").content
    assert "must not be empty" in tb.edit_file("pkg/nums.py", "", "b").content
    assert "identical" in tb.edit_file("pkg/nums.py", "add", "add").content
    assert "is a directory" in tb.edit_file("pkg", "a", "b").content


# -- run_tests -----------------------------------------------------------------------


def test_run_tests_reports_failure_then_success(tb):
    r = tb.run_tests()
    assert r.ok
    assert r.content.startswith("exit code: 1")
    assert "assert -1 == 3" in r.content
    tb.edit_file("pkg/nums.py", "a - b", "a + b")
    r = tb.run_tests("tests/test_visible.py::test_add")
    assert r.content.startswith("exit code: 0")
    assert "1 passed" in r.content
    assert not list(tb.root.rglob("*.pyc"))


def test_run_tests_truncates_keeping_head_and_tail(repo):
    big = "HEAD" + "x" * 20000 + "TAIL"
    tb = Toolbox(repo, test_runner=lambda root, argv: RunOutput(exit_code=1, output=big))
    r = tb.run_tests()
    assert len(r.content) <= 4000
    assert "HEAD" in r.content and r.content.endswith("TAIL")
    assert "characters omitted" in r.content


def test_run_tests_timeout(repo):
    tb = Toolbox(repo, test_runner=lambda root, argv: RunOutput(exit_code=None, output="partial", timed_out=True))
    r = tb.run_tests()
    assert r.content.startswith("Tests timed out")


def test_run_tests_errors(tb):
    assert "not found" in tb.run_tests("tests/nope.py").content
    assert "not an option" in tb.run_tests("--co").content
    assert "outside" in tb.run_tests("../x").content


def test_truncate_middle():
    assert truncate_middle("abc", 10) == "abc"
    out = truncate_middle("a" * 100 + "b" * 100, 80)
    assert len(out) <= 80 and out.startswith("a") and out.endswith("b")


# -- path escape ---------------------------------------------------------------------


ESCAPES = ["/etc/passwd", "../outside.txt", "pkg/../../outside.txt", "~/.bashrc", "link_out/secret.txt", "file_link.txt"]


@pytest.fixture
def escape_repo(repo: Path) -> Path:
    outside = repo.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET\n")
    (repo.parent / "outside.txt").write_text("SECRET\n")
    os.symlink(outside, repo / "link_out")
    os.symlink(outside / "secret.txt", repo / "file_link.txt")
    return repo


@pytest.mark.parametrize("path", ESCAPES)
def test_path_escape_rejected_by_every_tool(escape_repo, path):
    tb = Toolbox(escape_repo)
    results = [
        tb.read_file(path),
        tb.search_code("SECRET", path=path),
        tb.list_dir(path),
        tb.edit_file(path, "SECRET", "PWNED"),
        tb.run_tests(path),
    ]
    for r in results:
        assert not r.ok, r
        assert "SECRET" not in r.content
    assert (escape_repo.parent / "outside" / "secret.txt").read_text() == "SECRET\n"


def test_search_from_root_skips_symlinks_out(escape_repo):
    tb = Toolbox(escape_repo)
    assert tb.search_code("SECRET").content == "No matches found."


def test_symlink_inside_root_is_allowed(repo):
    os.symlink(repo / "pkg" / "nums.py", repo / "alias.py")
    assert Toolbox(repo).read_file("alias.py").ok


# -- dispatch / argument parsing -----------------------------------------------------


def test_call_dispatches_json_arguments(tb):
    r = tb.call("read_file", json.dumps({"path": "pkg/nums.py", "limit": "1"}))
    assert r.ok and r.content == "1| def add(a, b):\n[1 more lines; call read_file with offset=2 to continue]"


@pytest.mark.parametrize(
    "name, args, kind",
    [
        ("write_file", "{}", "unknown_tool"),
        ("read_file", "{not json", "invalid_json"),
        ("read_file", "[1, 2]", "invalid_arguments"),
        ("read_file", "{}", "invalid_arguments"),
        ("read_file", '{"path": 3}', "invalid_arguments"),
        ("read_file", '{"path": "a", "offset": "x"}', "invalid_arguments"),
        ("read_file", '{"path": "a", "offset": true}', "invalid_arguments"),
        ("list_dir", '{"dir": "."}', "invalid_arguments"),
    ],
)
def test_parse_arguments_errors(name, args, kind):
    with pytest.raises(ToolArgumentError) as info:
        parse_arguments(name, args)
    assert info.value.kind == kind


def test_call_never_raises_on_bad_input(tb):
    for name, args in [("nope", "{}"), ("read_file", "{"), ("edit_file", {"path": "x"})]:
        r = tb.call(name, args)
        assert not r.ok and r.content


def test_optional_args_may_be_omitted_or_null(tb):
    assert tb.call("list_dir", None).ok
    assert tb.call("list_dir", "").ok
    assert tb.call("search_code", {"pattern": "def", "path": None}).ok
