import ast
import json
import random
import re
from pathlib import Path

import pytest

from arl.env.task import VISIBLE_TEST_PATH, Task, load_tasks
from arl.taskgen.build import (
    ISSUE_TEMPLATES,
    PackageBuilder,
    assemble_task,
    build_from_functions,
    candidate_slots,
    check_no_leakage,
    check_split,
    split_clusters,
    split_functions,
)
from arl.taskgen.dedup import clusters
from arl.taskgen.mutate import BUG_TYPES, MutationSpec, apply_mutation, enumerate_mutants, infer_local_types, rename_names
from arl.taskgen.sources import Case, Discard, SourceFunction, load_sources, normalise_module, parse_assert
from arl.taskgen.validate import check_task_static, screen_function, validate_task

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TASKS = ROOT / "data" / "tasks"
has_raw = pytest.mark.skipif(not (RAW / "sanitized-mbpp.json").exists(), reason="raw data not downloaded")
has_tasks = pytest.mark.skipif(not (TASKS / "train.jsonl").exists(), reason="task set not built")


def stmt(src: str) -> ast.stmt:
    return ast.parse(src).body[0]


# -- assertion splitting -----------------------------------------------------------------------


def test_parse_assert_accepts_plain_equality():
    c = parse_assert(stmt("assert f(1, 'a', k=[2]) == (3, 'b')"), "f")
    assert c == Case(args="1, 'a', k=[2]", expected="(3, 'b')", is_float=False)
    assert c.check() == "result == (3, 'b')"
    assert c.call("g") == "g(1, 'a', k=[2])"


def test_parse_assert_float_uses_isclose():
    c = parse_assert(stmt("assert candidate(2) == 1.5"), "candidate")
    assert c.is_float
    assert c.check() == "math.isclose(result, 1.5, rel_tol=1e-06, abs_tol=1e-09)"


@pytest.mark.parametrize(
    "src, reason",
    [
        ("assert set(f(1)) == {1}", "call_wrapped_or_other_function"),
        ("assert math.isclose(f(1), 2.0)", "not_a_comparison"),
        ("assert f(1)", "not_a_comparison"),
        ("assert not f(1)", "not"),
        ("assert f(1) is None", "is"),
        ("assert f(1) is not None", "is"),
        ("assert 1 < f(1) < 3", "chained_comparison"),
        ("assert f(1) == 2 == 2", "chained_comparison"),
        ("assert f(1) != 2", "not_equality"),
        ("assert 2 == f(1)", "reversed_comparison"),
        ("assert f(1) == y", "expected_not_literal"),
        ("assert f(x) == 1", "args_not_literal"),
        ("assert f(*xs) == 1", "star_args"),
        ("assert f([1]) == [1.5]", "float_inside_container"),
        ("assert f(1) == 2 and f(2) == 3", "bool_op"),
        ("for i in range(3):\n    assert f(i) == i", "loop"),
        ("def helper(x):\n    return x", "helper_definition"),
        ("x = 1", "assignment"),
        ("import math", "import"),
        ("print(f(1))", "other_statement"),
    ],
)
def test_parse_assert_rejects(src, reason):
    with pytest.raises(Discard) as info:
        parse_assert(stmt(src), "f")
    assert info.value.reason == reason


def test_normalise_module():
    code = normalise_module("import re\n\ndef helper(x):\n    return x\n\ndef f(s):  # comment\n    return helper(s)\n", "f")
    assert "comment" not in code and "def f(s)" in code and "import re" in code
    with pytest.raises(Discard, match="unsupported_toplevel"):
        normalise_module("def f():\n    return 1\nprint(f())\n", "f")
    with pytest.raises(Discard, match="entry_function_not_found"):
        normalise_module("def g():\n    return 1\n", "f")


@has_raw
def test_real_sources_follow_the_rules():
    funcs, stats = load_sources(RAW)
    assert len(funcs) > 500
    for fn in funcs:
        assert len(fn.cases) >= 3
        calls = [c.args for c in fn.cases]
        assert len(calls) == len(set(calls))
        for c in fn.cases:
            check = c.check()
            assert "result" in check
            assert check.startswith("result == ") or check.startswith("math.isclose(result, ")
            ast.literal_eval(c.expected)
            assert "is not None" not in check and "bool(result)" not in check
    assert sum(stats.assert_discards.values()) > 0


# -- mutation ------------------------------------------------------------------------------------

SAMPLE = '''import math


def helper(v):
    return v + 1


def target(xs, n):
    total = 0
    count = 0
    for i in range(len(xs)):
        if xs[i] > n and not xs[i] == 0:
            total += xs[i] * 2
            count = count + 1
        elif xs[i] == n:
            return -1
        else:
            count -= 1
    if count < 1:
        return 0
    return total // count
'''


def test_every_bug_type_is_generated_and_compiles():
    mutants = enumerate_mutants(SAMPLE, "target", [[[1, 2, 3], 1]])
    types = {m.spec.bug_type for m in mutants}
    assert types == set(BUG_TYPES)
    original_helper = "def helper(v):\n    return v + 1"
    codes = set()
    for m in mutants:
        compile(m.code, "<m>", "exec")
        assert original_helper in m.code  # only the target function is mutated
        assert m.code not in codes
        codes.add(m.code)


@pytest.mark.parametrize(
    "spec_filter, before, after",
    [
        (lambda s: s.bug_type == "arith_swap" and s.variant == "*->//", "xs[i] * 2", "xs[i] // 2"),
        (lambda s: s.bug_type == "arith_swap" and s.variant == "+->-", "total += xs[i]", "total -= xs[i]"),
        (lambda s: s.bug_type == "compare_swap" and s.variant == "0:<-><=", "count < 1", "count <= 1"),
        (lambda s: s.bug_type == "const_off_by_one" and s.variant == "range-1", "range(len(xs))", "range(len(xs) - 1)"),
        (lambda s: s.bug_type == "bool_swap" and s.variant == "and->or", "xs[i] > n and not", "xs[i] > n or not"),
        (lambda s: s.bug_type == "bool_swap" and s.variant == "drop_not", "and not xs[i] == 0", "and xs[i] == 0"),
        (lambda s: s.bug_type == "remove_branch" and s.variant == "drop_return", "return -1", None),
    ],
)
def test_specific_mutations(spec_filter, before, after):
    mutants = [m for m in enumerate_mutants(SAMPLE, "target", [[[1, 2, 3], 1]]) if spec_filter(m.spec)]
    assert mutants and before in SAMPLE
    if after is None:
        assert any(before not in m.code for m in mutants)
    else:
        assert any(after in m.code for m in mutants)


def test_drop_return_never_removes_the_final_return():
    for m in enumerate_mutants(SAMPLE, "target"):
        if m.spec.variant == "drop_return":
            assert m.code.rstrip().endswith("return total // count")


def test_var_swap_respects_types():
    types = infer_local_types(ast.parse(SAMPLE).body[2], [[[1, 2, 3], 1]])
    assert types["xs"] == "list" and types["n"] == "int" and types["count"] == "int" and types["i"] == "int"
    assert "total" not in types  # `total += xs[i] * 2` has no inferable type, so total is never swapped
    for m in enumerate_mutants(SAMPLE, "target", [[[1, 2, 3], 1]]):
        if m.spec.bug_type == "var_swap":
            a, b = m.spec.variant.split("->")
            assert types[a] == types[b]
            assert "xs" not in (a, b)  # the only list local has no partner


def test_apply_mutation_replays_by_position():
    m = enumerate_mutants(SAMPLE, "target")[3]
    renamed = rename_names(SAMPLE, {"helper": "helper_2"})
    replay = apply_mutation(renamed, "target", m.spec)
    assert replay.code == rename_names(m.code, {"helper": "helper_2"})


def test_mutation_never_breaks_syntax_or_import(tmp_path):
    for m in enumerate_mutants(SAMPLE, "target"):
        ns: dict = {}
        exec(compile(m.code, "<m>", "exec"), ns)  # import (module execution) must not fail
        assert callable(ns["target"])


def test_rename_names():
    code = "def f(x):\n    return g(x) + f(x - 1) if x else 0\n\ndef g(y):\n    return y\n"
    out = rename_names(code, {"g": "g_2"})
    assert "def g_2(y)" in out and "g_2(x)" in out and "def f(x)" in out


# -- package assembly ----------------------------------------------------------------------------


def src(sid: str, entry: str, code: str, cases: list[tuple[str, str]]) -> SourceFunction:
    return SourceFunction(
        source_id=sid, entry=entry, code=normalise_module(code, entry),
        cases=[Case(args=a, expected=e, is_float=False) for a, e in cases],
    )


def test_package_builder_renames_collisions_and_builtins():
    b = PackageBuilder()
    a = b.try_place(src("x/1", "f", "def f(x):\n    return sum([x])\n", [("1", "1")] * 3), "core")
    c = b.try_place(src("x/2", "f", "def f(x):\n    return x + 1\n", [("1", "2")] * 3), "utils")
    d = b.try_place(src("x/3", "sum", "def sum(a, b):\n    return a + b\n", [("1, 2", "3")] * 3), "core")
    assert a.name == "f" and c.name == "f_2" and d.name == "sum_2"
    assert "def f_2(x)" in c.code and "def sum_2(a, b)" in d.code


def test_package_builder_rejects_conflicting_imports():
    b = PackageBuilder()
    assert b.try_place(src("x/1", "f", "import math as m\ndef f(x):\n    return m.sqrt(x)\n", [("4", "2.0")] * 3), "core")
    assert b.try_place(src("x/2", "g", "import re as m\ndef g(x):\n    return m.escape(x)\n", [("'a'", "'a'")] * 3), "core") is None
    assert b.try_place(src("x/3", "h", "import re as m\ndef h(x):\n    return m.escape(x)\n", [("'a'", "'a'")] * 3), "utils")


SYNTHETIC = [
    src(f"syn/{i}", name, code, cases)
    for i, (name, code, cases) in enumerate(
        [
            ("add_all", "def add_all(xs):\n    t = 0\n    for x in xs:\n        t = t + x\n    return t\n",
             [("[1, 2]", "3"), ("[5]", "5"), ("[1, 1, 1]", "3"), ("[2, 3, 4]", "9")]),
            ("count_big", "def count_big(xs, n):\n    c = 0\n    for x in xs:\n        if x > n:\n            c += 1\n    return c\n",
             [("[1, 5, 9], 4", "2"), ("[1, 2], 0", "2"), ("[3, 3, 3], 3", "0"), ("[7, 8], 7", "1")]),
            ("first_neg", "def first_neg(xs):\n    for i in range(len(xs)):\n        if xs[i] < 0:\n            return i\n    return -1\n",
             [("[1, -2, 3]", "1"), ("[1, 2]", "-1"), ("[-1]", "0"), ("[0, 0, -5]", "2")]),
            ("both_pos", "def both_pos(a, b):\n    if a > 0 and b > 0:\n        return True\n    return False\n",
             [("1, 2", "True"), ("-1, 2", "False"), ("1, -2", "False"), ("3, 3", "True")]),
            ("shout", "def shout(s):\n    return s.upper() + '!'\n", [("'a'", "'A!'"), ("'hi'", "'HI!'"), ("''", "'!'")]),
            ("span", "def span(xs):\n    return max(xs) - min(xs)\n", [("[1, 5]", "4"), ("[3]", "0"), ("[2, 9, 4]", "7")]),
            ("sum", "def sum(a, b):\n    return a * 2 + b\n", [("1, 2", "4"), ("0, 0", "0"), ("2, 1", "5")]),
            ("dbl", "def dbl(xs):\n    return [x * 2 for x in xs]\n", [("[1]", "[2]"), ("[]", "[]"), ("[1, 2]", "[2, 4]")]),
            ("mid", "def mid(xs):\n    return xs[len(xs) // 2]\n", [("[1, 2, 3]", "2"), ("[4]", "4"), ("[1, 2, 3, 4, 5]", "3")]),
            ("is_even", "def is_even(n):\n    return n % 2 == 0\n", [("2", "True"), ("3", "False"), ("10", "True")]),
            ("rev", "def rev(s):\n    return s[::-1]\n", [("'ab'", "'ba'"), ("''", "''"), ("'abc'", "'cba'")]),
            ("sq", "def sq(n):\n    return n * n\n", [("2", "4"), ("3", "9"), ("0", "0")]),
            ("total_len", "def total_len(xs):\n    return sum((len(x) for x in xs))\n", [("['a', 'bc']", "3"), ("[]", "0"), ("['xyz']", "3")]),
            ("cap", "def cap(n, hi):\n    if n > hi:\n        return hi\n    return n\n", [("5, 3", "3"), ("1, 3", "1"), ("3, 3", "3"), ("9, 0", "0")]),
        ]
    )
]


@pytest.fixture(scope="module")
def synthetic_build(tmp_path_factory):
    out = tmp_path_factory.mktemp("tasks")
    manifest = build_from_functions(
        SYNTHETIC, out, seed=1, workers=4, log=lambda *_: None, modules_range=(2, 2), functions_range=(1, 2)
    )
    return out, manifest


def test_synthetic_build_produces_valid_tasks(synthetic_build):
    out, manifest = synthetic_build
    train, heldout = load_tasks(out / "train.jsonl"), load_tasks(out / "heldout.jsonl")
    assert manifest["counts"]["total"] == len(train) + len(heldout) > 0
    assert len(heldout) > 0
    for t in train + heldout:
        assert check_task_static(t) is None
        assert t.meta["bug_file"] in t.solution and t.files[t.meta["bug_file"]] != t.solution[t.meta["bug_file"]]
        assert t.meta["issue_template"] in ISSUE_TEMPLATES


def test_synthetic_build_has_no_split_leakage(synthetic_build):
    out, _ = synthetic_build
    train, heldout = load_tasks(out / "train.jsonl"), load_tasks(out / "heldout.jsonl")
    train_ids = {sid for t in train for sid in t.meta["source_ids"]}
    heldout_ids = {sid for t in heldout for sid in t.meta["source_ids"]}
    assert train_ids and heldout_ids and not (train_ids & heldout_ids)


def test_synthetic_build_caps_per_function_and_bug_type(synthetic_build):
    out, _ = synthetic_build
    tasks = load_tasks(out / "train.jsonl") + load_tasks(out / "heldout.jsonl")
    slots: dict = {}
    for t in tasks:
        slots.setdefault((t.meta["bug_source"], t.meta["bug_type"]), []).append(t.meta["bug_position"])
    for positions in slots.values():
        assert len(positions) <= 2 and len(set(positions)) == len(positions)


def test_check_no_leakage_detects_overlap(task):
    a = Task.from_dict({**task.to_dict(), "meta": {"source_ids": ["mbpp/1", "mbpp/2"]}})
    b = Task.from_dict({**task.to_dict(), "meta": {"source_ids": ["mbpp/2", "mbpp/3"]}})
    check_no_leakage({"train": [a], "heldout": [Task.from_dict({**task.to_dict(), "meta": {"source_ids": ["mbpp/9"]}})]})
    with pytest.raises(ValueError, match="mbpp/2"):
        check_no_leakage({"train": [a], "heldout": [b]})


def test_split_is_deterministic():
    s1 = split_functions(SYNTHETIC, seed=3)
    assert s1 == split_functions(list(reversed(SYNTHETIC)), seed=3)
    assert set(s1.values()) == {"train", "heldout"}


def test_shadowing_builtin_is_renamed_in_the_package(synthetic_build):
    out, _ = synthetic_build
    tasks = load_tasks(out / "train.jsonl") + load_tasks(out / "heldout.jsonl")
    for t in tasks:
        for path, content in t.files.items():
            assert "def sum(" not in content


def _assert_distinct_calls(tasks: list[Task]) -> None:
    for t in tasks:
        for fn in t.meta["functions"]:
            hidden_calls = {t.hidden_cases[i].call for i in fn["hidden_case_indices"]}
            assert f"pkg.{fn['visible_call']}" not in hidden_calls, (t.task_id, fn["name"])
            assert fn["visible_call"] in t.files[VISIBLE_TEST_PATH]


def test_visible_and_hidden_calls_differ_synthetic(synthetic_build):
    out, _ = synthetic_build
    _assert_distinct_calls(load_tasks(out / "train.jsonl") + load_tasks(out / "heldout.jsonl"))


def test_issue_does_not_reveal_location_or_bug_type(synthetic_build):
    out, _ = synthetic_build
    for t in load_tasks(out / "train.jsonl") + load_tasks(out / "heldout.jsonl"):
        assert ".py" not in t.issue and "pkg" not in t.issue
        assert not re.search(r"\bline\s+\d", t.issue)
        for module in {f["module"] for f in t.meta["functions"]}:
            assert f"{module}." not in t.issue


@has_tasks
def test_generated_task_set_visible_and_hidden_calls_differ():
    _assert_distinct_calls(load_tasks(TASKS / "train.jsonl") + load_tasks(TASKS / "heldout.jsonl"))


@has_tasks
def test_generated_task_set_has_no_split_leakage():
    tasks = {s: load_tasks(TASKS / f"{s}.jsonl") for s in ("train", "heldout")}
    check_no_leakage(tasks)
    manifest = json.loads((TASKS / "manifest.json").read_text())
    from arl.taskgen.sources import sha256_file

    for name, digest in manifest["files"].items():
        assert sha256_file(TASKS / name) == digest


# -- cluster split (audit revision 3) ---------------------------------------------------------------


def test_split_clusters_keeps_clusters_together_and_reaches_fraction():
    cl = [[f"c{i}/{j}" for j in range(size)] for i, size in enumerate([5, 1, 1, 3, 2, 1, 1, 4, 1, 1])]
    split = split_clusters(cl, seed=0)
    for c in cl:
        assert len({split[s] for s in c}) == 1
    held = [s for s, v in split.items() if v == "heldout"]
    assert 0.3 * 20 <= len(held) < 0.3 * 20 + 5
    assert split == split_clusters(list(reversed(cl)), seed=0)


def test_near_duplicate_functions_never_straddle_the_split():
    dup = src("syn/99", "add_all_2", "def add_all_2(ys):\n    s = 0\n    for y in ys:\n        s = s + y\n    return s\n",
              [("[1]", "1"), ("[2, 2]", "4"), ("[]", "0")])
    funcs = SYNTHETIC + [dup]
    groups = clusters(funcs)
    assert ["syn/0", "syn/99"] in groups
    for seed in range(20):
        split = split_functions(funcs, seed)
        assert split["syn/0"] == split["syn/99"]
        assert check_split(funcs, split) == []


@has_raw
def test_real_cluster_split_has_no_cross_split_near_duplicates():
    funcs, _ = load_sources(RAW)
    split = split_functions(funcs, 0)
    for c in clusters(funcs):
        assert len({split[s] for s in c}) == 1
    assert check_split(funcs, split) == []


# -- strengthened hidden cases ----------------------------------------------------------------------


def test_synthetic_tasks_have_enough_hidden_cases_for_the_bug_function(synthetic_build):
    out, manifest = synthetic_build
    for t in load_tasks(out / "train.jsonl") + load_tasks(out / "heldout.jsonl"):
        bug = next(f for f in t.meta["functions"] if f["name"] == t.meta["bug_function"])
        assert len(bug["hidden_case_indices"]) >= 5
        assert sum(t.meta["bug_hidden_counts"].values()) == len(bug["hidden_case_indices"])
        assert len(t.meta["hidden_case_kinds"]) == len(t.hidden_cases)
        assert t.meta["bug_hidden_counts"].get("generated", 0) > 0
    assert manifest["augmentation"]["bug_generated_cases"] > 0


@has_tasks
def test_generated_task_set_bug_functions_have_at_least_five_hidden_cases():
    for t in load_tasks(TASKS / "train.jsonl") + load_tasks(TASKS / "heldout.jsonl"):
        bug = next(f for f in t.meta["functions"] if f["name"] == t.meta["bug_function"])
        assert len(bug["hidden_case_indices"]) >= 5, t.task_id
        kinds = [t.meta["hidden_case_kinds"][i] for i in bug["hidden_case_indices"]]
        assert kinds.count("generated") == t.meta["bug_hidden_counts"].get("generated", 0)
