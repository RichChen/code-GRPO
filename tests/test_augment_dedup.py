import json
import math

from arl.taskgen.augment import (
    MAX_EXTRA_LEN,
    MAX_GENERATED,
    augment_function,
    candidate_inputs,
    canonical,
    lit,
    literal_source,
    parse_args,
)
from arl.taskgen.dedup import jaccard, near_duplicates, normalised_body, trigrams
from arl.taskgen.sources import Case, SourceFunction, SourceStats, load_mbpp


def fn(code: str, entry: str, cases: list[tuple[str, str]], sid: str = "syn/1") -> SourceFunction:
    return SourceFunction(sid, entry, code, [Case(a, e, isinstance(eval(e), float)) for a, e in cases])


ADD = fn("def add(a, b):\n    return a + b\n", "add", [("1, 2", "3"), ("5, 7", "12"), ("0, 0", "0")])
COUNT = fn(
    "def count(xs, x):\n    c = 0\n    for v in xs:\n        if v == x:\n            c += 1\n    return c\n",
    "count",
    [("[1, 2, 1], 1", "2"), ("[], 3", "0"), ("['a', 'b'], 'b'", "1")],
)


# -- literals ---------------------------------------------------------------------------------------


def test_lit_sorts_set_elements_and_round_trips():
    assert lit({"b", "a", "c"}) == "{'a', 'b', 'c'}"
    assert lit((1,)) == "(1,)"
    assert literal_source([1, (2, "x"), {"k": [True]}]) == "[1, (2, 'x'), {'k': [True]}]"


def test_literal_source_rejects_values_that_do_not_round_trip():
    assert literal_source(set()) == "set()"  # literal_eval accepts it since Python 3.9
    assert literal_source(float("nan")) is None
    assert literal_source(object()) is None
    assert literal_source(frozenset({1})) is None


def test_canonical_normalises_argument_text():
    assert canonical('"abc", [1,2]') == "'abc', [1, 2]"
    assert parse_args("1, k=[2]") == ([1], [("k", [2])])


# -- candidate inputs -------------------------------------------------------------------------------


def test_candidates_are_deterministic_new_and_bounded():
    a, b = candidate_inputs(COUNT), candidate_inputs(COUNT)
    assert a == b and len(a) > 5
    existing = {canonical(c.args) for c in COUNT.cases}
    assert not existing & set(a)
    for args in a:
        xs, x = parse_args(args)[0]
        assert isinstance(xs, list) and len(xs) <= 3 + MAX_EXTRA_LEN


def test_candidates_preserve_argument_types():
    for args in candidate_inputs(ADD):
        a, b = parse_args(args)[0]
        assert type(a) is int and type(b) is int


# -- augmentation -----------------------------------------------------------------------------------


def test_augment_adds_hidden_cases_with_expected_from_original_code():
    out = augment_function(ADD).fn
    gen = [c for c in out.cases if c.kind == "generated"]
    assert 0 < len(gen) <= MAX_GENERATED
    assert out.cases[: len(ADD.cases)] == ADD.cases
    for c in gen:
        a, b = parse_args(c.args)[0]
        assert eval(c.expected) == a + b and c.check() == f"result == {c.expected}"


def test_augment_uses_isclose_for_float_results():
    halve = fn("def halve(x):\n    return x / 2\n", "halve", [("3", "1.5"), ("5", "2.5"), ("1", "0.5")])
    gen = [c for c in augment_function(halve).fn.cases if c.kind == "generated"]
    assert gen and all(c.is_float and c.check().startswith("math.isclose(result, ") for c in gen)
    for c in gen:
        assert math.isclose(eval(c.expected), parse_args(c.args)[0][0] / 2)


def test_augment_discards_raising_and_nondeterministic_inputs():
    picky = fn("def picky(x):\n    if x < 0:\n        raise ValueError\n    return x\n", "picky", [("1", "1"), ("2", "2"), ("3", "3")])
    res = augment_function(picky)
    for c in res.fn.cases:
        if c.kind == "generated":
            assert parse_args(c.args)[0][0] >= 0
    assert res.stats["discard:raises"] > 0
    noisy = fn("import random\n\ndef noisy(x):\n    return x + random.random()\n", "noisy", [("1", "1"), ("2", "2"), ("3", "3")])
    res = augment_function(noisy)
    assert res.fn.count("generated") == 0 and res.stats["discard:nondeterministic"] > 0


def test_augment_drops_challenge_cases_the_original_fails():
    f = SourceFunction("syn/2", "add", ADD.code, ADD.cases + [Case("2, 2", "5", False, "challenge"), Case("2, 3", "5", False, "challenge")])
    res = augment_function(f)
    assert [c.args for c in res.fn.cases if c.kind == "challenge"] == ["2, 3"]
    assert res.stats["challenge_original_fails"] == 1


# -- near duplicates --------------------------------------------------------------------------------


def test_renamed_copy_is_an_exact_near_duplicate():
    copy = fn("def tally(items, item):\n    n = 0\n    for e in items:\n        if e == item:\n            n += 1\n    return n\n",
              "tally", COUNT.cases and [(c.args, c.expected) for c in COUNT.cases], "syn/9")
    assert normalised_body(copy) == normalised_body(COUNT)
    assert jaccard(trigrams(copy), trigrams(COUNT)) == 1.0
    assert near_duplicates([copy], [COUNT, ADD]) == [{"source_id": "syn/9", "max_jaccard": 1.0, "closest_train": "syn/1"}]
    assert near_duplicates([ADD], [COUNT]) == []


def test_docstring_does_not_affect_similarity():
    doc = fn('def add(a, b):\n    """Add two numbers."""\n    return a + b\n', "add", [("1, 2", "3")])
    assert normalised_body(doc) == normalised_body(ADD)


# -- MBPP sanitized + full --------------------------------------------------------------------------


def test_load_mbpp_prefers_sanitized_and_drops_setup_code(tmp_path):
    sanitized = [{"task_id": 1, "code": "def f(x):\n    return x + 1\n",
                  "test_list": ["assert f(1) == 2", "assert f(2) == 3", "assert f(3) == 4"]}]
    full = [
        {"task_id": 1, "code": "def f(x):\n  return x+1", "test_setup_code": "",
         "test_list": ["assert f(9) == 10"], "challenge_test_list": ["assert f(10) == 11", "assert f(1) == 2"]},
        {"task_id": 2, "code": "def g(x):\n    return x * 2\n", "test_setup_code": "",
         "test_list": ["assert g(1) == 2", "assert g(2) == 4", "assert g(3) == 6"], "challenge_test_list": []},
        {"task_id": 3, "code": "def h(x):\n    return x\n", "test_setup_code": "import os",
         "test_list": ["assert h(1) == 1", "assert h(2) == 2", "assert h(3) == 3"], "challenge_test_list": []},
    ]
    (tmp_path / "s.json").write_text(json.dumps(sanitized))
    (tmp_path / "f.jsonl").write_text("\n".join(json.dumps(r) for r in full))
    stats = SourceStats()
    funcs = {f.source_id: f for f in load_mbpp(tmp_path / "s.json", tmp_path / "f.jsonl", stats)}
    assert set(funcs) == {"mbpp/1", "mbpp/2"}
    f1 = funcs["mbpp/1"]
    assert f1.origin == "mbpp_sanitized" and [c.args for c in f1.cases if c.kind == "test"] == ["1", "2", "3"]
    assert [c.args for c in f1.cases if c.kind == "challenge"] == ["10"]  # f(1) duplicates a test call
    assert funcs["mbpp/2"].origin == "mbpp_full"
    assert stats.function_discards["mbpp_full:test_setup_code"] == 1
