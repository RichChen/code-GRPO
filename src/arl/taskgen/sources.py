"""Source functions from MBPP (sanitized) and HumanEval, with their assertions split into cases.

Only the plain form ``assert f(args) == expected`` (MBPP) / ``assert candidate(args) == expected``
(HumanEval) is accepted, where every argument and ``expected`` can be evaluated with
``ast.literal_eval``. Everything else is discarded and counted by reason.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import urllib.request
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

RAW_FILES = {
    "mbpp": {
        "file": "sanitized-mbpp.json",
        "url": "https://raw.githubusercontent.com/google-research/google-research/"
        "f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json",
        "sha256": "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
    },
    "humaneval": {
        "file": "HumanEval.jsonl.gz",
        "url": "https://raw.githubusercontent.com/openai/human-eval/"
        "d321ec0b6c23dec317337be99f6d0c45ca73f3d5/data/HumanEval.jsonl.gz",
        "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    },
}
MIN_CASES = 3
FLOAT_CHECK = "math.isclose(result, {expected}, rel_tol=1e-06, abs_tol=1e-09)"
ALLOWED_TOPLEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


@dataclass(frozen=True)
class Case:
    args: str  # source of the argument list, e.g. "'Abc', 2"
    expected: str  # source of the expected literal
    is_float: bool

    def check(self) -> str:
        if self.is_float:
            return FLOAT_CHECK.format(expected=self.expected)
        return f"result == {self.expected}"

    def call(self, func: str) -> str:
        return f"{func}({self.args})"


@dataclass
class SourceFunction:
    source_id: str  # "mbpp/2", "humaneval/0"
    entry: str
    code: str  # normalised module source: imports, helpers and the entry function
    cases: list[Case]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceFunction":
        return cls(source_id=d["source_id"], entry=d["entry"], code=d["code"], cases=[Case(**c) for c in d["cases"]])


@dataclass
class SourceStats:
    functions_total: Counter = field(default_factory=Counter)
    functions_kept: Counter = field(default_factory=Counter)
    function_discards: Counter = field(default_factory=Counter)
    assert_total: Counter = field(default_factory=Counter)
    assert_kept: Counter = field(default_factory=Counter)
    assert_discards: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {name: dict(sorted(getattr(self, name).items())) for name in self.__dataclass_fields__}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(raw_dir: str | Path) -> dict[str, Path]:
    """Download the pinned raw files (honours HTTPS_PROXY) and verify their sha256."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, spec in RAW_FILES.items():
        path = raw_dir / spec["file"]
        if not path.exists() or sha256_file(path) != spec["sha256"]:
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(spec["url"], timeout=120) as resp, open(tmp, "wb") as out:
                out.write(resp.read())
            digest = sha256_file(tmp)
            if digest != spec["sha256"]:
                tmp.unlink()
                raise RuntimeError(f"sha256 mismatch for {spec['url']}: got {digest}")
            tmp.replace(path)
        paths[name] = path
    return paths


# -- assertion parsing ---------------------------------------------------------------------------


class Discard(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_float(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_float(k) or _contains_float(v) for k, v in value.items())
    if isinstance(value, complex):
        return True
    return False


def parse_assert(stmt: ast.stmt, func_name: str) -> Case:
    """Turn ``assert func_name(args) == expected`` into a ``Case`` or raise ``Discard``."""
    if isinstance(stmt, (ast.For, ast.While)):
        raise Discard("loop")
    if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
        raise Discard("helper_definition")
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        raise Discard("assignment")
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        raise Discard("import")
    if not isinstance(stmt, ast.Assert):
        raise Discard("other_statement")
    test = stmt.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        raise Discard("not")
    if isinstance(test, ast.BoolOp):
        raise Discard("bool_op")
    if not isinstance(test, ast.Compare):
        raise Discard("not_a_comparison")
    if len(test.ops) > 1:
        raise Discard("chained_comparison")
    op = test.ops[0]
    if isinstance(op, (ast.Is, ast.IsNot)):
        raise Discard("is")
    if not isinstance(op, ast.Eq):
        raise Discard("not_equality")
    left, right = test.left, test.comparators[0]
    if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == func_name):
        if isinstance(right, ast.Call) and isinstance(right.func, ast.Name) and right.func.id == func_name:
            raise Discard("reversed_comparison")
        raise Discard("call_wrapped_or_other_function")
    if any(isinstance(a, ast.Starred) for a in left.args) or any(k.arg is None for k in left.keywords):
        raise Discard("star_args")
    try:
        for node in list(left.args) + [k.value for k in left.keywords]:
            ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise Discard("args_not_literal") from None
    try:
        expected = ast.literal_eval(right)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise Discard("expected_not_literal") from None
    is_float = isinstance(expected, float)
    if not is_float and _contains_float(expected):
        raise Discard("float_inside_container")
    parts = [ast.unparse(a) for a in left.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in left.keywords]
    return Case(args=", ".join(parts), expected=ast.unparse(right), is_float=is_float)


def normalise_module(code: str, entry: str) -> str:
    """Parse ``code``, check its top level and return ``ast.unparse`` of it. Raises ``Discard``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise Discard("solution_syntax_error") from None
    body = []
    for i, stmt in enumerate(tree.body):
        if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue  # module docstring
        if not isinstance(stmt, ALLOWED_TOPLEVEL):
            raise Discard("unsupported_toplevel_statement")
        body.append(stmt)
    tree.body = body
    if not any(isinstance(s, ast.FunctionDef) and s.name == entry for s in body):
        raise Discard("entry_function_not_found")
    return ast.unparse(tree) + "\n"


def _split_cases(stmts: list[ast.stmt], func_name: str, origin: str, stats: SourceStats) -> list[Case]:
    cases: list[Case] = []
    seen_calls: set[str] = set()
    for stmt in stmts:
        stats.assert_total[origin] += 1
        try:
            case = parse_assert(stmt, func_name)
        except Discard as d:
            stats.assert_discards[f"{origin}:{d.reason}"] += 1
            continue
        if case.args in seen_calls:
            stats.assert_discards[f"{origin}:duplicate_call"] += 1
            continue
        seen_calls.add(case.args)
        stats.assert_kept[origin] += 1
        cases.append(case)
    return cases


def _finish(source_id: str, entry: str, code: str, cases: list[Case], origin: str, stats: SourceStats) -> SourceFunction | None:
    stats.functions_total[origin] += 1
    try:
        normalised = normalise_module(code, entry)
    except Discard as d:
        stats.function_discards[f"{origin}:{d.reason}"] += 1
        return None
    if len(cases) < MIN_CASES:
        stats.function_discards[f"{origin}:fewer_than_{MIN_CASES}_cases"] += 1
        return None
    stats.functions_kept[origin] += 1
    return SourceFunction(source_id=source_id, entry=entry, code=normalised, cases=cases)


def _entry_from_tests(tests: list[ast.stmt], defined: set[str]) -> str | None:
    """The function under test: the solution-defined function called most often in the tests."""
    names: Counter = Counter()
    for stmt in tests:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in defined:
                names[node.func.id] += 1
    return names.most_common(1)[0][0] if names else None


def load_mbpp(path: str | Path, stats: SourceStats) -> list[SourceFunction]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    out = []
    for rec in records:
        stmts = []
        for line in rec["test_list"]:
            try:
                stmts.extend(ast.parse(line).body)
            except SyntaxError:
                stats.assert_total["mbpp"] += 1
                stats.assert_discards["mbpp:test_syntax_error"] += 1
        try:
            defined = {s.name for s in ast.parse(rec["code"]).body if isinstance(s, ast.FunctionDef)}
        except SyntaxError:
            defined = set()
        entry = _entry_from_tests(stmts, defined) or ""
        cases = _split_cases(stmts, entry, "mbpp", stats)
        fn = _finish(f"mbpp/{rec['task_id']}", entry, rec["code"], cases, "mbpp", stats)
        if fn:
            out.append(fn)
    return out


def load_humaneval(path: str | Path, stats: SourceStats) -> list[SourceFunction]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    out = []
    for rec in records:
        test_tree = ast.parse(rec["test"])
        check = next((s for s in test_tree.body if isinstance(s, ast.FunctionDef) and s.name == "check"), None)
        body = check.body if check else []
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        cases = _split_cases(body, "candidate", "humaneval", stats)
        num = rec["task_id"].split("/")[1]
        fn = _finish(f"humaneval/{num}", rec["entry_point"], rec["prompt"] + rec["canonical_solution"], cases, "humaneval", stats)
        if fn:
            out.append(fn)
    return out


def load_sources(raw_dir: str | Path) -> tuple[list[SourceFunction], SourceStats]:
    raw_dir = Path(raw_dir)
    stats = SourceStats()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # invalid escape sequences in a few solutions
        funcs = load_mbpp(raw_dir / RAW_FILES["mbpp"]["file"], stats)
        funcs += load_humaneval(raw_dir / RAW_FILES["humaneval"]["file"], stats)
    return funcs, stats
