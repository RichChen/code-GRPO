"""Extra hidden cases for every source function: perturbed inputs, expected values from the original code.

Inputs are type-preserving perturbations of one existing case's arguments (ints shifted by a small
amount, negated or zeroed; strings, lists and tuples with an item inserted, deleted or swapped,
truncated or emptied; nested structure kept). Containers grow by at most ``MAX_EXTRA_LEN``.

Each candidate input runs twice against the original code in the sandbox. It is kept only if both
runs return the same value without raising or timing out, and the value survives a ``repr`` /
``ast.literal_eval`` round trip with identical types. Floats follow the assertion rules: a float
result is checked with ``math.isclose``; a container holding floats is discarded.
"""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from arl.env.grader import decode
from arl.env.sandbox import SandboxConfig
from arl.taskgen.sources import Case, SourceFunction, _contains_float
from arl.taskgen.validate import SCREEN_CONFIG, run_calls, run_jobs

MAX_GENERATED = 10
N_CANDIDATES = 25
MAX_ATTEMPTS = 400
MAX_EXTRA_LEN = 5
MAX_ARGS_CHARS = 500
MAX_EXPECTED_CHARS = 500
INSERT_CHARS = "ab z0"


Args = tuple[list[Any], list[tuple[str, Any]]]


def parse_args(args: str) -> Args:
    call = ast.parse(f"f({args})", mode="eval").body
    assert isinstance(call, ast.Call)
    return [ast.literal_eval(a) for a in call.args], [(k.arg, ast.literal_eval(k.value)) for k in call.keywords]


def lit(v: Any) -> str:
    """``repr`` with set elements sorted, so the text does not depend on the host's hash seed."""
    if isinstance(v, list):
        return "[" + ", ".join(lit(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(lit(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if type(v) is set:
        return "{" + ", ".join(sorted(lit(x) for x in v)) + "}" if v else "set()"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{lit(k)}: {lit(x)}" for k, x in v.items()) + "}"
    return repr(v)


def render_args(args: Args) -> str:
    pos, kw = args
    return ", ".join([lit(v) for v in pos] + [f"{k}={lit(v)}" for k, v in kw])


def canonical(args: str) -> str | None:
    try:
        return render_args(parse_args(args))
    except (ValueError, SyntaxError, TypeError, AssertionError):
        return None


def typed(v: Any) -> Any:
    """A hashable form of ``v`` that also distinguishes types (1 vs True vs 1.0, list vs tuple)."""
    if isinstance(v, list):
        return ("L", tuple(typed(x) for x in v))
    if isinstance(v, tuple):
        return ("T", tuple(typed(x) for x in v))
    if isinstance(v, (set, frozenset)):
        return (type(v).__name__, frozenset(typed(x) for x in v))
    if isinstance(v, dict):
        return ("D", frozenset((typed(k), typed(x)) for k, x in v.items()))
    return (type(v).__name__, v)


def literal_source(value: Any) -> str | None:
    """``lit(value)`` if it evaluates back to an identical value, else None."""
    try:
        src = lit(value)
        back = ast.literal_eval(src)
    except Exception:
        return None
    try:
        return src if typed(back) == typed(value) else None
    except TypeError:  # unhashable inside a set or dict key
        return None


# -- perturbation -----------------------------------------------------------------------------------


class Perturber:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def value(self, v: Any, seed_len: int | None = None) -> Any:
        r = self.rng
        if isinstance(v, bool):
            return not v
        if isinstance(v, int):
            op = r.randrange(4)
            if op == 0:
                return v + r.choice([-3, -2, -1, 1, 2, 3])
            if op == 1:
                return -v
            if op == 2:
                return 0
            return -abs(v) - r.randint(1, 3)
        if isinstance(v, float):
            op = r.randrange(3)
            if op == 0:
                return v + r.choice([-2.5, -1.0, -0.5, 0.5, 1.0, 2.5])
            if op == 1:
                return -v
            return 0.0
        if isinstance(v, str):
            return self._sequence(v, seed_len)
        if isinstance(v, (list, tuple)):
            return self._sequence(v, seed_len)
        if isinstance(v, dict):
            return self._dict(v)
        if isinstance(v, set) and v:
            items = sorted(v, key=repr)
            op = r.randrange(2)
            if op == 0 and len(items) > 1:
                items.pop(r.randrange(len(items)))
            else:
                items.append(self.value(r.choice(items)))
            try:
                return set(items)
            except TypeError:
                return v
        return v

    def _sequence(self, v: Any, seed_len: int | None) -> Any:
        r = self.rng
        limit = (len(v) if seed_len is None else seed_len) + MAX_EXTRA_LEN
        items = list(v)
        ops = ["empty", "truncate", "delete", "insert", "swap"]
        if not isinstance(v, str) and items:
            ops.append("element")
        op = r.choice(ops)
        if op == "empty":
            items = []
        elif op == "truncate" and items:
            items = items[: r.randrange(len(items))]
        elif op == "delete" and items:
            items.pop(r.randrange(len(items)))
        elif op == "insert" and len(items) < limit:
            if isinstance(v, str):
                pool = sorted(set(v)) or list(INSERT_CHARS)
                new: Any = r.choice(pool)
            elif items:
                new = self.value(r.choice(items))
            else:
                return v
            items.insert(r.randrange(len(items) + 1), new)
        elif op == "swap" and len(items) >= 2:
            i, j = r.sample(range(len(items)), 2)
            items[i], items[j] = items[j], items[i]
        elif op == "element":
            i = r.randrange(len(items))
            items[i] = self.value(items[i])
        if isinstance(v, str):
            return "".join(items)
        return tuple(items) if isinstance(v, tuple) else items

    def _dict(self, v: dict) -> dict:
        r = self.rng
        out = dict(v)
        keys = list(out)
        op = r.randrange(3)
        if op == 0 and keys:
            del out[r.choice(keys)]
        elif op == 1 and keys:
            k = r.choice(keys)
            out[k] = self.value(out[k])
        else:
            out = {}
        return out

    def args(self, seed: Args) -> Args:
        pos, kw = list(seed[0]), list(seed[1])
        slots = list(range(len(pos) + len(kw)))
        if not slots:
            return seed
        n = 2 if len(slots) > 1 and self.rng.random() < 0.3 else 1
        for i in self.rng.sample(slots, n):
            if i < len(pos):
                pos[i] = self.value(pos[i])
            else:
                k, val = kw[i - len(pos)]
                kw[i - len(pos)] = (k, self.value(val))
        return pos, kw


def candidate_inputs(fn: SourceFunction, n: int = N_CANDIDATES) -> list[str]:
    """Up to ``n`` distinct perturbed argument lists, none equal to an existing case's arguments."""
    seeds = []
    seen = set()
    for c in fn.cases:
        key = canonical(c.args)
        if key is None:
            continue
        seen.add(key)
        seeds.append(parse_args(c.args))
    if not seeds:
        return []
    rng = random.Random(f"augment-{fn.source_id}")
    perturb = Perturber(rng)
    out: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        if len(out) >= n:
            break
        try:
            args = perturb.args(rng.choice(seeds))
            text = render_args(args)
        except (TypeError, ValueError, RecursionError):
            continue
        if len(text) > MAX_ARGS_CHARS or canonical(text) != text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


# -- generation -------------------------------------------------------------------------------------


@dataclass
class AugmentResult:
    fn: SourceFunction
    stats: Counter = field(default_factory=Counter)


def _expected(rec1: dict, rec2: dict) -> tuple[str | None, bool, str]:
    """(expected source, is_float, reason) for one candidate from its two runs."""
    if not rec1.get("ok") or not rec2.get("ok"):
        return None, False, "raises"
    if json.dumps(rec1["v"], sort_keys=True) != json.dumps(rec2["v"], sort_keys=True):
        return None, False, "nondeterministic"
    try:
        value = decode(rec1["v"])
    except Exception:
        return None, False, "not_literal"
    if isinstance(value, complex):
        return None, False, "complex"
    is_float = isinstance(value, float)
    if not is_float and _contains_float(value):
        return None, False, "float_inside_container"
    src = literal_source(value)
    if src is None:
        return None, False, "not_literal"
    if len(src) > MAX_EXPECTED_CHARS:
        return None, False, "expected_too_long"
    return src, is_float, "ok"


def augment_function(fn: SourceFunction, config: SandboxConfig = SCREEN_CONFIG) -> AugmentResult:
    """Drop challenge cases the original code fails, then append up to ``MAX_GENERATED`` generated cases."""
    stats: Counter = Counter()
    [orig] = run_jobs(fn, [fn.code], config)
    cases = list(fn.cases)
    if orig.status == "ok":
        kept = []
        for c, o in zip(cases, orig.outcomes):
            if c.kind == "challenge" and not o.passed:
                stats["challenge_original_fails"] += 1
                continue
            kept.append(c)
        cases = kept
    fn = replace(fn, cases=cases)

    candidates = candidate_inputs(fn)
    stats["candidates"] += len(candidates)
    if not candidates:
        return AugmentResult(fn, stats)
    jobs = [(fn.code, [args]) for args in candidates]
    first = run_calls(fn.entry, jobs, config)
    second = run_calls(fn.entry, jobs, config)
    generated: list[Case] = []
    for args, a, b in zip(candidates, first, second):
        if a.status != "ok" or b.status != "ok":
            stats[f"discard:{a.status if a.status != 'ok' else b.status}"] += 1
            continue
        expected, is_float, reason = _expected(a.records[0], b.records[0])
        if expected is None:
            stats[f"discard:{reason}"] += 1
            continue
        if len(generated) >= MAX_GENERATED:
            stats["discard:over_limit"] += 1
            continue
        generated.append(Case(args=args, expected=expected, is_float=is_float, kind="generated"))
    stats["generated"] += len(generated)
    return AugmentResult(replace(fn, cases=cases + generated), stats)
