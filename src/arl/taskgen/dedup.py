"""Near-duplicate check between held-out and training source functions.

The entry function body is normalised (docstring dropped, every non-builtin name and argument
replaced by a placeholder), unparsed and tokenised; similarity is the Jaccard index of the token
3-gram sets.
"""

from __future__ import annotations

import ast
import builtins
import io
import tokenize

from arl.taskgen.sources import SourceFunction

PLACEHOLDER = "_v"
NGRAM = 3
JACCARD_THRESHOLD = 0.8
MAX_MOVED_FRACTION = 0.10
_BUILTINS = frozenset(dir(builtins))
_SKIP_TOKENS = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.COMMENT}


class _Normalise(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id not in _BUILTINS:
            node.id = PLACEHOLDER
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = PLACEHOLDER
        node.annotation = None
        return node


def normalised_body(fn: SourceFunction) -> str:
    tree = ast.parse(fn.code)
    func = next(s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == fn.entry)
    body = list(func.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:] or [ast.Pass()]
    func.body = body
    func.name = PLACEHOLDER
    func.returns = None
    func.decorator_list = []
    _Normalise().visit(func)
    return ast.unparse(func)


def trigrams(fn: SourceFunction) -> frozenset[tuple[str, ...]]:
    toks = [t.string for t in tokenize.generate_tokens(io.StringIO(normalised_body(fn)).readline) if t.type not in _SKIP_TOKENS]
    toks = [t if t else "<dedent>" for t in toks]
    return frozenset(tuple(toks[i : i + NGRAM]) for i in range(len(toks) - NGRAM + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def clusters(funcs: list[SourceFunction], threshold: float = JACCARD_THRESHOLD) -> list[list[str]]:
    """Connected components of the graph with an edge between functions whose Jaccard exceeds ``threshold``.

    Each cluster is a sorted list of source ids; clusters are sorted by their first member.
    """
    ids = sorted(f.source_id for f in funcs)
    grams = {f.source_id: trigrams(f) for f in funcs}
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        ga = grams[a]
        for b in ids[i + 1 :]:
            if jaccard(ga, grams[b]) > threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def near_duplicates(heldout: list[SourceFunction], train: list[SourceFunction],
                    threshold: float = JACCARD_THRESHOLD) -> list[dict]:
    """Held-out functions whose max Jaccard similarity to any training function exceeds ``threshold``."""
    train_grams = [(f.source_id, trigrams(f)) for f in train]
    out = []
    for fn in heldout:
        g = trigrams(fn)
        best, best_id = 0.0, None
        for sid, tg in train_grams:
            j = jaccard(g, tg)
            if j > best:
                best, best_id = j, sid
        if best > threshold:
            out.append({"source_id": fn.source_id, "max_jaccard": round(best, 3), "closest_train": best_id})
    return out
