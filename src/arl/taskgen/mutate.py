"""AST-based bug injection into one function.

A mutant is identified by ``(bug_type, position, variant)``; ``position`` is the index of the
mutated node in ``ast.walk`` order over the target function, so the same mutation can be
replayed after the module has been renamed or re-parsed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterator

BUG_TYPES = ("arith_swap", "compare_swap", "const_off_by_one", "bool_swap", "var_swap", "remove_branch")

_ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}
_COMPARE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_SYMBOL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
    ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=", ast.And: "and", ast.Or: "or",
}


@dataclass(frozen=True)
class MutationSpec:
    bug_type: str
    position: int
    variant: str


@dataclass(frozen=True)
class Mutant:
    spec: MutationSpec
    code: str
    lineno: int  # line of the mutated node in ``code`` (analysis only)


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return stmt
    raise KeyError(name)


def _walk(func: ast.FunctionDef) -> list[ast.AST]:
    return list(ast.walk(func))


# -- type inference for var_swap ---------------------------------------------------------------

_ANNOTATION_TYPES = {
    "int": "int", "float": "float", "str": "str", "bool": "bool", "list": "list", "List": "list",
    "dict": "dict", "Dict": "dict", "tuple": "tuple", "Tuple": "tuple", "set": "set", "Set": "set",
}
_CALL_TYPES = {
    "len": "int", "int": "int", "ord": "int", "str": "str", "chr": "str", "float": "float", "list": "list",
    "sorted": "list", "dict": "dict", "set": "set", "tuple": "tuple", "bool": "bool",
}


def _annotation_type(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return _ANNOTATION_TYPES.get(node.id)
    if isinstance(node, ast.Attribute):
        return _ANNOTATION_TYPES.get(node.attr)
    return None


def _value_type(value: object) -> str | None:
    for t in (bool, int, float, str, list, dict, tuple, set):
        if type(value) is t:
            return t.__name__
    return None


def _expr_type(node: ast.expr, known: dict[str, str | None]) -> str | None:
    if isinstance(node, ast.Constant):
        return _value_type(node.value)
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Name):
        return known.get(node.id)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return _CALL_TYPES.get(node.func.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            return "str"
    if isinstance(node, ast.BinOp):
        left, right = _expr_type(node.left, known), _expr_type(node.right, known)
        if left == right and left in ("int", "str", "list", "float") and not isinstance(node.op, ast.Div):
            return left
    return None


def infer_local_types(func: ast.FunctionDef, arg_values: list[list[object]] | None = None) -> dict[str, str]:
    """Best-effort type of each local variable; names with unknown or conflicting types are omitted."""
    param_seen: dict[str, set[str | None]] = {}
    params = [a.arg for a in func.args.posonlyargs + func.args.args]
    for a in func.args.posonlyargs + func.args.args + func.args.kwonlyargs:
        t = _annotation_type(a.annotation)
        if t:
            param_seen.setdefault(a.arg, set()).add(t)
    for values in arg_values or []:
        for name, value in zip(params, values):
            param_seen.setdefault(name, set()).add(_value_type(value))

    def resolved(seen: dict[str, set[str | None]]) -> dict[str, str]:
        return {n: next(iter(ts)) for n, ts in seen.items() if len(ts) == 1 and None not in ts}

    def tentative(seen: dict[str, set[str | None]]) -> dict[str, str]:
        out = {}
        for n, ts in seen.items():
            concrete = ts - {None}
            if len(concrete) == 1:
                out[n] = next(iter(concrete))
        return out

    known: dict[str, str | None] = dict(resolved(param_seen))
    seen: dict[str, set[str | None]] = {}
    # Optimistic fixpoint: passes evaluate bindings under tentative types (unknowns ignored) so
    # self-referential chains like `c = 0; c = c + 1` resolve; the final pass must be fully consistent.
    for _ in range(4):
        seen = {n: set(ts) for n, ts in param_seen.items()}

        def note(name: str, t: str | None) -> None:
            seen.setdefault(name, set()).add(t)

        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                t = _expr_type(node.value, known)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        note(target.id, t)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                note(node.target.id, _annotation_type(node.annotation) or (_expr_type(node.value, known) if node.value else None))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                note(node.target.id, _expr_type(node.value, known) if not isinstance(node.op, ast.Div) else "float")
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                it = node.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                    note(node.target.id, "int")
                elif _expr_type(it, known) == "str":
                    note(node.target.id, "str")
                else:
                    note(node.target.id, None)
            elif isinstance(node, (ast.For, ast.With, ast.AsyncFor)):
                targets = [node.target] if isinstance(node, ast.For) else [i.optional_vars for i in node.items]
                for t in targets:
                    for n in ast.walk(t) if t is not None else []:
                        if isinstance(n, ast.Name):
                            note(n.id, None)
        known = dict(tentative(seen))
    return resolved(seen)


def _first_binding_lines(func: ast.FunctionDef) -> dict[str, int]:
    lines: dict[str, int] = {}
    for a in func.args.posonlyargs + func.args.args + func.args.kwonlyargs:
        lines[a.arg] = 0
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            lines[node.id] = min(lines.get(node.id, node.lineno), node.lineno)
    return lines


def _comprehension_names(func: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.comprehension):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


def _nested_scope_nodes(func: ast.FunctionDef) -> set[int]:
    """ids of nodes inside nested functions, lambdas or comprehensions (other scopes)."""
    inside: set[int] = set()
    for node in ast.walk(func):
        if node is func:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for sub in ast.walk(node):
                if sub is not node:
                    inside.add(id(sub))
    return inside


# -- enumeration -------------------------------------------------------------------------------


def _specs(func: ast.FunctionDef, arg_values: list[list[object]] | None) -> Iterator[MutationSpec]:
    nodes = _walk(func)
    types = infer_local_types(func, arg_values)
    bind_lines = _first_binding_lines(func)
    comp_names = _comprehension_names(func)
    nested = _nested_scope_nodes(func)
    body_ids = {id(s) for s in func.body}
    docstring = func.body[0] if func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant) else None
    parent_of: dict[int, ast.AST] = {}
    for node in nodes:
        for child in ast.iter_child_nodes(node):
            parent_of[id(child)] = node

    for pos, node in enumerate(nodes):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _ARITH:
            yield MutationSpec("arith_swap", pos, f"{_SYMBOL[type(node.op)]}->{_SYMBOL[_ARITH[type(node.op)]]}")
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                if type(op) in _COMPARE:
                    yield MutationSpec("compare_swap", pos, f"{k}:{_SYMBOL[type(op)]}->{_SYMBOL[_COMPARE[type(op)]]}")
        if isinstance(node, ast.Constant) and type(node.value) is int and node is not docstring:
            yield MutationSpec("const_off_by_one", pos, "+1")
            yield MutationSpec("const_off_by_one", pos, "-1")
        parent = parent_of.get(id(node))
        if (
            isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "range"
            and node in parent.args and not isinstance(node, ast.Constant)
        ):
            yield MutationSpec("const_off_by_one", pos, "range+1")
            yield MutationSpec("const_off_by_one", pos, "range-1")
        if isinstance(node, ast.BoolOp):
            yield MutationSpec("bool_swap", pos, "and->or" if isinstance(node.op, ast.And) else "or->and")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield MutationSpec("bool_swap", pos, "drop_not")
        if (
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in types
            and node.id not in comp_names and id(node) not in nested
        ):
            for other in sorted(types):
                if (
                    other != node.id and types[other] == types[node.id] and other not in comp_names
                    and bind_lines.get(other, 10**9) < node.lineno
                ):
                    yield MutationSpec("var_swap", pos, f"{node.id}->{other}")
        if isinstance(node, ast.If):
            yield MutationSpec("remove_branch", pos, "drop_if")
            if node.orelse and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)):
                yield MutationSpec("remove_branch", pos, "drop_else")
        if isinstance(node, ast.Return) and id(node) not in body_ids and id(node) not in nested:
            yield MutationSpec("remove_branch", pos, "drop_return")


def _replace_stmt(parent_nodes: list[ast.AST], target: ast.stmt, replacement: list[ast.stmt]) -> bool:
    for parent in parent_nodes:
        for fld in ("body", "orelse", "finalbody", "handlers"):
            seq = getattr(parent, fld, None)
            if isinstance(seq, list) and any(s is target for s in seq):
                idx = next(i for i, s in enumerate(seq) if s is target)
                new = seq[:idx] + replacement + seq[idx + 1:]
                setattr(parent, fld, new or [ast.Pass()])
                return True
    return False


def _apply(func: ast.FunctionDef, spec: MutationSpec) -> int:
    """Apply ``spec`` to ``func`` in place; return the line of the mutated node."""
    nodes = _walk(func)
    node = nodes[spec.position]
    line = getattr(node, "lineno", func.lineno)
    t, v = spec.bug_type, spec.variant
    if t == "arith_swap":
        node.op = _ARITH[type(node.op)]()
    elif t == "compare_swap":
        k = int(v.split(":")[0])
        node.ops[k] = _COMPARE[type(node.ops[k])]()
    elif t == "const_off_by_one":
        if v in ("+1", "-1"):
            node.value = node.value + (1 if v == "+1" else -1)
        else:
            parent = next(p for p in nodes if isinstance(p, ast.Call) and any(a is node for a in p.args))
            i = next(i for i, a in enumerate(parent.args) if a is node)
            parent.args[i] = ast.BinOp(left=node, op=ast.Add() if v == "range+1" else ast.Sub(), right=ast.Constant(1))
    elif t == "bool_swap":
        if v == "drop_not":
            parent = next(p for p in nodes if any(c is node for c in ast.iter_child_nodes(p)))
            for fld, val in ast.iter_fields(parent):
                if val is node:
                    setattr(parent, fld, node.operand)
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        if item is node:
                            val[i] = node.operand
        else:
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif t == "var_swap":
        node.id = v.split("->")[1]
    elif t == "remove_branch":
        if v == "drop_if":
            ok = _replace_stmt(nodes, node, list(node.orelse))
        elif v == "drop_else":
            node.orelse = []
            ok = True
        else:
            ok = _replace_stmt(nodes, node, [])
        if not ok:
            raise ValueError(f"cannot apply {spec}")
    else:
        raise ValueError(f"unknown bug type {t}")
    return line


def apply_mutation(code: str, entry: str, spec: MutationSpec) -> Mutant:
    tree = ast.parse(code)
    func = find_function(tree, entry)
    line = _apply(func, spec)
    ast.fix_missing_locations(tree)
    new_code = ast.unparse(tree) + "\n"
    compile(new_code, "<mutant>", "exec")
    return Mutant(spec=spec, code=new_code, lineno=line)


def enumerate_mutants(code: str, entry: str, arg_values: list[list[object]] | None = None) -> list[Mutant]:
    """All syntactically valid mutants of ``entry`` that change the source text."""
    tree = ast.parse(code)
    func = find_function(tree, entry)
    original = ast.unparse(tree) + "\n"
    mutants = []
    seen = {original}
    for spec in _specs(func, arg_values):
        try:
            m = apply_mutation(code, entry, spec)
        except (SyntaxError, ValueError, StopIteration):
            continue
        if m.code in seen:
            continue
        seen.add(m.code)
        mutants.append(m)
    return mutants


def rename_names(code: str, mapping: dict[str, str]) -> str:
    """Rename module-level names (defs, classes, assignments and their uses) per ``mapping``."""
    if not mapping:
        return code
    tree = ast.parse(code)

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in mapping:
                node.id = mapping[node.id]
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            if node.name in mapping:
                node.name = mapping[node.name]
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
            if node.name in mapping:
                node.name = mapping[node.name]
            self.generic_visit(node)
            return node

    Renamer().visit(tree)
    return ast.unparse(tree) + "\n"
