"""Assemble validated bugs into small multi-module packages and write the task set.

Usage: ``uv run python -m arl.taskgen.build --raw data/raw --out data/tasks``
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from arl.env.task import VISIBLE_TEST_PATH, HiddenCase, Task
from arl.taskgen.augment import augment_function
from arl.taskgen.dedup import MAX_MOVED_FRACTION, clusters, near_duplicates
from arl.taskgen.mutate import BUG_TYPES, enumerate_mutants, rename_names
from arl.taskgen.sources import RAW_FILES, Case, SourceFunction, load_sources, sha256_file
from arl.taskgen.validate import FunctionScreen, MutantVerdict, count_reasons, screen_function, validate_task

BUILTIN_NAMES = frozenset(dir(builtins))
MODULE_NAMES = ("core", "utils", "helpers", "ops", "tools", "misc", "base", "common", "extras", "funcs", "routines", "shared")
HELDOUT_FRACTION = 0.3
MODULES_RANGE = (3, 5)
FUNCTIONS_RANGE = (2, 4)
MAX_PER_FUNCTION_BUG = 2
MAX_CLUSTER_SIZE = 30
MIN_BUG_HIDDEN = 5
PYTEST_INI = "[pytest]\ntestpaths = tests\n"
ACTUAL_MAX_CHARS = 120
ISSUE_PART_MAX_CHARS = 300

ISSUE_TEMPLATES = {
    "returns_a": "`{call}` returns `{actual}`, but it should return `{expected}`.",
    "returns_b": "Calling `{call}` gives `{actual}` instead of `{expected}`.",
    "returns_c": "`{call}` should return `{expected}`, but it returns `{actual}`.",
    "raises": "`{call}` raises `{exc}`, but it should return `{expected}`.",
    "wrong_no_actual": "`{call}` returns a wrong result; the expected result is `{expected}`.",
    "long_input": "`{name}` returns a wrong result for the input used in its visible test.",
}


class BuildStopped(RuntimeError):
    """A condition the plan says must be reported instead of worked around."""


# -- split ----------------------------------------------------------------------------------------


def split_clusters(cluster_list: list[list[str]], seed: int, heldout_fraction: float = HELDOUT_FRACTION) -> dict[str, str]:
    """Map source_id -> "train" | "heldout", whole clusters at a time.

    Clusters are shuffled with ``split-{seed}`` and added to heldout until it holds at least
    ``heldout_fraction`` of all functions (it may overshoot by less than one cluster).
    """
    order = sorted(cluster_list, key=lambda c: c[0])
    random.Random(f"split-{seed}").shuffle(order)
    target = heldout_fraction * sum(len(c) for c in order)
    out: dict[str, str] = {}
    held = 0
    for c in order:
        split = "heldout" if held < target else "train"
        if split == "heldout":
            held += len(c)
        for sid in c:
            out[sid] = split
    return out


def split_functions(funcs: list[SourceFunction], seed: int, heldout_fraction: float = HELDOUT_FRACTION) -> dict[str, str]:
    """Cluster near-duplicate functions, then split by cluster. Fixed by seed, decided before any mutation."""
    return split_clusters(clusters(funcs), seed, heldout_fraction)


def check_split(funcs: list[SourceFunction], split_of: dict[str, str]) -> list[dict]:
    """Near duplicates left in heldout after the cluster split; by construction there should be none."""
    held = [f for f in funcs if split_of[f.source_id] == "heldout"]
    train = [f for f in funcs if split_of[f.source_id] == "train"]
    return near_duplicates(held, train)


def check_no_leakage(tasks_by_split: dict[str, list[Task]]) -> None:
    """Raise if any source function appears (as bug or distractor) in more than one split."""
    owner: dict[str, str] = {}
    for split, tasks in tasks_by_split.items():
        for t in tasks:
            for sid in t.meta["source_ids"]:
                if owner.setdefault(sid, split) != split:
                    raise ValueError(f"source function {sid} appears in both {owner[sid]} and {split}")


# -- selection ------------------------------------------------------------------------------------


def candidate_slots(screen: FunctionScreen, seed: int, cap: int = MAX_PER_FUNCTION_BUG) -> dict[str, list[MutantVerdict]]:
    """Per bug type, valid mutants in a seeded order, one per position (positions must differ)."""
    by_type: dict[str, dict[int, list[MutantVerdict]]] = defaultdict(lambda: defaultdict(list))
    for v in screen.verdicts:
        if v.ok:
            by_type[v.mutant.spec.bug_type][v.mutant.spec.position].append(v)
    out = {}
    for bug_type, by_pos in sorted(by_type.items()):
        rng = random.Random(f"select-{seed}-{screen.fn.source_id}-{bug_type}")
        positions = sorted(by_pos)
        rng.shuffle(positions)
        out[bug_type] = [rng.choice(by_pos[p]) for p in positions]
    return out


# -- package assembly -------------------------------------------------------------------------------


@dataclass
class Placed:
    fn: SourceFunction
    module: str
    name: str  # entry name after renaming
    code: str  # renamed original code
    mapping: dict[str, str]
    visible_index: int = 0


def _top_level(code: str) -> tuple[list[str], list[str], set[str], set[str]]:
    """(import statements, other statements, names bound by imports, names defined otherwise)."""
    tree = ast.parse(code)
    imports, others = [], []
    import_names, defined = set(), set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(stmt))
            for alias in stmt.names:
                import_names.add(alias.asname or alias.name.split(".")[0])
        else:
            others.append(ast.unparse(stmt))
            if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
                defined.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name):
                            defined.add(n.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
    return imports, others, import_names, defined


def _module_source(parts: list[str]) -> str:
    imports: list[str] = []
    bodies: list[str] = []
    for code in parts:
        imp, other, _, _ = _top_level(code)
        imports += [i for i in imp if i not in imports]
        bodies.append("\n\n\n".join(other))
    head = "\n".join(imports)
    return (head + "\n\n\n" if head else "") + "\n\n\n".join(bodies) + "\n"


class PackageBuilder:
    """Places functions into modules, renaming top-level names so they are unique package-wide."""

    def __init__(self) -> None:
        self.placed: list[Placed] = []
        self.defined: set[str] = set()
        self.module_imports: dict[str, dict[str, str]] = defaultdict(dict)  # module -> name -> import stmt
        self.renames = 0

    def try_place(self, fn: SourceFunction, module: str) -> Placed | None:
        imports, _, import_names, defined = _top_level(fn.code)
        bound = self.module_imports[module]
        stmt_for = {}
        for stmt in imports:
            tree = ast.parse(stmt).body[0]
            for alias in tree.names:
                stmt_for[alias.asname or alias.name.split(".")[0]] = stmt
        for name in import_names:
            other = bound.get(name)
            if other is not None and other != stmt_for[name]:
                return None  # same name imported differently in this module
            if name in self.defined:
                return None
        mapping = {}
        # Builtins count as taken: a source defining `sum` would shadow the builtin for its neighbours.
        taken = self.defined | set(bound) | import_names | BUILTIN_NAMES
        for name in sorted(defined):
            if name in taken:
                k = 2
                while f"{name}_{k}" in taken or f"{name}_{k}" in defined:
                    k += 1
                mapping[name] = f"{name}_{k}"
                taken.add(mapping[name])
        code = rename_names(fn.code, mapping)
        self.defined |= {mapping.get(n, n) for n in defined}
        for name in import_names:
            bound[name] = stmt_for[name]
        self.renames += len(mapping)
        placed = Placed(fn=fn, module=module, name=mapping.get(fn.entry, fn.entry), code=code, mapping=mapping)
        self.placed.append(placed)
        return placed


def _visible_call(p: Placed, case: Case) -> str:
    # Module-qualified, so source functions named test_* are not collected by pytest.
    return f"{p.module}.{case.call(p.name)}"


def _visible_assert(p: Placed, case: Case) -> str:
    call = _visible_call(p, case)
    if case.is_float:
        return f"    assert math.isclose({call}, {case.expected}, rel_tol=1e-06, abs_tol=1e-09)"
    return f"    assert {call} == {case.expected}"


def _issue(bug: Placed, case: Case, actual: str, raised: str | None, rng: random.Random) -> tuple[str, str]:
    call = case.call(bug.name)
    if len(call) > ISSUE_PART_MAX_CHARS or len(case.expected) > ISSUE_PART_MAX_CHARS:
        key = "long_input"
    elif raised:
        key = "raises"
    elif len(actual) > ACTUAL_MAX_CHARS or actual.startswith("<"):
        key = "wrong_no_actual"
    else:
        key = rng.choice(["returns_a", "returns_b", "returns_c"])
    text = ISSUE_TEMPLATES[key].format(call=call, actual=actual, expected=case.expected, exc=raised, name=bug.name)
    return text, key


def assemble_task(
    verdict: MutantVerdict, bug_src: SourceFunction, pool: list[SourceFunction], split: str, task_id: str, seed: int,
    modules_range: tuple[int, int] = MODULES_RANGE, functions_range: tuple[int, int] = FUNCTIONS_RANGE,
) -> tuple[Task | None, str]:
    """Build one task around ``verdict`` (a mutant of ``bug_src``) with distractors from ``pool``."""
    rng = random.Random(f"assemble-{seed}-{task_id}")
    n_modules = rng.randint(*modules_range)
    sizes = [rng.randint(*functions_range) for _ in range(n_modules)]
    modules = rng.sample(MODULE_NAMES, n_modules)
    slots = [m for m, size in zip(modules, sizes) for _ in range(size)]
    rng.shuffle(slots)

    builder = PackageBuilder()
    bug_module = slots[0]
    bug = builder.try_place(bug_src, bug_module)
    assert bug is not None
    candidates = [f for f in pool if f.source_id != bug_src.source_id]
    rng.shuffle(candidates)
    rejected = 0
    for module in slots[1:]:
        while candidates:
            fn = candidates.pop()
            if builder.try_place(fn, module):
                break
            rejected += 1
        else:
            return None, "not_enough_distractors"

    # Module order inside a file is the placement order; shuffle it so the bug is not always first.
    by_module: dict[str, list[Placed]] = defaultdict(list)
    for p in builder.placed:
        by_module[p.module].append(p)
    for items in by_module.values():
        rng.shuffle(items)

    mutant_code = rename_names(verdict.mutant.code, bug.mapping)
    outcomes = verdict.outcomes
    bug.visible_index = next(i for i, o in enumerate(outcomes) if not o.passed and bug_src.cases[i].kind == "test")

    files: dict[str, str] = {"pkg/__init__.py": "", "pytest.ini": PYTEST_INI}
    solution: dict[str, str] = {}
    for module in sorted(by_module):
        items = by_module[module]
        original = _module_source([p.code for p in items])
        shipped = _module_source([mutant_code if p is bug else p.code for p in items])
        files[f"pkg/{module}.py"] = shipped
        if module == bug_module:
            solution[f"pkg/{module}.py"] = original

    ordered = [p for module in sorted(by_module) for p in by_module[module]]
    uses_math = any(p.fn.cases[p.visible_index].is_float for p in ordered)
    lines = ["import math", ""] if uses_math else []
    lines.append(f"from pkg import {', '.join(sorted(by_module))}")
    for p in ordered:
        lines += ["", "", f"def test_{p.name}():", _visible_assert(p, p.fn.cases[p.visible_index])]
    files[VISIBLE_TEST_PATH] = "\n".join(lines) + "\n"

    hidden: list[HiddenCase] = []
    hidden_kinds: list[str] = []
    functions_meta = []
    for p in ordered:
        indices = []
        for i, case in enumerate(p.fn.cases):
            if i == p.visible_index:
                continue
            indices.append(len(hidden))
            hidden.append(HiddenCase(call=f"pkg.{p.module}.{case.call(p.name)}", check=case.check()))
            hidden_kinds.append(case.kind)
        functions_meta.append(
            {
                "name": p.name,
                "module": p.module,
                "source_id": p.fn.source_id,
                "renamed": p.mapping,
                "visible_call": _visible_call(p, p.fn.cases[p.visible_index]),
                "hidden_case_indices": indices,
            }
        )

    visible_case = bug_src.cases[bug.visible_index]
    out = outcomes[bug.visible_index]
    issue, template = _issue(bug, visible_case, out.actual, out.raised, rng)
    shipped_lines = files[f"pkg/{bug_module}.py"].splitlines()
    original_lines = solution[f"pkg/{bug_module}.py"].splitlines()
    bug_line = next(
        (n for n, (a, b) in enumerate(zip(shipped_lines, original_lines), 1) if a != b),
        min(len(shipped_lines), len(original_lines)) + 1,
    )
    spec = verdict.mutant.spec
    bug_hidden = [i for i in range(len(bug_src.cases)) if i != bug.visible_index]
    bug_hidden_counts = dict(Counter(bug_src.cases[i].kind for i in bug_hidden))
    bug_hidden_failing = dict(Counter(bug_src.cases[i].kind for i in bug_hidden if not outcomes[i].passed))
    task = Task(
        task_id=task_id,
        issue=issue,
        files=files,
        hidden_setup="\n".join(f"import pkg.{m}" for m in sorted(by_module)),
        hidden_cases=hidden,
        solution=solution,
        meta={
            "split": split,
            "bug_type": spec.bug_type,
            "bug_variant": spec.variant,
            "bug_position": spec.position,
            "bug_file": f"pkg/{bug_module}.py",
            "bug_function": bug.name,
            "bug_line": bug_line,
            "bug_source": bug_src.source_id,
            "source_ids": sorted(p.fn.source_id for p in builder.placed),
            "functions": functions_meta,
            "hidden_case_kinds": hidden_kinds,
            "bug_hidden_counts": bug_hidden_counts,
            "bug_hidden_failing": bug_hidden_failing,
            "issue_template": template,
            "n_modules": len(by_module),
            "n_functions": len(builder.placed),
            "renames": builder.renames,
            "distractors_rejected": rejected,
        },
    )
    return task, "ok"


# -- pipeline ---------------------------------------------------------------------------------------


def _arg_values(fn: SourceFunction) -> list[list[object]]:
    """Positional argument values of each case (for var_swap type inference)."""
    values = []
    for c in fn.cases:
        try:
            values.append(list(ast.literal_eval(f"({c.args},)")))
        except (ValueError, SyntaxError):
            values.append([])
    return values


def build(raw_dir: Path, out_dir: Path, seed: int = 0, workers: int = 8, cap: int = MAX_PER_FUNCTION_BUG,
          limit_functions: int | None = None, log=print) -> dict:
    funcs, source_stats = load_sources(raw_dir)
    funcs.sort(key=lambda f: f.source_id)
    if limit_functions:
        funcs = random.Random(seed).sample(funcs, limit_functions)
        funcs.sort(key=lambda f: f.source_id)
    manifest = build_from_functions(funcs, out_dir, seed=seed, workers=workers, cap=cap, log=log)
    manifest["config"]["limit_functions"] = limit_functions
    manifest["discards"]["assertion_split"] = source_stats.to_dict()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_from_functions(funcs: list[SourceFunction], out_dir: Path, seed: int = 0, workers: int = 8,
                         cap: int = MAX_PER_FUNCTION_BUG, log=print,
                         modules_range: tuple[int, int] = MODULES_RANGE,
                         functions_range: tuple[int, int] = FUNCTIONS_RANGE, augment: bool = True) -> dict:
    t0 = time.monotonic()
    cluster_list = clusters(funcs)
    largest = max(len(c) for c in cluster_list)
    if largest > MAX_CLUSTER_SIZE:
        raise BuildStopped(f"largest near-duplicate cluster has {largest} functions (> {MAX_CLUSTER_SIZE})")
    split_of = split_clusters(cluster_list, seed)
    n_held = sum(1 for s in split_of.values() if s == "heldout")
    leftover = check_split(funcs, split_of)
    if leftover:
        raise BuildStopped(f"{len(leftover)} near duplicates remain after the cluster split: {leftover[:5]}")
    if n_held and len(leftover) > MAX_MOVED_FRACTION * n_held:
        raise BuildStopped("near duplicates above the 10% backstop")
    log(f"[build] {len(funcs)} source functions in {len(cluster_list)} clusters (largest {largest}), {n_held} held out")

    aug_stats: Counter = Counter()
    if augment:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(augment_function, funcs))
        funcs = [r.fn for r in results]
        for r in results:
            aug_stats.update(r.stats)
        log(f"[build] augmented hidden cases in {time.monotonic() - t0:.0f}s: {aug_stats['generated']} generated")
    t_augment = time.monotonic() - t0

    def screen(fn: SourceFunction) -> FunctionScreen:
        return screen_function(fn, enumerate_mutants(fn.code, fn.entry, _arg_values(fn)))

    with ThreadPoolExecutor(workers) as ex:
        screens = list(ex.map(screen, funcs))
    t_screen = time.monotonic() - t0 - t_augment
    log(f"[build] screened in {t_screen:.0f}s")

    function_discards = Counter(s.original_reason for s in screens if not s.original_ok)
    mutant_reasons: Counter = Counter()
    generated: Counter = Counter()
    for s in screens:
        mutant_reasons.update(count_reasons(s.verdicts))
        generated.update(v.mutant.spec.bug_type for v in s.verdicts)
    usable = [s for s in screens if s.original_ok]
    pools = {split: [s.fn for s in usable if split_of[s.fn.source_id] == split] for split in ("train", "heldout")}

    # Slots: per (function, bug type) up to `cap` tasks at distinct positions. Candidates beyond
    # the first `cap` are replacements used only if a selected one fails full validation.
    queue: list[tuple[str, SourceFunction, str, list[MutantVerdict]]] = []
    task_reasons: Counter = Counter()
    for s in usable:
        for bug_type, cands in candidate_slots(s, seed, cap).items():
            if len(s.fn.cases) - 1 < MIN_BUG_HIDDEN:
                task_reasons[f"{bug_type}:too_few_hidden_cases"] += min(cap, len(cands))
                continue
            queue.append((split_of[s.fn.source_id], s.fn, bug_type, cands))

    accepted: dict[str, list[Task]] = {"train": [], "heldout": []}
    grade_times: list[float] = []
    by_id = {f.source_id: f for f in funcs}

    def attempt(item: tuple[str, SourceFunction, str, list[MutantVerdict]]) -> tuple[list[Task], Counter, list[float]]:
        split, fn, bug_type, cands = item
        reasons: Counter = Counter()
        times: list[float] = []
        kept: list[Task] = []
        for v in cands:
            if len(kept) >= cap:
                break
            task_id = f"{split}-{fn.source_id.replace('/', '-')}-{bug_type}-{v.mutant.spec.position}"
            task, why = assemble_task(v, fn, pools[split], split, task_id, seed, modules_range, functions_range)
            if task is None:
                reasons[f"{bug_type}:{why}"] += 1
                continue
            verdict = validate_task(task)
            times += verdict.grade_seconds
            reasons[f"{bug_type}:{verdict.reason}"] += 1
            if verdict.ok:
                kept.append(task)
        return kept, reasons, times

    t1 = time.monotonic()
    with ThreadPoolExecutor(workers) as ex:
        for kept, reasons, times in ex.map(attempt, queue):
            task_reasons.update(reasons)
            grade_times += times
            for t in kept:
                accepted[t.meta["split"]].append(t)
    t_validate = time.monotonic() - t1
    log(f"[build] assembled and validated in {t_validate:.0f}s")

    for split in accepted:
        accepted[split].sort(key=lambda t: t.task_id)
    check_no_leakage(accepted)

    out_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for split, tasks in accepted.items():
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(t.to_json() + "\n")
        file_hashes[path.name] = sha256_file(path)

    def per_type(tasks: list[Task]) -> dict[str, int]:
        c = Counter(t.meta["bug_type"] for t in tasks)
        return {b: c.get(b, 0) for b in BUG_TYPES}

    all_tasks = accepted["train"] + accepted["heldout"]
    final = Counter(t.meta["bug_type"] for t in all_tasks)
    pass_rates = {}
    for b in BUG_TYPES:
        screened_ok = mutant_reasons.get(f"{b}:ok", 0)
        attempted = sum(v for k, v in task_reasons.items() if k.startswith(f"{b}:") and not k.endswith(":too_few_hidden_cases"))
        validated_ok = task_reasons.get(f"{b}:ok", 0)
        pass_rates[b] = {
            "mutants_generated": generated.get(b, 0),
            "mutants_passing_screening": screened_ok,
            "screening_pass_rate": round(screened_ok / generated[b], 3) if generated.get(b) else None,
            "tasks_validated": attempted,
            "tasks_passing_validation": validated_ok,
            "validation_pass_rate": round(validated_ok / attempted, 3) if attempted else None,
            "final_tasks": final.get(b, 0),
        }

    def dist(values: list[int]) -> dict[str, int]:
        return {str(k): v for k, v in sorted(Counter(values).items())}

    def hidden_of(t: Task, kinds: tuple[str, ...]) -> int:
        return sum(v for k, v in t.meta["bug_hidden_counts"].items() if k in kinds)

    gen_total = sum(t.meta["bug_hidden_counts"].get("generated", 0) for t in all_tasks)
    gen_failing = sum(t.meta["bug_hidden_failing"].get("generated", 0) for t in all_tasks)
    sizes = [len(c) for c in cluster_list]
    origin_of = {f.source_id: f.origin for f in funcs}
    manifest = {
        "seed": seed,
        "config": {
            "heldout_fraction": HELDOUT_FRACTION,
            "max_per_function_bug": cap,
            "case_timeout_s": 1.0,
            "modules_per_task": list(modules_range),
            "functions_per_module": list(functions_range),
            "min_bug_hidden_cases": MIN_BUG_HIDDEN,
            "max_cluster_size": MAX_CLUSTER_SIZE,
            "augment": augment,
        },
        "raw_files": {name: {"url": spec["url"], "sha256": spec["sha256"]} for name, spec in RAW_FILES.items()},
        "counts": {split: len(tasks) for split, tasks in accepted.items()} | {"total": len(all_tasks)},
        "bug_types": {split: per_type(tasks) for split, tasks in accepted.items()},
        "pass_rate_by_bug_type": pass_rates,
        "source_functions": {
            "loaded": len(funcs),
            "loaded_by_origin": dict(sorted(Counter(f.origin for f in funcs).items())),
            "usable_after_screening": len(usable),
            "usable_by_origin": dict(sorted(Counter(s.fn.origin for s in usable).items())),
            "by_split": {split: len(pool) for split, pool in pools.items()},
            "by_split_and_origin": {split: dict(sorted(Counter(f.origin for f in pool).items())) for split, pool in pools.items()},
        },
        "clustering": {
            "jaccard_threshold": 0.8,
            "clusters": len(cluster_list),
            "cluster_size_distribution": dist(sizes),
            "largest_5": [
                {"size": len(c), "split": split_of[c[0]], "members": [f"{sid} {by_id[sid].entry}" for sid in c]}
                for c in sorted(cluster_list, key=lambda c: (-len(c), c[0]))[:5]
            ],
            "heldout_functions": n_held,
            "near_duplicates_after_split": len(leftover),
        },
        "augmentation": {
            "stats": dict(sorted(aug_stats.items())),
            "bug_hidden_cases_distribution": {
                "test_only": dist([hidden_of(t, ("test",)) for t in all_tasks]),
                "test_and_challenge": dist([hidden_of(t, ("test", "challenge")) for t in all_tasks]),
                "after_augmentation": dist([hidden_of(t, ("test", "challenge", "generated")) for t in all_tasks]),
            },
            "bug_generated_cases": gen_total,
            "bug_generated_cases_failing_on_buggy": gen_failing,
            "bug_generated_failing_fraction": round(gen_failing / gen_total, 3) if gen_total else None,
            "tasks_where_some_generated_case_fails": sum(1 for t in all_tasks if t.meta["bug_hidden_failing"].get("generated")),
        },
        "bug_sources_by_origin": {
            split: dict(sorted(Counter(origin_of[t.meta["bug_source"]] for t in tasks).items())) for split, tasks in accepted.items()
        },
        "distinct_bug_functions": {split: len({t.meta["bug_source"] for t in tasks}) for split, tasks in accepted.items()},
        "issue_templates": dict(Counter(t.meta["issue_template"] for t in all_tasks)),
        "discards": {
            "function_screening": dict(function_discards),
            "mutant_generated": {b: generated.get(b, 0) for b in BUG_TYPES},
            "mutant_screening": dict(sorted(mutant_reasons.items())),
            "task_validation": dict(sorted(task_reasons.items())),
        },
        "timing_s": {
            "augmentation": round(t_augment, 1),
            "screening": round(t_screen, 1),
            "assembly_and_validation": round(t_validate, 1),
            "grade_median": round(sorted(grade_times)[len(grade_times) // 2], 3) if grade_times else None,
        },
        "files": file_hashes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/tasks"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-functions", type=int, default=None)
    ap.add_argument("--download", action="store_true", help="download and verify the raw files first")
    args = ap.parse_args(argv)
    if args.download:
        from arl.taskgen.sources import download_raw

        download_raw(args.raw)
    try:
        manifest = build(args.raw, args.out, seed=args.seed, workers=args.workers, limit_functions=args.limit_functions)
    except BuildStopped as e:
        print(f"[build] STOPPED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
