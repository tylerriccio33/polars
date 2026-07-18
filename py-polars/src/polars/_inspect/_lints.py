"""The inspections. Each takes a `Plan` and yields `Finding`s.

These are maintainability lints, not correctness or performance lints: everything here
is something the optimizer already handles at execution time. The point is to tell the
*author* about it, so the source can be cleaned up.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from polars._inspect._plan import Expression, Plan

# with_columns blocks larger than this are hard to read as a single unit.
WIDE_BLOCK = 50
# Dependency chains deeper than this are hard to follow.
DEEP_CHAIN = 8
# A column read by more than this many expressions is a hub.
HIGH_FAN_OUT = 10
# Ignore common subexpressions smaller than this; inlining them is not a win.
MIN_CSE_SIZE = 3


@dataclass
class Finding:
    code: str
    title: str
    items: list[str] = field(default_factory=list)
    detail: str | None = None
    suggestion: str | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        if self.count is None:
            self.count = len(self.items)


Inspection = Callable[["Plan"], Iterator[Finding]]
_REGISTRY: list[Inspection] = []


def inspection(fn: Inspection) -> Inspection:
    _REGISTRY.append(fn)
    return fn


def run_all(plan: Plan) -> list[Finding]:
    return [w for fn in _REGISTRY for w in fn(plan)]


def live_columns(plan: Plan) -> set[str]:
    """Columns that actually affect the result.

    A column is live if it reaches the output, or if it feeds something else that is
    live. This has to be a fixpoint, not a single step: `mgr` feeding `mgr_upper` is
    not a real use if `mgr_upper` is itself dead.

    Expressions that are not definitions -- filter predicates, sort keys, join keys --
    affect *which rows* come out, so their inputs are always live even though they
    produce no column. Same for group-by keys and aggregates.
    """
    live = set(plan.output_columns)
    while True:
        grown = set(live)
        for e in plan.exprs:
            feeds_result = (
                not e.is_definition
                or e.kind == "GroupBy"
                or e.output_name in live
            )
            if feeds_result:
                grown |= e.inputs
        if grown == live:
            return live
        live = grown


def dependency_graph(plan: Plan) -> dict[str, set[str]]:
    """Derived column -> the columns it is computed from.

    Only the *last* definition of a name counts: if a column is shadowed, the earlier
    definition's inputs are not what the final value depends on. Bare projections
    (`select("a")`, which re-bind `a` to itself) are not definitions at all.
    """
    graph: dict[str, set[str]] = {}
    for e in sorted(plan.definitions, key=lambda e: e.stage):
        if e.size == 1 and e.inputs == {e.output_name}:
            continue  # a passthrough projection, not a computation
        graph[e.output_name] = e.inputs - {e.output_name}
    return graph


def depth_of(name: str, graph: dict[str, set[str]]) -> int:
    """Longest path from `name` back to a source column."""
    memo: dict[str, int] = {}

    def go(col: str, path: frozenset[str]) -> int:
        if col in memo:
            return memo[col]
        if col in path or col not in graph or not graph[col]:
            return 0
        depth = 1 + max(go(parent, path | {col}) for parent in graph[col])
        memo[col] = depth
        return depth

    return go(name, frozenset())


def chain_to_root(name: str, graph: dict[str, set[str]]) -> list[str]:
    """The single deepest dependency path starting at `name`."""
    chain = [name]
    seen = {name}
    current = name
    while graph.get(current):
        parents = [p for p in graph[current] if p not in seen]
        if not parents:
            break
        current = max(parents, key=lambda p: depth_of(p, graph))
        chain.append(current)
        seen.add(current)
    return chain


@inspection
def duplicate_expressions(plan: Plan) -> Iterator[Finding]:
    """Two derived columns computing the exact same thing under different names."""
    by_fp: dict[tuple, list[Expression]] = defaultdict(list)
    for e in plan.definitions:
        if e.size > 1:
            by_fp[e.fingerprint].append(e)

    groups = [
        exprs
        for exprs in by_fp.values()
        if len({e.output_name for e in exprs}) > 1
    ]
    if not groups:
        return
    items = []
    for exprs in groups:
        for e in sorted({e.output_name: e for e in exprs}.values(), key=lambda e: e.stage):
            items.append(f"{e.output_name} = {e.text}")
        items.append("")
    yield Finding(
        code="duplicate-expressions",
        title="Duplicate expressions",
        count=len(groups),
        items=items[:-1],
        suggestion="Compute once and reuse.",
    )


@inspection
def common_subexpressions(plan: Plan) -> Iterator[Finding]:
    """A non-trivial subexpression repeated across several definitions."""
    owners: dict[tuple, set[str]] = defaultdict(set)
    for e in plan.definitions:
        for fp, size in e.subexprs.items():
            if size >= MIN_CSE_SIZE and fp != e.fingerprint:
                owners[fp].add(e.output_name)

    shared = {fp: cols for fp, cols in owners.items() if len(cols) > 1}
    if not shared:
        return
    items = [
        f"{_describe_fp(fp)}  ({', '.join(sorted(cols))})"
        for fp, cols in sorted(shared.items(), key=lambda kv: -len(kv[1]))
    ]
    yield Finding(
        code="common-subexpressions",
        title="Common subexpressions",
        count=len(shared),
        items=items,
        detail="The same sub-computation appears in several columns.",
        suggestion="Factor into one intermediate column.",
    )


def _describe_fp(fp: tuple) -> str:
    """Best-effort rendering of a fingerprint (structure only, no names lost)."""
    if not isinstance(fp, tuple):
        return str(fp)
    if fp and fp[0] == "col":
        return str(fp[1])
    head, *rest = fp
    if head == "BinaryExpr" and len(rest) == 3:
        from polars._inspect._plan import binop_symbol

        left, right = _describe_fp(rest[1]), _describe_fp(rest[2])
        return f"{left} {binop_symbol(rest[0])} {right}"
    args = ", ".join(_describe_fp(r) for r in rest if isinstance(r, tuple))
    return f"{head}({args})" if args else str(head)


@inspection
def dead_columns(plan: Plan) -> Iterator[Finding]:
    """Derived columns that cannot affect the result.

    Liveness is transitive: a column read only by another dead column is itself dead.
    """
    live = live_columns(plan)
    dead = [name for name in sorted(plan.derived_columns) if name not in live]
    if dead:
        yield Finding(
            code="dead-columns",
            title="Dead columns",
            items=dead,
            detail="Never referenced after creation, and not in the output.",
            suggestion="Remove.",
        )


@inspection
def single_use_intermediates(plan: Plan) -> Iterator[Finding]:
    """Derived columns read exactly once and dropped before the output."""
    output = set(plan.output_columns)
    items = [
        name
        for name in sorted(plan.derived_columns)
        if name not in output and len(plan.uses_of(name)) == 1
    ]
    if items:
        yield Finding(
            code="single-use-intermediates",
            title="Single-use intermediates",
            items=items,
            detail="Created only to be immediately consumed.",
            suggestion="Inline, unless the name earns its keep as documentation.",
        )


@inspection
def unused_input_columns(plan: Plan) -> Iterator[Finding]:
    """Source columns that are neither read nor returned."""
    output = set(plan.output_columns)
    read = {col for e in plan.exprs for col in e.inputs}
    unused = sorted(plan.source_columns - read - output)
    if unused:
        yield Finding(
            code="unused-input-columns",
            title="Unused input columns",
            items=unused,
            detail="Read from the source but never used.",
            suggestion="Narrow the projection at the scan.",
        )


# Joins whose whole purpose is filtering; they contribute no columns by design.
_FILTER_ONLY_JOINS = frozenset({"Semi", "Anti"})

# What removing the join would do to the row count, per join type. An inner join that
# contributes no columns is still doing work -- it filters. A left join is not.
_ROW_EFFECT = {
    "Left": (
        "Contributes nothing, and a left join cannot drop rows.",
        'Remove the join. (If the right keys are not unique it is *adding* rows -- in '
        'that case you want `.unique()` on the right, not a join.)',
    ),
    "Inner": (
        "Contributes no columns, but an inner join still drops non-matching rows.",
        'You are joining to filter. Use `how="semi"`: same rows, no columns, no row '
        "duplication if the right keys repeat.",
    ),
    "Full": (
        "Contributes no columns, but a full join can still add unmatched rows.",
        'Probably `how="left"` or no join at all -- check whether the extra rows matter.',
    ),
    "Cross": (
        "Contributes no columns, and multiplies the row count.",
        "Almost certainly a mistake.",
    ),
}


@inspection
def pointless_joins(plan: Plan) -> Iterator[Finding]:
    """A join whose right-hand side contributes no column that anything reads.

    The whole right subtree -- scan, filters, aggregations and all -- is being computed
    to produce columns that are then thrown away.
    """
    live = live_columns(plan)
    for join in plan.joins:
        if join.how in _FILTER_ONLY_JOINS:
            continue

        # A contributed column counts only if it is *live* -- being read by a column
        # that is itself dead does not save the join.
        consumers = plan.ancestors(join.node)
        read_above = plan.columns_read_by(consumers) | set(plan.output_columns)
        used = join.contributed & live & read_above
        if used or not join.contributed:
            continue

        effect, suggestion = _ROW_EFFECT.get(
            join.how,
            ("Contributes no columns that anything reads.", "Check whether it is needed."),
        )
        items = [f"{join.how.lower()} join, contributing: {', '.join(sorted(join.contributed))}"]
        if join.right_sources:
            items.append("")
            items.append("The right-hand side is read only to be discarded:")
            items += [f"  {s}" for s in join.right_sources]

        yield Finding(
            code="pointless-join",
            title="Pointless join",
            count=1,
            items=items,
            detail=effect,
            suggestion=suggestion,
        )


@inspection
def unused_sources(plan: Plan) -> Iterator[Finding]:
    """A scan whose columns are never read and never reach the output."""
    output = set(plan.output_columns)
    read = {c for e in plan.exprs for c in e.inputs}
    items = []
    for node in plan.nodes.values():
        if node.source is None:
            continue
        if not (set(node.schema) & (read | output)):
            items.append(f"{node.source}  ({len(node.schema)} columns, none used)")
    if items:
        yield Finding(
            code="unused-source",
            title="Unused source",
            items=sorted(items),
            detail="Scanned, but not one of its columns is read or returned.",
            suggestion="Drop the source and whatever joins it in.",
        )


@inspection
def dependency_depth(plan: Plan) -> Iterator[Finding]:
    graph = dependency_graph(plan)
    if not graph:
        return
    deepest = max(graph, key=lambda c: depth_of(c, graph))
    depth = depth_of(deepest, graph)
    if depth <= DEEP_CHAIN:
        return
    chain = chain_to_root(deepest, graph)
    items = [
        f"{'     ' * i}{'└── ' if i else ''}{col}" for i, col in enumerate(chain)
    ]
    yield Finding(
        code="dependency-depth",
        title="Long dependency chain",
        count=1,
        items=items,
        detail=f"Depth: {depth}",
        suggestion="Consider splitting into stages.",
    )


@inspection
def high_fan_out(plan: Plan) -> Iterator[Finding]:
    counts = {
        name: len(plan.uses_of(name))
        for name in plan.derived_columns | plan.source_columns
    }
    hubs = sorted(
        ((n, c) for n, c in counts.items() if c > HIGH_FAN_OUT),
        key=lambda nc: -nc[1],
    )
    for name, count in hubs:
        yield Finding(
            code="high-fan-out",
            title="High fan-out",
            count=1,
            items=[name],
            detail=f"Used by {count} downstream expressions.",
            suggestion="A central calculation; may deserve its own section.",
        )


@inspection
def wide_blocks(plan: Plan) -> Iterator[Finding]:
    for _node, kind, n_exprs in plan.ir_nodes:
        if kind == "HStack" and n_exprs > WIDE_BLOCK:
            yield Finding(
                code="wide-block",
                title="Giant with_columns",
                count=1,
                items=[f"{n_exprs} expressions in one with_columns"],
                suggestion="Split into logically grouped blocks.",
            )


@inspection
def shadowed_aliases(plan: Plan) -> Iterator[Finding]:
    """A column redefined later without reading its previous value."""
    definitions: dict[str, list[Expression]] = defaultdict(list)
    for e in plan.definitions:
        definitions[e.output_name].append(e)

    items = []
    for name, defs in sorted(definitions.items()):
        if len(defs) < 2:
            continue
        for prev, curr in zip(defs, defs[1:]):
            if curr.stage == prev.stage:
                continue  # duplicate-aliases covers this
            if name in curr.inputs:
                continue  # a genuine update: x = x + 1
            used_between = any(
                name in e.inputs and prev.stage < e.stage < curr.stage
                for e in plan.exprs
            )
            if not used_between:
                items.append(f"{name} = {curr.text}  (overwrites {prev.text})")
    if items:
        yield Finding(
            code="shadowed-aliases",
            title="Shadowed aliases",
            items=items,
            detail="Overwritten before the previous value was ever read.",
            suggestion="Accidental overwrite, or a redundant first definition.",
        )


@inspection
def constant_expressions(plan: Plan) -> Iterator[Finding]:
    """Definitions that read no columns at all.

    Note that literal arithmetic (`lit(1.1) * lit(0.9)`) is folded during DSL -> IR
    conversion, so by the time we see it the whole expression is a single literal.
    We therefore report the resulting constant *column*, not the arithmetic.
    """
    items = [
        f"{e.output_name} = {e.text}" for e in plan.definitions if e.is_constant
    ]
    if items:
        yield Finding(
            code="constant-expressions",
            title="Constant expressions",
            items=sorted(set(items)),
            detail="Broadcast to every row but depend on no column.",
            suggestion="Move to a literal or config value, or apply after collection.",
        )


def stages(plan: Plan) -> list[list[str]]:
    """Group columns into dependency levels: everything in a stage is independent."""
    graph = dependency_graph(plan)
    known = set(plan.source_columns)
    for deps in graph.values():
        known |= {d for d in deps if d not in graph}

    levels: list[list[str]] = []
    if known:
        levels.append(sorted(known))

    placed = set(known)
    remaining = {c: set(d) for c, d in graph.items() if c not in placed}
    while remaining:
        ready = sorted(
            col for col, deps in remaining.items() if not (deps - placed) - {col}
        )
        if not ready:  # cycle: emit the rest and stop
            levels.append(sorted(remaining))
            break
        levels.append(ready)
        placed |= set(ready)
        for col in ready:
            del remaining[col]
    return levels
