"""Read an unoptimized LazyFrame plan into a column-level dependency graph.

Everything here works on the IR *as the user wrote it* (`_ldf.visit(optimized=False)`).
Running the optimizer first would defeat the purpose: CSE would collapse the duplicate
expressions we want to report, and projection pushdown would delete the dead columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polars.lazyframe.frame import LazyFrame

# Which fields of each expression node are references into the expression arena.
# `...` marks a variadic field (a list of child nodes).
_EXPR_CHILDREN: dict[str, tuple[tuple[str, bool], ...]] = {
    "Alias": (("expr", False),),
    "Column": (),
    "Literal": (),
    "Len": (),
    "BinaryExpr": (("left", False), ("right", False)),
    "Cast": (("expr", False),),
    "Sort": (("expr", False),),
    "Gather": (("expr", False), ("idx", False)),
    "Filter": (("input", False), ("by", False)),
    "SortBy": (("expr", False), ("by", True)),
    "Agg": (("arguments", True),),
    "Ternary": (("predicate", False), ("truthy", False), ("falsy", False)),
    "Function": (("input", True),),
    "Slice": (("input", False), ("offset", False), ("length", False)),
    "Window": (("function", False), ("partition_by", True), ("order_by", False)),
    "Rolling": (("function", False),),
}

# Fields that identify a node beyond its children, used to fingerprint expressions.
_EXPR_TAGS: dict[str, tuple[str, ...]] = {
    "Column": ("name",),
    "Literal": ("value", "dtype"),
    "BinaryExpr": ("op",),
    "Cast": ("dtype", "options"),
    "Sort": ("options",),
    "SortBy": ("sort_options",),
    "Agg": ("name",),
    "Function": ("function_data",),
    "Window": ("options",),
}

# IR nodes whose expressions bind new (or replacement) column names.
DEFINING_NODES = frozenset({"Select", "HStack", "Reduce", "GroupBy"})


@dataclass
class Expression:
    """One expression from the plan, flattened into a fingerprintable form."""

    node: int
    output_name: str
    kind: str
    """Name of the IR node the expression hangs off, e.g. ``HStack``."""
    stage: int
    """Index of the owning IR node in bottom-up execution order."""
    fingerprint: tuple[Any, ...]
    """Structural identity, ignoring the outermost alias."""
    inputs: set[str]
    """Columns this expression reads."""
    size: int
    """Number of expression nodes."""
    subexprs: dict[tuple[Any, ...], int]
    """Fingerprint -> size, for every non-trivial subexpression."""
    text: str
    is_definition: bool
    ir_node: int = -1
    """Id of the IR node this expression hangs off."""

    @property
    def is_constant(self) -> bool:
        return not self.inputs and self.kind != "GroupBy"


@dataclass
class Node:
    """One IR node."""

    id: int
    kind: str
    stage: int
    inputs: list[int]
    schema: list[str]
    n_exprs: int
    source: str | None = None
    """For scans, something identifying where the data comes from."""


@dataclass
class Join:
    """A join, and the columns it actually contributes."""

    node: int
    how: str
    left: int
    right: int
    contributed: set[str]
    """Columns present after the join but not before it: what the right side adds."""
    right_sources: list[str]
    """The scans feeding the right-hand side."""


@dataclass
class Plan:
    """A column-level view of a query plan."""

    exprs: list[Expression] = field(default_factory=list)
    nodes: dict[int, Node] = field(default_factory=dict)
    joins: list[Join] = field(default_factory=list)
    root: int = -1
    source_columns: set[str] = field(default_factory=set)
    output_columns: list[str] = field(default_factory=list)

    @property
    def ir_nodes(self) -> list[tuple[int, str, int]]:
        return [(n.id, n.kind, n.n_exprs) for n in self.nodes.values()]

    def ancestors(self, node: int) -> set[int]:
        """Every node that consumes `node`'s output, directly or transitively."""
        out: set[int] = set()
        stack = [self.root]
        path: list[int] = []

        def walk(current: int) -> bool:
            """True if `node` is reachable from `current`."""
            if current == node:
                return True
            found = False
            for child in self.nodes[current].inputs:
                if walk(child):
                    found = True
            if found:
                out.add(current)
            return found

        walk(self.root)
        out.discard(node)
        return out

    def columns_read_by(self, nodes: set[int]) -> set[str]:
        return {c for e in self.exprs if e.ir_node in nodes for c in e.inputs}

    @property
    def definitions(self) -> list[Expression]:
        return [e for e in self.exprs if e.is_definition]

    @property
    def derived_columns(self) -> set[str]:
        return {e.output_name for e in self.definitions} - self.source_columns

    def uses_of(self, name: str) -> list[Expression]:
        """Every expression that reads `name`, other than that column's own definition."""
        return [
            e
            for e in self.exprs
            if name in e.inputs and not (e.is_definition and e.output_name == name)
        ]


class _ExprReader:
    def __init__(self, nt: Any) -> None:
        self._nt = nt
        self._cache: dict[int, Any] = {}

    def view(self, node: int) -> Any:
        if node not in self._cache:
            self._cache[node] = self._nt.view_expression(node)
        return self._cache[node]

    def read(self, node: int) -> tuple[tuple[Any, ...], set[str], int, dict, str]:
        """Fingerprint an expression subtree, collecting inputs, size and subexpressions."""
        expr = self.view(node)
        kind = type(expr).__name__
        spec = _EXPR_CHILDREN.get(kind)

        inputs: set[str] = set()
        subexprs: dict[tuple[Any, ...], int] = {}
        size = 1
        child_prints: list[Any] = []
        child_texts: list[str] = []

        if spec is None:
            # An IR node we don't have a schema for. Fingerprint it opaquely by
            # identity so it never collides with anything, rather than guessing.
            return (kind, node), inputs, size, subexprs, kind

        for name, variadic in spec:
            value = getattr(expr, name)
            if value is None:
                child_prints.append(None)
                continue
            children = value if variadic else [value]
            prints = []
            for child in children:
                cp, cin, csize, csub, ctext = self.read(child)
                prints.append(cp)
                inputs |= cin
                size += csize
                subexprs.update(csub)
                child_texts.append(ctext)
                if csize > 1:
                    subexprs[cp] = csize
            child_prints.append(tuple(prints) if variadic else prints[0])

        tags = tuple(_tag(getattr(expr, t)) for t in _EXPR_TAGS.get(kind, ()))

        if kind == "Column":
            inputs.add(str(expr.name))
            return ("col", str(expr.name)), inputs, 1, subexprs, str(expr.name)
        if kind == "Alias":
            # An alias is a name, not a computation: fingerprint through it so that
            # `(price * rate).alias("tax")` and `.alias("tax_amt")` are recognised as
            # the same computation.
            return child_prints[0], inputs, size, subexprs, child_texts[0]

        fp = (kind, *tags, *child_prints)
        return fp, inputs, size, subexprs, _render(kind, tags, child_texts)

    def read_expr_ir(self, expr_ir: Any, kind: str, stage: int) -> Expression:
        fp, inputs, size, subexprs, text = self.read(expr_ir.node)
        return Expression(
            node=expr_ir.node,
            output_name=expr_ir.output_name,
            kind=kind,
            stage=stage,
            fingerprint=fp,
            inputs=inputs,
            size=size,
            subexprs=subexprs,
            text=text,
            is_definition=kind in DEFINING_NODES,
        )


def _tag(value: Any) -> Any:
    """Reduce an arbitrary option object to something hashable and comparable."""
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


_BINOP_SYMBOLS = {
    "Plus": "+",
    "Minus": "-",
    "Multiply": "*",
    "Divide": "/",
    "TrueDivide": "/",
    "Modulus": "%",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtEq": "<=",
    "Gt": ">",
    "GtEq": ">=",
    "And": "&",
    "Or": "|",
}


def binop_symbol(op: Any) -> str:
    """`Operator.Multiply` -> `*`."""
    name = str(op).rsplit(".", 1)[-1]
    return _BINOP_SYMBOLS.get(name, name)


def _parenthesize(child: str) -> str:
    """Parenthesize a compound child so precedence reads correctly."""
    return f"({child})" if " " in child.strip() else child


def _render(kind: str, tags: tuple[Any, ...], children: list[str]) -> str:
    """A short, readable rendering of an expression -- for report output only."""
    if kind == "Literal":
        return str(tags[0]) if tags else "lit"
    if kind == "BinaryExpr" and len(children) == 2:
        op = binop_symbol(tags[0])
        left, right = (_parenthesize(c) for c in children)
        return f"{left} {op} {right}"
    if kind == "Cast" and children:
        return f"{children[0]}.cast({tags[0]})"
    if kind == "Agg" and children:
        return f"{children[0]}.{str(tags[0]).lower()}()"
    if kind == "Ternary" and len(children) == 3:
        return f"when({children[0]}).then({children[1]}).otherwise({children[2]})"
    if kind == "Function":
        name = str(tags[0]) if tags else "fn"
        return f"{name}({', '.join(children)})"
    if kind == "Window" and children:
        return f"{children[0]}.over(...)"
    if kind == "Len":
        return "len()"
    if children:
        return f"{kind.lower()}({', '.join(children)})"
    return kind.lower()


def read_plan(lf: LazyFrame) -> Plan:
    """Walk the unoptimized IR of `lf` and collect its expressions."""
    nt = lf._ldf.visit(optimized=False)
    reader = _ExprReader(nt)
    plan = Plan()

    root = nt.get_node()
    nt.set_node(root)
    plan.output_columns = list(nt.get_schema().keys())

    # Post-order walk: a node's inputs are visited before the node itself, so
    # `stage` increases in execution order and definitions always precede their uses.
    order: list[int] = []
    seen: set[int] = set()

    def walk(node: int) -> None:
        if node in seen:
            return
        seen.add(node)
        nt.set_node(node)
        for child in nt.get_inputs():
            walk(child)
        order.append(node)

    walk(root)

    plan.root = root

    for stage, node in enumerate(order):
        nt.set_node(node)
        ir = nt.view_current_node()
        kind = type(ir).__name__
        exprs = nt.get_exprs()
        schema = list(nt.get_schema().keys())
        inputs = list(nt.get_inputs())

        source = None
        if kind in SCAN_NODES:
            plan.source_columns |= set(schema)
            source = _scan_source(ir, kind)

        plan.nodes[node] = Node(
            id=node,
            kind=kind,
            stage=stage,
            inputs=inputs,
            schema=schema,
            n_exprs=len(exprs),
            source=source,
        )
        for expr_ir in exprs:
            e = reader.read_expr_ir(expr_ir, kind, stage)
            e.ir_node = node
            plan.exprs.append(e)

    for node in order:
        info = plan.nodes[node]
        if info.kind != "Join":
            continue
        left, right = info.inputs[0], info.inputs[1]
        nt.set_node(node)
        how = str(nt.view_current_node().options[0])
        # What the right-hand side adds: the columns that exist after the join but
        # not before it. Reading it this way means coalesced keys and suffixed
        # collisions come out right without special-casing either.
        contributed = set(info.schema) - set(plan.nodes[left].schema)
        plan.joins.append(
            Join(
                node=node,
                how=how,
                left=left,
                right=right,
                contributed=contributed,
                right_sources=_sources_under(plan, right),
            )
        )

    nt.set_node(root)
    return plan


SCAN_NODES = frozenset({"Scan", "DataFrameScan", "PythonScan"})


def _scan_source(ir: Any, kind: str) -> str:
    """Best-effort identification of where a scan reads from."""
    paths = getattr(ir, "paths", None)
    if paths:
        return ", ".join(str(p) for p in paths)
    return {"DataFrameScan": "in-memory frame", "PythonScan": "python source"}.get(
        kind, "scan"
    )


def _sources_under(plan: Plan, node: int) -> list[str]:
    """The scans in the subtree rooted at `node`."""
    out: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        info = plan.nodes[current]
        if info.source is not None:
            out.append(info.source)
        stack.extend(info.inputs)
    return out
