"""Report object returned by `pl.inspect`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars._inspect._lints import (
    Finding,
    dependency_graph,
    run_all,
    stages,
)
from polars._inspect._plan import read_plan

if TYPE_CHECKING:
    from polars.lazyframe.frame import LazyFrame
    from polars._inspect._plan import Plan

RULE = "-" * 60


class Report:
    """The result of statically analysing a `LazyFrame`.

    Nothing here changes what the query computes; these are readability findings
    about the plan as written.
    """

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.warnings: list[Finding] = run_all(plan)

    @property
    def n_nodes(self) -> int:
        return len(self._plan.ir_nodes)

    @property
    def n_exprs(self) -> int:
        return len(self._plan.exprs)

    @property
    def n_derived(self) -> int:
        return len(self._plan.derived_columns)

    def summary(self) -> str:
        graph = dependency_graph(self._plan)
        lines = [
            "Pipeline",
            "========",
            "",
            f"Nodes:               {self.n_nodes:>5}",
            f"Expressions:         {self.n_exprs:>5}",
            f"Derived columns:     {self.n_derived:>5}",
            f"Output columns:      {len(self._plan.output_columns):>5}",
            f"Warnings:            {len(self.warnings):>5}",
        ]
        if graph:
            from polars._inspect._lints import depth_of

            deepest = max(graph, key=lambda c: depth_of(c, graph))
            lines.append(f"Max chain depth:     {depth_of(deepest, graph):>5}")
        return "\n".join(lines)

    def warnings_report(self) -> str:
        if not self.warnings:
            return "Warnings\n========\n\nNone. Nothing to clean up.\n"

        blocks = []
        for w in self.warnings:
            body = [f"⚠ {w.title} ({w.count})", ""]
            body += [f"    {item}" if item else "" for item in w.items]
            if w.detail:
                body += ["", w.detail]
            if w.suggestion:
                body += ["", "Suggestion:", f"    {w.suggestion}"]
            blocks.append("\n".join(body))
        return "Warnings\n========\n\n" + f"\n\n{RULE}\n\n".join(blocks) + "\n"

    def stages(self) -> list[list[str]]:
        """Columns grouped into dependency levels; each level is independent."""
        return stages(self._plan)

    def stages_report(self) -> str:
        out = ["Stages", "======", ""]
        for i, level in enumerate(self.stages(), start=1):
            out += [f"Stage {i}", "-" * 8, ""]
            out += [f"  {col}" for col in level]
            out.append("")
        return "\n".join(out)

    def graph(self) -> str:
        """An ASCII dependency graph of the derived columns."""
        deps = dependency_graph(self._plan)
        if not deps:
            return ""
        out = ["Dependencies", "============", ""]
        for col in sorted(deps):
            parents = sorted(deps[col])
            if not parents:
                continue
            out.append(f"  {col}")
            for i, parent in enumerate(parents):
                branch = "└──" if i == len(parents) - 1 else "├──"
                marker = " *" if deps.get(parent) else ""
                out.append(f"    {branch} {parent}{marker}")
            out.append("")
        out.append("  (* = itself derived)")
        return "\n".join(out)

    def render(self) -> str:
        """The full report."""
        parts = [self.summary(), self.warnings_report(), self.stages_report(), self.graph()]
        return "\n\n".join(p for p in parts if p)

    def __repr__(self) -> str:
        return self.render()


def inspect(lf: LazyFrame) -> Report:
    """Statically analyse a `LazyFrame` and report on its readability.

    This is a linter, not an optimizer pass. It reads the plan *as written*, before
    any optimization, and reports things a human would want to clean up in the source:
    duplicated expressions, dead columns, single-use temporaries, deep dependency
    chains, and so on. The query itself is unchanged, and the optimizer would have
    handled most of these at execution time anyway.

    Examples
    --------
    >>> report = pl.inspect(lf)  # doctest: +SKIP
    >>> print(report.summary())  # doctest: +SKIP
    >>> print(report.render())  # doctest: +SKIP
    """
    return Report(read_plan(lf))
