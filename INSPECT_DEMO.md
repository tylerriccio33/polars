# `pl.inspect` — a static analyzer for LazyFrame plans

A working prototype of the linter/IDE-inspection idea. It reads a `LazyFrame` plan
**as written**, before any optimization, and reports things a *human* would want to
clean up. It never changes what the query computes.

```python
report = pl.inspect(lf)

print(report.summary())   # counts
print(report.render())    # the full report
report.stages()           # topological grouping -> list[list[str]]
print(report.graph())     # ASCII dependency graph
```

The `Report` object's `__repr__` is the full report, so in a REPL `pl.inspect(lf)` just
prints.

---

## Why it can't be an optimizer feature

This is the load-bearing design decision, and it's the reason the prototype needed a
(small) Rust change.

Polars already exposes `lf._ldf.visit()`, which hands Python a `NodeTraverser` over the
IR arenas — it's how the GPU engine walks plans. But it runs the **full optimizer
first**. That's fatal for a linter: common-subexpression elimination collapses the
duplicate expressions you want to report, and projection pushdown deletes the dead
columns. By the time you can see the plan, every finding has been optimized away.

So the one Rust change is `crates/polars-python/src/lazyframe/visit.rs`:

```rust
#[pyo3(signature = (optimized = true))]
fn visit(&self, optimized: bool) -> PyResult<NodeTraverser> {
    ...
    } else {
        // The plan as written by the user: only DSL -> IR conversion, no optimization
        // passes.
        let plan = ldf.to_alp().map_err(PyPolarsErr::from)?;
        (plan.lp_top, plan.lp_arena, plan.expr_arena)
    };
```

`LazyFrame::to_alp()` already existed — it does DSL→IR conversion and schema resolution
and nothing else. `visit()` keeps its old behaviour by default, so nothing breaks.

**Everything else is pure Python** (`py-polars/src/polars/_inspect/`, ~450 lines):

| file | role |
| --- | --- |
| `_plan.py` | walks the unoptimized IR, flattens each expression into a structural **fingerprint** |
| `_lints.py` | the inspections, registered via an `@inspection` decorator |
| `_report.py` | `pl.inspect()` and the `Report` object |

The fingerprint is what makes this work. Each expression is reduced to a canonical tuple
of its structure — and **aliases fingerprint *through***, so `(price * rate).alias("tax")`
and `(price * rate).alias("tax_amt")` produce the *same* fingerprint. That single choice
gives you duplicate-expression and common-subexpression detection for free.

---

## The demo pipeline

`py-polars/tests/unit/test_inspect.py`. Every issue below is deliberately planted:

```python
# Joined in, but nothing that survives to the output ever reads a column from it.
# `mgr` *is* read below -- by `mgr_upper`, which is itself dropped.
sales_reps = pl.LazyFrame(
    schema={"customer_key": pl.Int64, "mgr": pl.String, "rep_email": pl.String}
)

q = (
    lf.join(sales_reps, on="customer_key", how="left")   # pointless join
    .with_columns(mgr_upper=pl.col("mgr").str.to_uppercase())  # dead, and the only reader of `mgr`
    .with_columns(
        subtotal=pl.col("price") * pl.col("quantity"),
        # duplicate expression: same computation, two names
        tax=pl.col("price") * pl.col("tax_rate"),
        tax_amt=pl.col("price") * pl.col("tax_rate"),
        fx_rate=pl.lit(1.1) * pl.lit(0.9),          # constant
        _tmp1=pl.col("price") + 1,                   # dead
        normalized_name=pl.col("name").str.to_lowercase(),  # dead
    )
    .with_columns(
        tmp_discount=pl.col("subtotal") * 0.1,       # single-use
        total=pl.col("subtotal") + pl.col("tax"),
    )
    .with_columns(
        # (subtotal + tax) is a common subexpression across these two
        net=(pl.col("subtotal") + pl.col("tax")) - pl.col("tmp_discount"),
        gross=(pl.col("subtotal") + pl.col("tax")) * pl.col("fx_rate"),
    )
    .with_columns(gross=pl.col("net") * 1.2)         # shadows `gross` without reading it
    .with_columns(revenue=pl.col("gross") * pl.col("fx_rate"))
    .with_columns(revenue_after_tax=pl.col("revenue") - pl.col("tax"))
    .with_columns(profit=pl.col("revenue_after_tax") * 0.8)
    .with_columns(margin=pl.col("profit") / pl.col("revenue"))
    .with_columns(margin_pct=pl.col("margin") * 100)  # -> chain of depth 9
    .with_columns(**{f"cust_feat_{i}": pl.col("customer_key") + i for i in range(12)})
    .filter(pl.col("region") == "EU")
    .select(...)
)
print(pl.inspect(q).render())
```

The source frame also has `legacy_id` and `notes`, which are never touched.

## The output

All of this is real, copied from a run.

```
Pipeline
========

Nodes:                  16
Expressions:            53
Derived columns:        28
Output columns:         21
Warnings:               10
Max chain depth:         9
```

### Pointless join

The two lints below are best read together — they're the same finding seen from two ends.

```
⚠ Pointless join (1)

    left join, contributing: mgr, rep_email

    The right-hand side is read only to be discarded:
      in-memory frame

Contributes nothing, and a left join cannot drop rows.

Suggestion:
    Remove the join. (If the right keys are not unique it is *adding* rows -- in
    that case you want `.unique()` on the right, not a join.)
```

Note what had to happen for this to fire. `mgr` **is** read — by `mgr_upper`. A naive
"does anything above the join reference these columns?" check says the join is used and
stays quiet. Only the liveness fixpoint (below) works out that `mgr_upper` is itself dead,
therefore `mgr` is dead, therefore the whole join is dead weight.

### Duplicate expressions

The two are *structurally identical* — the alias is not part of the fingerprint.

```
⚠ Duplicate expressions (1)

    tax = price * tax_rate
    tax_amt = price * tax_rate

Suggestion:
    Compute once and reuse.
```

### Common subexpressions

`(subtotal + tax)` is a shared *sub*-tree, not a whole column, and it's found inside two
different definitions.

```
⚠ Common subexpressions (1)

    subtotal + tax  (gross, net)

Suggestion:
    Factor into one intermediate column.
```

### Dead columns

Cannot affect the result. Note `tax_amt` shows up here *as well as* in the duplicate
report — it's both. And `mgr_upper` is the column that makes the pointless join above
detectable: it's the only reader of `mgr`, and it's dead, so `mgr` is dead too.

```
⚠ Dead columns (4)

    _tmp1
    mgr_upper
    normalized_name
    tax_amt
```

### Single-use intermediates

Read exactly once, then dropped.

```
⚠ Single-use intermediates (2)

    revenue_after_tax
    tmp_discount

Suggestion:
    Inline, unless the name earns its keep as documentation.
```

### Unused input columns

```
⚠ Unused input columns (2)

    legacy_id
    notes

Suggestion:
    Narrow the projection at the scan.
```

### Long dependency chain

Exactly the tree you sketched, walked back to the source column:

```
⚠ Long dependency chain (1)

    margin_pct
         └── margin
              └── profit
                   └── revenue_after_tax
                        └── revenue
                             └── gross
                                  └── net
                                       └── tmp_discount
                                            └── subtotal
                                                 └── price

Depth: 9

Suggestion:
    Consider splitting into stages.
```

### High fan-out

```
⚠ High fan-out (1)

    customer_key

Used by 12 downstream expressions.
```

### Shadowed aliases

The overwritten value is rendered too, so you can see what you lost:

```
⚠ Shadowed aliases (1)

    gross = net * 1.2  (overwrites (subtotal + tax) * fx_rate)

Overwritten before the previous value was ever read.
```

### Constant expressions

```
⚠ Constant expressions (1)

    fx_rate = 0.9900000000000001
```

Worth a note: `pl.lit(1.1) * pl.lit(0.9)` is **already constant-folded during DSL→IR
conversion** — by the time the analyzer sees it, it's a single literal node, not a
multiply. So this lint reports the resulting constant *column*, not the arithmetic. My
first version looked for literal arithmetic and could never have fired.

### Stages and the dependency graph

```
Stages
======

Stage 1          Stage 2            Stage 3        Stage 4 ... Stage 10
--------         --------           --------
  customer_key     _tmp1              tmp_discount   net         margin_pct
  legacy_id        cust_feat_0..11    total
  name             fx_rate
  notes            normalized_name
  price            subtotal
  quantity         tax
  region           tax_amt
  tax_rate
```
*(shown side by side here; the real output is vertical)*

Everything within a stage is independent and could be computed in one block.

```
Dependencies
============

  gross
    └── net *

  margin
    ├── profit *
    └── revenue *

  net
    ├── subtotal *
    ├── tax *
    └── tmp_discount *

  cust_feat_0
    └── customer_key

  (* = itself derived)
```

Two subtleties in there that I got wrong on the first pass:

- `gross` lists only `net`, not the inputs of its *shadowed* first definition. The graph
  uses the **last** definition of each name — that's what the final value actually
  depends on.
- `customer_key` has no `*` because it's a source column. A bare `.select("customer_key")`
  re-binds the name to itself, and I was initially counting that as a "definition".

---

## Pointless joins and unused sources

The highest-value inspection, and the one that most needs the pre-optimization view.

**How it identifies what a join contributes:** `schema(join_node) − schema(left_input)`.
Whatever exists after the join but not before it is exactly what the right side added.
Coalesced keys and `_right`-suffixed collisions fall out of that for free — no
special-casing.

**But "is it used?" is a liveness question, not a lookup.** My first version checked
whether any node above the join read the contributed columns. That misses the case that
matters most:

```python
orders.join(regions, on="cust_id", how="left")
      .with_columns(mgr_upper=pl.col("mgr").str.to_uppercase())  # reads `mgr`!
      .select("order_id", "amount")                              # ...but drops it
```

`mgr` *is* read — by a column that is itself dead. The join is still pointless. So
`live_columns()` is a proper backward dataflow fixpoint: a column is live if it reaches
the output or feeds something else that's live. Iterated to convergence. That also made
the dead-column lint transitive, which it wasn't before.

**The join type decides what advice is honest.** An inner join that contributes no
columns is *still doing work* — it filters rows — so "delete it" would be flat wrong:

```
1. Pointless LEFT join
⚠ Pointless join (1)

    left join, contributing: mgr, region

    The right-hand side is read only to be discarded:
      in-memory frame

Contributes nothing, and a left join cannot drop rows.
Suggestion:
    Remove the join. (If the right keys are not unique it is *adding* rows -- in
    that case you want `.unique()` on the right, not a join.)
```

```
2. INNER join used only to filter
⚠ Pointless join (1)

    inner join, contributing: since

Contributes no columns, but an inner join still drops non-matching rows.
Suggestion:
    You are joining to filter. Use `how="semi"`: same rows, no columns, no row
    duplication if the right keys repeat.
```

That second one is the money finding — a `semi` join is both faster *and* immune to the
row-duplication bug you get when the right keys aren't unique.

The full table of behaviour:

| join | contributes nothing → | why |
| --- | --- | --- |
| `left` | **remove it** | cannot drop rows; can only *add* them if right keys repeat |
| `inner` | **use `how="semi"`** | still filters, so it can't just be deleted |
| `full` | check the extra rows | can add unmatched rows |
| `cross` | almost certainly a bug | multiplies the row count |
| `semi` / `anti` | *never flagged* | contributing no columns is their entire purpose |

Verified against all five shapes (including two that **must not** fire — a join whose
columns are genuinely used, and a semi join). Both stay silent.

**Unused sources** falls out of the same machinery: any scan none of whose columns are
live gets reported with its path, so you can delete the source *and* the join that
drags it in.

---

## Inspections: what shipped, and what I dropped

| Inspection | Status |
| --- | --- |
| Duplicate expressions | ✅ |
| Common subexpressions | ✅ |
| Dead derived columns | ✅ |
| Single-use temps | ✅ |
| Unused input columns | ✅ |
| Dependency depth | ✅ |
| Fan-in/fan-out | ✅ |
| Giant `with_columns` (>50) | ✅ (not tripped by this demo) |
| Constant expressions | ✅ (reframed, see above) |
| Shadowed aliases | ✅ |
| **Pointless joins** | ✅ (added — see above) |
| **Unused sources** | ✅ (added — see above) |
| **Duplicate aliases** | ❌ **unreachable** |
| **Circular dependencies** | ❌ **unreachable** |

I deleted the last two rather than ship lints that can never fire:

- **Duplicate aliases.** Polars already rejects this at schema resolution:
  `ComputeError: the name 'x' passed to 'LazyFrame.with_columns' is duplicate`. It's a
  hard error, not a lint.
- **Circular dependencies.** The plan is a DAG by construction, so a genuine cycle can't
  be expressed. Worse, a *name-based* dependency graph like mine would **false-positive**
  on perfectly legal shadowing — `.with_columns(y=col("x"))` followed by
  `.with_columns(x=col("y"))` looks circular but is fine, because those are two different
  versions of `x`. Catching this properly needs SSA-style versioning of column names,
  which is real work for a lint with no true positives.

Thresholds live at the top of `_lints.py` (`WIDE_BLOCK = 50`, `DEEP_CHAIN = 8`,
`HIGH_FAN_OUT = 10`, `MIN_CSE_SIZE = 3`) and should become arguments to `inspect()`.

---

## Known limitations

- **Names, not versions.** The column graph keys on name. Shadowing is handled by taking
  the last definition, but a fully correct model would version each binding (SSA). This
  is the single biggest thing to fix, and it's what would unlock a trustworthy
  circular-dependency check and better shadowing analysis.
- **Joins and unions** are walked (the traversal is a proper post-order over all inputs),
  and the join lint reasons about the two sides correctly via schema difference. But
  elsewhere, columns from two branches that share a name are still conflated by the
  name-keyed graph. Same root cause as above.
- **Join-key uniqueness is unknowable statically.** That's why the `left` join advice is
  hedged rather than a flat "delete this" — if the right keys repeat, the join is
  *adding* rows, and removing it would change the result.
- **`Expression.text`** is a hand-rolled pretty-printer for the report. It covers the
  common node types and falls back to `kind(args...)` otherwise. It is *not* a
  round-trippable rendering, and shouldn't grow into one.
- Unknown expression node types fingerprint opaquely by identity, so they never collide
  with anything. That's deliberate — a wrong-but-confident fingerprint would produce
  false "duplicate expression" reports, which is the one thing that would make people
  stop trusting the tool.

## Not done: `pl.reorder`

The auto-format idea is a genuinely different animal, and I'd keep it separate. Reading a
plan is easy; `reorder` has to **rebuild** one. That needs the IR→DSL direction (or
plan-rewriting through the traverser's mutation API), and it has to preserve semantics
across `filter`/`join`/`group_by` boundaries — you can't hoist a `with_columns` above a
filter that changes what rows exist without thinking hard about it. Worth doing, but as
its own piece of work rather than bolted onto the analyzer.

## Naming

`pl.inspect` shadows the stdlib `inspect` module name at the top level. It's harmless in
practice (`pl.inspect`, never `import inspect`), but `pl.analyze` from your second sketch
sidesteps the question entirely. Trivial to switch — say the word.
