import polars as pl

lf = pl.LazyFrame(
    schema={
        "price": pl.Float64,
        "quantity": pl.Int64,
        "tax_rate": pl.Float64,
        "customer_key": pl.Int64,
        "region": pl.String,
        "name": pl.String,
        "legacy_id": pl.Int64,  # never used -> unused input column
        "notes": pl.String,  # never used -> unused input column
    }
)

# Joined in, but nothing that survives to the output ever reads a column from it.
# `mgr` *is* read below -- by `mgr_upper`, which is itself dropped.
sales_reps = pl.LazyFrame(
    schema={"customer_key": pl.Int64, "mgr": pl.String, "rep_email": pl.String}
)

q = (
    lf.join(sales_reps, on="customer_key", how="left")
    .with_columns(mgr_upper=pl.col("mgr").str.to_uppercase())
    .with_columns(
        subtotal=pl.col("price") * pl.col("quantity"),
        # duplicate expression: same computation, two names
        tax=pl.col("price") * pl.col("tax_rate"),
        tax_amt=pl.col("price") * pl.col("tax_rate"),
        # constant expression
        fx_rate=pl.lit(1.1) * pl.lit(0.9),
        # dead columns
        _tmp1=pl.col("price") + 1,
        normalized_name=pl.col("name").str.to_lowercase(),
    )
    .with_columns(
        # single-use intermediate: consumed once, never output
        tmp_discount=pl.col("subtotal") * 0.1,
        # common subexpression: (subtotal + tax) appears twice below
        total=pl.col("subtotal") + pl.col("tax"),
    )
    .with_columns(
        net=(pl.col("subtotal") + pl.col("tax")) - pl.col("tmp_discount"),
        gross=(pl.col("subtotal") + pl.col("tax")) * pl.col("fx_rate"),
    )
    # shadowed alias: overwrites `gross` without reading it
    .with_columns(gross=pl.col("net") * 1.2)
    .with_columns(revenue=pl.col("gross") * pl.col("fx_rate"))
    .with_columns(revenue_after_tax=pl.col("revenue") - pl.col("tax"))
    .with_columns(profit=pl.col("revenue_after_tax") * 0.8)
    .with_columns(margin=pl.col("profit") / pl.col("revenue"))
    .with_columns(margin_pct=pl.col("margin") * 100)
    # fan-out hub: customer_key read by many expressions
    .with_columns(
        **{
            f"cust_feat_{i}": pl.col("customer_key") + i
            for i in range(12)
        }
    )
    .filter(pl.col("region") == "EU")
    .select(
        "customer_key",
        "subtotal",
        "total",
        "net",
        "gross",
        "revenue",
        "margin",
        "margin_pct",
        "profit",
        *[f"cust_feat_{i}" for i in range(12)],
    )
)

report = pl.inspect(q)
print(report.render())

orders = pl.LazyFrame(schema={"order_id": pl.Int64, "cust_id": pl.Int64, "amount": pl.Float64})
customers = pl.LazyFrame(schema={"cust_id": pl.Int64, "name": pl.String, "tier": pl.String})
regions = pl.LazyFrame(schema={"cust_id": pl.Int64, "region": pl.String, "mgr": pl.String})
active = pl.LazyFrame(schema={"cust_id": pl.Int64, "since": pl.Date})


def report(name, q):
    print("=" * 70)
    print(name)
    print("=" * 70)
    r = pl.inspect(q)
    joins = [w for w in r.warnings if w.code in ("pointless-join", "unused-source")]
    if not joins:
        print("no join findings\n")
        return
    for w in joins:
        print(f"\n⚠ {w.title} ({w.count})\n")
        for i in w.items:
            print(f"    {i}" if i else "")
        print(f"\n{w.detail}\nSuggestion:\n    {w.suggestion}\n")


# 1. Left join whose columns are never used: genuinely removable.
report(
    "1. Pointless LEFT join (regions never used)",
    orders.join(regions, on="cust_id", how="left").select("order_id", "amount"),
)

# 2. Inner join used purely as a filter -> should suggest semi.
report(
    "2. INNER join used only to filter (should suggest semi)",
    orders.join(active, on="cust_id", how="inner").select("order_id", "amount"),
)

# 3. Join whose columns ARE used -> must not fire.
report(
    "3. Join that is actually used (must not fire)",
    orders.join(customers, on="cust_id", how="left").select("order_id", "tier"),
)

# 4. Semi join -> filter by design, must never fire.
report(
    "4. Semi join (must never fire)",
    orders.join(active, on="cust_id", how="semi").select("order_id", "amount"),
)

# 5. Column used only in an intermediate that is itself dropped.
report(
    "5. Right column used, but only by a dead intermediate",
    orders.join(regions, on="cust_id", how="left")
    .with_columns(mgr_upper=pl.col("mgr").str.to_uppercase())
    .select("order_id", "amount"),
)
