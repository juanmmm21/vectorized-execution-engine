# vectorized-execution-engine

Part of [`strata-database-engine`](https://github.com/juanmmm21/strata-database-engine), a from-scratch relational database engine built as a set of independent, individually-usable components. This repository is the query engine's execution layer: [`juanmmm21/vectorized-execution-engine`](https://github.com/juanmmm21/vectorized-execution-engine).

## What it is and what problem it solves

A query engine that only knows how to interpret a physical plan tree can still be fast or slow depending on *how* it walks that tree. The naive approach — a Volcano-style iterator that produces one row at a time, with a Python function call per operator per row — pays interpreter overhead on every single value. A vectorized engine instead processes whole batches of rows as NumPy arrays: a filter predicate, a join key, an aggregate sum all become one array operation over thousands of rows instead of thousands of scalar operations.

Concretely, this module:

- Executes the nine physical operator types a cost-based optimizer would produce (`PhysicalTableScan`, `PhysicalIndexScan`, `PhysicalFilter`, `PhysicalNestedLoopJoin`, `PhysicalHashJoin`, `PhysicalHashAggregate`, `PhysicalProject`, `PhysicalSort`, `PhysicalLimit`) as a tree of composable `Operator`s.
- Evaluates every expression — comparisons, arithmetic, `BETWEEN`/`IN`/`LIKE`/`IS NULL`, and SQL's three-valued `AND`/`OR`/`NOT` logic — as NumPy array operations over an entire batch, never a Python loop over rows, in the hot path.
- Streams non-blocking operators (scan, filter, project, limit) batch by batch, and bounds the memory of blocking operators (hash join, hash aggregate, sort) with an explicit, configurable, shared budget that fails loudly instead of growing without limit.
- Validates its own correctness against an independent, deliberately non-vectorized row-by-row reference implementation, over randomized data with fixed seeds — not just against its own expectations.

## Role in `strata-database-engine`

```text
sql-query-parser ──▶ cost-based-query-optimizer ──▶ vectorized-execution-engine ──┐
                                                              ▲                    │
                                                              │  rows              ▼
                                          mvcc-transaction-manager /          query results
                                          lock-manager-deadlock-detector
                                                              ▲
                                                              │
                                    bplus-tree-storage-engine / lsm-tree-engine
```

`cost-based-query-optimizer` produces a physical plan — a tree of `Physical*` nodes, each already carrying its chosen algorithm (nested-loop vs. hash join, table vs. index scan) and estimated cost. This module consumes a plan of that shape (see "Data format" below) and a source of row batches — in the assembled system, rows read through `mvcc-transaction-manager` or `lock-manager-deadlock-detector` over whichever storage engine is active — and executes it, producing the final batches of result rows that `nanosql` returns to its caller. As with every pair of sibling repositories in this ecosystem, there is no direct import between them: each repository defines its own equivalent types, and the real wiring happens only inside `nanosql`.

## Goal / skills demonstrated

- **Vectorized execution as a performance discipline, not a slogan**: every operator's hot path is a NumPy array operation over a whole batch — filter masks, join key gathers, aggregate reductions, sort keys — with the one unavoidable exception (hash table construction, which is inherently entry-by-entry even in production engines) called out explicitly in the source rather than hidden.
- **SQL's three-valued logic done properly**: `NULL` is neither true nor false. `FALSE AND NULL` is `FALSE`, not `NULL`, because a false operand already decides the result without needing the other — implemented as an actual truth table over `(values, nulls)` array pairs, not a shortcut that happens to work on the common cases.
- **A pluggable `Operator` interface**: every physical node maps to one operator implementing a two-method `Protocol` (`output_schema`, `execute`) — adding a new operator kind is a new class plus one line in `pipeline.build_operator_tree`, never a change to an existing operator.
- **Bounded memory as a first-class constraint**: `MemoryBudget` is shared across every blocking operator of a single plan execution (not one independent limit per operator, which would let the total still grow unboundedly) and fails with a typed error rather than paging to disk — a documented, deliberate scope boundary, not an oversight.
- **Correctness verified independently, not just tested**: `tests/naive_reference.py` re-implements every operator's semantics — predicate evaluation, all four join types, `GROUP BY` grouping, multi-key sort — from scratch, row by row, in plain Python, sharing no code path with `src/`. The property-based test suite compares the two over randomized data across five fixed seeds.

## How it works

### Columnar batches, not row objects

`batch.RowBatch` holds each column as a NumPy array (`dtype=object` for text, native dtypes otherwise) plus a parallel boolean array marking which cells are null — a null is never encoded as a sentinel value inside the data array itself, so a numeric column stays a clean `int64`/`float64` array. `DEFAULT_BATCH_SIZE` (2048 rows, the same default DuckDB uses) is the unit every operator streams in and out.

### Expression evaluation

`expressions.evaluate` walks an expression tree once and returns a `(values, nulls)` pair of arrays for the whole batch. Three-valued `AND`/`OR`/`NOT` are implemented as boolean-array truth tables (`_three_valued_and`/`_three_valued_or`/`_three_valued_not`) so that, for example, `FALSE AND NULL` resolves to `FALSE` — a false operand decides the result regardless of what the other side is — exactly matching SQL semantics rather than a simpler "any null makes it null" approximation. Arithmetic and comparisons wrap NumPy's ufuncs and convert an incompatible-type `TypeError` into a typed `UnsupportedExpressionError` instead of letting it escape raw; a zero divisor is detected up front and raised as `DivisionByZeroError` before it can silently produce `inf`/`nan`.

### Joins: vectorized cross product, and vectorized gather over a scalar hash table

`NestedLoopJoinOperator` builds the cross product of two batches with `np.repeat`/`np.tile` over row indices — every pairing materializes at once, never a Python double loop — then evaluates the join condition as a single array operation over the combined batch. `HashJoinOperator` builds its hash table on the (fully materialized) left side by key, which is inherently an entry-by-entry operation no different from any real hash join implementation; what *is* vectorized is everything downstream of the probe: once matching `(left_index, right_index)` pairs are known, gathering both sides into the output batch is a single `RowBatch.take` fancy-index operation per probed batch, not a row-by-row copy. Both operators handle all four `JoinType`s (`INNER`/`LEFT`/`RIGHT`/`FULL`) by tracking which rows on each side were ever matched and emitting a null-filled batch (`batch.null_batch`) for the unmatched remainder.

### Aggregation and sort: the same "scalar bucketing, vectorized math" split

`HashAggregateOperator` groups rows into buckets by a Python `dict` keyed on the (normalized) group values — grouping by an arbitrary key is exactly as inherently scalar as hash-join key construction — but each aggregate's actual reduction (`SUM`/`AVG`/`MIN`/`MAX`/`COUNT`, with or without `DISTINCT`, always skipping `NULL`s except for `COUNT(*)`) runs as one NumPy call over that bucket's row indices. `SortOperator` converts every sort key (including text, via `np.unique`'s order-preserving integer codes) into a numeric array, negates it for `DESC`, and drives a single `np.lexsort` call over all keys at once — nulls always sort last regardless of direction, via a dedicated null-flag key placed ahead of the value in significance.

## Architecture

```text
src/vectorized_execution_engine/
├── models.py         # Expression AST subset + the nine PhysicalOperator
│                       node types (mirrors cost-based-query-optimizer's
│                       plan.py shape, no import) + ColumnType/Schema
├── errors.py         # Typed exception hierarchy (ExecutionEngineError and subclasses)
├── batch.py          # RowBatch (columnar batch), TableSource Protocol +
│                       InMemoryTableSource, schema/batch combinators
├── expressions.py    # Vectorized expression evaluator, three-valued logic
├── protocols.py      # Operator Protocol + MemoryBudget
├── operators.py      # Scan, IndexScan, Filter, Project, Limit (streaming)
├── joins.py          # NestedLoopJoin, HashJoin (all four JoinTypes)
├── aggregate.py      # HashAggregate (GROUP BY + COUNT/SUM/AVG/MIN/MAX)
├── sort.py           # Multi-key stable sort via np.lexsort
├── pipeline.py       # build_operator_tree() + execute_plan()/collect()/explain()
└── __main__.py       # CLI: demo, benchmark
```

`Operator` (in `protocols.py`) is the only abstraction `pipeline.build_operator_tree` needs: any new physical node type is one `isinstance` branch there plus a class implementing `output_schema`/`execute`, never a change to an existing operator — the extensibility the DoD asks for.

## Requirements and installation

- Python `>=3.11`
- Runtime dependency: `numpy>=1.26` — the one module in this ecosystem where NumPy is a justified runtime dependency (see `../CLAUDE.md`), since vectorized array operations are the entire point.

```bash
git clone https://github.com/juanmmm21/vectorized-execution-engine.git
cd vectorized-execution-engine
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
vectorized-execution-engine demo
vectorized-execution-engine benchmark --rows 500000 --seed 42
```

`demo` builds a handful of illustrative physical plans directly with `models.py`'s dataclasses — a scan/filter/project, a hash join, a join feeding a hash aggregate, and a sort/limit — against an in-memory sample dataset, printing each plan's `EXPLAIN` and its resulting rows. `benchmark` measures throughput (rows/second) for each operator kind at a configurable data volume.

### Programmatic

```python
from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.models import (
    BinaryOp, BinaryOperator, ColumnRef, ColumnType, Literal, LiteralType,
    PhysicalFilter, PhysicalTableScan, TableRef,
)
from vectorized_execution_engine.pipeline import collect

source = InMemoryTableSource()
source.add_table(
    "orders",
    {"id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
    [{"id": 1, "amount": 42.5}, {"id": 2, "amount": 8.0}],
)

plan = PhysicalFilter(
    PhysicalTableScan(TableRef("orders", "o"), estimated_rows=2, estimated_cost=2),
    BinaryOp(ColumnRef("amount", "o"), BinaryOperator.GT, Literal(10.0, LiteralType.FLOAT)),
    estimated_rows=1,
    estimated_cost=2,
)

result = collect(plan, source)  # a single materialized RowBatch
print(result.columns["o.id"])   # array([1])
```

`execute_plan(plan, source)` is the streaming equivalent — an iterator of `RowBatch`, for consumers that shouldn't materialize the whole result at once.

## Data format / interface exposed to `nanosql`

Per `../AGENTS.md`, there is no cross-repo import between this module and `cost-based-query-optimizer`: `models.py` **redefines**, independently, the nine `Physical*` node types and the expression subset they reference, matching `cost_based_query_optimizer.plan`'s shape one-for-one field by field, so `nanosql` can feed this module's `pipeline.build_operator_tree` the optimizer's own output directly, with no adapter layer.

The other half of the input contract is `batch.TableSource` — a minimal `Protocol` (`schema(table_name)`, `scan(table_name)`) that plays the same role `RowStore` plays for `mvcc-transaction-manager`: this module never imports a storage or transaction engine, it only requires whatever backs it to hand back rows already grouped into `RowBatch`es. `InMemoryTableSource` is the reference implementation used by the CLI and every test; any adapter over `mvcc-transaction-manager`'s visible rows (or `lock-manager-deadlock-detector`'s) that batches them into `RowBatch`es can replace it without touching `pipeline.py` — that integration happens only inside `nanosql`.

The output of any operator (and of `pipeline.execute_plan`/`collect`) is `batch.RowBatch`: a `Schema` (ordered, qualified column names — `"o.amount"` after scanning `orders o` — plus a `ColumnType` per column) and two `dict[str, np.ndarray]`, one holding values and one holding null masks, both keyed by the schema's column names.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy --strict src/
```

195 tests across twelve files:

- `test_batch.py` (17) / `test_expressions.py` (38): `RowBatch` construction, gather/slice/concat/null-fill primitives, and every expression form in isolation — three-valued `AND`/`OR` truth tables, `BETWEEN`/`IN`/`LIKE` (including escaped wildcards) /`IS NULL`, column resolution (qualified, unqualified-unique, ambiguous), division by zero, and incompatible-type errors.
- `test_operators.py` (12) / `test_joins.py` (13) / `test_aggregate.py` (7) / `test_sort_limit.py` (6): scan column qualification, filter/project/limit streaming semantics, all four `JoinType`s on both `NestedLoopJoinOperator` and `HashJoinOperator` (including a direct cross-check that the two agree), `NULL`-key grouping, `DISTINCT` aggregates, multi-key sort with mixed directions and nulls-last.
- `test_memory_limits.py` (6): each blocking operator (hash join, hash aggregate, sort) raises `MemoryLimitExceededError` under a deliberately tiny budget, and a generous budget does not.
- `test_pipeline.py` (6): `build_operator_tree` end to end for every physical node kind, `UnsupportedOperatorError` on an unrecognized node, and that a `MemoryBudget` is shared (not reset) across every blocking operator of one tree.
- `test_properties.py` (90): a seeded (`random.Random(seed)`, 5 fixed seeds) generator of `customers`/`orders` data — including `NULL`s and foreign keys with no match, to force outer-join edge cases — checked against `tests/naive_reference.py`, an independent row-by-row implementation sharing no code path with `src/`. Covers filters (comparison, `OR`, `BETWEEN`, `IN`, `LIKE`, `IS NULL`), all four join types on both join operators, aliased projection, a full join → aggregate → sort pipeline, and `SUM`/`AVG`/`COUNT` compared with floating-point tolerance (`np.sum`'s internal summation order can legitimately differ from left-to-right Python `sum()` in the last bit).

## Benchmarks

```text
1,000,000 rows in 'orders', seed=42:

Scan + Filter:       1,000,000 rows processed (500,808 result) in 0.1474s -> 6,783,476 rows/s
Hash Join:           1,000,000 rows processed (1,000,000 result) in 0.5121s -> 1,952,753 rows/s
Join + Aggregate:    1,000,000 rows processed (4 result) in 0.8946s -> 1,117,795 rows/s
Sort:                1,000,000 rows processed (1,000,000 result) in 0.2194s -> 4,557,929 rows/s
```

Run `vectorized-execution-engine benchmark --rows 1000000` to reproduce (numbers above are one representative run; expect variance of a few percent between runs on the same machine).

A real, slightly humbling finding from building this benchmark: the naive comparison to reach for first — regenerate the same data as a `list[dict]` and time a plain Python `for` loop doing the identical filter — made the vectorized engine look *slower*, not faster. The reason has nothing to do with the filter itself: `InMemoryTableSource.scan()` rebuilds NumPy arrays from Python lists on every call, and that one-time columnarization cost dominated a batch as simple as a single numeric comparison. Once the comparison is made fair — materialize the batches once outside the timer, then measure only `FilterOperator.execute()` against a plain Python loop over the *same already-extracted* values (see `_ReplayOperator` in `__main__.py`) — vectorization wins, but only by roughly 1.4–1.6x for this single simple predicate, not by orders of magnitude. The real payoff shows up further down the pipeline: `Hash Join`, `Join + Aggregate`, and `Sort` all sustain multi-million-rows/second throughput precisely because every row that would otherwise pay Python's per-row interpreter overhead at *every* operator of a multi-step plan instead flows through as one array operation per operator — the advantage compounds with pipeline depth, it does not show up on a single trivial filter measured in isolation.

## Troubleshooting

- **`UnresolvedColumnError`**: an unqualified `ColumnRef` (no `table=`) matched zero or more than one column of the batch it was evaluated against. Qualify the column (`t.column`) or ensure only one joined side's schema contains that bare name.
- **`UnsupportedExpressionError`**: either a `FunctionCall` (aggregate) appeared somewhere other than a `PhysicalHashAggregate`'s `aggregates` or a `PhysicalProject`/`PhysicalFilter` referencing an already-computed aggregate column by the same expression, or an arithmetic/comparison operator was applied between incompatible column types (e.g. adding a text column to a numeric one) — this engine does not type-check expressions ahead of evaluation, matching `cost-based-query-optimizer`'s and `sql-query-parser`'s own documented scope.
- **`UnsupportedJoinConditionError`**: a `PhysicalHashJoin.condition` wasn't a plain equality between one column on each side. Hash join requires an equi-condition; a non-equi condition needs `PhysicalNestedLoopJoin` instead — exactly the choice `cost-based-query-optimizer` is responsible for making upstream.
- **`DivisionByZeroError`**: a `/` or `%` found a zero divisor in at least one non-null row of the batch. This engine deliberately raises rather than letting NumPy silently produce `inf`/`nan`.
- **`MemoryLimitExceededError`**: a hash join's build side, a hash aggregate, or a sort tried to materialize more than its `MemoryBudget.max_bytes` allows. Raise the budget passed to `pipeline.build_operator_tree`/`execute_plan`/`collect`, or reduce the input size — this engine does not spill to disk (see "How it works" and the DoD's documented scope).
- **A query's rows come back in an unexpected order when there's no `PhysicalSort` in the plan**: expected — only `PhysicalSort` guarantees order. Every other operator's row order is an implementation detail of batch processing, not a contract.

## License

MIT — see [`LICENSE`](LICENSE).
