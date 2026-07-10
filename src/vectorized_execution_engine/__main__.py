"""CLI de demostración: `vectorized-execution-engine {demo,benchmark}`.

Este motor no incluye ni un parser SQL ni un optimizador (por diseño — ver
`../AGENTS.md`, sin imports cruzados con `sql-query-parser` ni con
`cost-based-query-optimizer`), así que su CLI no puede aceptar texto SQL
como las de sus hermanos. En su lugar, `demo` construye a mano un puñado de
`PhysicalOperator`/`InMemoryTableSource` ilustrativos — directamente con
los dataclasses de `models.py`, exactamente la forma en que `nanosql` los
construiría a partir del plan físico real de `cost-based-query-optimizer`
— y ejecuta cada uno mostrando su `EXPLAIN` y su resultado; `benchmark`
mide el throughput (filas/segundo) de cada operador vectorizado sobre un
volumen de datos configurable.
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .batch import InMemoryTableSource, RowBatch
from .models import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    FunctionCall,
    JoinType,
    Literal,
    LiteralType,
    OrderByItem,
    OrderDirection,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalLimit,
    PhysicalOperator,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
    Schema,
    SelectItem,
    Star,
    TableRef,
)
from .operators import FilterOperator, ScanOperator
from .pipeline import execute_plan, explain


def _demo_source() -> InMemoryTableSource:
    source = InMemoryTableSource()
    source.add_table(
        "customers",
        {"id": ColumnType.INTEGER, "name": ColumnType.STRING, "country": ColumnType.STRING},
        [
            {"id": 1, "name": "Ada Lovelace", "country": "UK"},
            {"id": 2, "name": "Grace Hopper", "country": "US"},
            {"id": 3, "name": "Alan Turing", "country": "UK"},
            {"id": 4, "name": "Margaret Hamilton", "country": "US"},
        ],
    )
    source.add_table(
        "orders",
        {
            "id": ColumnType.INTEGER,
            "customer_id": ColumnType.INTEGER,
            "amount": ColumnType.FLOAT,
            "status": ColumnType.STRING,
        },
        [
            {"id": 100, "customer_id": 1, "amount": 42.5, "status": "PENDING"},
            {"id": 101, "customer_id": 2, "amount": 17.0, "status": "SHIPPED"},
            {"id": 102, "customer_id": 1, "amount": 8.25, "status": "SHIPPED"},
            {"id": 103, "customer_id": 3, "amount": 99.9, "status": "PENDING"},
            {"id": 104, "customer_id": 2, "amount": 120.0, "status": "PENDING"},
            {"id": 105, "customer_id": 4, "amount": 5.5, "status": "CANCELLED"},
        ],
    )
    return source


def _print_batches(title: str, plan_explanation: str, batches: list[RowBatch]) -> None:
    print(f"=== {title} ===\n")
    print(plan_explanation)
    print()
    total_rows = sum(b.row_count for b in batches)
    if total_rows == 0:
        print("(sin filas)")
    else:
        header_batch: RowBatch = batches[0]
        print(" | ".join(header_batch.schema.columns))
        for batch in batches:
            for row in range(batch.row_count):
                cells = []
                for name in batch.schema.columns:
                    if batch.nulls[name][row]:
                        cells.append("NULL")
                    else:
                        cells.append(str(batch.columns[name][row]))
                print(" | ".join(cells))
    print()


def _demo_filter_project() -> None:
    scan = PhysicalTableScan(TableRef("orders", "o"), estimated_rows=6, estimated_cost=6)
    filtered = PhysicalFilter(
        scan,
        BinaryOp(
            ColumnRef("status", "o"), BinaryOperator.EQ, Literal("PENDING", LiteralType.STRING)
        ),
        estimated_rows=3,
        estimated_cost=6,
    )
    plan = PhysicalProject(
        filtered,
        (
            SelectItem(ColumnRef("id", "o"), alias="order_id"),
            SelectItem(ColumnRef("amount", "o")),
        ),
        estimated_rows=3,
        estimated_cost=6,
    )
    source = _demo_source()
    _print_batches(
        "1. Scan + Filter + Project (WHERE status = 'PENDING')",
        explain(plan),
        list(execute_plan(plan, source)),
    )


def _demo_hash_join() -> None:
    orders = PhysicalTableScan(TableRef("orders", "o"), estimated_rows=6, estimated_cost=6)
    customers = PhysicalTableScan(TableRef("customers", "c"), estimated_rows=4, estimated_cost=4)
    joined = PhysicalHashJoin(
        customers,
        orders,
        JoinType.INNER,
        BinaryOp(ColumnRef("id", "c"), BinaryOperator.EQ, ColumnRef("customer_id", "o")),
        estimated_rows=6,
        estimated_cost=10,
    )
    plan = PhysicalProject(
        joined,
        (SelectItem(ColumnRef("name", "c")), SelectItem(ColumnRef("amount", "o"))),
        estimated_rows=6,
        estimated_cost=10,
    )
    source = _demo_source()
    _print_batches(
        "2. Hash Join (customers.id = orders.customer_id)",
        explain(plan),
        list(execute_plan(plan, source)),
    )


def _demo_aggregate() -> None:
    orders = PhysicalTableScan(TableRef("orders", "o"), estimated_rows=6, estimated_cost=6)
    customers = PhysicalTableScan(TableRef("customers", "c"), estimated_rows=4, estimated_cost=4)
    joined = PhysicalHashJoin(
        customers,
        orders,
        JoinType.INNER,
        BinaryOp(ColumnRef("id", "c"), BinaryOperator.EQ, ColumnRef("customer_id", "o")),
        estimated_rows=6,
        estimated_cost=10,
    )
    aggregated = PhysicalHashAggregate(
        joined,
        group_by=(ColumnRef("country", "c"),),
        aggregates=(
            FunctionCall("SUM", arguments=(ColumnRef("amount", "o"),)),
            FunctionCall("COUNT", star_argument=True),
        ),
        estimated_rows=2,
        estimated_cost=10,
    )
    plan = PhysicalSort(
        aggregated,
        order_by=(OrderByItem(ColumnRef("country", "c")),),
        estimated_rows=2,
        estimated_cost=10,
    )
    source = _demo_source()
    _print_batches(
        "3. Hash Join + Hash Aggregate (SUM(amount)/COUNT(*) GROUP BY country)",
        explain(plan),
        list(execute_plan(plan, source)),
    )


def _demo_sort_limit() -> None:
    scan = PhysicalTableScan(TableRef("orders", "o"), estimated_rows=6, estimated_cost=6)
    sorted_plan = PhysicalSort(
        scan,
        order_by=(OrderByItem(ColumnRef("amount", "o"), OrderDirection.DESC),),
        estimated_rows=6,
        estimated_cost=6,
    )
    limited = PhysicalLimit(sorted_plan, limit=3, offset=0, estimated_rows=3, estimated_cost=6)
    plan = PhysicalProject(
        limited,
        (SelectItem(Star()),),
        estimated_rows=3,
        estimated_cost=6,
    )
    source = _demo_source()
    _print_batches(
        "4. Sort DESC + Limit 3 (top 3 pedidos por importe)",
        explain(plan),
        list(execute_plan(plan, source)),
    )


def run_demo() -> None:
    _demo_filter_project()
    _demo_hash_join()
    _demo_aggregate()
    _demo_sort_limit()


def _build_benchmark_source(row_count: int, seed: int) -> InMemoryTableSource:
    rng = random.Random(seed)
    source = InMemoryTableSource()
    source.add_table(
        "customers",
        {"id": ColumnType.INTEGER, "country": ColumnType.STRING},
        [{"id": i, "country": rng.choice(["UK", "US", "ES", "DE"])} for i in range(1000)],
    )
    source.add_table(
        "orders",
        {"id": ColumnType.INTEGER, "customer_id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [
            {
                "id": i,
                "customer_id": rng.randrange(1000),
                "amount": round(rng.uniform(1, 500), 2),
            }
            for i in range(row_count)
        ],
    )
    return source


def _time_plan(plan: PhysicalOperator, source: InMemoryTableSource) -> tuple[float, int]:
    start = time.perf_counter()
    total_rows = sum(batch.row_count for batch in execute_plan(plan, source))
    elapsed = time.perf_counter() - start
    return elapsed, total_rows


def _orders_scan(row_count: int) -> PhysicalTableScan:
    return PhysicalTableScan(
        TableRef("orders", "o"), estimated_rows=row_count, estimated_cost=row_count
    )


def _customers_scan() -> PhysicalTableScan:
    return PhysicalTableScan(TableRef("customers", "c"), estimated_rows=1000, estimated_cost=1000)


def _report(label: str, processed_rows: int, output_rows: int, elapsed: float) -> None:
    """`processed_rows` es el volumen de entrada empujado por el pipeline
    (siempre `row_count` de `orders` en este benchmark); se usa como
    denominador del throughput en vez de `output_rows` porque un
    `HashAggregate` reduce miles de filas a un puñado de grupos — dividir
    por esas pocas filas de salida daría un "filas/s" sin sentido, sin
    reflejar el trabajo vectorizado realmente hecho."""
    throughput = processed_rows / elapsed if elapsed > 0 else float("inf")
    print(
        f"{label:<18} {processed_rows:>9} filas procesadas ({output_rows} resultado) "
        f"en {elapsed:.4f}s -> {throughput:,.0f} filas/s"
    )


@dataclass(slots=True)
class _ReplayOperator:
    """`Operator` ad-hoc que sólo repite batches ya materializados.

    Usado exclusivamente por el benchmark para medir el coste de
    `FilterOperator` en aislado: si se cronometrase `ScanOperator` +
    `FilterOperator` juntos, el resultado incluiría el coste (pagado una
    única vez, no repetido por cada fila) de construir arrays de NumPy a
    partir de los `list[dict]` internos de `InMemoryTableSource` — un coste
    de "columnarización" que no existe en un motor de producción con
    almacenamiento ya columnar, y que dominaría por completo una
    comparación honesta contra un filtro fila a fila en Python puro sobre
    los MISMOS valores ya extraídos.
    """

    schema: Schema
    batches: list[RowBatch]

    @property
    def output_schema(self) -> Schema:
        return self.schema

    def execute(self) -> Iterator[RowBatch]:
        yield from self.batches


def _naive_filter(amounts: list[float], threshold: float) -> float:
    """Filtra `amounts` fila a fila con un bucle Python explícito — la
    comparación honesta que motiva este motor: cuánto se gana vectorizando
    `amount > umbral` sobre NumPy frente a Python puro, sobre los mismos
    valores ya extraídos (ver `_ReplayOperator`)."""
    start = time.perf_counter()
    _ = [amount for amount in amounts if amount > threshold]
    return time.perf_counter() - start


def run_benchmark(row_count: int, seed: int) -> None:
    source = _build_benchmark_source(row_count, seed)
    print(f"Benchmark: {row_count} filas en 'orders', semilla={seed}\n")

    filtered = PhysicalFilter(
        _orders_scan(row_count),
        BinaryOp(ColumnRef("amount", "o"), BinaryOperator.GT, Literal(250.0, LiteralType.FLOAT)),
        estimated_rows=row_count // 2,
        estimated_cost=row_count,
    )
    elapsed, rows = _time_plan(filtered, source)
    _report("Scan + Filter:", row_count, rows, elapsed)

    scan_operator = ScanOperator(TableRef("orders", "o"), source)
    prebuilt_batches = list(scan_operator.execute())
    replay = _ReplayOperator(scan_operator.output_schema, prebuilt_batches)
    filter_only = FilterOperator(
        replay,
        BinaryOp(ColumnRef("amount", "o"), BinaryOperator.GT, Literal(250.0, LiteralType.FLOAT)),
    )
    vectorized_start = time.perf_counter()
    vectorized_rows = sum(batch.row_count for batch in filter_only.execute())
    vectorized_elapsed = time.perf_counter() - vectorized_start

    amounts = [
        float(value)
        for batch in prebuilt_batches
        for i, value in enumerate(batch.columns["o.amount"])
        if not batch.nulls["o.amount"][i]
    ]
    naive_elapsed = _naive_filter(amounts, 250.0)
    speedup = naive_elapsed / vectorized_elapsed if vectorized_elapsed > 0 else float("inf")
    print(
        f"  (filtro aislado sobre batches ya materializados: vectorizado "
        f"{vectorized_elapsed:.4f}s ({vectorized_rows} filas) vs. Python puro "
        f"{naive_elapsed:.4f}s -> {speedup:.1f}x más rápido vectorizado)"
    )

    equi_condition = BinaryOp(
        ColumnRef("id", "c"), BinaryOperator.EQ, ColumnRef("customer_id", "o")
    )
    joined = PhysicalHashJoin(
        _customers_scan(),
        _orders_scan(row_count),
        JoinType.INNER,
        equi_condition,
        estimated_rows=row_count,
        estimated_cost=row_count,
    )
    elapsed, rows = _time_plan(joined, source)
    _report("Hash Join:", row_count, rows, elapsed)

    aggregated = PhysicalHashAggregate(
        PhysicalHashJoin(
            _customers_scan(),
            _orders_scan(row_count),
            JoinType.INNER,
            equi_condition,
            estimated_rows=row_count,
            estimated_cost=row_count,
        ),
        group_by=(ColumnRef("country", "c"),),
        aggregates=(FunctionCall("SUM", arguments=(ColumnRef("amount", "o"),)),),
        estimated_rows=4,
        estimated_cost=row_count,
    )
    elapsed, rows = _time_plan(aggregated, source)
    _report("Join + Aggregate:", row_count, rows, elapsed)

    sorted_plan = PhysicalSort(
        _orders_scan(row_count),
        order_by=(OrderByItem(ColumnRef("amount", "o"), OrderDirection.DESC),),
        estimated_rows=row_count,
        estimated_cost=row_count,
    )
    elapsed, rows = _time_plan(sorted_plan, source)
    _report("Sort:", row_count, rows, elapsed)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vectorized-execution-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="ejecuta un puñado de planes físicos ilustrativos")

    benchmark_parser = subparsers.add_parser("benchmark", help="mide throughput (filas/segundo)")
    benchmark_parser.add_argument("--rows", type=int, default=200_000)
    benchmark_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo()
    elif args.command == "benchmark":
        run_benchmark(args.rows, args.seed)


if __name__ == "__main__":
    main()
