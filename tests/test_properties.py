"""Tests de propiedades: datos aleatorios de semilla fija, comparados entre
el motor vectorizado (`pipeline.execute_plan`) y la referencia fila a fila
independiente (`naive_reference.naive_execute`) sobre exactamente los
mismos datos — ver DoD del proyecto y `../CLAUDE.md`, regla B.

Igual que en `cost-based-query-optimizer`/`lock-manager-deadlock-detector`,
la referencia no reutiliza ningún camino de código del motor real: si algún
día diverge un bug de correctness (p. ej. un caso de `NULL` mal propagado
en un join, o un orden de columnas equivocado), estos tests deberían
detectarlo aquí, no en producción.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence

import pytest
from naive_reference import Row, batch_to_rows, naive_execute

from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.models import (
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    FunctionCall,
    InList,
    IsNull,
    JoinType,
    Like,
    Literal,
    LiteralType,
    OrderByItem,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalLimit,
    PhysicalNestedLoopJoin,
    PhysicalOperator,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
    SelectItem,
    TableRef,
)
from vectorized_execution_engine.pipeline import execute_plan

SEEDS = (1, 2, 3, 4, 5)


def _random_source(seed: int) -> InMemoryTableSource:
    rng = random.Random(seed)
    # Tamaño de batch deliberadamente pequeño y no alineado con el volumen de
    # datos generado, para forzar que scan/filter/join troceen en más de un
    # batch — un tamaño que coincidiera con el total de filas ocultaría bugs
    # de frontera entre batches.
    source = InMemoryTableSource(batch_size=7)

    customer_count = rng.randint(5, 15)
    countries = ["US", "UK", "ES", "DE", None]
    customers_rows = [
        {"id": i, "country": rng.choice(countries), "name": f"cust{i}"}
        for i in range(customer_count)
    ]
    source.add_table(
        "customers",
        {"id": ColumnType.INTEGER, "country": ColumnType.STRING, "name": ColumnType.STRING},
        customers_rows,
    )

    order_count = rng.randint(20, 60)
    statuses = ["PENDING", "SHIPPED", "CANCELLED"]
    # 999 nunca es un id de cliente real: fuerza filas de `orders` sin
    # coincidencia en los joins externos (RIGHT/FULL).
    customer_ids = [None, 999, *range(customer_count)]
    orders_rows = []
    for i in range(order_count):
        amount = None if rng.random() < 0.1 else round(rng.uniform(1, 200), 2)
        orders_rows.append(
            {
                "id": i,
                "customer_id": rng.choice(customer_ids),
                "amount": amount,
                "status": rng.choice(statuses),
            }
        )
    source.add_table(
        "orders",
        {
            "id": ColumnType.INTEGER,
            "customer_id": ColumnType.INTEGER,
            "amount": ColumnType.FLOAT,
            "status": ColumnType.STRING,
        },
        orders_rows,
    )
    return source


def _values_close(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _row_equal(a: Row, b: Row) -> bool:
    return a.keys() == b.keys() and all(_values_close(a[k], b[k]) for k in a)


def _real_rows(plan: PhysicalOperator, source: InMemoryTableSource) -> list[Row]:
    rows: list[Row] = []
    for batch in execute_plan(plan, source):
        rows.extend(batch_to_rows(batch))
    return rows


def _assert_ordered_match(plan: PhysicalOperator, source: InMemoryTableSource) -> None:
    real_rows = _real_rows(plan, source)
    naive_rows, _ = naive_execute(plan, source)
    assert len(real_rows) == len(naive_rows)
    for real_row, naive_row in zip(real_rows, naive_rows, strict=True):
        assert _row_equal(real_row, naive_row), (real_row, naive_row)


def _assert_unordered_match(plan: PhysicalOperator, source: InMemoryTableSource) -> None:
    """Sólo válido cuando ninguna columna del resultado es fruto de una
    reducción numérica (`SUM`/`AVG`): compara exactamente vía multiconjunto,
    sin tolerancia — no hay margen de error de coma flotante posible cuando
    las filas sólo se mueven/filtran/combinan, nunca se recalculan."""

    def canonical(rows: Sequence[Row]) -> Counter[tuple[tuple[str, object], ...]]:
        return Counter(tuple(sorted(row.items())) for row in rows)

    real_rows = _real_rows(plan, source)
    naive_rows, _ = naive_execute(plan, source)
    assert canonical(real_rows) == canonical(naive_rows)


def _customers_scan() -> PhysicalTableScan:
    return PhysicalTableScan(TableRef("customers", "c"), estimated_rows=1, estimated_cost=1)


def _orders_scan() -> PhysicalTableScan:
    return PhysicalTableScan(TableRef("orders", "o"), estimated_rows=1, estimated_cost=1)


def _equi_condition() -> BinaryOp:
    return BinaryOp(ColumnRef("id", "c"), BinaryOperator.EQ, ColumnRef("customer_id", "o"))


@pytest.mark.parametrize("seed", SEEDS)
def test_filter_gt_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    plan = PhysicalFilter(
        _orders_scan(),
        BinaryOp(ColumnRef("amount", "o"), BinaryOperator.GT, Literal(50.0, LiteralType.FLOAT)),
        estimated_rows=1,
        estimated_cost=1,
    )
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_filter_or_status_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    predicate = BinaryOp(
        BinaryOp(
            ColumnRef("status", "o"), BinaryOperator.EQ, Literal("PENDING", LiteralType.STRING)
        ),
        BinaryOperator.OR,
        BinaryOp(
            ColumnRef("status", "o"), BinaryOperator.EQ, Literal("SHIPPED", LiteralType.STRING)
        ),
    )
    plan = PhysicalFilter(_orders_scan(), predicate, estimated_rows=1, estimated_cost=1)
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_filter_between_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    predicate = Between(
        ColumnRef("amount", "o"),
        Literal(20.0, LiteralType.FLOAT),
        Literal(100.0, LiteralType.FLOAT),
    )
    plan = PhysicalFilter(_orders_scan(), predicate, estimated_rows=1, estimated_cost=1)
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_filter_in_list_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    predicate = InList(
        ColumnRef("status", "o"),
        (Literal("PENDING", LiteralType.STRING), Literal("CANCELLED", LiteralType.STRING)),
    )
    plan = PhysicalFilter(_orders_scan(), predicate, estimated_rows=1, estimated_cost=1)
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_filter_like_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    predicate = Like(ColumnRef("name", "c"), Literal("cust1%", LiteralType.STRING))
    plan = PhysicalFilter(_customers_scan(), predicate, estimated_rows=1, estimated_cost=1)
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_filter_is_null_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    predicate = IsNull(ColumnRef("country", "c"))
    plan = PhysicalFilter(_customers_scan(), predicate, estimated_rows=1, estimated_cost=1)
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "join_type", [JoinType.INNER, JoinType.LEFT, JoinType.RIGHT, JoinType.FULL]
)
@pytest.mark.parametrize("join_cls", [PhysicalNestedLoopJoin, PhysicalHashJoin])
def test_join_matches_naive(seed: int, join_type: JoinType, join_cls: type) -> None:
    source = _random_source(seed)
    plan = join_cls(
        _customers_scan(),
        _orders_scan(),
        join_type,
        _equi_condition(),
        estimated_rows=1,
        estimated_cost=1,
    )
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_project_with_alias_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    plan = PhysicalProject(
        _orders_scan(),
        (
            SelectItem(ColumnRef("id", "o"), alias="order_id"),
            SelectItem(ColumnRef("amount", "o"), alias="total"),
        ),
        estimated_rows=1,
        estimated_cost=1,
    )
    _assert_unordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_join_aggregate_sorted_matches_naive_with_float_tolerance(seed: int) -> None:
    source = _random_source(seed)
    aggregated = PhysicalHashAggregate(
        PhysicalHashJoin(
            _customers_scan(),
            _orders_scan(),
            JoinType.INNER,
            _equi_condition(),
            estimated_rows=1,
            estimated_cost=1,
        ),
        group_by=(ColumnRef("country", "c"),),
        aggregates=(
            FunctionCall("SUM", arguments=(ColumnRef("amount", "o"),)),
            FunctionCall("COUNT", star_argument=True),
            FunctionCall("AVG", arguments=(ColumnRef("amount", "o"),)),
        ),
        estimated_rows=1,
        estimated_cost=1,
    )
    plan = PhysicalSort(
        aggregated,
        order_by=(OrderByItem(ColumnRef("country", "c")),),
        estimated_rows=1,
        estimated_cost=1,
    )
    _assert_ordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_sort_desc_and_limit_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    sorted_plan = PhysicalSort(
        _orders_scan(),
        order_by=(OrderByItem(ColumnRef("amount", "o")),),
        estimated_rows=1,
        estimated_cost=1,
    )
    plan = PhysicalLimit(sorted_plan, limit=5, offset=2, estimated_rows=1, estimated_cost=1)
    _assert_ordered_match(plan, source)


@pytest.mark.parametrize("seed", SEEDS)
def test_full_pipeline_matches_naive(seed: int) -> None:
    source = _random_source(seed)
    joined = PhysicalHashJoin(
        _customers_scan(),
        _orders_scan(),
        JoinType.LEFT,
        _equi_condition(),
        estimated_rows=1,
        estimated_cost=1,
    )
    filtered = PhysicalFilter(
        joined,
        IsNull(ColumnRef("amount", "o"), negated=True),
        estimated_rows=1,
        estimated_cost=1,
    )
    projected = PhysicalProject(
        filtered,
        (SelectItem(ColumnRef("name", "c")), SelectItem(ColumnRef("amount", "o"), alias="amount")),
        estimated_rows=1,
        estimated_cost=1,
    )
    sorted_plan = PhysicalSort(
        projected, order_by=(OrderByItem(ColumnRef("amount")),), estimated_rows=1, estimated_cost=1
    )
    _assert_ordered_match(sorted_plan, source)
