from dataclasses import dataclass

import pytest

from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.errors import MemoryLimitExceededError, UnsupportedOperatorError
from vectorized_execution_engine.models import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    JoinType,
    Literal,
    LiteralType,
    OrderByItem,
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalLimit,
    PhysicalOperator,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
    SelectItem,
    Star,
    TableRef,
)
from vectorized_execution_engine.pipeline import build_operator_tree, collect, execute_plan, explain
from vectorized_execution_engine.protocols import MemoryBudget


def _source() -> InMemoryTableSource:
    source = InMemoryTableSource()
    source.add_table(
        "orders",
        {"id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [{"id": 1, "amount": 10.0}, {"id": 2, "amount": 20.0}, {"id": 3, "amount": 30.0}],
    )
    return source


def test_build_operator_tree_scan_filter_project() -> None:
    source = _source()
    plan = PhysicalProject(
        PhysicalFilter(
            PhysicalTableScan(TableRef("orders", "o"), estimated_rows=3, estimated_cost=3),
            BinaryOp(ColumnRef("amount", "o"), BinaryOperator.GT, Literal(15.0, LiteralType.FLOAT)),
            estimated_rows=2,
            estimated_cost=3,
        ),
        (SelectItem(Star()),),
        estimated_rows=2,
        estimated_cost=3,
    )
    result = collect(plan, source)
    assert result.row_count == 2


def test_execute_plan_streams_batches() -> None:
    source = _source()
    plan = PhysicalTableScan(TableRef("orders", "o"), estimated_rows=3, estimated_cost=3)
    total = sum(batch.row_count for batch in execute_plan(plan, source))
    assert total == 3


def test_build_operator_tree_rejects_unknown_node() -> None:
    @dataclass(frozen=True, slots=True)
    class FakePhysicalNode(PhysicalOperator):
        estimated_rows: float
        estimated_cost: float

    source = _source()
    with pytest.raises(UnsupportedOperatorError):
        build_operator_tree(FakePhysicalNode(1, 1), source)


def test_explain_includes_all_nodes() -> None:
    plan = PhysicalLimit(
        PhysicalSort(
            PhysicalTableScan(TableRef("orders", "o"), estimated_rows=3, estimated_cost=3),
            order_by=(OrderByItem(ColumnRef("amount", "o")),),
            estimated_rows=3,
            estimated_cost=3,
        ),
        limit=1,
        offset=0,
        estimated_rows=1,
        estimated_cost=3,
    )
    text = explain(plan)
    assert "PhysicalLimit" in text
    assert "PhysicalSort" in text
    assert "PhysicalTableScan" in text


def test_full_pipeline_join_aggregate_sort_limit() -> None:
    source = InMemoryTableSource()
    source.add_table(
        "customers",
        {"id": ColumnType.INTEGER, "country": ColumnType.STRING},
        [{"id": 1, "country": "US"}],
    )
    source.add_table(
        "orders",
        {"customer_id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [{"customer_id": 1, "amount": 5.0}, {"customer_id": 1, "amount": 15.0}],
    )
    plan = PhysicalLimit(
        PhysicalSort(
            PhysicalHashAggregate(
                PhysicalHashJoin(
                    PhysicalTableScan(
                        TableRef("customers", "c"), estimated_rows=1, estimated_cost=1
                    ),
                    PhysicalTableScan(TableRef("orders", "o"), estimated_rows=2, estimated_cost=2),
                    JoinType.INNER,
                    BinaryOp(
                        ColumnRef("id", "c"), BinaryOperator.EQ, ColumnRef("customer_id", "o")
                    ),
                    estimated_rows=2,
                    estimated_cost=3,
                ),
                group_by=(ColumnRef("country", "c"),),
                aggregates=(),
                estimated_rows=1,
                estimated_cost=3,
            ),
            order_by=(OrderByItem(ColumnRef("country", "c")),),
            estimated_rows=1,
            estimated_cost=3,
        ),
        limit=1,
        offset=0,
        estimated_rows=1,
        estimated_cost=3,
    )
    result = collect(plan, source)
    assert result.row_count == 1
    assert result.columns["c.country"][0] == "US"


def test_memory_budget_is_shared_across_all_blocking_operators_in_one_tree() -> None:
    source = InMemoryTableSource()
    source.add_table("t", {"a": ColumnType.INTEGER}, [{"a": i} for i in range(500)])
    plan = PhysicalSort(
        PhysicalHashAggregate(
            PhysicalTableScan(TableRef("t", "t"), estimated_rows=500, estimated_cost=500),
            group_by=(ColumnRef("a", "t"),),
            aggregates=(),
            estimated_rows=500,
            estimated_cost=500,
        ),
        order_by=(OrderByItem(ColumnRef("a", "t")),),
        estimated_rows=500,
        estimated_cost=500,
    )
    budget = MemoryBudget(max_bytes=100)
    operator = build_operator_tree(plan, source, memory_budget=budget)
    with pytest.raises(MemoryLimitExceededError):
        list(operator.execute())
