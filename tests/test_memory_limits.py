import pytest

from vectorized_execution_engine.aggregate import HashAggregateOperator
from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.errors import MemoryLimitExceededError
from vectorized_execution_engine.joins import HashJoinOperator
from vectorized_execution_engine.models import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    FunctionCall,
    JoinType,
    OrderByItem,
    TableRef,
)
from vectorized_execution_engine.operators import ScanOperator
from vectorized_execution_engine.protocols import MemoryBudget
from vectorized_execution_engine.sort import SortOperator


def _wide_source(row_count: int) -> InMemoryTableSource:
    source = InMemoryTableSource()
    source.add_table(
        "t",
        {"a": ColumnType.INTEGER, "b": ColumnType.INTEGER},
        [{"a": i, "b": i} for i in range(row_count)],
    )
    return source


def test_hash_join_raises_when_build_side_exceeds_budget() -> None:
    source = _wide_source(1000)
    left = ScanOperator(TableRef("t", "l"), source)
    right = ScanOperator(TableRef("t", "r"), source)
    condition = BinaryOp(ColumnRef("a", "l"), BinaryOperator.EQ, ColumnRef("a", "r"))
    join = HashJoinOperator(
        left, right, JoinType.INNER, condition, memory_budget=MemoryBudget(max_bytes=16)
    )
    with pytest.raises(MemoryLimitExceededError):
        list(join.execute())


def test_hash_aggregate_raises_when_input_exceeds_budget() -> None:
    source = _wide_source(1000)
    scan = ScanOperator(TableRef("t", "t"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(ColumnRef("a", "t"),),
        aggregates=(FunctionCall("COUNT", star_argument=True),),
        memory_budget=MemoryBudget(max_bytes=16),
    )
    with pytest.raises(MemoryLimitExceededError):
        list(agg.execute())


def test_sort_raises_when_input_exceeds_budget() -> None:
    source = _wide_source(1000)
    scan = ScanOperator(TableRef("t", "t"), source)
    sort_op = SortOperator(
        scan, (OrderByItem(ColumnRef("a", "t")),), memory_budget=MemoryBudget(max_bytes=16)
    )
    with pytest.raises(MemoryLimitExceededError):
        list(sort_op.execute())


def test_generous_budget_does_not_raise() -> None:
    source = _wide_source(100)
    scan = ScanOperator(TableRef("t", "t"), source)
    sort_op = SortOperator(scan, (OrderByItem(ColumnRef("a", "t")),))
    total = sum(batch.row_count for batch in sort_op.execute())
    assert total == 100


def test_memory_budget_reserve_accumulates_and_raises() -> None:
    budget = MemoryBudget(max_bytes=100)
    budget.reserve("op1", 50)
    assert budget.used_bytes == 50
    with pytest.raises(MemoryLimitExceededError):
        budget.reserve("op2", 60)


def test_memory_budget_release_frees_capacity() -> None:
    budget = MemoryBudget(max_bytes=100)
    budget.reserve("op1", 80)
    budget.release(80)
    budget.reserve("op2", 80)
    assert budget.used_bytes == 80
