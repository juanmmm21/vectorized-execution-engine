import pytest

from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.errors import UnsupportedJoinConditionError
from vectorized_execution_engine.joins import HashJoinOperator, NestedLoopJoinOperator
from vectorized_execution_engine.models import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    JoinType,
    Literal,
    LiteralType,
    TableRef,
)
from vectorized_execution_engine.operators import ScanOperator


def _source() -> InMemoryTableSource:
    source = InMemoryTableSource(batch_size=2)
    source.add_table(
        "customers",
        {"id": ColumnType.INTEGER, "name": ColumnType.STRING},
        [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}, {"id": 3, "name": "Alan"}],
    )
    source.add_table(
        "orders",
        {"id": ColumnType.INTEGER, "customer_id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [
            {"id": 100, "customer_id": 1, "amount": 10.0},
            {"id": 101, "customer_id": 1, "amount": 20.0},
            {"id": 102, "customer_id": 2, "amount": 30.0},
            {"id": 103, "customer_id": 99, "amount": 40.0},
        ],
    )
    return source


def _customers(source: InMemoryTableSource) -> ScanOperator:
    return ScanOperator(TableRef("customers", "c"), source)


def _orders(source: InMemoryTableSource) -> ScanOperator:
    return ScanOperator(TableRef("orders", "o"), source)


def _equi_condition() -> BinaryOp:
    return BinaryOp(ColumnRef("id", "c"), BinaryOperator.EQ, ColumnRef("customer_id", "o"))


def _pairs(join: object) -> set[tuple[object, object]]:
    result = set()
    for batch in join.execute():  # type: ignore[attr-defined]
        for i in range(batch.row_count):
            c_id = None if batch.nulls["c.id"][i] else batch.columns["c.id"][i]
            o_id = None if batch.nulls["o.id"][i] else batch.columns["o.id"][i]
            result.add((c_id, o_id))
    return result


@pytest.mark.parametrize("operator_cls", [NestedLoopJoinOperator, HashJoinOperator])
def test_inner_join_matches_only(operator_cls: type) -> None:
    source = _source()
    condition = _equi_condition()
    join = operator_cls(_customers(source), _orders(source), JoinType.INNER, condition)
    pairs = _pairs(join)
    assert pairs == {(1, 100), (1, 101), (2, 102)}


@pytest.mark.parametrize("operator_cls", [NestedLoopJoinOperator, HashJoinOperator])
def test_left_join_keeps_unmatched_left_rows(operator_cls: type) -> None:
    source = _source()
    condition = _equi_condition()
    join = operator_cls(_customers(source), _orders(source), JoinType.LEFT, condition)
    pairs = _pairs(join)
    assert (3, None) in pairs
    assert (1, 100) in pairs


@pytest.mark.parametrize("operator_cls", [NestedLoopJoinOperator, HashJoinOperator])
def test_right_join_keeps_unmatched_right_rows(operator_cls: type) -> None:
    source = _source()
    condition = _equi_condition()
    join = operator_cls(_customers(source), _orders(source), JoinType.RIGHT, condition)
    pairs = _pairs(join)
    assert (None, 103) in pairs


@pytest.mark.parametrize("operator_cls", [NestedLoopJoinOperator, HashJoinOperator])
def test_full_join_keeps_both_unmatched_sides(operator_cls: type) -> None:
    source = _source()
    condition = _equi_condition()
    join = operator_cls(_customers(source), _orders(source), JoinType.FULL, condition)
    pairs = _pairs(join)
    assert (3, None) in pairs
    assert (None, 103) in pairs
    assert (1, 100) in pairs


def test_nested_loop_join_cartesian_product_without_condition() -> None:
    source = _source()
    join = NestedLoopJoinOperator(_customers(source), _orders(source), JoinType.INNER, None)
    total = sum(batch.row_count for batch in join.execute())
    assert total == 3 * 4


def test_hash_join_rejects_non_equi_condition() -> None:
    source = _source()
    condition = BinaryOp(ColumnRef("id", "c"), BinaryOperator.LT, ColumnRef("customer_id", "o"))
    join = HashJoinOperator(_customers(source), _orders(source), JoinType.INNER, condition)
    with pytest.raises(UnsupportedJoinConditionError):
        list(join.execute())


def test_hash_join_rejects_condition_not_between_both_sides() -> None:
    source = _source()
    condition = BinaryOp(ColumnRef("id", "c"), BinaryOperator.EQ, Literal(1, LiteralType.INTEGER))
    join = HashJoinOperator(_customers(source), _orders(source), JoinType.INNER, condition)
    with pytest.raises(UnsupportedJoinConditionError):
        list(join.execute())


def test_hash_join_and_nested_loop_join_agree() -> None:
    source1 = _source()
    source2 = _source()
    condition = _equi_condition()
    nested = NestedLoopJoinOperator(_customers(source1), _orders(source1), JoinType.FULL, condition)
    hashed = HashJoinOperator(_customers(source2), _orders(source2), JoinType.FULL, condition)
    assert _pairs(nested) == _pairs(hashed)


def test_join_output_schema_is_left_columns_then_right_columns() -> None:
    source = _source()
    condition = _equi_condition()
    join = HashJoinOperator(_customers(source), _orders(source), JoinType.INNER, condition)
    assert join.output_schema.columns == ("c.id", "c.name", "o.id", "o.customer_id", "o.amount")
