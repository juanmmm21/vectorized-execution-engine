from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.models import (
    ColumnRef,
    ColumnType,
    OrderByItem,
    OrderDirection,
    TableRef,
)
from vectorized_execution_engine.operators import ScanOperator
from vectorized_execution_engine.sort import SortOperator


def _values(sort_operator: SortOperator, column: str) -> list[object]:
    result = []
    for batch in sort_operator.execute():
        for i in range(batch.row_count):
            result.append(None if batch.nulls[column][i] else batch.columns[column][i])
    return result


def test_sort_ascending() -> None:
    source = InMemoryTableSource()
    source.add_table("t", {"a": ColumnType.INTEGER}, [{"a": 3}, {"a": 1}, {"a": 2}])
    scan = ScanOperator(TableRef("t"), source)
    sort_op = SortOperator(scan, (OrderByItem(ColumnRef("a")),))
    assert _values(sort_op, "t.a") == [1, 2, 3]


def test_sort_descending() -> None:
    source = InMemoryTableSource()
    source.add_table("t", {"a": ColumnType.INTEGER}, [{"a": 3}, {"a": 1}, {"a": 2}])
    scan = ScanOperator(TableRef("t"), source)
    sort_op = SortOperator(scan, (OrderByItem(ColumnRef("a"), OrderDirection.DESC),))
    assert _values(sort_op, "t.a") == [3, 2, 1]


def test_sort_nulls_last_ascending_and_descending() -> None:
    source = InMemoryTableSource()
    source.add_table("t", {"a": ColumnType.INTEGER}, [{"a": 2}, {"a": None}, {"a": 1}])
    scan_asc = ScanOperator(TableRef("t"), source)
    sort_asc = SortOperator(scan_asc, (OrderByItem(ColumnRef("a")),))
    assert _values(sort_asc, "t.a") == [1, 2, None]

    scan_desc = ScanOperator(TableRef("t"), source)
    sort_desc = SortOperator(scan_desc, (OrderByItem(ColumnRef("a"), OrderDirection.DESC),))
    assert _values(sort_desc, "t.a") == [2, 1, None]


def test_sort_is_stable_on_ties() -> None:
    source = InMemoryTableSource()
    source.add_table(
        "t",
        {"key": ColumnType.INTEGER, "seq": ColumnType.INTEGER},
        [{"key": 1, "seq": 0}, {"key": 1, "seq": 1}, {"key": 1, "seq": 2}],
    )
    scan = ScanOperator(TableRef("t"), source)
    sort_op = SortOperator(scan, (OrderByItem(ColumnRef("key")),))
    assert _values(sort_op, "t.seq") == [0, 1, 2]


def test_sort_multi_key() -> None:
    source = InMemoryTableSource()
    source.add_table(
        "t",
        {"a": ColumnType.INTEGER, "b": ColumnType.INTEGER},
        [{"a": 1, "b": 2}, {"a": 1, "b": 1}, {"a": 0, "b": 5}],
    )
    scan = ScanOperator(TableRef("t"), source)
    sort_op = SortOperator(
        scan, (OrderByItem(ColumnRef("a")), OrderByItem(ColumnRef("b"), OrderDirection.DESC))
    )
    pairs = []
    for batch in sort_op.execute():
        for i in range(batch.row_count):
            pairs.append((batch.columns["t.a"][i], batch.columns["t.b"][i]))
    assert pairs == [(0, 5), (1, 2), (1, 1)]


def test_sort_on_string_column() -> None:
    source = InMemoryTableSource()
    source.add_table(
        "t",
        {"name": ColumnType.STRING},
        [{"name": "banana"}, {"name": "apple"}, {"name": "cherry"}],
    )
    scan = ScanOperator(TableRef("t"), source)
    sort_op = SortOperator(scan, (OrderByItem(ColumnRef("name")),))
    assert _values(sort_op, "t.name") == ["apple", "banana", "cherry"]
