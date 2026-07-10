from vectorized_execution_engine.aggregate import HashAggregateOperator
from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.models import ColumnRef, ColumnType, FunctionCall, TableRef
from vectorized_execution_engine.operators import ScanOperator


def _orders_source() -> InMemoryTableSource:
    source = InMemoryTableSource()
    source.add_table(
        "orders",
        {"customer_id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [
            {"customer_id": 1, "amount": 10.0},
            {"customer_id": 1, "amount": 20.0},
            {"customer_id": 2, "amount": 30.0},
            {"customer_id": 2, "amount": None},
            {"customer_id": None, "amount": 5.0},
            {"customer_id": None, "amount": 7.0},
        ],
    )
    return source


def _rows(operator: HashAggregateOperator) -> list[dict[str, object]]:
    result = []
    for batch in operator.execute():
        for i in range(batch.row_count):
            row = {
                name: (None if batch.nulls[name][i] else batch.columns[name][i])
                for name in batch.schema.columns
            }
            result.append(row)
    return result


def test_group_by_sums_and_counts_per_group() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(ColumnRef("customer_id", "o"),),
        aggregates=(
            FunctionCall("SUM", arguments=(ColumnRef("amount", "o"),)),
            FunctionCall("COUNT", star_argument=True),
        ),
    )
    rows_by_customer = {row["o.customer_id"]: row for row in _rows(agg)}
    assert rows_by_customer[1]["SUM(o.amount)"] == 30.0
    assert rows_by_customer[1]["COUNT(*)"] == 2
    assert rows_by_customer[2]["SUM(o.amount)"] == 30.0
    assert rows_by_customer[2]["COUNT(*)"] == 2


def test_null_group_key_groups_together() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(ColumnRef("customer_id", "o"),),
        aggregates=(FunctionCall("COUNT", star_argument=True),),
    )
    rows = _rows(agg)
    null_group_rows = [row for row in rows if row["o.customer_id"] is None]
    assert len(null_group_rows) == 1
    assert null_group_rows[0]["COUNT(*)"] == 2


def test_sum_skips_nulls() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(ColumnRef("customer_id", "o"),),
        aggregates=(FunctionCall("SUM", arguments=(ColumnRef("amount", "o"),)),),
    )
    rows_by_customer = {row["o.customer_id"]: row for row in _rows(agg)}
    assert rows_by_customer[2]["SUM(o.amount)"] == 30.0


def test_avg_of_all_null_group_is_null() -> None:
    source = InMemoryTableSource()
    source.add_table(
        "orders",
        {"customer_id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [{"customer_id": 1, "amount": None}],
    )
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(ColumnRef("customer_id", "o"),),
        aggregates=(FunctionCall("AVG", arguments=(ColumnRef("amount", "o"),)),),
    )
    rows = _rows(agg)
    assert rows[0]["AVG(o.amount)"] is None


def test_global_aggregate_without_group_by_on_empty_table_returns_one_row() -> None:
    source = InMemoryTableSource()
    source.add_table("orders", {"amount": ColumnType.FLOAT}, [])
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(),
        aggregates=(FunctionCall("COUNT", star_argument=True),),
    )
    rows = _rows(agg)
    assert len(rows) == 1
    assert rows[0]["COUNT(*)"] == 0


def test_distinct_count() -> None:
    source = InMemoryTableSource()
    source.add_table(
        "orders",
        {"customer_id": ColumnType.INTEGER, "amount": ColumnType.FLOAT},
        [
            {"customer_id": 1, "amount": 10.0},
            {"customer_id": 1, "amount": 10.0},
            {"customer_id": 1, "amount": 20.0},
        ],
    )
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(ColumnRef("customer_id", "o"),),
        aggregates=(FunctionCall("COUNT", arguments=(ColumnRef("amount", "o"),), distinct=True),),
    )
    rows = _rows(agg)
    assert rows[0]["COUNT(DISTINCT o.amount)"] == 2


def test_min_and_max() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    agg = HashAggregateOperator(
        scan,
        group_by=(),
        aggregates=(
            FunctionCall("MIN", arguments=(ColumnRef("amount", "o"),)),
            FunctionCall("MAX", arguments=(ColumnRef("amount", "o"),)),
        ),
    )
    rows = _rows(agg)
    assert rows[0]["MIN(o.amount)"] == 5.0
    assert rows[0]["MAX(o.amount)"] == 30.0
