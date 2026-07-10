import pytest

from vectorized_execution_engine.batch import InMemoryTableSource
from vectorized_execution_engine.errors import SchemaConflictError
from vectorized_execution_engine.models import (
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    Literal,
    LiteralType,
    SelectItem,
    Star,
    TableRef,
)
from vectorized_execution_engine.operators import (
    FilterOperator,
    IndexScanOperator,
    LimitOperator,
    ProjectOperator,
    ScanOperator,
)


def _orders_source() -> InMemoryTableSource:
    source = InMemoryTableSource(batch_size=2)
    source.add_table(
        "orders",
        {"id": ColumnType.INTEGER, "amount": ColumnType.FLOAT, "status": ColumnType.STRING},
        [
            {"id": 1, "amount": 10.0, "status": "PENDING"},
            {"id": 2, "amount": 20.0, "status": "SHIPPED"},
            {"id": 3, "amount": 30.0, "status": "PENDING"},
        ],
    )
    return source


def test_scan_qualifies_columns_with_alias() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    assert scan.output_schema.columns == ("o.id", "o.amount", "o.status")
    batches = list(scan.execute())
    total_rows = sum(b.row_count for b in batches)
    assert total_rows == 3


def test_scan_uses_table_name_without_alias() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders"), source)
    assert scan.output_schema.columns[0] == "orders.id"


def test_filter_keeps_only_matching_rows() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    predicate = BinaryOp(
        ColumnRef("status", "o"), BinaryOperator.EQ, Literal("PENDING", LiteralType.STRING)
    )
    filtered = FilterOperator(scan, predicate)
    ids = [value for batch in filtered.execute() for value in batch.columns["o.id"].tolist()]
    assert ids == [1, 3]


def test_filter_skips_empty_result_batches() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    predicate = BinaryOp(
        ColumnRef("status", "o"), BinaryOperator.EQ, Literal("MISSING", LiteralType.STRING)
    )
    filtered = FilterOperator(scan, predicate)
    assert list(filtered.execute()) == []


def test_index_scan_behaves_like_scan_plus_filter() -> None:
    source = _orders_source()
    predicate = BinaryOp(
        ColumnRef("status", "o"), BinaryOperator.EQ, Literal("SHIPPED", LiteralType.STRING)
    )
    index_scan = IndexScanOperator(TableRef("orders", "o"), predicate, source)
    ids = [value for batch in index_scan.execute() for value in batch.columns["o.id"].tolist()]
    assert ids == [2]


def test_project_star_passes_through() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    project = ProjectOperator(scan, (SelectItem(Star()),))
    assert project.output_schema.columns == scan.output_schema.columns


def test_project_aliases_columns() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    project = ProjectOperator(scan, (SelectItem(ColumnRef("amount", "o"), alias="total"),))
    assert project.output_schema.columns == ("total",)
    values = [v for batch in project.execute() for v in batch.columns["total"].tolist()]
    assert values == [10.0, 20.0, 30.0]


def test_project_default_name_for_bare_column() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    project = ProjectOperator(scan, (SelectItem(ColumnRef("amount", "o")),))
    assert project.output_schema.columns == ("amount",)


def test_project_duplicate_output_name_raises() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    project = ProjectOperator(
        scan,
        (
            SelectItem(ColumnRef("id", "o"), alias="x"),
            SelectItem(ColumnRef("amount", "o"), alias="x"),
        ),
    )
    with pytest.raises(SchemaConflictError):
        _ = project.output_schema


def test_limit_applies_offset_and_limit_across_batches() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    limited = LimitOperator(scan, limit=1, offset=1)
    ids = [value for batch in limited.execute() for value in batch.columns["o.id"].tolist()]
    assert ids == [2]


def test_limit_none_only_applies_offset() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    limited = LimitOperator(scan, limit=None, offset=2)
    ids = [value for batch in limited.execute() for value in batch.columns["o.id"].tolist()]
    assert ids == [3]


def test_limit_zero_yields_nothing() -> None:
    source = _orders_source()
    scan = ScanOperator(TableRef("orders", "o"), source)
    limited = LimitOperator(scan, limit=0, offset=0)
    assert list(limited.execute()) == []
