import numpy as np
import pytest

from vectorized_execution_engine.batch import (
    DEFAULT_BATCH_SIZE,
    InMemoryTableSource,
    RowBatch,
    chunk_batch,
    columns_from_python,
    concat_batches,
    concat_schemas,
    dtype_for,
    hstack_batches,
    null_batch,
)
from vectorized_execution_engine.errors import SchemaConflictError
from vectorized_execution_engine.models import ColumnType, Schema


def test_dtype_for_maps_all_column_types() -> None:
    assert dtype_for(ColumnType.INTEGER) == np.dtype(np.int64)
    assert dtype_for(ColumnType.FLOAT) == np.dtype(np.float64)
    assert dtype_for(ColumnType.BOOLEAN) == np.dtype(np.bool_)
    assert dtype_for(ColumnType.STRING) == np.dtype(object)


def test_columns_from_python_marks_none_as_null() -> None:
    data, nulls = columns_from_python([1, None, 3], ColumnType.INTEGER)
    assert list(nulls) == [False, True, False]
    assert data[0] == 1
    assert data[2] == 3


def test_row_batch_row_count() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    data, nulls = columns_from_python([1, 2, 3], ColumnType.INTEGER)
    batch = RowBatch(schema, {"a": data}, {"a": nulls})
    assert batch.row_count == 3


def test_row_batch_take_gathers_rows_including_negative_as_null() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    data, nulls = columns_from_python([10, 20, 30], ColumnType.INTEGER)
    batch = RowBatch(schema, {"a": data}, {"a": nulls})
    gathered = batch.take(np.array([2, -1, 0]))
    assert gathered.nulls["a"].tolist() == [False, True, False]
    assert gathered.columns["a"][0] == 30
    assert gathered.columns["a"][2] == 10


def test_row_batch_slice_rows() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    data, nulls = columns_from_python([1, 2, 3, 4], ColumnType.INTEGER)
    batch = RowBatch(schema, {"a": data}, {"a": nulls})
    sliced = batch.slice_rows(1, 3)
    assert sliced.columns["a"].tolist() == [2, 3]


def test_concat_schemas_rejects_overlapping_columns() -> None:
    left = Schema(("a",), {"a": ColumnType.INTEGER})
    right = Schema(("a",), {"a": ColumnType.INTEGER})
    with pytest.raises(SchemaConflictError):
        concat_schemas(left, right)


def test_concat_schemas_preserves_order() -> None:
    left = Schema(("a", "b"), {"a": ColumnType.INTEGER, "b": ColumnType.STRING})
    right = Schema(("c",), {"c": ColumnType.FLOAT})
    combined = concat_schemas(left, right)
    assert combined.columns == ("a", "b", "c")


def test_concat_batches_handles_empty_sequence() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    result = concat_batches(schema, [])
    assert result.row_count == 0


def test_concat_batches_joins_multiple_batches() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    data1, nulls1 = columns_from_python([1, 2], ColumnType.INTEGER)
    data2, nulls2 = columns_from_python([3], ColumnType.INTEGER)
    b1 = RowBatch(schema, {"a": data1}, {"a": nulls1})
    b2 = RowBatch(schema, {"a": data2}, {"a": nulls2})
    result = concat_batches(schema, [b1, b2])
    assert result.columns["a"].tolist() == [1, 2, 3]


def test_null_batch_is_all_null() -> None:
    schema = Schema(("a",), {"a": ColumnType.STRING})
    batch = null_batch(schema, 3)
    assert batch.row_count == 3
    assert all(batch.nulls["a"])


def test_hstack_batches_combines_columns() -> None:
    left_schema = Schema(("a",), {"a": ColumnType.INTEGER})
    right_schema = Schema(("b",), {"b": ColumnType.STRING})
    combined_schema = concat_schemas(left_schema, right_schema)
    left_data, left_nulls = columns_from_python([1], ColumnType.INTEGER)
    right_data, right_nulls = columns_from_python(["x"], ColumnType.STRING)
    left = RowBatch(left_schema, {"a": left_data}, {"a": left_nulls})
    right = RowBatch(right_schema, {"b": right_data}, {"b": right_nulls})
    combined = hstack_batches(left, right, combined_schema)
    assert combined.columns["a"][0] == 1
    assert combined.columns["b"][0] == "x"


def test_chunk_batch_splits_into_requested_size() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    data, nulls = columns_from_python(list(range(5)), ColumnType.INTEGER)
    batch = RowBatch(schema, {"a": data}, {"a": nulls})
    chunks = list(chunk_batch(batch, batch_size=2))
    assert [c.row_count for c in chunks] == [2, 2, 1]


def test_chunk_batch_yields_nothing_for_empty_batch() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    data, nulls = columns_from_python([], ColumnType.INTEGER)
    batch = RowBatch(schema, {"a": data}, {"a": nulls})
    assert list(chunk_batch(batch)) == []


def test_default_batch_size_is_positive() -> None:
    assert DEFAULT_BATCH_SIZE > 0


def test_in_memory_table_source_scan_respects_batch_size() -> None:
    source = InMemoryTableSource(batch_size=2)
    source.add_table(
        "t",
        {"id": ColumnType.INTEGER},
        [{"id": i} for i in range(5)],
    )
    batches = list(source.scan("t"))
    assert [b.row_count for b in batches] == [2, 2, 1]
    all_ids = [value for b in batches for value in b.columns["id"].tolist()]
    assert all_ids == [0, 1, 2, 3, 4]


def test_in_memory_table_source_marks_nulls() -> None:
    source = InMemoryTableSource()
    source.add_table("t", {"x": ColumnType.INTEGER}, [{"x": 1}, {"x": None}])
    (batch,) = list(source.scan("t"))
    assert batch.nulls["x"].tolist() == [False, True]


def test_in_memory_table_source_empty_table_yields_no_batches() -> None:
    source = InMemoryTableSource()
    source.add_table("t", {"x": ColumnType.INTEGER}, [])
    assert list(source.scan("t")) == []
