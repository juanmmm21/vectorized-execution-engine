"""`DistinctOperator`: deduplicación de filas completas."""

from __future__ import annotations

import pytest

from vectorized_execution_engine.batch import InMemoryTableSource, concat_batches
from vectorized_execution_engine.distinct import DistinctOperator
from vectorized_execution_engine.errors import MemoryLimitExceededError
from vectorized_execution_engine.models import (
    ColumnType,
    PhysicalDistinct,
    PhysicalTableScan,
    TableRef,
)
from vectorized_execution_engine.operators import ScanOperator
from vectorized_execution_engine.pipeline import collect
from vectorized_execution_engine.protocols import MemoryBudget


def _source(rows: list[dict[str, object]]) -> InMemoryTableSource:
    source = InMemoryTableSource()
    source.add_table(
        "t",
        {"a": ColumnType.INTEGER, "b": ColumnType.STRING},
        rows,  # type: ignore[arg-type]
    )
    return source


def _distinct_rows(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    source = _source(rows)
    operator = DistinctOperator(ScanOperator(TableRef("t"), source))
    batch = concat_batches(operator.output_schema, list(operator.execute()))
    out: list[tuple[object, ...]] = []
    for i in range(batch.row_count):
        out.append(
            tuple(
                None
                if batch.nulls[name][i]
                else batch.columns[name][i].item()
                if hasattr(batch.columns[name][i], "item")
                else batch.columns[name][i]
                for name in batch.schema.columns
            )
        )
    return out


class TestDistinctOperator:
    def test_removes_duplicate_rows_keeping_first_occurrence_order(self) -> None:
        rows = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 1, "b": "x"},
            {"a": 1, "b": "z"},
            {"a": 2, "b": "y"},
        ]
        assert _distinct_rows(rows) == [(1, "x"), (2, "y"), (1, "z")]

    def test_nulls_are_equal_for_distinct(self) -> None:
        # Semántica SQL de deduplicación: NULL agrupa con NULL (a diferencia
        # de una comparación '=', donde NULL = NULL es NULL).
        rows = [
            {"a": None, "b": None},
            {"a": None, "b": None},
            {"a": 1, "b": None},
        ]
        assert _distinct_rows(rows) == [(None, None), (1, None)]

    def test_all_unique_input_passes_through(self) -> None:
        rows = [{"a": n, "b": f"v{n}"} for n in range(5)]
        assert len(_distinct_rows(rows)) == 5

    def test_empty_input_yields_no_rows(self) -> None:
        assert _distinct_rows([]) == []

    def test_memory_budget_is_enforced(self) -> None:
        rows = [{"a": n, "b": "relleno" * 50} for n in range(500)]
        source = _source(rows)
        operator = DistinctOperator(
            ScanOperator(TableRef("t"), source), memory_budget=MemoryBudget(1024)
        )
        with pytest.raises(MemoryLimitExceededError):
            list(operator.execute())


class TestPipelineIntegration:
    def test_build_operator_tree_handles_physical_distinct(self) -> None:
        source = _source([{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        plan = PhysicalDistinct(
            input=PhysicalTableScan(table=TableRef("t"), estimated_rows=3.0, estimated_cost=3.0),
            estimated_rows=2.0,
            estimated_cost=6.0,
        )
        batch = collect(plan, source)
        assert batch.row_count == 2
