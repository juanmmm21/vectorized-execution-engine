"""Operadores de streaming: `Scan`, `IndexScan`, `Filter`, `Project`, `Limit`.

Ninguno de los cinco es bloqueante: cada uno consume los batches de su
entrada uno a uno y produce su salida sin necesitar materializar nada por
completo, así que no llevan `MemoryBudget` (ver `protocols.py` — reservado
para `joins.py`/`aggregate.py`/`sort.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .batch import RowBatch, TableSource
from .errors import SchemaConflictError
from .expressions import evaluate, expression_key, infer_expression_type
from .models import ColumnRef, ColumnType, Expression, Schema, SelectItem, Star, TableRef
from .protocols import Operator


def _default_output_name(expr: Expression) -> str:
    """Nombre de columna de salida cuando un `SelectItem` no lleva alias:
    el nombre desnudo (sin cualificar) para una referencia de columna
    simple, la representación canónica de la expresión en cualquier otro
    caso — igual que haría `EXPLAIN` para etiquetar una columna calculada."""
    if isinstance(expr, ColumnRef):
        return expr.name
    return expression_key(expr)


@dataclass(slots=True)
class ScanOperator:
    """Recorrido completo de `table`, cualificando cada columna física con
    `table.effective_name` (alias si existe, si no el nombre real) para que
    los joins de más arriba puedan distinguir columnas del mismo nombre
    procedentes de tablas distintas."""

    table: TableRef
    source: TableSource

    @property
    def output_schema(self) -> Schema:
        physical = self.source.schema(self.table.name)
        prefix = self.table.effective_name
        columns = tuple(f"{prefix}.{name}" for name in physical.columns)
        types = {f"{prefix}.{name}": physical.type_of(name) for name in physical.columns}
        return Schema(columns, types)

    def execute(self) -> Iterator[RowBatch]:
        physical = self.source.schema(self.table.name)
        prefix = self.table.effective_name
        out_schema = self.output_schema
        for batch in self.source.scan(self.table.name):
            columns = {f"{prefix}.{name}": batch.columns[name] for name in physical.columns}
            nulls = {f"{prefix}.{name}": batch.nulls[name] for name in physical.columns}
            yield RowBatch(out_schema, columns, nulls)


@dataclass(slots=True)
class IndexScanOperator:
    """Ejecutado exactamente igual que `ScanOperator` más `index_predicate`
    aplicado como filtro (ver `models.PhysicalIndexScan`): el recorrido
    físico por páginas de índice es responsabilidad del storage engine
    activo, no de este motor — aquí sólo se honra el contrato de filas que
    produce ese acceso (delegar en `FilterOperator` reutiliza exactamente la
    misma evaluación vectorizada de predicado que un `PhysicalFilter`)."""

    table: TableRef
    index_predicate: Expression
    source: TableSource

    @property
    def output_schema(self) -> Schema:
        return ScanOperator(self.table, self.source).output_schema

    def execute(self) -> Iterator[RowBatch]:
        scan = ScanOperator(self.table, self.source)
        yield from FilterOperator(scan, self.index_predicate).execute()


@dataclass(slots=True)
class FilterOperator:
    input: Operator
    predicate: Expression

    @property
    def output_schema(self) -> Schema:
        return self.input.output_schema

    def execute(self) -> Iterator[RowBatch]:
        for batch in self.input.execute():
            values, nulls = evaluate(self.predicate, batch)
            mask = values & ~nulls
            if not np.any(mask):
                continue
            indices = np.nonzero(mask)[0]
            yield batch.take(indices)


@dataclass(slots=True)
class ProjectOperator:
    input: Operator
    items: tuple[SelectItem, ...]

    @property
    def _is_star(self) -> bool:
        return len(self.items) == 1 and isinstance(self.items[0].expression, Star)

    @property
    def output_schema(self) -> Schema:
        if self._is_star:
            return self.input.output_schema
        input_schema = self.input.output_schema
        columns: list[str] = []
        types: dict[str, ColumnType] = {}
        seen: set[str] = set()
        for item in self.items:
            name = item.alias if item.alias is not None else _default_output_name(item.expression)
            if name in seen:
                raise SchemaConflictError(name)
            seen.add(name)
            columns.append(name)
            types[name] = infer_expression_type(item.expression, input_schema)
        return Schema(tuple(columns), types)

    def execute(self) -> Iterator[RowBatch]:
        out_schema = self.output_schema
        for batch in self.input.execute():
            if self._is_star:
                yield batch
                continue
            columns = {}
            nulls = {}
            for item, name in zip(self.items, out_schema.columns, strict=True):
                values, null_mask = evaluate(item.expression, batch)
                columns[name] = values
                nulls[name] = null_mask
            yield RowBatch(out_schema, columns, nulls)


@dataclass(slots=True)
class LimitOperator:
    """`limit=None` significa sin tope (sólo aplica `offset`); ambos se
    aplican en streaming, cortando la extracción de batches de `input` en
    cuanto se alcanza `limit` — nunca se materializa más de lo necesario."""

    input: Operator
    limit: int | None
    offset: int | None

    @property
    def output_schema(self) -> Schema:
        return self.input.output_schema

    def execute(self) -> Iterator[RowBatch]:
        skip_remaining = self.offset or 0
        take_remaining = self.limit
        for batch in self.input.execute():
            if take_remaining is not None and take_remaining <= 0:
                return
            rows = batch.row_count
            start = 0
            if skip_remaining > 0:
                if skip_remaining >= rows:
                    skip_remaining -= rows
                    continue
                start = skip_remaining
                skip_remaining = 0
            stop = rows
            if take_remaining is not None and (stop - start) > take_remaining:
                stop = start + take_remaining
            sliced = batch.slice_rows(start, stop)
            if sliced.row_count > 0:
                if take_remaining is not None:
                    take_remaining -= sliced.row_count
                yield sliced
            if take_remaining is not None and take_remaining <= 0:
                return
