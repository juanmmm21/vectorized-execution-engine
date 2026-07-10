"""`HashAggregateOperator`: agrupación por hash + agregados vectorizados.

La construcción de los grupos en sí (asignar cada fila a su cubo por
`GROUP BY`) es, igual que en `joins.HashJoinOperator`, una operación
entrada-a-entrada por naturaleza — no hay forma de particionar por clave
arbitraria (incluidas columnas de texto) con NumPy puro sin fijar de
antemano el conjunto de claves posibles. Lo que sí se vectoriza es el
cálculo del propio agregado (`SUM`/`AVG`/`MIN`/`MAX`/`COUNT`) sobre cada
grupo: una vez que se conoce el array de índices de fila de un grupo, la
reducción numérica corre como una única llamada de NumPy sobre ese array
(`np.sum`, `np.mean`, ...), nunca acumulando valor a valor en Python.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .batch import RowBatch, chunk_batch, columns_from_python, concat_batches
from .errors import SchemaConflictError, UnsupportedExpressionError
from .expressions import (
    EvaluatedColumn,
    evaluate,
    expression_key,
    infer_expression_type,
    resolve_column,
    to_native,
)
from .models import (
    AGGREGATE_FUNCTION_NAMES,
    ColumnRef,
    ColumnType,
    Expression,
    FunctionCall,
    LiteralValue,
    Schema,
)
from .protocols import MemoryBudget, Operator, estimate_batch_bytes

#: Marca de "clave de grupo nula": todas las filas cuyo valor de un mismo
#: `GROUP BY` sea nulo deben terminar en el mismo grupo (regla SQL estándar
#: — `NULL` agrupa consigo mismo aunque `NULL = NULL` sea `NULL`, no
#: `TRUE`). Un único objeto centinela por identidad basta como componente
#: de la tupla-clave del `dict` de agrupación.
_NULL_GROUP_KEY = object()


def _group_output_name(expr: Expression, schema: Schema) -> str:
    if isinstance(expr, ColumnRef):
        return resolve_column(schema, expr)
    return expression_key(expr)


def _aggregate_output_type(func: FunctionCall, input_schema: Schema) -> ColumnType:
    if func.name == "COUNT":
        return ColumnType.INTEGER
    if func.name == "AVG":
        return ColumnType.FLOAT
    if func.star_argument or not func.arguments:
        return ColumnType.FLOAT
    return infer_expression_type(func.arguments[0], input_schema)


def _compute_aggregate(
    func: FunctionCall, arg: EvaluatedColumn | None, indices: np.ndarray
) -> LiteralValue:
    if func.name not in AGGREGATE_FUNCTION_NAMES:
        raise UnsupportedExpressionError(f"función agregada no soportada: {func.name}")
    if func.star_argument:
        if func.name != "COUNT":
            raise UnsupportedExpressionError(f"{func.name}(*) no es una agregada soportada")
        return int(indices.size)
    if arg is None:
        raise UnsupportedExpressionError(f"{func.name} necesita un argumento explícito")
    values, nulls = arg
    group_values = values[indices]
    group_nulls = nulls[indices]
    non_null = group_values[~group_nulls]
    if func.distinct and non_null.size:
        non_null = np.unique(non_null)
    if func.name == "COUNT":
        return int(non_null.size)
    if non_null.size == 0:
        return None
    if func.name == "SUM":
        return to_native(np.sum(non_null))
    if func.name == "AVG":
        return float(np.mean(non_null))
    if func.name == "MIN":
        return to_native(np.min(non_null))
    return to_native(np.max(non_null))


@dataclass(slots=True)
class HashAggregateOperator:
    """Bloqueante: materializa toda su entrada bajo `memory_budget` antes de
    poder saber qué grupos existen — no hay forma de emitir un grupo antes
    de haber visto todas las filas que podrían pertenecer a él."""

    input: Operator
    group_by: tuple[Expression, ...]
    aggregates: tuple[FunctionCall, ...]
    memory_budget: MemoryBudget = field(default_factory=MemoryBudget)

    def _group_names(self, input_schema: Schema) -> tuple[str, ...]:
        return tuple(_group_output_name(expr, input_schema) for expr in self.group_by)

    def _aggregate_names(self) -> tuple[str, ...]:
        return tuple(expression_key(func) for func in self.aggregates)

    @property
    def output_schema(self) -> Schema:
        input_schema = self.input.output_schema
        group_names = self._group_names(input_schema)
        aggregate_names = self._aggregate_names()
        columns = group_names + aggregate_names
        if len(set(columns)) != len(columns):
            raise SchemaConflictError(next(name for name in columns if columns.count(name) > 1))
        types: dict[str, ColumnType] = {}
        for expr, name in zip(self.group_by, group_names, strict=True):
            types[name] = infer_expression_type(expr, input_schema)
        for func, name in zip(self.aggregates, aggregate_names, strict=True):
            types[name] = _aggregate_output_type(func, input_schema)
        return Schema(columns, types)

    def execute(self) -> Iterator[RowBatch]:
        input_schema = self.input.output_schema
        materialized = concat_batches(input_schema, list(self.input.execute()))
        self.memory_budget.reserve("HashAggregateOperator", estimate_batch_bytes(materialized))

        group_arrays = [evaluate(expr, materialized) for expr in self.group_by]
        aggregate_args = [
            None if func.star_argument else evaluate(func.arguments[0], materialized)
            for func in self.aggregates
        ]

        row_count = materialized.row_count
        groups: dict[tuple[object, ...], list[int]] = {}
        if row_count == 0 and not self.group_by:
            groups[()] = []
        for row in range(row_count):
            key = tuple(
                _NULL_GROUP_KEY if nulls[row] else to_native(values[row])
                for values, nulls in group_arrays
            )
            groups.setdefault(key, []).append(row)

        out_schema = self.output_schema
        group_names = self._group_names(input_schema)
        aggregate_names = self._aggregate_names()
        output_values: dict[str, list[LiteralValue]] = {name: [] for name in out_schema.columns}

        for row_indices in groups.values():
            idx_arr = np.array(row_indices, dtype=np.int64)
            for gi, name in enumerate(group_names):
                values, nulls = group_arrays[gi]
                row0 = row_indices[0]
                output_values[name].append(None if nulls[row0] else to_native(values[row0]))
            for func, name, arg in zip(
                self.aggregates, aggregate_names, aggregate_args, strict=True
            ):
                output_values[name].append(_compute_aggregate(func, arg, idx_arr))

        columns: dict[str, np.ndarray] = {}
        nulls_out: dict[str, np.ndarray] = {}
        for name in out_schema.columns:
            data, null_mask = columns_from_python(output_values[name], out_schema.type_of(name))
            columns[name] = data
            nulls_out[name] = null_mask
        yield from chunk_batch(RowBatch(out_schema, columns, nulls_out))
