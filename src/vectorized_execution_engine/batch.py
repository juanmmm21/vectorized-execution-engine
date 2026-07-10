"""Batch columnar (`RowBatch`) y la interfaz de lectura de filas
(`TableSource`) sobre la que corre el operador de scan.

`TableSource` redefine, sin import cruzado (ver `../AGENTS.md`), el mismo
papel que `RowStore` cumple en `mvcc-transaction-manager`: una interfaz
"tonta" que sólo sabe entregar filas de una tabla, sin conocer nada de
transacciones ni planes. La diferencia deliberada es que aquí la unidad de
entrega es un `RowBatch` columnar (no una versión de fila individual): este
motor consume lo que sea que el storage/transacciones activos produzcan,
pero necesita que llegue ya en batches para poder vectorizar — el adaptador
real que agrupa filas de `RowStore` en `RowBatch`es vive en `nanosql`, nunca
aquí (mismo principio que el resto del ecosistema).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import SchemaConflictError
from .models import ColumnType, LiteralValue, Schema

#: Tamaño de batch por defecto para cualquier operador que produzca/trocee
#: filas (scan, resultado final de un operador bloqueante). Un número
#: explícito y documentado, no disperso como constante mágica en cada sitio
#: — ver `../CLAUDE.md`, sección C (aplicado aquí al tamaño de batch en vez
#: de al tamaño de página, ya que este motor no persiste nada). 2048 filas es
#: un punto intermedio habitual en motores vectorizados reales (DuckDB usa
#: 2048 por defecto): suficientemente grande para amortizar el overhead de
#: por-batch de NumPy, suficientemente pequeño para que un batch quepa
#: cómodamente en cache de CPU para la mayoría de anchos de fila.
DEFAULT_BATCH_SIZE = 2048


def dtype_for(column_type: ColumnType) -> np.dtype:
    """Dtype de NumPy usado para almacenar los valores de una columna.

    Las columnas de texto usan `dtype=object` (cada celda es un `str` de
    Python): NumPy no tiene un tipo de cadena de longitud variable nativo
    utilizable de forma general sin fijar de antemano una longitud máxima,
    y fijar una trunca datos reales de forma silenciosa — se prefiere pagar
    el coste de un array de objetos a cambio de corrección.
    """
    if column_type is ColumnType.INTEGER:
        return np.dtype(np.int64)
    if column_type is ColumnType.FLOAT:
        return np.dtype(np.float64)
    if column_type is ColumnType.BOOLEAN:
        return np.dtype(np.bool_)
    return np.dtype(object)


def _placeholder_for(column_type: ColumnType) -> LiteralValue:
    """Valor de relleno usado en la posición de una celda nula.

    Nunca se lee sin comprobar antes la máscara de nulos correspondiente
    (ver todos los usos en `expressions.py`/operadores): es puro relleno
    para que el array tenga un dtype homogéneo, no un valor con significado.
    """
    if column_type is ColumnType.INTEGER:
        return 0
    if column_type is ColumnType.FLOAT:
        return 0.0
    if column_type is ColumnType.BOOLEAN:
        return False
    return ""


def columns_from_python(
    values: Sequence[LiteralValue], column_type: ColumnType
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte una secuencia Python (con `None` marcando nulos) en el par
    `(valores, nulos)` que usa `RowBatch` internamente."""
    placeholder = _placeholder_for(column_type)
    nulls = np.fromiter((value is None for value in values), dtype=np.bool_, count=len(values))
    dense = [placeholder if value is None else value for value in values]
    data = np.array(dense, dtype=dtype_for(column_type))
    return data, nulls


@dataclass(slots=True)
class RowBatch:
    """Un batch de filas en formato columnar.

    `columns[c]` y `nulls[c]` tienen siempre la misma longitud para toda
    columna `c` de `schema.columns` — es el invariante que toda operación
    vectorizada de este módulo asume sin volver a comprobar en cada paso.
    `nulls[c][i] is True` significa que `columns[c][i]` es sólo relleno
    (ver `_placeholder_for`) y no debe leerse directamente.
    """

    schema: Schema
    columns: dict[str, np.ndarray]
    nulls: dict[str, np.ndarray]

    @property
    def row_count(self) -> int:
        if not self.schema.columns:
            return 0
        return len(self.columns[self.schema.columns[0]])

    def take(self, indices: np.ndarray) -> RowBatch:
        """Reúne las filas en `indices` (fancy indexing vectorizado, no
        bucle fila a fila). Un índice de `-1` selecciona una fila nula en
        todas sus columnas — el mecanismo que usan los joins externos
        (`LEFT`/`RIGHT`/`FULL`) para rellenar el lado sin coincidencia."""
        valid = indices >= 0
        safe_indices = np.where(valid, indices, 0)
        result_columns: dict[str, np.ndarray] = {}
        result_nulls: dict[str, np.ndarray] = {}
        for name in self.schema.columns:
            gathered = self.columns[name][safe_indices]
            gathered_nulls = self.nulls[name][safe_indices].copy()
            gathered_nulls[~valid] = True
            result_columns[name] = gathered
            result_nulls[name] = gathered_nulls
        return RowBatch(self.schema, result_columns, result_nulls)

    def slice_rows(self, start: int, stop: int) -> RowBatch:
        return RowBatch(
            self.schema,
            {name: arr[start:stop] for name, arr in self.columns.items()},
            {name: arr[start:stop] for name, arr in self.nulls.items()},
        )


def concat_schemas(left: Schema, right: Schema) -> Schema:
    """Esquema combinado de un join: columnas de `left` seguidas de las de
    `right`, en ese orden — convención compartida por los cuatro
    `JoinType` (ver `models.PhysicalHashJoin`/`PhysicalNestedLoopJoin`:
    sólo qué lado puede llevar nulos cambia con el tipo de join, nunca el
    orden de columnas)."""
    overlap = set(left.columns) & set(right.columns)
    if overlap:
        raise SchemaConflictError(next(iter(overlap)))
    return Schema(
        columns=left.columns + right.columns,
        types={**left.types, **right.types},
    )


def concat_batches(schema: Schema, batches: Sequence[RowBatch]) -> RowBatch:
    """Materializa una secuencia de batches (mismo esquema) en uno solo.

    Usado por los operadores bloqueantes (`HashJoinOperator` en su lado de
    construcción, `HashAggregateOperator`, `SortOperator`) para tener toda
    su entrada disponible como arrays contiguos antes de operar — el propio
    `MemoryBudget` (ver `protocols.py`) es quien limita cuánto se les deja
    crecer.
    """
    non_empty = [batch for batch in batches if batch.row_count > 0]
    if not non_empty:
        return RowBatch(
            schema,
            {name: np.array([], dtype=dtype_for(schema.type_of(name))) for name in schema.columns},
            {name: np.array([], dtype=np.bool_) for name in schema.columns},
        )
    return RowBatch(
        schema,
        {name: np.concatenate([b.columns[name] for b in non_empty]) for name in schema.columns},
        {name: np.concatenate([b.nulls[name] for b in non_empty]) for name in schema.columns},
    )


def null_batch(schema: Schema, row_count: int) -> RowBatch:
    """Batch de `row_count` filas, todas nulas en todas sus columnas — el
    relleno que usan los joins externos (`LEFT`/`RIGHT`/`FULL`) para el lado
    sin coincidencia."""
    columns = {
        name: np.zeros(row_count, dtype=dtype_for(schema.type_of(name))) for name in schema.columns
    }
    nulls = {name: np.ones(row_count, dtype=np.bool_) for name in schema.columns}
    return RowBatch(schema, columns, nulls)


def hstack_batches(left: RowBatch, right: RowBatch, schema: Schema) -> RowBatch:
    """Combina dos batches con el mismo número de filas, columna a columna
    (sin solapar nombres), en uno solo con `schema` — el mecanismo con el
    que los joins arman su fila de salida a partir de una fila de cada
    lado ya seleccionada (ver `joins.py`)."""
    return RowBatch(schema, {**left.columns, **right.columns}, {**left.nulls, **right.nulls})


def chunk_batch(batch: RowBatch, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[RowBatch]:
    """Trocea un `RowBatch` ya materializado en piezas de como mucho
    `batch_size` filas, para que el resultado de un operador bloqueante
    vuelva a fluir en batches hacia el operador de encima."""
    total = batch.row_count
    if total == 0:
        return
    for start in range(0, total, batch_size):
        yield batch.slice_rows(start, min(start + batch_size, total))


@runtime_checkable
class TableSource(Protocol):
    """Fuente de filas de una tabla física, indexada por nombre de tabla.

    Deliberadamente "tonta" (mismo espíritu que `RowStore` de
    `mvcc-transaction-manager`): no conoce alias, joins ni el resto del
    plan físico — sólo expone el esquema físico de una tabla y sus filas en
    batches. La cualificación por alias (`TableRef.effective_name`) la
    aplica `operators.ScanOperator`, no esta interfaz.
    """

    def schema(self, table_name: str) -> Schema:
        """Esquema físico (columnas sin cualificar) de `table_name`."""
        ...

    def scan(self, table_name: str) -> Iterator[RowBatch]:
        """Todas las filas de `table_name`, en batches de tamaño acotado."""
        ...


class InMemoryTableSource:
    """Implementación en memoria de `TableSource`.

    Sirve tanto para los tests como para la CLI de demostración — igual que
    `InMemoryRowStore` en `mvcc-transaction-manager`, no es la única
    implementación posible: cualquier adaptador que cumpla `TableSource`
    (por ejemplo uno que agrupe en batches las filas visibles que entrega
    `mvcc-transaction-manager` sobre `bplus-tree-storage-engine`) puede
    sustituirla sin tocar `pipeline.py` — esa integración real ocurre
    siempre dentro de `nanosql`.
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._batch_size = batch_size
        self._schemas: dict[str, Schema] = {}
        self._data: dict[str, dict[str, list[LiteralValue]]] = {}

    def add_table(
        self,
        table_name: str,
        columns: dict[str, ColumnType],
        rows: Sequence[dict[str, LiteralValue]],
    ) -> None:
        """Registra una tabla completa a partir de filas como `dict`s
        Python (`None` marca un valor nulo) — la forma más natural de
        construir datos de prueba/demo sin tener que pensar en NumPy."""
        column_names = tuple(columns.keys())
        self._schemas[table_name] = Schema(columns=column_names, types=dict(columns))
        self._data[table_name] = {name: [row.get(name) for row in rows] for name in column_names}

    def schema(self, table_name: str) -> Schema:
        return self._schemas[table_name]

    def scan(self, table_name: str) -> Iterator[RowBatch]:
        schema = self._schemas[table_name]
        table_data = self._data[table_name]
        row_count = len(next(iter(table_data.values()))) if table_data else 0
        for start in range(0, row_count, self._batch_size):
            stop = min(start + self._batch_size, row_count)
            columns: dict[str, np.ndarray] = {}
            nulls: dict[str, np.ndarray] = {}
            for name in schema.columns:
                values, null_mask = columns_from_python(
                    table_data[name][start:stop], schema.type_of(name)
                )
                columns[name] = values
                nulls[name] = null_mask
            yield RowBatch(schema, columns, nulls)
