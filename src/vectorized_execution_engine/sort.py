"""`SortOperator`: ordenación multi-clave, estable, vectorizada con
`np.lexsort`.

Convención de nulos: siempre al final, tanto en `ASC` como en `DESC` — una
simplificación deliberada (SQL estándar/dialectos reales varían y algunos
exponen `NULLS FIRST`/`NULLS LAST` explícito, pero `OrderByItem` de este
ecosistema —ver `models.py`— no lleva ese campo, así que no hay forma de
que el plan físico de entrada pida lo contrario).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .batch import RowBatch, chunk_batch, concat_batches
from .expressions import evaluate
from .models import OrderByItem, OrderDirection, Schema
from .protocols import MemoryBudget, Operator, estimate_batch_bytes


def _sort_key(values: np.ndarray, descending: bool) -> np.ndarray:
    """Convierte `values` en un array numérico cuyo orden natural coincide
    con el de `values` (para texto: códigos de `np.unique` en orden
    lexicográfico), negado si la dirección es `DESC` — así una única
    llamada a `np.lexsort` puede combinar columnas de distinto tipo y
    distinta dirección sin casos especiales por dtype."""
    if values.dtype == object:
        _, codes = np.unique(values, return_inverse=True)
        key = codes.astype(np.int64)
    elif values.dtype == np.bool_:
        key = values.astype(np.int64)
    else:
        key = values
    return -key if descending else key


def _compute_order_indices(order_by: tuple[OrderByItem, ...], batch: RowBatch) -> np.ndarray:
    """`np.lexsort` trata el último array de la lista como clave primaria;
    para que el primer `OrderByItem` (el más significativo) gane, se
    recorren en orden inverso. Dentro de cada item, la máscara de nulos se
    añade DESPUÉS de la clave de valor, para que "es nulo" decida antes que
    el propio valor (nulos siempre al final, sin que la dirección de ese
    item la afecte)."""
    if not order_by:
        return np.arange(batch.row_count, dtype=np.int64)
    keys: list[np.ndarray] = []
    for item in reversed(order_by):
        values, nulls = evaluate(item.expression, batch)
        keys.append(_sort_key(values, item.direction is OrderDirection.DESC))
        keys.append(nulls.astype(np.int64))
    result: np.ndarray = np.lexsort(keys)
    return result


@dataclass(slots=True)
class SortOperator:
    """Bloqueante: no hay forma de saber la posición final de una fila sin
    haber visto ya todas las que podrían ir antes o después — materializa
    toda su entrada bajo `memory_budget` antes de ordenar."""

    input: Operator
    order_by: tuple[OrderByItem, ...]
    memory_budget: MemoryBudget = field(default_factory=MemoryBudget)

    @property
    def output_schema(self) -> Schema:
        return self.input.output_schema

    def execute(self) -> Iterator[RowBatch]:
        schema = self.input.output_schema
        materialized = concat_batches(schema, list(self.input.execute()))
        self.memory_budget.reserve("SortOperator", estimate_batch_bytes(materialized))
        order_indices = _compute_order_indices(self.order_by, materialized)
        yield from chunk_batch(materialized.take(order_indices))
