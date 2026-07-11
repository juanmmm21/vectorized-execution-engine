"""`DistinctOperator`: deduplicación de filas completas (`SELECT DISTINCT`).

Convención de nulos: para DISTINCT dos NULL se consideran el mismo valor
(igual que en `GROUP BY`, y a diferencia de una comparación `=`, donde
NULL = NULL es NULL) — es la semántica de SQL estándar para deduplicación.

Como el hash join y el hash aggregate, la tabla de vistos se construye
entrada a entrada con un `set` de Python (inherente a cualquier
deduplicación por hashing: cada clave de fila debe consultarse en una
estructura asociativa); la salida, en cambio, se materializa con
`RowBatch.take` (fancy indexing vectorizado) sobre los índices de primera
aparición, nunca copiando fila a fila.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .batch import RowBatch, chunk_batch, concat_batches
from .expressions import to_native
from .models import LiteralValue, Schema
from .protocols import MemoryBudget, Operator, estimate_batch_bytes


@dataclass(slots=True)
class DistinctOperator:
    """Bloqueante: una fila solo puede emitirse sabiendo que ninguna
    anterior era idéntica, así que materializa toda su entrada bajo
    `memory_budget` (mismo contrato que `SortOperator`). El orden de salida
    es el de primera aparición — determinista dado un input determinista."""

    input: Operator
    memory_budget: MemoryBudget = field(default_factory=MemoryBudget)

    @property
    def output_schema(self) -> Schema:
        return self.input.output_schema

    def execute(self) -> Iterator[RowBatch]:
        schema = self.input.output_schema
        materialized = concat_batches(schema, list(self.input.execute()))
        self.memory_budget.reserve("DistinctOperator", estimate_batch_bytes(materialized))

        columns = [materialized.columns[name] for name in schema.columns]
        nulls = [materialized.nulls[name] for name in schema.columns]
        seen: set[tuple[LiteralValue, ...]] = set()
        first_occurrences: list[int] = []
        for row in range(materialized.row_count):
            key = tuple(
                None if nulls[c][row] else to_native(columns[c][row]) for c in range(len(columns))
            )
            if key not in seen:
                seen.add(key)
                first_occurrences.append(row)
        indices = np.array(first_occurrences, dtype=np.int64)
        yield from chunk_batch(materialized.take(indices))
