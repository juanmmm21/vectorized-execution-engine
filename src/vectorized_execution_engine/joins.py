"""`NestedLoopJoinOperator` y `HashJoinOperator`.

Los cuatro `JoinType` comparten la misma convención de esquema de salida
(columnas de `left` seguidas de las de `right`, ver `batch.concat_schemas`)
y el mismo mecanismo de "fila nula" para el lado sin coincidencia
(`batch.null_batch`); sólo cambia qué lado necesita ese relleno y cuándo se
sabe que una fila se quedó sin pareja.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .batch import RowBatch, concat_batches, concat_schemas, hstack_batches, null_batch
from .errors import UnresolvedColumnError, UnsupportedJoinConditionError
from .expressions import evaluate, resolve_column, to_native
from .models import BinaryOp, BinaryOperator, ColumnRef, Expression, JoinType, Schema
from .protocols import MemoryBudget, Operator, estimate_batch_bytes


@dataclass(slots=True)
class NestedLoopJoinOperator:
    """Aplicable a cualquier condición (o a ninguna: `condition=None`
    representa el producto cartesiano). El propio bucle anidado se
    vectoriza generando de golpe, vía `np.repeat`/`np.tile`, todos los
    pares `(fila_left, fila_right)` de un par de batches (nunca comparando
    fila a fila en Python) y evaluando la condición sobre ese batch
    combinado de una sola vez.

    Bloqueante en el lado `right`: necesita reproducirlo íntegro por cada
    batch de `left` (y, para `RIGHT`/`FULL`, agotar primero todo `left`
    antes de poder emitir las filas de `right` que nunca coincidieron), así
    que lo materializa por completo en una lista de batches. No lleva
    `MemoryBudget` — a diferencia de `HashJoinOperator`, el DoD de este
    motor sólo exige límite de memoria explícito para hash join, sort y
    aggregate; un nested-loop join sin condición de igualdad aprovechable
    ya es, por construcción, la opción más costosa que el optimizador
    escoge sólo cuando no hay alternativa mejor.
    """

    left: Operator
    right: Operator
    join_type: JoinType
    condition: Expression | None

    @property
    def output_schema(self) -> Schema:
        return concat_schemas(self.left.output_schema, self.right.output_schema)

    def execute(self) -> Iterator[RowBatch]:
        out_schema = self.output_schema
        right_schema = self.right.output_schema
        left_schema = self.left.output_schema
        right_batches = list(self.right.execute())
        right_matched = [np.zeros(b.row_count, dtype=np.bool_) for b in right_batches]
        needs_left_unmatched = self.join_type in (JoinType.LEFT, JoinType.FULL)
        needs_right_unmatched = self.join_type in (JoinType.RIGHT, JoinType.FULL)

        for left_batch in self.left.execute():
            left_matched = np.zeros(left_batch.row_count, dtype=np.bool_)
            for right_pos, right_batch in enumerate(right_batches):
                l_count = left_batch.row_count
                r_count = right_batch.row_count
                if l_count == 0 or r_count == 0:
                    continue
                left_idx = np.repeat(np.arange(l_count), r_count)
                right_idx = np.tile(np.arange(r_count), l_count)
                combined = hstack_batches(
                    left_batch.take(left_idx), right_batch.take(right_idx), out_schema
                )
                if self.condition is not None:
                    values, nulls = evaluate(self.condition, combined)
                    match_mask = values & ~nulls
                else:
                    match_mask = np.ones(combined.row_count, dtype=np.bool_)
                if not np.any(match_mask):
                    continue
                left_matched[left_idx[match_mask]] = True
                right_matched[right_pos][right_idx[match_mask]] = True
                yield combined.take(np.nonzero(match_mask)[0])
            if needs_left_unmatched:
                unmatched = np.nonzero(~left_matched)[0]
                if unmatched.size > 0:
                    yield hstack_batches(
                        left_batch.take(unmatched),
                        null_batch(right_schema, unmatched.size),
                        out_schema,
                    )

        if needs_right_unmatched:
            for right_pos, right_batch in enumerate(right_batches):
                unmatched = np.nonzero(~right_matched[right_pos])[0]
                if unmatched.size > 0:
                    yield hstack_batches(
                        null_batch(left_schema, unmatched.size),
                        right_batch.take(unmatched),
                        out_schema,
                    )


def _column_in_schema(ref: ColumnRef, schema: Schema) -> bool:
    try:
        resolve_column(schema, ref)
        return True
    except UnresolvedColumnError:
        return False


def _split_equi_condition(
    condition: Expression, left_schema: Schema, right_schema: Schema
) -> tuple[ColumnRef, ColumnRef]:
    """Determina qué lado de la igualdad de `condition` corresponde a
    `left_schema` y cuál a `right_schema`, sin asumir un orden fijo en el
    árbol de la expresión (`a.x = b.y` y `b.y = a.x` son equivalentes)."""
    if not isinstance(condition, BinaryOp) or condition.operator is not BinaryOperator.EQ:
        raise UnsupportedJoinConditionError(
            "PhysicalHashJoin.condition debe ser una igualdad (=) entre columnas"
        )
    if not isinstance(condition.left, ColumnRef) or not isinstance(condition.right, ColumnRef):
        raise UnsupportedJoinConditionError(
            "PhysicalHashJoin.condition debe comparar dos referencias de columna"
        )
    if _column_in_schema(condition.left, left_schema) and _column_in_schema(
        condition.right, right_schema
    ):
        return condition.left, condition.right
    if _column_in_schema(condition.left, right_schema) and _column_in_schema(
        condition.right, left_schema
    ):
        return condition.right, condition.left
    raise UnsupportedJoinConditionError(
        "las columnas de la igualdad no casan una con cada lado del join"
    )


@dataclass(slots=True)
class HashJoinOperator:
    """Sólo válido cuando `condition` es una igualdad de columna a columna,
    una por cada lado. `left` es el lado de construcción de la tabla hash
    (se materializa por completo, bajo `memory_budget`); `right` se sondea
    en streaming, batch a batch.

    El propio hashing es inherentemente una operación entrada-a-entrada (no
    vectorizable con NumPy puro sin sacrificar generalidad de tipos — ni
    siquiera pandas/DuckDB vectorizan la construcción de la tabla hash en
    sí); lo que sí se vectoriza es la materialización de las filas de
    salida ya emparejadas, mediante `RowBatch.take` (fancy indexing) sobre
    los arrays de índices acumulados durante el sondeo, nunca copiando
    valor a valor en un bucle.
    """

    left: Operator
    right: Operator
    join_type: JoinType
    condition: Expression
    memory_budget: MemoryBudget = field(default_factory=MemoryBudget)

    @property
    def output_schema(self) -> Schema:
        return concat_schemas(self.left.output_schema, self.right.output_schema)

    def execute(self) -> Iterator[RowBatch]:
        left_schema = self.left.output_schema
        right_schema = self.right.output_schema
        out_schema = concat_schemas(left_schema, right_schema)
        left_key_ref, right_key_ref = _split_equi_condition(
            self.condition, left_schema, right_schema
        )

        left_materialized = concat_batches(left_schema, list(self.left.execute()))
        self.memory_budget.reserve("HashJoinOperator", estimate_batch_bytes(left_materialized))

        left_key_values, left_key_nulls = evaluate(left_key_ref, left_materialized)
        hash_table: dict[object, list[int]] = {}
        for i in range(left_materialized.row_count):
            if left_key_nulls[i]:
                continue
            hash_table.setdefault(to_native(left_key_values[i]), []).append(i)

        left_matched = np.zeros(left_materialized.row_count, dtype=np.bool_)
        needs_left_unmatched = self.join_type in (JoinType.LEFT, JoinType.FULL)
        needs_right_unmatched = self.join_type in (JoinType.RIGHT, JoinType.FULL)

        for right_batch in self.right.execute():
            if right_batch.row_count == 0:
                continue
            right_key_values, right_key_nulls = evaluate(right_key_ref, right_batch)
            left_indices: list[int] = []
            right_indices: list[int] = []
            right_matched_batch = np.zeros(right_batch.row_count, dtype=np.bool_)
            for j in range(right_batch.row_count):
                if right_key_nulls[j]:
                    continue
                matches = hash_table.get(to_native(right_key_values[j]))
                if not matches:
                    continue
                right_matched_batch[j] = True
                for i in matches:
                    left_indices.append(i)
                    right_indices.append(j)
                    left_matched[i] = True
            if left_indices:
                left_idx_arr = np.array(left_indices, dtype=np.int64)
                right_idx_arr = np.array(right_indices, dtype=np.int64)
                yield hstack_batches(
                    left_materialized.take(left_idx_arr),
                    right_batch.take(right_idx_arr),
                    out_schema,
                )
            if needs_right_unmatched:
                unmatched = np.nonzero(~right_matched_batch)[0]
                if unmatched.size > 0:
                    yield hstack_batches(
                        null_batch(left_schema, unmatched.size),
                        right_batch.take(unmatched),
                        out_schema,
                    )

        if needs_left_unmatched:
            unmatched = np.nonzero(~left_matched)[0]
            if unmatched.size > 0:
                yield hstack_batches(
                    left_materialized.take(unmatched),
                    null_batch(right_schema, unmatched.size),
                    out_schema,
                )
