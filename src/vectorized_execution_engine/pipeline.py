"""Traduce un plan físico (`models.PhysicalOperator`) en un árbol de
`Operator` ejecutable y ofrece las funciones de conveniencia que usan la
CLI y los tests para correr un plan de principio a fin.

`build_operator_tree` es el único sitio del motor que conoce los nueve
tipos concretos `Physical*` — cualquier operador nuevo (ver DoD:
"extensible sin tocar el motor") sólo necesita una entrada aquí, no
cambios en `operators.py`/`joins.py`/`aggregate.py`/`sort.py`.

Presupuesto de memoria compartido: todas las llamadas recursivas de una
misma invocación de `build_operator_tree` reciben el mismo `MemoryBudget`,
para que el límite configurado represente el pico de memoria conjunto de
todos los operadores bloqueantes de la consulta (hash join + aggregate +
sort), no un límite independiente por operador que permitiría que la suma
total creciera sin control — ver `protocols.MemoryBudget`. Ninguna reserva
se libera durante la ejecución: los arrays materializados siguen vivos
mientras el generador de un operador bloqueante no se haya agotado, así
que "liberar" antes de tiempo no reflejaría la memoria realmente en uso.
"""

from __future__ import annotations

from collections.abc import Iterator

from .aggregate import HashAggregateOperator
from .batch import RowBatch, TableSource, concat_batches
from .errors import UnsupportedOperatorError
from .joins import HashJoinOperator, NestedLoopJoinOperator
from .models import (
    PhysicalFilter,
    PhysicalHashAggregate,
    PhysicalHashJoin,
    PhysicalIndexScan,
    PhysicalLimit,
    PhysicalNestedLoopJoin,
    PhysicalOperator,
    PhysicalProject,
    PhysicalSort,
    PhysicalTableScan,
)
from .operators import (
    FilterOperator,
    IndexScanOperator,
    LimitOperator,
    ProjectOperator,
    ScanOperator,
)
from .protocols import DEFAULT_MEMORY_BUDGET_BYTES, MemoryBudget, Operator
from .sort import SortOperator


def build_operator_tree(
    plan: PhysicalOperator,
    source: TableSource,
    *,
    memory_budget: MemoryBudget | None = None,
) -> Operator:
    """Construye recursivamente el operador correspondiente a `plan`.

    `memory_budget` no debería pasarse explícitamente desde fuera salvo en
    llamadas recursivas internas: para ejecutar un plan completo, usar
    `execute_plan`/`collect`, que crean un `MemoryBudget` nuevo por
    ejecución (ver su docstring sobre por qué no se comparte entre
    ejecuciones distintas).
    """
    budget = memory_budget if memory_budget is not None else MemoryBudget()

    if isinstance(plan, PhysicalTableScan):
        return ScanOperator(plan.table, source)
    if isinstance(plan, PhysicalIndexScan):
        return IndexScanOperator(plan.table, plan.index_predicate, source)
    if isinstance(plan, PhysicalFilter):
        return FilterOperator(
            build_operator_tree(plan.input, source, memory_budget=budget), plan.predicate
        )
    if isinstance(plan, PhysicalNestedLoopJoin):
        return NestedLoopJoinOperator(
            build_operator_tree(plan.left, source, memory_budget=budget),
            build_operator_tree(plan.right, source, memory_budget=budget),
            plan.join_type,
            plan.condition,
        )
    if isinstance(plan, PhysicalHashJoin):
        return HashJoinOperator(
            build_operator_tree(plan.left, source, memory_budget=budget),
            build_operator_tree(plan.right, source, memory_budget=budget),
            plan.join_type,
            plan.condition,
            memory_budget=budget,
        )
    if isinstance(plan, PhysicalHashAggregate):
        return HashAggregateOperator(
            build_operator_tree(plan.input, source, memory_budget=budget),
            plan.group_by,
            plan.aggregates,
            memory_budget=budget,
        )
    if isinstance(plan, PhysicalProject):
        return ProjectOperator(
            build_operator_tree(plan.input, source, memory_budget=budget), plan.items
        )
    if isinstance(plan, PhysicalSort):
        return SortOperator(
            build_operator_tree(plan.input, source, memory_budget=budget),
            plan.order_by,
            memory_budget=budget,
        )
    if isinstance(plan, PhysicalLimit):
        return LimitOperator(
            build_operator_tree(plan.input, source, memory_budget=budget), plan.limit, plan.offset
        )
    raise UnsupportedOperatorError(plan)


def execute_plan(
    plan: PhysicalOperator,
    source: TableSource,
    *,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> Iterator[RowBatch]:
    """Construye y ejecuta `plan` de principio a fin, en streaming."""
    operator = build_operator_tree(plan, source, memory_budget=MemoryBudget(memory_budget_bytes))
    yield from operator.execute()


def collect(
    plan: PhysicalOperator,
    source: TableSource,
    *,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> RowBatch:
    """Ejecuta `plan` y materializa todo su resultado en un único `RowBatch`
    — conveniente para tests y para la CLI de demostración; no usar sobre
    resultados que puedan no caber en memoria (para eso está el streaming
    de `execute_plan`)."""
    operator = build_operator_tree(plan, source, memory_budget=MemoryBudget(memory_budget_bytes))
    return concat_batches(operator.output_schema, list(operator.execute()))


def _children(plan: PhysicalOperator) -> tuple[PhysicalOperator, ...]:
    if isinstance(
        plan,
        PhysicalFilter | PhysicalHashAggregate | PhysicalProject | PhysicalSort | PhysicalLimit,
    ):
        return (plan.input,)
    if isinstance(plan, PhysicalNestedLoopJoin | PhysicalHashJoin):
        return (plan.left, plan.right)
    return ()


def explain(plan: PhysicalOperator, indent: int = 0) -> str:
    """Representación textual, indentada por nivel, del árbol de plan
    físico de entrada — usada por la CLI de demostración (`EXPLAIN`)."""
    prefix = "  " * indent
    node_name = type(plan).__name__
    header = f"{node_name} (rows={plan.estimated_rows:.0f}, cost={plan.estimated_cost:.1f})"
    lines = [f"{prefix}{header}"]
    for child in _children(plan):
        lines.append(explain(child, indent + 1))
    return "\n".join(lines)
