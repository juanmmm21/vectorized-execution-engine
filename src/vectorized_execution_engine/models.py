"""Tipos de entrada del motor de ejecución: subconjunto del AST de expresiones
y el árbol de plan físico de `cost-based-query-optimizer` redefinidos de
forma independiente (mismo nombre y forma de campos, cero imports cruzados —
ver `../AGENTS.md` y la regla de integración de `../CLAUDE.md`), más los
tipos columnares (`ColumnType`/`Schema`) propios de este motor.

Alcance deliberado: sólo lo que aparece dentro de un plan físico de
`SELECT` ya optimizado — el motor de ejecución no vuelve a decidir nada
(qué índice usar, qué orden de join), sólo ejecuta el árbol que le entrega
el optimizador. Por eso aquí no hay `SelectStatement` ni plan lógico, sólo
las expresiones que cuelgan de los nodos físicos (`predicate`, `condition`,
`items`, `order_by`, `group_by`/`aggregates`) y los propios nueve tipos
`Physical*` de `cost_based_query_optimizer.plan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# ---------------------------------------------------------------------------
# Expresiones (subconjunto de cost_based_query_optimizer/models.py, que a su
# vez redefine sql_query_parser/models.py)
# ---------------------------------------------------------------------------


class Expression:
    """Clase marcadora para cualquier nodo de expresión del plan físico de entrada."""


class LiteralType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    NULL = auto()


LiteralValue = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    value: LiteralValue
    literal_type: LiteralType


@dataclass(frozen=True, slots=True)
class ColumnRef(Expression):
    """Referencia a una columna, opcionalmente cualificada por tabla/alias
    (`t.col`). Sin cualificar sólo es resoluble si exactamente una columna
    del batch de entrada termina en `.col` (ver `expressions.resolve_column`)."""

    name: str
    table: str | None = None


@dataclass(frozen=True, slots=True)
class Star(Expression):
    """El comodín `*`, sólo válido como único `SelectItem` de un `PhysicalProject`."""


class BinaryOperator(Enum):
    OR = auto()
    AND = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()


@dataclass(frozen=True, slots=True)
class BinaryOp(Expression):
    left: Expression
    operator: BinaryOperator
    right: Expression


class UnaryOperator(Enum):
    NEG = auto()
    POS = auto()
    NOT = auto()


@dataclass(frozen=True, slots=True)
class UnaryOp(Expression):
    operator: UnaryOperator
    operand: Expression


@dataclass(frozen=True, slots=True)
class Between(Expression):
    operand: Expression
    low: Expression
    high: Expression
    negated: bool = False


@dataclass(frozen=True, slots=True)
class InList(Expression):
    operand: Expression
    values: tuple[Expression, ...]
    negated: bool = False


@dataclass(frozen=True, slots=True)
class Like(Expression):
    operand: Expression
    pattern: Expression
    negated: bool = False


@dataclass(frozen=True, slots=True)
class IsNull(Expression):
    operand: Expression
    negated: bool = False


@dataclass(frozen=True, slots=True)
class FunctionCall(Expression):
    """Llamada a función. Este motor sólo sabe *evaluar* las cinco agregadas
    clásicas (`AGGREGATE_FUNCTION_NAMES`), y sólo dentro de un
    `PhysicalHashAggregate` (ver `aggregate.py`); si una `FunctionCall`
    aparece en un `PhysicalProject`/`PhysicalFilter`, se busca como columna
    ya precomputada por el `HashAggregate` de debajo (ver
    `expressions.expression_key`), nunca se reevalúa desde cero."""

    name: str
    arguments: tuple[Expression, ...] = ()
    distinct: bool = False
    star_argument: bool = False


AGGREGATE_FUNCTION_NAMES = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


# ---------------------------------------------------------------------------
# Fragmentos compartidos por los nodos físicos
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectItem:
    expression: Expression
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class TableRef:
    name: str
    alias: str | None = None

    @property
    def effective_name(self) -> str:
        """Nombre por el que se cualifican las columnas de esta tabla en el
        batch de salida de su scan: el alias si existe, si no el nombre real."""
        return self.alias if self.alias is not None else self.name


class JoinType(Enum):
    INNER = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()


class OrderDirection(Enum):
    ASC = auto()
    DESC = auto()


@dataclass(frozen=True, slots=True)
class OrderByItem:
    expression: Expression
    direction: OrderDirection = OrderDirection.ASC


# ---------------------------------------------------------------------------
# Plan físico (idéntico en forma a cost_based_query_optimizer.plan)
# ---------------------------------------------------------------------------
#
# Misma convención que el optimizador: `estimated_rows`/`estimated_cost` se
# aceptan aquí por fidelidad de forma con la entrada real que produce
# `cost-based-query-optimizer`, pero este motor no los usa para decidir nada
# (ya no hay ninguna decisión de coste que tomar en tiempo de ejecución) —
# se conservan de puro paso, por si `nanosql` quiere mostrarlos junto al
# resultado (p.ej. en un `EXPLAIN ANALYZE`).


class PhysicalOperator:
    """Clase marcadora para cualquier nodo del plan físico de entrada."""

    __slots__ = ()
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalTableScan(PhysicalOperator):
    table: TableRef
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalIndexScan(PhysicalOperator):
    """Ejecutado exactamente igual que `PhysicalTableScan` más el filtrado de
    `index_predicate`: el recorrido físico por páginas de índice es
    responsabilidad del storage engine activo, no de este motor — aquí sólo
    importa el contrato de filas que produce (ver `operators.IndexScanOperator`)."""

    table: TableRef
    index_name: str
    index_predicate: Expression
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalFilter(PhysicalOperator):
    input: PhysicalOperator
    predicate: Expression
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalNestedLoopJoin(PhysicalOperator):
    left: PhysicalOperator
    right: PhysicalOperator
    join_type: JoinType
    condition: Expression | None
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalHashJoin(PhysicalOperator):
    """`left` es el lado de construcción de la tabla hash, `right` el de
    sondeo (misma convención que el optimizador); `condition` debe ser una
    igualdad de columna a columna, una por cada lado."""

    left: PhysicalOperator
    right: PhysicalOperator
    join_type: JoinType
    condition: Expression
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalHashAggregate(PhysicalOperator):
    input: PhysicalOperator
    group_by: tuple[Expression, ...]
    aggregates: tuple[FunctionCall, ...]
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalProject(PhysicalOperator):
    input: PhysicalOperator
    items: tuple[SelectItem, ...]
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalDistinct(PhysicalOperator):
    input: PhysicalOperator
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalSort(PhysicalOperator):
    input: PhysicalOperator
    order_by: tuple[OrderByItem, ...]
    estimated_rows: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class PhysicalLimit(PhysicalOperator):
    input: PhysicalOperator
    limit: int | None
    offset: int | None
    estimated_rows: float
    estimated_cost: float


# ---------------------------------------------------------------------------
# Tipos columnares propios de este motor (no vienen del optimizador)
# ---------------------------------------------------------------------------


class ColumnType(Enum):
    """Tipo lógico de una columna. Determina el dtype de NumPy usado para su
    array de valores (ver `batch.dtype_for`) — separado de `LiteralType`
    porque una columna puede ser `INTEGER` aunque nunca aparezca un literal
    entero en la consulta."""

    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()


@dataclass(frozen=True, slots=True)
class Schema:
    """Esquema de un `RowBatch`: nombres de columna ya cualificados (p.ej.
    `"o.customer_id"` tras un scan de `orders o`) en el orden en que se
    exponen, más el tipo lógico de cada una.

    Se guarda `columns` como tupla ordenada (no basta el `dict`, que en
    Python 3.11 preserva orden de inserción pero no lo comunica a quien lee
    el tipo) porque el orden es observable: es el orden de columnas de la
    fila de salida final.
    """

    columns: tuple[str, ...]
    types: dict[str, ColumnType] = field(default_factory=dict)

    def type_of(self, column: str) -> ColumnType:
        return self.types[column]
