"""Jerarquía de excepciones tipadas del motor de ejecución.

Todas heredan de `ExecutionEngineError` para que un consumidor (p. ej.
`nanosql`) pueda capturar cualquier fallo del motor con un único `except`,
pero cada subclase es lo bastante específica como para decidir una
recuperación distinta — ver `../CLAUDE.md`, manejo de errores explícito.
"""

from __future__ import annotations


class ExecutionEngineError(Exception):
    """Base de toda excepción propia de este módulo."""


class UnresolvedColumnError(ExecutionEngineError):
    """Una `ColumnRef` no se pudo resolver de forma única a una columna del
    batch de entrada: o ninguna columna termina en `.nombre` (probable
    error de la propia consulta o del plan físico recibido), o más de una
    lo hace y la referencia no está cualificada (`t.columna`) para
    desambiguar."""

    def __init__(self, column_name: str) -> None:
        self.column_name = column_name
        super().__init__(f"no se pudo resolver la columna '{column_name}' a una única columna")


class UnsupportedExpressionError(ExecutionEngineError):
    """Una expresión del plan físico usa una forma que este motor no sabe
    evaluar en el contexto en el que aparece — p. ej. una `FunctionCall`
    fuera de un `PhysicalHashAggregate` que no coincide con ninguna columna
    ya precomputada, o un `Star` fuera de un `PhysicalProject`."""


class UnsupportedOperatorError(ExecutionEngineError):
    """El nodo de plan físico recibido no es ninguno de los nueve tipos
    `Physical*` que este motor sabe traducir a un operador (ver
    `pipeline.build_operator_tree`). Defensivo: no debería dispararse nunca
    con un plan producido por `cost-based-query-optimizer`."""

    def __init__(self, node: object) -> None:
        self.node = node
        super().__init__(f"tipo de nodo de plan físico no soportado: {type(node).__name__}")


class UnsupportedJoinConditionError(ExecutionEngineError):
    """La `condition` de un `PhysicalHashJoin` no es una igualdad de columna
    a columna (una por cada lado) — el único caso en el que un hash join es
    aplicable. El optimizador nunca debería producir esta forma, pero el
    motor la valida en tiempo de ejecución en vez de asumirlo ciegamente."""


class DivisionByZeroError(ExecutionEngineError):
    """Una división o módulo (`/`, `%`) encontró un divisor cero en al menos
    una fila del batch evaluado. Se prefiere fallar explícitamente (ver
    `../CLAUDE.md`) antes que producir silenciosamente `inf`/`NaN` de NumPy,
    que se propagarían sin avisar a comparaciones y agregados posteriores."""


class SchemaConflictError(ExecutionEngineError):
    """Dos columnas del mismo nombre cualificado terminaron en el mismo
    esquema de salida (p. ej. un join entre dos `TableRef` con el mismo
    alias efectivo). El plan físico de entrada debería garantizar alias
    únicos por tabla; este motor lo valida en vez de dejar que una columna
    pise silenciosamente a la otra."""

    def __init__(self, column_name: str) -> None:
        self.column_name = column_name
        super().__init__(f"columna duplicada en el esquema combinado: '{column_name}'")


class MemoryLimitExceededError(ExecutionEngineError):
    """Un operador bloqueante (`HashJoinOperator`, `HashAggregateOperator`,
    `SortOperator`) superó su `MemoryBudget` configurado al intentar
    materializar su entrada. Este motor no implementa spill a disco (ver
    README, sección de arquitectura): ante un límite excedido, falla de
    forma explícita en vez de crecer sin límite (regla D de `../CLAUDE.md`)."""

    def __init__(self, operator_name: str, requested_bytes: int, limit_bytes: int) -> None:
        self.operator_name = operator_name
        self.requested_bytes = requested_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"{operator_name}: límite de memoria excedido "
            f"({requested_bytes} bytes solicitados, límite {limit_bytes} bytes)"
        )
