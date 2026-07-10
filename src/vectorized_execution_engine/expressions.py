"""Evaluación vectorizada de expresiones sobre un `RowBatch`.

Toda función de este módulo trabaja sobre columnas completas (arrays de
NumPy), nunca fila a fila — es la pieza que hace que `FilterOperator`,
`ProjectOperator` y las condiciones de join sean vectorizados de verdad,
no sólo los operadores "de fachada" que llaman a NumPy una vez y ya.

Convención de valores: cada expresión se evalúa a un par
`(values, nulls)` — dos arrays de la misma longitud que el batch de
entrada. `nulls[i] is True` significa que `values[i]` es sólo relleno (no
se lee, ver `batch._placeholder_for`); todo el módulo respeta ese contrato
al combinar sub-expresiones, incluida la lógica de tres valores de SQL
para `AND`/`OR`/`NOT` (`NULL` no es ni verdadero ni falso: `FALSE AND NULL`
es `FALSE`, no `NULL`, porque un operando falso ya decide el resultado sin
necesitar el otro).
"""

from __future__ import annotations

import re
from typing import cast

import numpy as np

from .batch import RowBatch, dtype_for
from .errors import DivisionByZeroError, UnresolvedColumnError, UnsupportedExpressionError
from .models import (
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    LiteralType,
    LiteralValue,
    Schema,
    Star,
    UnaryOp,
    UnaryOperator,
)

EvaluatedColumn = tuple[np.ndarray, np.ndarray]

_COMPARISON_OPERATORS = frozenset(
    {
        BinaryOperator.EQ,
        BinaryOperator.NEQ,
        BinaryOperator.LT,
        BinaryOperator.LE,
        BinaryOperator.GT,
        BinaryOperator.GE,
    }
)

_LOGICAL_OPERATORS = frozenset({BinaryOperator.AND, BinaryOperator.OR})

_LITERAL_TYPE_TO_COLUMN_TYPE = {
    LiteralType.INTEGER: ColumnType.INTEGER,
    LiteralType.FLOAT: ColumnType.FLOAT,
    LiteralType.STRING: ColumnType.STRING,
    LiteralType.BOOLEAN: ColumnType.BOOLEAN,
}


def to_native(value: object) -> LiteralValue:
    """Normaliza un escalar de NumPy (`np.int64`/`np.float64`/`np.bool_`) a
    su equivalente nativo de Python. Los tipos escalares de NumPy ya
    implementan `__hash__`/`__eq__` consistentes con sus tipos Python, pero
    normalizar antes de usarlos como clave de `dict` (agrupación de
    `HashJoinOperator`/`HashAggregateOperator`) evita cualquier sorpresa al
    mezclar claves provenientes de columnas de distinto dtype pero mismo
    valor lógico. `cast` aquí es un límite de tipado deliberado: NumPy no
    tipa `.item()` más allá de `Any`, pero en este motor sólo se invoca
    sobre celdas ya validadas como `LiteralValue` por `batch.dtype_for`."""
    if isinstance(value, np.generic):
        return cast(LiteralValue, value.item())
    return cast(LiteralValue, value)


def resolve_column(schema: Schema, ref: ColumnRef) -> str:
    """Nombre de columna ya cualificado del batch que satisface `ref`.

    Si `ref.table` está presente, se exige coincidencia exacta con
    `"table.name"`. Si no, se acepta cualquier columna cuyo sufijo tras el
    último `.` sea `ref.name`, siempre que sea la única — mismo criterio de
    desambiguación que `cost-based-query-optimizer` aplica sobre el AST de
    entrada.
    """
    if ref.table is not None:
        key = f"{ref.table}.{ref.name}"
        if key in schema.types:
            return key
        raise UnresolvedColumnError(f"{ref.table}.{ref.name}")
    suffix = f".{ref.name}"
    matches = [c for c in schema.columns if c == ref.name or c.endswith(suffix)]
    if len(matches) != 1:
        raise UnresolvedColumnError(ref.name)
    return matches[0]


def expression_key(expr: Expression) -> str:
    """Representación canónica en texto de una expresión.

    Dos usos: (1) nombre de columna por defecto de un `SelectItem` sin
    alias en `ProjectOperator`; (2) clave de columna precomputada que
    `HashAggregateOperator` usa para exponer el resultado de una
    `FunctionCall` agregada, de forma que un `PhysicalProject` por encima
    pueda referenciarla sin volver a evaluarla (ver docstring de
    `FunctionCall` en `models.py`).
    """
    if isinstance(expr, Literal):
        return "NULL" if expr.literal_type is LiteralType.NULL else repr(expr.value)
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table is not None else expr.name
    if isinstance(expr, Star):
        return "*"
    if isinstance(expr, BinaryOp):
        return f"({expression_key(expr.left)} {expr.operator.name} {expression_key(expr.right)})"
    if isinstance(expr, UnaryOp):
        return f"({expr.operator.name} {expression_key(expr.operand)})"
    if isinstance(expr, Between):
        negation = "NOT " if expr.negated else ""
        return (
            f"({expression_key(expr.operand)} {negation}BETWEEN "
            f"{expression_key(expr.low)} AND {expression_key(expr.high)})"
        )
    if isinstance(expr, InList):
        negation = "NOT " if expr.negated else ""
        values = ", ".join(expression_key(v) for v in expr.values)
        return f"({expression_key(expr.operand)} {negation}IN ({values}))"
    if isinstance(expr, Like):
        negation = "NOT " if expr.negated else ""
        return f"({expression_key(expr.operand)} {negation}LIKE {expression_key(expr.pattern)})"
    if isinstance(expr, IsNull):
        negation = "NOT " if expr.negated else ""
        return f"({expression_key(expr.operand)} IS {negation}NULL)"
    if isinstance(expr, FunctionCall):
        if expr.star_argument:
            arguments = "*"
        else:
            arguments = ", ".join(expression_key(a) for a in expr.arguments)
            if expr.distinct:
                arguments = f"DISTINCT {arguments}"
        return f"{expr.name}({arguments})"
    raise UnsupportedExpressionError(f"no se puede generar una clave para {type(expr).__name__}")


def infer_expression_type(expr: Expression, schema: Schema) -> ColumnType:
    """Tipo lógico de salida de `expr` sin evaluar ningún dato — usado para
    construir el `Schema` de salida de `ProjectOperator`/`HashAggregateOperator`
    antes de que corra ningún batch. Best-effort: no reproduce todas las
    reglas de promoción de tipos de SQL, sólo lo suficiente para etiquetar
    la columna de salida; cualquier combinación de tipos realmente
    incompatible se descubre en tiempo de evaluación real (`evaluate`), no
    aquí — ver el manejo de `TypeError` en `_apply_binary`."""
    if isinstance(expr, Literal):
        if expr.literal_type is LiteralType.NULL:
            return ColumnType.BOOLEAN
        return _LITERAL_TYPE_TO_COLUMN_TYPE[expr.literal_type]
    if isinstance(expr, ColumnRef):
        return schema.type_of(resolve_column(schema, expr))
    if isinstance(expr, BinaryOp):
        if expr.operator in _LOGICAL_OPERATORS or expr.operator in _COMPARISON_OPERATORS:
            return ColumnType.BOOLEAN
        if expr.operator is BinaryOperator.DIV:
            return ColumnType.FLOAT
        left_type = infer_expression_type(expr.left, schema)
        right_type = infer_expression_type(expr.right, schema)
        if ColumnType.FLOAT in (left_type, right_type):
            return ColumnType.FLOAT
        return ColumnType.INTEGER
    if isinstance(expr, UnaryOp):
        if expr.operator is UnaryOperator.NOT:
            return ColumnType.BOOLEAN
        return infer_expression_type(expr.operand, schema)
    if isinstance(expr, Between | InList | Like | IsNull):
        return ColumnType.BOOLEAN
    if isinstance(expr, FunctionCall):
        key = expression_key(expr)
        if key in schema.types:
            return schema.type_of(key)
        raise UnsupportedExpressionError(
            f"la función '{key}' sólo se puede referenciar tras un HashAggregate"
        )
    raise UnsupportedExpressionError(f"no se puede inferir el tipo de {type(expr).__name__}")


def _three_valued_and(
    left_values: np.ndarray,
    left_nulls: np.ndarray,
    right_values: np.ndarray,
    right_nulls: np.ndarray,
) -> EvaluatedColumn:
    false_left = (~left_nulls) & (~left_values)
    false_right = (~right_nulls) & (~right_values)
    result_false = false_left | false_right
    result_null = (~result_false) & (left_nulls | right_nulls)
    data = (~result_false) & (~result_null)
    return data, result_null


def _three_valued_or(
    left_values: np.ndarray,
    left_nulls: np.ndarray,
    right_values: np.ndarray,
    right_nulls: np.ndarray,
) -> EvaluatedColumn:
    true_left = (~left_nulls) & left_values
    true_right = (~right_nulls) & right_values
    result_true = true_left | true_right
    result_null = (~result_true) & (left_nulls | right_nulls)
    return result_true, result_null


def _three_valued_not(values: np.ndarray, nulls: np.ndarray) -> EvaluatedColumn:
    return ~values, nulls


def _broadcast_literal(expr: Literal, row_count: int) -> EvaluatedColumn:
    if expr.literal_type is LiteralType.NULL:
        return np.zeros(row_count, dtype=np.bool_), np.ones(row_count, dtype=np.bool_)
    column_type = _LITERAL_TYPE_TO_COLUMN_TYPE[expr.literal_type]
    data = np.full(row_count, expr.value, dtype=dtype_for(column_type))
    nulls = np.zeros(row_count, dtype=np.bool_)
    return data, nulls


def _apply_binary(op_name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Aplica el ufunc de NumPy correspondiente, envolviendo cualquier
    `TypeError`/`ValueError` (p. ej. sumar un array de texto con uno
    numérico) en la excepción tipada del módulo — nunca se deja escapar un
    error crudo de NumPy/Python, ver `../CLAUDE.md`."""
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            if op_name == "ADD":
                result = left + right
            elif op_name == "SUB":
                result = left - right
            elif op_name == "MUL":
                result = left * right
            elif op_name == "DIV":
                result = left / right
            elif op_name == "MOD":
                result = left % right
            elif op_name == "EQ":
                result = left == right
            elif op_name == "NEQ":
                result = left != right
            elif op_name == "LT":
                result = left < right
            elif op_name == "LE":
                result = left <= right
            elif op_name == "GT":
                result = left > right
            else:
                result = left >= right
    except TypeError as exc:
        raise UnsupportedExpressionError(
            f"no se pudo aplicar '{op_name}' entre columnas de tipos incompatibles: {exc}"
        ) from exc
    return np.asarray(result)


def _evaluate_binary_op(expr: BinaryOp, batch: RowBatch) -> EvaluatedColumn:
    if expr.operator is BinaryOperator.AND:
        lv, ln = evaluate(expr.left, batch)
        rv, rn = evaluate(expr.right, batch)
        return _three_valued_and(lv, ln, rv, rn)
    if expr.operator is BinaryOperator.OR:
        lv, ln = evaluate(expr.left, batch)
        rv, rn = evaluate(expr.right, batch)
        return _three_valued_or(lv, ln, rv, rn)

    left_values, left_nulls = evaluate(expr.left, batch)
    right_values, right_nulls = evaluate(expr.right, batch)
    combined_nulls = left_nulls | right_nulls

    if expr.operator in (BinaryOperator.DIV, BinaryOperator.MOD):
        valid = ~combined_nulls
        if np.any((right_values == 0) & valid):
            raise DivisionByZeroError(
                "división o módulo por cero en al menos una fila del batch evaluado"
            )

    data = _apply_binary(expr.operator.name, left_values, right_values)
    return data, combined_nulls


def _evaluate_unary_op(expr: UnaryOp, batch: RowBatch) -> EvaluatedColumn:
    values, nulls = evaluate(expr.operand, batch)
    if expr.operator is UnaryOperator.NOT:
        return _three_valued_not(values, nulls)
    try:
        data = -values if expr.operator is UnaryOperator.NEG else +values
    except TypeError as exc:
        raise UnsupportedExpressionError(
            f"no se pudo aplicar '{expr.operator.name}' sobre esta columna: {exc}"
        ) from exc
    return data, nulls


def _evaluate_between(expr: Between, batch: RowBatch) -> EvaluatedColumn:
    operand_values, operand_nulls = evaluate(expr.operand, batch)
    low_values, low_nulls = evaluate(expr.low, batch)
    high_values, high_nulls = evaluate(expr.high, batch)
    ge_low = _apply_binary("GE", operand_values, low_values)
    le_high = _apply_binary("LE", operand_values, high_values)
    data, nulls = _three_valued_and(
        ge_low, operand_nulls | low_nulls, le_high, operand_nulls | high_nulls
    )
    if expr.negated:
        data, nulls = _three_valued_not(data, nulls)
    return data, nulls


def _evaluate_in_list(expr: InList, batch: RowBatch) -> EvaluatedColumn:
    operand_values, operand_nulls = evaluate(expr.operand, batch)
    row_count = batch.row_count
    result_values = np.zeros(row_count, dtype=np.bool_)
    result_nulls = np.zeros(row_count, dtype=np.bool_)
    for candidate in expr.values:
        candidate_values, candidate_nulls = evaluate(candidate, batch)
        eq_values = _apply_binary("EQ", operand_values, candidate_values)
        eq_nulls = operand_nulls | candidate_nulls
        result_values, result_nulls = _three_valued_or(
            result_values, result_nulls, eq_values, eq_nulls
        )
    if expr.negated:
        result_values, result_nulls = _three_valued_not(result_values, result_nulls)
    return result_values, result_nulls


def _like_pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Traduce los comodines SQL `%`/`_` a una regex ancla completa.

    `\\` escapa el carácter siguiente (incluidos los propios comodines),
    igual que la mayoría de dialectos SQL; cualquier otro carácter especial
    de regex se escapa literalmente vía `re.escape` carácter a carácter."""
    result = ["^"]
    escaped = False
    for char in pattern:
        if escaped:
            result.append(re.escape(char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "%":
            result.append(".*")
        elif char == "_":
            result.append(".")
        else:
            result.append(re.escape(char))
    result.append("$")
    return re.compile("".join(result), re.DOTALL)


def _evaluate_like(expr: Like, batch: RowBatch) -> EvaluatedColumn:
    """No vectorizado con NumPy puro: no existe una operación nativa de
    coincidencia de patrón con comodines SQL sobre arrays de NumPy sin
    recurrir, de todos modos, a compilar una regex por fila. Se compensa
    cacheando la regex compilada por patrón (`_pattern_cache`) para que un
    `LIKE` con patrón constante (el caso común) sólo compile una vez por
    batch, no una vez por fila."""
    operand_values, operand_nulls = evaluate(expr.operand, batch)
    pattern_values, pattern_nulls = evaluate(expr.pattern, batch)
    row_count = batch.row_count
    combined_nulls = operand_nulls | pattern_nulls
    result = np.zeros(row_count, dtype=np.bool_)
    pattern_cache: dict[str, re.Pattern[str]] = {}
    for i in range(row_count):
        if combined_nulls[i]:
            continue
        pattern_text = str(pattern_values[i])
        compiled = pattern_cache.get(pattern_text)
        if compiled is None:
            compiled = _like_pattern_to_regex(pattern_text)
            pattern_cache[pattern_text] = compiled
        result[i] = compiled.fullmatch(str(operand_values[i])) is not None
    if expr.negated:
        result, combined_nulls = _three_valued_not(result, combined_nulls)
    return result, combined_nulls


def _evaluate_is_null(expr: IsNull, batch: RowBatch) -> EvaluatedColumn:
    _, operand_nulls = evaluate(expr.operand, batch)
    data = ~operand_nulls if expr.negated else operand_nulls
    return data, np.zeros(batch.row_count, dtype=np.bool_)


def evaluate(expr: Expression, batch: RowBatch) -> EvaluatedColumn:
    """Evalúa `expr` contra `batch` devolviendo `(values, nulls)`."""
    if isinstance(expr, Literal):
        return _broadcast_literal(expr, batch.row_count)
    if isinstance(expr, ColumnRef):
        name = resolve_column(batch.schema, expr)
        return batch.columns[name], batch.nulls[name]
    if isinstance(expr, FunctionCall):
        key = expression_key(expr)
        if key in batch.schema.types:
            return batch.columns[key], batch.nulls[key]
        raise UnsupportedExpressionError(
            f"la función '{key}' sólo se puede evaluar dentro de un PhysicalHashAggregate"
        )
    if isinstance(expr, BinaryOp):
        return _evaluate_binary_op(expr, batch)
    if isinstance(expr, UnaryOp):
        return _evaluate_unary_op(expr, batch)
    if isinstance(expr, Between):
        return _evaluate_between(expr, batch)
    if isinstance(expr, InList):
        return _evaluate_in_list(expr, batch)
    if isinstance(expr, Like):
        return _evaluate_like(expr, batch)
    if isinstance(expr, IsNull):
        return _evaluate_is_null(expr, batch)
    if isinstance(expr, Star):
        raise UnsupportedExpressionError(
            "'*' sólo es válido como único SelectItem de un PhysicalProject"
        )
    raise UnsupportedExpressionError(f"expresión no soportada: {type(expr).__name__}")
