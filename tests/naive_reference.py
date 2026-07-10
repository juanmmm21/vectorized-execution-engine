"""Implementación de referencia, fila a fila y en Python puro.

Usada únicamente por los tests de correctness/propiedades para comparar
contra el motor vectorizado sobre los mismos datos (ver DoD del proyecto).
Deliberadamente NO reutiliza `expressions.evaluate`, `operators.py`,
`joins.py`, `aggregate.py` ni `sort.py` del motor: es un camino de código
completamente independiente, escrito sobre `list[dict[str, LiteralValue]]`
(una fila = un `dict`), nunca sobre arrays de NumPy — mismo principio que
el test de propiedades de `cost-based-query-optimizer`, que compara contra
una búsqueda de fuerza bruta que no reimplementa las fórmulas de coste del
propio optimizador.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Sequence

from vectorized_execution_engine.batch import RowBatch, TableSource
from vectorized_execution_engine.models import (
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    JoinType,
    Like,
    Literal,
    LiteralType,
    LiteralValue,
    OrderByItem,
    OrderDirection,
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
    SelectItem,
    Star,
    TableRef,
    UnaryOp,
    UnaryOperator,
)

Row = dict[str, LiteralValue]
_NULL_GROUP_KEY = object()


def _native(value: object) -> LiteralValue:
    return value.item() if hasattr(value, "item") else value  # type: ignore[no-any-return]


def _resolve_in_names(names: Sequence[str], ref: ColumnRef) -> str:
    if ref.table is not None:
        key = f"{ref.table}.{ref.name}"
        if key in names:
            return key
        raise KeyError(key)
    matches = [name for name in names if name == ref.name or name.endswith(f".{ref.name}")]
    if len(matches) != 1:
        raise KeyError(ref.name)
    return matches[0]


def _like_match(text: str, pattern: str) -> bool:
    """Compara `text` contra `pattern` (comodines SQL `%`/`_`) con un motor
    de coincidencia manual, no vía `re` (para que ni siquiera el mecanismo
    de comparación de patrones se comparta con `expressions._evaluate_like`)."""
    return _like_match_from(text, pattern, 0, 0)


def _like_match_from(text: str, pattern: str, ti: int, pi: int) -> bool:
    if pi == len(pattern):
        return ti == len(text)
    char = pattern[pi]
    if char == "\\" and pi + 1 < len(pattern):
        return (
            ti < len(text)
            and text[ti] == pattern[pi + 1]
            and _like_match_from(text, pattern, ti + 1, pi + 2)
        )
    if char == "%":
        for skip in range(len(text) - ti + 1):
            if _like_match_from(text, pattern, ti + skip, pi + 1):
                return True
        return False
    if char == "_":
        return ti < len(text) and _like_match_from(text, pattern, ti + 1, pi + 1)
    return ti < len(text) and text[ti] == char and _like_match_from(text, pattern, ti + 1, pi + 1)


def _argument_label(expr: Expression) -> str:
    """Etiqueta de un argumento dentro de `_function_key` — debe coincidir
    con `expressions.expression_key` del motor real: una referencia de
    columna se etiqueta CUALIFICADA aquí (p. ej. `SUM(o.amount)`), a
    diferencia del nombre de columna por defecto de un `SelectItem` sin
    alias (ver `_default_name`), que es desnudo."""
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table is not None else expr.name
    if isinstance(expr, FunctionCall):
        return _function_key(expr)
    if isinstance(expr, Literal):
        return "NULL" if expr.literal_type is LiteralType.NULL else repr(expr.value)
    return type(expr).__name__


def _function_key(func: FunctionCall) -> str:
    if func.star_argument:
        arguments = "*"
    else:
        arguments = ", ".join(_argument_label(a) for a in func.arguments)
        if func.distinct:
            arguments = f"DISTINCT {arguments}"
    return f"{func.name}({arguments})"


def _default_name(expr: Expression) -> str:
    """Nombre por defecto de un `SelectItem` sin alias — debe coincidir con
    `operators._default_output_name` del motor real: nombre desnudo (sin
    cualificar) para una referencia de columna simple, aunque venga
    cualificada por tabla en la expresión de entrada."""
    if isinstance(expr, ColumnRef):
        return expr.name
    if isinstance(expr, FunctionCall):
        return _function_key(expr)
    if isinstance(expr, Literal):
        return "NULL" if expr.literal_type is LiteralType.NULL else repr(expr.value)
    return type(expr).__name__


def eval_expr(expr: Expression, row: Row) -> tuple[LiteralValue, bool]:
    """Evalúa `expr` contra una única fila, devolviendo `(valor, es_nulo)`."""
    if isinstance(expr, Literal):
        return (None, True) if expr.literal_type is LiteralType.NULL else (expr.value, False)
    if isinstance(expr, ColumnRef):
        key = _resolve_in_names(list(row.keys()), expr)
        value = row[key]
        return value, value is None
    if isinstance(expr, FunctionCall):
        key = _function_key(expr)
        value = row[key]
        return value, value is None
    if isinstance(expr, BinaryOp):
        return _eval_binary(expr, row)
    if isinstance(expr, UnaryOp):
        value, is_null = eval_expr(expr.operand, row)
        if expr.operator is UnaryOperator.NOT:
            return (None, True) if is_null else (not value, False)
        if is_null:
            return None, True
        return (-value if expr.operator is UnaryOperator.NEG else +value), False  # type: ignore[operator]
    if isinstance(expr, Between):
        return _eval_between(expr, row)
    if isinstance(expr, InList):
        return _eval_in_list(expr, row)
    if isinstance(expr, Like):
        operand_value, operand_null = eval_expr(expr.operand, row)
        pattern_value, pattern_null = eval_expr(expr.pattern, row)
        if operand_null or pattern_null:
            return None, True
        matched = _like_match(str(operand_value), str(pattern_value))
        return (not matched if expr.negated else matched), False
    if isinstance(expr, IsNull):
        _, is_null = eval_expr(expr.operand, row)
        return (not is_null if expr.negated else is_null), False
    raise AssertionError(f"expresión no soportada en la referencia naive: {type(expr).__name__}")


def _eval_binary(expr: BinaryOp, row: Row) -> tuple[LiteralValue, bool]:
    if expr.operator is BinaryOperator.AND:
        lv, ln = eval_expr(expr.left, row)
        rv, rn = eval_expr(expr.right, row)
        if (not ln and lv is False) or (not rn and rv is False):
            return False, False
        if ln or rn:
            return None, True
        return True, False
    if expr.operator is BinaryOperator.OR:
        lv, ln = eval_expr(expr.left, row)
        rv, rn = eval_expr(expr.right, row)
        if (not ln and lv is True) or (not rn and rv is True):
            return True, False
        if ln or rn:
            return None, True
        return False, False

    left_value, left_null = eval_expr(expr.left, row)
    right_value, right_null = eval_expr(expr.right, row)
    if left_null or right_null:
        return None, True
    op = expr.operator
    if op is BinaryOperator.EQ:
        return left_value == right_value, False
    if op is BinaryOperator.NEQ:
        return left_value != right_value, False
    if op is BinaryOperator.LT:
        return left_value < right_value, False  # type: ignore[operator]
    if op is BinaryOperator.LE:
        return left_value <= right_value, False  # type: ignore[operator]
    if op is BinaryOperator.GT:
        return left_value > right_value, False  # type: ignore[operator]
    if op is BinaryOperator.GE:
        return left_value >= right_value, False  # type: ignore[operator]
    if op is BinaryOperator.ADD:
        return left_value + right_value, False  # type: ignore[operator]
    if op is BinaryOperator.SUB:
        return left_value - right_value, False  # type: ignore[operator]
    if op is BinaryOperator.MUL:
        return left_value * right_value, False  # type: ignore[operator]
    if op is BinaryOperator.DIV:
        return left_value / right_value, False  # type: ignore[operator]
    return left_value % right_value, False  # type: ignore[operator]


def _eval_between(expr: Between, row: Row) -> tuple[LiteralValue, bool]:
    operand_value, operand_null = eval_expr(expr.operand, row)
    low_value, low_null = eval_expr(expr.low, row)
    high_value, high_null = eval_expr(expr.high, row)
    if operand_null or low_null or high_null:
        result_value, result_null = None, True
    else:
        result_value = low_value <= operand_value <= high_value  # type: ignore[operator]
        result_null = False
    if expr.negated and not result_null:
        return not result_value, False
    return result_value, result_null


def _eval_in_list(expr: InList, row: Row) -> tuple[LiteralValue, bool]:
    operand_value, operand_null = eval_expr(expr.operand, row)
    any_true = False
    any_null = operand_null
    for candidate in expr.values:
        candidate_value, candidate_null = eval_expr(candidate, row)
        if operand_null or candidate_null:
            any_null = True
            continue
        if operand_value == candidate_value:
            any_true = True
    if any_true:
        result_value, result_null = True, False
    elif any_null:
        result_value, result_null = None, True
    else:
        result_value, result_null = False, False
    if expr.negated and not result_null:
        return not result_value, False
    return result_value, result_null


def naive_scan(source: TableSource, table: TableRef) -> tuple[list[Row], tuple[str, ...]]:
    prefix = table.effective_name
    physical_schema = source.schema(table.name)
    columns = tuple(f"{prefix}.{name}" for name in physical_schema.columns)
    rows: list[Row] = []
    for batch in source.scan(table.name):
        for i in range(batch.row_count):
            row: Row = {}
            for name in physical_schema.columns:
                qualified = f"{prefix}.{name}"
                row[qualified] = None if batch.nulls[name][i] else _native(batch.columns[name][i])
            rows.append(row)
    return rows, columns


def naive_filter(rows: list[Row], predicate: Expression) -> list[Row]:
    result = []
    for row in rows:
        value, is_null = eval_expr(predicate, row)
        if not is_null and value:
            result.append(row)
    return result


def naive_project(
    rows: list[Row], items: tuple[SelectItem, ...], columns: tuple[str, ...]
) -> tuple[list[Row], tuple[str, ...]]:
    if len(items) == 1 and isinstance(items[0].expression, Star):
        return [dict(row) for row in rows], columns
    output_columns = tuple(
        item.alias if item.alias is not None else _default_name(item.expression) for item in items
    )
    result = []
    for row in rows:
        new_row: Row = {}
        for item, name in zip(items, output_columns, strict=True):
            value, is_null = eval_expr(item.expression, row)
            new_row[name] = None if is_null else value
        result.append(new_row)
    return result, output_columns


def naive_join(
    left_rows: list[Row],
    left_columns: tuple[str, ...],
    right_rows: list[Row],
    right_columns: tuple[str, ...],
    join_type: JoinType,
    condition: Expression | None,
) -> tuple[list[Row], tuple[str, ...]]:
    combined_columns = left_columns + right_columns
    result: list[Row] = []
    left_matched = [False] * len(left_rows)
    right_matched = [False] * len(right_rows)
    for li, left_row in enumerate(left_rows):
        for ri, right_row in enumerate(right_rows):
            combined = {**left_row, **right_row}
            if condition is not None:
                value, is_null = eval_expr(condition, combined)
                matched = (not is_null) and bool(value)
            else:
                matched = True
            if matched:
                left_matched[li] = True
                right_matched[ri] = True
                result.append(combined)
    if join_type in (JoinType.LEFT, JoinType.FULL):
        for li, left_row in enumerate(left_rows):
            if not left_matched[li]:
                result.append({**left_row, **{c: None for c in right_columns}})
    if join_type in (JoinType.RIGHT, JoinType.FULL):
        for ri, right_row in enumerate(right_rows):
            if not right_matched[ri]:
                result.append({**{c: None for c in left_columns}, **right_row})
    return result, combined_columns


def _compute_naive_aggregate(func: FunctionCall, group_rows: list[Row]) -> LiteralValue:
    if func.star_argument:
        if func.name != "COUNT":
            raise AssertionError(f"{func.name}(*) no soportado en la referencia naive")
        return len(group_rows)
    values: list[LiteralValue] = []
    for row in group_rows:
        value, is_null = eval_expr(func.arguments[0], row)
        if not is_null:
            values.append(value)
    if func.distinct:
        seen: list[LiteralValue] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        values = seen
    if func.name == "COUNT":
        return len(values)
    if not values:
        return None
    if func.name == "SUM":
        return sum(values)  # type: ignore[arg-type]
    if func.name == "AVG":
        return sum(values) / len(values)  # type: ignore[arg-type]
    if func.name == "MIN":
        return min(values)  # type: ignore[type-var]
    if func.name == "MAX":
        return max(values)  # type: ignore[type-var]
    raise AssertionError(f"agregada no soportada en la referencia naive: {func.name}")


def naive_aggregate(
    rows: list[Row],
    columns: tuple[str, ...],
    group_by: tuple[Expression, ...],
    aggregates: tuple[FunctionCall, ...],
) -> tuple[list[Row], tuple[str, ...]]:
    group_names = tuple(
        _resolve_in_names(columns, expr) if isinstance(expr, ColumnRef) else _default_name(expr)
        for expr in group_by
    )
    aggregate_names = tuple(_function_key(func) for func in aggregates)

    groups: dict[tuple[object, ...], list[Row]] = {}
    if not rows and not group_by:
        groups[()] = []
    for row in rows:
        key = tuple(
            (_NULL_GROUP_KEY if is_null else value)
            for value, is_null in (eval_expr(expr, row) for expr in group_by)
        )
        groups.setdefault(key, []).append(row)

    result: list[Row] = []
    for group_rows in groups.values():
        out: Row = {}
        for name, expr in zip(group_names, group_by, strict=True):
            if group_rows:
                value, is_null = eval_expr(expr, group_rows[0])
                out[name] = None if is_null else value
            else:
                out[name] = None
        for name, func in zip(aggregate_names, aggregates, strict=True):
            out[name] = _compute_naive_aggregate(func, group_rows)
        result.append(out)
    return result, group_names + aggregate_names


def naive_sort(rows: list[Row], order_by: tuple[OrderByItem, ...]) -> list[Row]:
    def compare(a: Row, b: Row) -> int:
        for item in order_by:
            a_value, a_null = eval_expr(item.expression, a)
            b_value, b_null = eval_expr(item.expression, b)
            if a_null and b_null:
                continue
            if a_null:
                return 1
            if b_null:
                return -1
            if a_value == b_value:
                continue
            ascending = a_value < b_value  # type: ignore[operator]
            if item.direction is OrderDirection.DESC:
                return -1 if not ascending else 1
            return -1 if ascending else 1
        return 0

    return sorted(rows, key=functools.cmp_to_key(compare))


def naive_limit(rows: list[Row], limit: int | None, offset: int | None) -> list[Row]:
    start = offset or 0
    if limit is None:
        return rows[start:]
    return rows[start : start + limit]


def naive_execute(plan: PhysicalOperator, source: TableSource) -> tuple[list[Row], tuple[str, ...]]:
    """Ejecuta `plan` fila a fila, devolviendo `(filas, columnas)` — el
    equivalente naive de `pipeline.build_operator_tree` + `execute`."""
    if isinstance(plan, PhysicalTableScan):
        return naive_scan(source, plan.table)
    if isinstance(plan, PhysicalIndexScan):
        rows, columns = naive_scan(source, plan.table)
        return naive_filter(rows, plan.index_predicate), columns
    if isinstance(plan, PhysicalFilter):
        rows, columns = naive_execute(plan.input, source)
        return naive_filter(rows, plan.predicate), columns
    if isinstance(plan, PhysicalNestedLoopJoin | PhysicalHashJoin):
        left_rows, left_columns = naive_execute(plan.left, source)
        right_rows, right_columns = naive_execute(plan.right, source)
        return naive_join(
            left_rows, left_columns, right_rows, right_columns, plan.join_type, plan.condition
        )
    if isinstance(plan, PhysicalHashAggregate):
        rows, columns = naive_execute(plan.input, source)
        return naive_aggregate(rows, columns, plan.group_by, plan.aggregates)
    if isinstance(plan, PhysicalProject):
        rows, columns = naive_execute(plan.input, source)
        return naive_project(rows, plan.items, columns)
    if isinstance(plan, PhysicalSort):
        rows, columns = naive_execute(plan.input, source)
        return naive_sort(rows, plan.order_by), columns
    if isinstance(plan, PhysicalLimit):
        rows, columns = naive_execute(plan.input, source)
        return naive_limit(rows, plan.limit, plan.offset), columns
    raise AssertionError(f"nodo de plan no soportado en la referencia naive: {type(plan).__name__}")


def batch_to_rows(batch: RowBatch) -> list[Row]:
    """Convierte un `RowBatch` del motor real a `list[Row]`, para poder
    comparar su resultado contra `naive_execute` con el mismo tipo de dato."""
    rows: list[Row] = []
    for i in range(batch.row_count):
        row: Row = {}
        for name in batch.schema.columns:
            row[name] = None if batch.nulls[name][i] else _native(batch.columns[name][i])
        rows.append(row)
    return rows


def rows_as_multiset(rows: list[Row]) -> Counter[tuple[tuple[str, LiteralValue], ...]]:
    """Representación como multiconjunto de `rows`, para comparar dos
    resultados sin importar el orden relativo (salvo que el plan incluya un
    `PhysicalSort`, donde el test debe comparar la lista tal cual, no esto)."""
    return Counter(tuple(sorted(row.items())) for row in rows)
