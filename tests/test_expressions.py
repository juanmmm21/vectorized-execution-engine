import pytest

from vectorized_execution_engine.batch import RowBatch, columns_from_python
from vectorized_execution_engine.errors import (
    DivisionByZeroError,
    UnresolvedColumnError,
    UnsupportedExpressionError,
)
from vectorized_execution_engine.expressions import (
    evaluate,
    expression_key,
    infer_expression_type,
    resolve_column,
)
from vectorized_execution_engine.models import (
    Between,
    BinaryOp,
    BinaryOperator,
    ColumnRef,
    ColumnType,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    LiteralType,
    Schema,
    UnaryOp,
    UnaryOperator,
)


def _batch(columns: dict[str, tuple[list[object], ColumnType]]) -> RowBatch:
    schema_columns = tuple(columns.keys())
    types = {name: column_type for name, (_, column_type) in columns.items()}
    schema = Schema(schema_columns, types)
    data = {}
    nulls = {}
    for name, (values, column_type) in columns.items():
        arr, null_mask = columns_from_python(values, column_type)
        data[name] = arr
        nulls[name] = null_mask
    return RowBatch(schema, data, nulls)


def test_resolve_column_qualified() -> None:
    schema = Schema(("o.id", "c.id"), {"o.id": ColumnType.INTEGER, "c.id": ColumnType.INTEGER})
    assert resolve_column(schema, ColumnRef("id", "o")) == "o.id"


def test_resolve_column_unqualified_unique_suffix() -> None:
    schema = Schema(("o.amount",), {"o.amount": ColumnType.FLOAT})
    assert resolve_column(schema, ColumnRef("amount")) == "o.amount"


def test_resolve_column_ambiguous_raises() -> None:
    schema = Schema(("o.id", "c.id"), {"o.id": ColumnType.INTEGER, "c.id": ColumnType.INTEGER})
    with pytest.raises(UnresolvedColumnError):
        resolve_column(schema, ColumnRef("id"))


def test_resolve_column_missing_raises() -> None:
    schema = Schema(("o.id",), {"o.id": ColumnType.INTEGER})
    with pytest.raises(UnresolvedColumnError):
        resolve_column(schema, ColumnRef("missing"))


def test_evaluate_literal_broadcasts() -> None:
    batch = _batch({"a": ([1, 2, 3], ColumnType.INTEGER)})
    values, nulls = evaluate(Literal(7, LiteralType.INTEGER), batch)
    assert values.tolist() == [7, 7, 7]
    assert not any(nulls)


def test_evaluate_null_literal_is_all_null() -> None:
    batch = _batch({"a": ([1, 2], ColumnType.INTEGER)})
    _, nulls = evaluate(Literal(None, LiteralType.NULL), batch)
    assert all(nulls)


@pytest.mark.parametrize(
    ("left", "left_null", "right", "right_null", "expected_value", "expected_null"),
    [
        (True, False, True, False, True, False),
        (False, False, True, False, False, False),
        (True, False, False, False, False, False),
        (False, False, False, False, False, False),
        (False, False, None, True, False, False),
        (True, False, None, True, None, True),
        (None, True, None, True, None, True),
    ],
)
def test_and_three_valued_truth_table(
    left: object,
    left_null: bool,
    right: object,
    right_null: bool,
    expected_value: object,
    expected_null: bool,
) -> None:
    batch = _batch(
        {
            "l": ([None if left_null else left], ColumnType.BOOLEAN),
            "r": ([None if right_null else right], ColumnType.BOOLEAN),
        }
    )
    values, nulls = evaluate(BinaryOp(ColumnRef("l"), BinaryOperator.AND, ColumnRef("r")), batch)
    assert bool(nulls[0]) == expected_null
    if not expected_null:
        assert bool(values[0]) == expected_value


@pytest.mark.parametrize(
    ("left", "left_null", "right", "right_null", "expected_value", "expected_null"),
    [
        (True, False, False, False, True, False),
        (False, False, False, False, False, False),
        (True, False, None, True, True, False),
        (False, False, None, True, None, True),
        (None, True, None, True, None, True),
    ],
)
def test_or_three_valued_truth_table(
    left: object,
    left_null: bool,
    right: object,
    right_null: bool,
    expected_value: object,
    expected_null: bool,
) -> None:
    batch = _batch(
        {
            "l": ([None if left_null else left], ColumnType.BOOLEAN),
            "r": ([None if right_null else right], ColumnType.BOOLEAN),
        }
    )
    values, nulls = evaluate(BinaryOp(ColumnRef("l"), BinaryOperator.OR, ColumnRef("r")), batch)
    assert bool(nulls[0]) == expected_null
    if not expected_null:
        assert bool(values[0]) == expected_value


def test_not_propagates_null() -> None:
    batch = _batch({"a": ([True, False, None], ColumnType.BOOLEAN)})
    values, nulls = evaluate(UnaryOp(UnaryOperator.NOT, ColumnRef("a")), batch)
    assert nulls.tolist() == [False, False, True]
    assert values[0] == False  # noqa: E712
    assert values[1] == True  # noqa: E712


def test_comparison_propagates_null() -> None:
    batch = _batch({"a": ([1, None], ColumnType.INTEGER)})
    values, nulls = evaluate(
        BinaryOp(ColumnRef("a"), BinaryOperator.EQ, Literal(1, LiteralType.INTEGER)), batch
    )
    assert nulls.tolist() == [False, True]
    assert values[0] == True  # noqa: E712


def test_division_by_zero_raises_typed_error() -> None:
    batch = _batch({"a": ([10], ColumnType.INTEGER), "b": ([0], ColumnType.INTEGER)})
    with pytest.raises(DivisionByZeroError):
        evaluate(BinaryOp(ColumnRef("a"), BinaryOperator.DIV, ColumnRef("b")), batch)


def test_division_by_zero_at_null_row_does_not_raise() -> None:
    batch = _batch({"a": ([None], ColumnType.INTEGER), "b": ([0], ColumnType.INTEGER)})
    values, nulls = evaluate(BinaryOp(ColumnRef("a"), BinaryOperator.DIV, ColumnRef("b")), batch)
    assert nulls[0]


def test_arithmetic_type_mismatch_raises_unsupported_expression() -> None:
    batch = _batch({"a": (["x"], ColumnType.STRING), "b": ([1], ColumnType.INTEGER)})
    with pytest.raises(UnsupportedExpressionError):
        evaluate(BinaryOp(ColumnRef("a"), BinaryOperator.ADD, ColumnRef("b")), batch)


def test_between_inclusive() -> None:
    batch = _batch({"a": ([1, 5, 10], ColumnType.INTEGER)})
    values, nulls = evaluate(
        Between(ColumnRef("a"), Literal(2, LiteralType.INTEGER), Literal(8, LiteralType.INTEGER)),
        batch,
    )
    assert values.tolist() == [False, True, False]


def test_between_negated() -> None:
    batch = _batch({"a": ([5], ColumnType.INTEGER)})
    values, _ = evaluate(
        Between(
            ColumnRef("a"),
            Literal(2, LiteralType.INTEGER),
            Literal(8, LiteralType.INTEGER),
            negated=True,
        ),
        batch,
    )
    assert values[0] == False  # noqa: E712


def test_in_list_true_dominates_null() -> None:
    batch = _batch({"a": ([5], ColumnType.INTEGER)})
    values, nulls = evaluate(
        InList(
            ColumnRef("a"),
            (
                Literal(1, LiteralType.INTEGER),
                Literal(None, LiteralType.NULL),
                Literal(5, LiteralType.INTEGER),
            ),
        ),
        batch,
    )
    assert not nulls[0]
    assert values[0] == True  # noqa: E712


def test_in_list_null_without_match_is_null() -> None:
    batch = _batch({"a": ([3], ColumnType.INTEGER)})
    values, nulls = evaluate(
        InList(
            ColumnRef("a"),
            (Literal(1, LiteralType.INTEGER), Literal(None, LiteralType.NULL)),
        ),
        batch,
    )
    assert nulls[0]


def test_like_wildcards() -> None:
    batch = _batch({"a": (["hello", "world", "help"], ColumnType.STRING)})
    values, _ = evaluate(Like(ColumnRef("a"), Literal("hel%", LiteralType.STRING)), batch)
    assert values.tolist() == [True, False, True]


def test_like_underscore_matches_single_char() -> None:
    batch = _batch({"a": (["cat", "cot", "coat"], ColumnType.STRING)})
    values, _ = evaluate(Like(ColumnRef("a"), Literal("c_t", LiteralType.STRING)), batch)
    assert values.tolist() == [True, True, False]


def test_like_escaped_wildcard() -> None:
    batch = _batch({"a": (["50%", "50x"], ColumnType.STRING)})
    values, _ = evaluate(Like(ColumnRef("a"), Literal("50\\%", LiteralType.STRING)), batch)
    assert values.tolist() == [True, False]


def test_is_null() -> None:
    batch = _batch({"a": ([1, None], ColumnType.INTEGER)})
    values, nulls = evaluate(IsNull(ColumnRef("a")), batch)
    assert values.tolist() == [False, True]
    assert not any(nulls)


def test_is_not_null() -> None:
    batch = _batch({"a": ([1, None], ColumnType.INTEGER)})
    values, _ = evaluate(IsNull(ColumnRef("a"), negated=True), batch)
    assert values.tolist() == [True, False]


def test_function_call_reads_precomputed_column() -> None:
    schema = Schema(("COUNT(*)",), {"COUNT(*)": ColumnType.INTEGER})
    data, nulls = columns_from_python([5], ColumnType.INTEGER)
    batch = RowBatch(schema, {"COUNT(*)": data}, {"COUNT(*)": nulls})
    values, _ = evaluate(FunctionCall("COUNT", star_argument=True), batch)
    assert values[0] == 5


def test_function_call_without_precomputed_column_raises() -> None:
    batch = _batch({"a": ([1], ColumnType.INTEGER)})
    with pytest.raises(UnsupportedExpressionError):
        evaluate(FunctionCall("COUNT", star_argument=True), batch)


def test_expression_key_formats_function_call() -> None:
    func = FunctionCall("SUM", arguments=(ColumnRef("amount", "o"),))
    assert expression_key(func) == "SUM(o.amount)"


def test_expression_key_formats_count_star() -> None:
    assert expression_key(FunctionCall("COUNT", star_argument=True)) == "COUNT(*)"


def test_infer_expression_type_comparison_is_boolean() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    expr = BinaryOp(ColumnRef("a"), BinaryOperator.EQ, Literal(1, LiteralType.INTEGER))
    assert infer_expression_type(expr, schema) is ColumnType.BOOLEAN


def test_infer_expression_type_division_is_float() -> None:
    schema = Schema(("a",), {"a": ColumnType.INTEGER})
    expr = BinaryOp(ColumnRef("a"), BinaryOperator.DIV, Literal(2, LiteralType.INTEGER))
    assert infer_expression_type(expr, schema) is ColumnType.FLOAT
