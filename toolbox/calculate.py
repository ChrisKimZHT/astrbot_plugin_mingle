"""Bounded decimal arithmetic without executing Python expressions."""

import ast
import json
from decimal import Decimal, Underflow, localcontext

_MAX_EXPRESSION_LENGTH = 256
_MAX_AST_NODES = 64
_MAX_EXPONENT = 100
_MIN_MAGNITUDE = Decimal("1e-100")
_MAX_MAGNITUDE = Decimal("1e100")


def _evaluate(node: ast.AST, expression: str) -> Decimal:
    """Evaluate only numeric literals and supported arithmetic AST nodes.

    Args:
        node: Node from a size-limited expression tree.
        expression: Source used to preserve decimal literal precision.

    Returns:
        Zero or a finite decimal with magnitude between 1e-100 and 1e100.

    Raises:
        ValueError: If a node, exponent or result is unsupported.
        ArithmeticError: If arithmetic fails.
    """
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        result = Decimal(ast.get_source_segment(expression, node))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, expression)
        result = value if isinstance(node.op, ast.UAdd) else value.copy_negate()
    elif isinstance(node, ast.BinOp):
        left = _evaluate(node.left, expression)
        right = _evaluate(node.right, expression)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            # Floor the exact ratio: rounded decimal division can cross an
            # integer boundary and produce an incorrect quotient or remainder.
            left_numerator, left_denominator = left.as_integer_ratio()
            right_numerator, right_denominator = right.as_integer_ratio()
            numerator = left_numerator * right_denominator
            denominator = left_denominator * right_numerator
            quotient = numerator // denominator
            result = (
                Decimal(quotient)
                if isinstance(node.op, ast.FloorDiv)
                else Decimal(numerator - quotient * denominator)
                / Decimal(left_denominator * right_denominator)
            )
        elif isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or right.copy_abs() > _MAX_EXPONENT:
                raise ValueError("Exponents must be integers between -100 and 100.")
            result = left ** int(right)
        else:
            raise ValueError("Unsupported arithmetic operator.")
    else:
        raise ValueError(
            "Only numeric literals, parentheses and arithmetic are supported."
        )
    if not result.is_finite() or (
        result != 0 and not _MIN_MAGNITUDE <= result.copy_abs() <= _MAX_MAGNITUDE
    ):
        raise ValueError("Nonzero magnitudes must be between 1e-100 and 1e100.")
    return result


def calculate(expression: str) -> str:
    """Calculate bounded arithmetic with 28 significant decimal digits.

    Args:
        expression: At most 256 characters of decimal arithmetic.

    Returns:
        JSON with the expression and a decimal result string, or an error.
    """
    try:
        if (
            not isinstance(expression, str)
            or not 1 <= len(expression) <= _MAX_EXPRESSION_LENGTH
        ):
            raise ValueError("Expression must contain 1 to 256 characters.")
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError(
                "Expression is too complex; at most 64 AST nodes are allowed."
            )
        with localcontext() as context:
            context.prec = 28
            context.Emax = 100
            context.Emin = -100
            context.traps[Underflow] = True
            result = _evaluate(tree.body, expression)
            result_text = format(result, "f")
            if "." in result_text:
                result_text = result_text.rstrip("0").rstrip(".")
            if result == 0:
                result_text = "0"
        return json.dumps({"expression": expression, "result": result_text})
    except (ValueError, SyntaxError, ArithmeticError, RecursionError):
        return json.dumps(
            {
                "error": "Invalid or excessive arithmetic. Use decimal numbers, "
                "parentheses, + - * / // % and **; integer exponents must be "
                "within -100 to 100, nonzero magnitudes between 1e-100 and 1e100, "
                "length at most 256 and AST nodes at most 64."
            }
        )
