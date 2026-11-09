"""Safe evaluation of time-varying parameter expressions.

Dynamic parameters (``{'__dynamic__': True, 'mode': 'function', 'expression': ...}``)
carry user-supplied math expressions such as ``'50 + 25 * sin(t * 0.1)'``.
These can arrive from untrusted callers (API / MCP parameter payloads), so they
are never passed to ``eval`` directly: ``eval`` with ``{'__builtins__': {}}`` is
not a sandbox and is trivially escaped via attribute access
(``().__class__.__base__.__subclasses__()`` ...).

Instead the expression is parsed with :mod:`ast` and checked against a strict
whitelist before it is compiled:

* numeric literals (ints are promoted to float, so ``9**9**9`` overflows
  immediately instead of building a huge integer)
* the variable ``t`` and the constants ``pi`` and ``e``
* arithmetic ``+ - * / // % **``, unary ``+ - not``
* comparisons, ``and`` / ``or`` and ``x if cond else y``
* calls to whitelisted math functions by bare name, positional args only

Everything else (attributes, subscripts, strings, comprehensions, lambdas,
keyword args, other names, ...) is rejected with :class:`UnsafeExpressionError`.
With no attribute access, subscripts or non-numeric literals, the checked
expression has no path back to Python internals.

This module has no dependencies beyond the standard library so it can be
vendored into standalone generated models.
"""

import ast
import math
from typing import Callable, Dict

MAX_EXPRESSION_LENGTH = 1000

SAFE_FUNCTIONS: Dict[str, Callable] = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
    'abs': abs, 'min': min, 'max': max, 'pow': math.pow,
    # floor/ceil return floats so results can never become unbounded ints
    'floor': lambda x: float(math.floor(x)), 'ceil': lambda x: float(math.ceil(x)),
}
SAFE_CONSTANTS: Dict[str, float] = {'pi': math.pi, 'e': math.e}
TIME_VARIABLE = 't'

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Load, ast.Constant,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses syntax outside the safe whitelist."""


class _Validate(ast.NodeTransformer):
    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpressionError(
                f"Unsupported syntax in expression: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Constant(self, node):
        value = node.value
        # bool is a subclass of int; only plain numbers are allowed.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsafeExpressionError(
                f"Only numeric literals are allowed, got {value!r}")
        return ast.copy_location(ast.Constant(value=float(value)), node)

    def visit_Name(self, node):
        if node.id != TIME_VARIABLE and node.id not in SAFE_CONSTANTS \
                and node.id not in SAFE_FUNCTIONS:
            raise UnsafeExpressionError(f"Unknown name in expression: {node.id!r}")
        return node

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise UnsafeExpressionError("Only whitelisted math functions may be called")
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not allowed")
        node.args = [self.visit(arg) for arg in node.args]
        return node


def compile_expression(expression: str) -> Callable[[float], float]:
    """Validate and compile an expression into a function of ``t``.

    Raises:
        UnsafeExpressionError: if the expression is too long or uses
            syntax outside the whitelist.
        SyntaxError: if the expression is not valid Python syntax.
    """
    if not isinstance(expression, str):
        raise UnsafeExpressionError("Expression must be a string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(
            f"Expression longer than {MAX_EXPRESSION_LENGTH} characters")

    tree = ast.parse(expression.strip(), mode='eval')
    tree = ast.fix_missing_locations(_Validate().visit(tree))
    code = compile(tree, '<veydra-expression>', 'eval')
    namespace = {**SAFE_FUNCTIONS, **SAFE_CONSTANTS}

    def evaluate(t: float) -> float:
        return float(eval(code, {'__builtins__': {}}, {**namespace, TIME_VARIABLE: float(t)}))

    return evaluate


def evaluate_expression(expression: str, t: float) -> float:
    """One-shot helper: validate, compile and evaluate ``expression`` at ``t``."""
    return compile_expression(expression)(t)
