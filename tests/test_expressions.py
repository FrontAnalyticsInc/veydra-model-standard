"""Tests for the safe dynamic-parameter expression evaluator."""

import math
import time
from datetime import datetime

import pytest

from veydra_model_standard import (
    SimulationContext,
    UnsafeExpressionError,
    compile_expression,
    evaluate_expression,
)


def dynamic_param(expression, default=-1.0, points=None):
    spec = {'__dynamic__': True, 'mode': 'function', 'expression': expression,
            'defaultValue': default}
    if points is not None:
        spec['points'] = points
    return spec


def ctx_value(expression, t=0.0, **kw):
    ctx = SimulationContext(t, datetime(2024, 1, 1), {'p': dynamic_param(expression, **kw)}, {})
    return ctx.get_param('p')


# --- expressions that must keep working ------------------------------------

@pytest.mark.parametrize('expression,t,expected', [
    ('sin(t * 0.1) * 50 + 100', 10.0, math.sin(1.0) * 50 + 100),   # base.py docstring
    ('sin(t * 0.1) * 50 + 200', 0.0, 200.0),                        # veydra-docs
    ('10 * sin(t / 5) + 50', 5.0, 10 * math.sin(1.0) + 50),         # veydra-docs
    ('50 + 25 * sin(t * 0.1)', 0.0, 50.0),                           # UI placeholder
    ('100.0 + 12.50 * sin(t * 0.1)', 0.0, 100.0),                    # UI "Sine wave" preset
    ('0 + (100 - 0) * t / 365', 73.0, 20.0),                         # UI "Linear ramp" preset
    ('100 * exp(-t * 0.05)', 20.0, 100 * math.exp(-1.0)),            # UI "Decay" preset
    ('sqrt(abs(-16)) + log(e) + pow(2, 3) + floor(1.7) + ceil(1.2)', 0.0, 4 + 1 + 8 + 1 + 2),
    ('max(1, t, 3) - min(t, 2)', 5.0, 5 - 2),
    ('cos(pi) + tan(0)', 0.0, -1.0),
    ('2 ** 10 // 3 % 7', 0.0, (1024.0 // 3) % 7),
    ('-t + +t', 4.0, 0.0),
    ('100 if t < 50 else 200', 60.0, 200.0),                         # step, Python syntax
    ('300 if (t > 30 and t < 50) else 100', 40.0, 300.0),            # pulse, Python syntax
    ('1 if not (t >= 10 or t == 3) else 0', 2.0, 1.0),
    ('t != 1', 2.0, 1.0),
])
def test_valid_expressions(expression, t, expected):
    assert evaluate_expression(expression, t) == pytest.approx(expected)
    assert ctx_value(expression, t) == pytest.approx(expected)


def test_result_is_float():
    assert isinstance(evaluate_expression('1 + 1', 0), float)
    assert isinstance(evaluate_expression('t > 1', 2), float)


# --- sandbox escapes must be rejected without executing ----------------------

ESCAPES = [
    "().__class__.__base__.__subclasses__()",
    "[c for c in ().__class__.__base__.__subclasses__() if c.__name__=='BuiltinImporter'][0]"
    ".load_module('os').system('true')",
    "__import__('os').system('true')",
    "(lambda: 1)()",
    "sin.__self__",
    "getattr(sin, '__self__')",
    "eval('1')",
    "open('/etc/passwd')",
    "'abc'",
    "b'abc'",
    "[1, 2][0]",
    "{1: 2}",
    "(1, 2)",
    "x",
    "t.real",
    "max(*[1, 2])",
    "max(key=abs)",
    "(y := 1)",
    "True",
    "None",
    "f'{t}'",
    "...",
    "1 if t in (1,) else 0",
    "t is t",
    "~1",
    "1 << 100",
    "1 ^ 2",
    "sin(t)(1)",
]


@pytest.mark.parametrize('expression', ESCAPES)
def test_unsafe_expressions_rejected(expression):
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        compile_expression(expression)


@pytest.mark.parametrize('expression', ESCAPES)
def test_unsafe_expressions_fall_back_to_default(expression):
    assert ctx_value(expression, default=-1.0) == -1.0


def test_escape_payload_does_not_execute(tmp_path):
    marker = tmp_path / 'pwned'
    payload = (
        "[c for c in ().__class__.__base__.__subclasses__() "
        "if c.__name__=='BuiltinImporter'][0].load_module('os')"
        f".system('touch {marker}')"
    )
    assert ctx_value(payload, default=7.0) == 7.0
    assert not marker.exists()


# --- resource exhaustion ------------------------------------------------------

@pytest.mark.parametrize('expression', [
    '9 ** 9 ** 9',
    'pow(9, pow(9, 9))',
    'floor(1e300) ** floor(1e300)',
    'ceil(1e300) ** 99',
    'exp(1000)',
])
def test_huge_numbers_fail_fast(expression):
    start = time.monotonic()
    assert ctx_value(expression, default=5.0) in (5.0, math.inf)
    assert time.monotonic() - start < 1.0


def test_length_limit():
    with pytest.raises(UnsafeExpressionError):
        compile_expression('1+' * 600 + '1')


def test_deep_nesting_fails_safely():
    assert ctx_value('(' * 5000 + '1' + ')' * 5000, default=3.0) == 3.0


def test_non_string_expression_rejected():
    with pytest.raises(UnsafeExpressionError):
        compile_expression(123)


# --- existing behaviour preserved --------------------------------------------

def test_precomputed_points_take_priority():
    # JS-only syntax (frontend presets) still works when points are supplied.
    value = ctx_value('t < 50 ? 1 : 2', t=5.0, points=[[0, 0.0], [10, 10.0]])
    assert value == 5.0


def test_empty_expression_returns_default():
    assert ctx_value('', default=9.0) == 9.0


def test_invalid_syntax_returns_default():
    assert ctx_value('sin(', default=9.0) == 9.0
