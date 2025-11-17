"""Tests for string parsing."""

import pytest

from reno import ops, parser
from reno.components import Distribution, Flow, Piecewise, Scalar, Stock, Variable
from reno.model import Model


@pytest.mark.parametrize(
    "string,expected,expected_type",
    [
        ("5.0", 5.0, float),
        (" 5.0", 5.0, float),
        ("5.0 ", 5.0, float),
        (" 5.0 ", 5.0, float),
        ("13", 13, int),
        (" 13", 13, int),
        (" 13 ", 13, int),
        ("10", 10, int),
        ("something", "something", str),
    ],
)
def test_int_float_parsing(string, expected, expected_type):
    val = parser.parse_value(string)
    assert isinstance(val, expected_type)
    assert val == expected


@pytest.mark.parametrize(
    "string,expected_args,expected_kwargs",
    [
        ("Normal(5.0, 10)", [5.0, 10], {}),
        ("Normal(5.0, std=10)", [5.0], {"std": 10}),
        ("Normal(5.0, std =13.2)", [5.0], {"std": 13.2}),
        ("Scalar(1.0)", [1.0], {}),
        ("Scalar([1.0, 6.0, 2.0])", [[1.0, 6.0, 2.0]], {}),
    ],
)
def test_param_parsing(string, expected_args, expected_kwargs):
    args, kwargs, *_ = parser.parse_function_args(string)
    assert args == expected_args
    assert kwargs == expected_kwargs


def test_full_op_parse():
    out = parser.parse_class_or_scalar(" Normal(5.0, std =13.2)")
    assert isinstance(out, ops.Normal)
    assert out.sub_equation_parts[0].value == 5.0
    assert out.sub_equation_parts[1].value == 13.2


def test_blank_parse():
    assert parser.parse_class_or_scalar(" ") is None


@pytest.mark.parametrize(
    "string,expected_name,expected_args",
    [
        ("(+ 5 3)", "+", ["5", "3"]),
        ("(+ (5 3))", "+", ["(5 3)"]),
        ("(+ (5) 3)", "+", ["(5)", "3"]),
        ("(+ ((5)) 3)", "+", ["((5))", "3"]),
        ("(+ 'thing' 3)", "+", ["'thing'", "3"]),
        ("(+ (+ 2 'thing') 3)", "+", ["(+ 2 'thing')", "3"]),
        ("+ 5 3", "+", ["5", "3"]),
        ("+ 5 (+ 3 6)", "+", ["5", "(+ 3 6)"]),
        ("(+ 5 (+ 3 6))", "+", ["5", "(+ 3 6)"]),
    ],
)
def test_prefix_op_str(string, expected_name, expected_args):
    """We should be able to separate out arguments in raw strings
    of prefix syntax."""
    name, args = parser.parse_op_str(string)

    assert name == name
    assert args == expected_args


@pytest.mark.parametrize(
    "string,expected_result",
    [
        ("(+ 5 3)", 8),
        ("(+ (- 5 2) 3)", 6),
        ("(+ (- 5 (maximum 3 4)) (minimum 2 1))", 2),
    ],
)
def test_prefix_parsing_math(string, expected_result):
    """Basic math operations should parse and evaluate correctly."""
    op = parser.parse(string)
    assert op.eval(0) == expected_result


def test_prefix_parsing_w_resolution():
    """References should correctly resolve during prefix parsing."""
    v = Variable(Scalar(2))
    v.name = "v"
    v.populate(1, 1)

    op = parser.parse("(+ 3 'v')", {"v": v})
    assert op.eval(0) == 5


def test_prefix_parsing_w_piecewise_example():
    m = Model()
    m.decision = Variable(ops.Bernoulli(0.5))
    m.f = Flow(
        Piecewise(
            [Scalar(100), Scalar(0)],
            [ops.eq(m.decision, Scalar(1)), ops.eq(m.decision, Scalar(0))],
        )
    )
    m.s = Stock()
    m.s += m.f

    p_op = parser.parse(str(m.f.eq), {r.qual_name(): r for r in m.all_refs()})
    assert len(p_op.conditions) == 2
    assert len(p_op.equations) == 2

    assert m.decision in p_op.conditions[0].seek_refs()


def test_prefix_mixed_classsyntax():
    """The distribution reprs are formatted like python functions, parsing
    should handle this appropriately."""
    op = parser.parse(str(ops.sum(ops.Bernoulli(0.5))))
    assert isinstance(op.sub_equation_parts[0], Distribution)
