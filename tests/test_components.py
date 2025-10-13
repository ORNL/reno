"""Tests for individual operations and/or SD components."""

import pytest

from reno import ops, utils
from reno.components import Flow, PostMeasurement, Scalar, Stock, TimeRef, Variable
from reno.model import Model


@pytest.mark.parametrize(
    "input_list,sample_size,expected_output",
    [
        ([1, 2, 3], 2, [1, 2]),
        ([1, 2, 3], 3, [1, 2, 3]),
        ([1, 2, 3], 4, [1, 2, 3, 1]),
        ([1, 2, 3], 5, [1, 2, 3, 1, 2]),
        ([1, 2, 3], 6, [1, 2, 3, 1, 2, 3]),
    ],
)
def test_list_dist(input_list, sample_size, expected_output):
    """Specifying a list distribution and populating with various sample sizes
    should tile appropriately."""

    ldist = ops.List(input_list)
    if sample_size < len(input_list):
        with pytest.warns(RuntimeWarning):
            ldist.populate(sample_size)
    else:
        ldist.populate(sample_size)
    assert (ldist.value == expected_output).all()


def test_static_check_scalar():
    """Flows and variables whose equations are just a scalar should be static."""

    f0 = Flow(Scalar(5))
    f0.populate(5, 5)
    assert f0._static


def test_static_check_dist():
    """Flows and variables whose equations are just distributions should be static."""
    v0 = Variable(ops.List([1, 2]))
    v0.populate(5, 5)
    assert v0._static

    f0 = Flow(ops.List([1, 2]))
    f0.populate(5, 5)
    assert f0._static


def test_static_check_static_eq():
    """Flows and variables whose equations are purely static should be static."""
    v0 = Variable(Scalar(2) + 1)
    v0.populate(5, 5)
    assert v0._static

    f0 = Flow(Scalar(2) + 1)
    f0.populate(5, 5)
    assert f0._static


def test_static_check_static_eq_w_refs():
    """Flows and variables whose equations are purely static (containing refs to other
    static refs) should be static."""

    m = Model()
    m.v = Variable(Scalar(3))

    m.v0 = Variable(Scalar(2) + 1 + m.v)

    m.f0 = Flow(Scalar(2) + 1 + m.v)

    m._populate(5, 5)

    assert m.v0._static
    assert m.f0._static

    assert m.v0.is_static()
    assert m.f0.is_static()


def test_static_check_eq_w_refs():
    """Flows and variables whose equations are not static (contain non-static
    refs) should not be static."""

    t = TimeRef()

    v0 = Variable(Scalar(2) + 1 + t)
    v0.populate(5, 5)
    assert not v0._static

    f0 = Flow(Scalar(2) + 1 + t)
    f0.populate(5, 5)
    assert not f0._static


def test_static_check_eq_w_nested_refs():
    """Flows and variables whose equations are not static (contain refs that
    contain refs that are non-static, e.g. time) should not be static."""

    t = TimeRef()
    v = Variable(t)

    v0 = Variable(Scalar(2) + 1 + v)
    assert not v0.is_static()

    f0 = Flow(Scalar(2) + 1 + v)
    assert not f0.is_static()


def test_static_check_eq_w_nested_refs_to_stocks():
    """Flows and variables whose equations are not static (contain refs that
    contain refs that are non-static, e.g. a stock) should not be static."""

    s = Stock()
    v = Variable(s)

    v0 = Variable(Scalar(2) + 1 + v)
    assert not v0.is_static()

    f0 = Flow(Scalar(2) + 1 + v)
    assert not f0.is_static()


def test_static_check_static_flow_but_dynamic_limits():
    """Flows or variables whose equations are static but have non-static
    limits should not be static."""

    m = Model()
    t = TimeRef()
    m.v = Variable(Scalar(1), max=t)
    m.f = Flow(Scalar(1), max=t)

    m.v.populate(5, 5)
    m.f.populate(5, 5)

    assert not m.v._static
    assert not m.f._static

    assert not m.v.is_static()
    assert not m.f.is_static()

    m.x = Variable(Scalar(1), max=ops.minimum(Scalar(0), t))
    m.y = Variable(Scalar(1), max=ops.minimum(Scalar(0), m.f))

    assert not m.x.is_static()
    assert not m.y.is_static()


def test_static_check_eq_with_historical_value():
    """Flows or variables that reference a historical value are inherently
    based in time and can't be static."""

    v0 = Variable(Scalar(1))
    v1 = v0.history(Scalar(1))
    f0 = Flow(v1)

    v0.name = "v0"
    v1.name = "v1"
    f0.name = "f0"

    v0.populate(5, 5)
    # v1.populate(5, 5)
    f0.populate(5, 5)

    assert not v1.is_static()
    assert not f0.is_static()


def test_single_ref_eq_appears_in_seek_refs():
    """An equation that is just another reference, e.g. flow1 = var1, should
    correctly return var1 when seek_refs is called on flow1.eq"""

    m = Model()
    m.v0 = Variable(Scalar(1))
    m.f0 = Flow(m.v0)
    # m.f0 = Flow()
    # m.f0.eq = m.v0

    assert m.f0.seek_refs() == [m.v0]
    assert m.f0.eq.seek_refs() == [m.v0]


def test_depencency_ordering_metrics():
    """Calling dependency_compute_order on a specific set of metrics should
    correctly order them."""

    m = Model()
    m.v0 = Variable(Scalar(1))
    m.f0 = Flow(m.v0)
    m.s0 = Stock()
    m.s0 += m.f0

    m.metric1 = PostMeasurement()
    m.metric2 = PostMeasurement()

    m.metric1.eq = m.metric2 + m.f0
    m.metric2.eq = m.s0

    ordered = utils.dependency_compute_order([m.metric1, m.metric2])
    assert ordered == [m.metric2, m.metric1]


def test_submodel_getattrs():
    """Getting attributes of submodels (e.g. getattr(my_model, "submodel.attr"))
    should work."""

    m = Model(name="parent")
    s = Model(name="child")
    s.v0 = Variable(Scalar(1))
    m.s = s

    assert m.s.v0 == s.v0
    assert getattr(s, "v0") == s.v0
    assert getattr(m, "s.v0") == s.v0


def test_min_max_respected():
    """A min/max on a variable should mean that that range isn't exceeded."""
    m = Model()
    m.limited = Variable(min=0, max=5)
    m.change = Variable(6)
    m.limited.eq = m.change
    m()
    assert m.limited.value[0] == 5

    m.change.eq = -1
    m()
    assert m.limited.value[0] == 0

    m.limited.min = -2
    m()
    assert m.limited.value[0] == -1


def test_init_without_explicit_scalar():
    """Creating a stock with an initial condition that is implicitly a scalar should be auto
    converted."""
    m = Model()
    m.thing = Stock(init=5)
    m.inflow = Flow(1)
    m.thing += m.inflow
    ds = m()
    assert ds.thing.values[-1][-1] == 14.0
