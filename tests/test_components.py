"""Tests for individual operations and/or SD components."""

import pytest

from reno import ops, utils
from reno.components import Flow, Metric, Piecewise, Scalar, Stock, TimeRef, Variable
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


def test_seq_normal_not_static():
    """A normal distribution with per_timestep should not be considered static."""
    m = Model()
    m.v0 = Variable(ops.Normal(1, 2, per_timestep=True))
    assert not m.v0.is_static()


def test_interpolate_on_static_still_static():
    """An interpolate call on a static variable should still be static."""
    m = Model()
    m.v0 = Variable(Scalar([0, 1, 1.5, 2.72, 3.14]))
    m.v1 = Variable(ops.interpolate(m.v0, [1, 2, 3], [3, 2, 0]))
    assert m.v0.is_static()
    assert m.v1.is_static()


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

    m.metric1 = Metric()
    m.metric2 = Metric()

    m.metric1.eq = m.metric2 + m.f0
    m.metric2.eq = m.s0

    ordered = utils.dependency_compute_order([m.metric1, m.metric2])
    assert ordered == [m.metric2, m.metric1]


def test_dependency_ordering_w_inits():
    """An init equation that is just another reference should correctly gather it as a seek_refs and compute order"""
    m = Model()
    m.v0 = Variable(1)
    m.f0 = Flow(m.v0)
    m.s = Stock(init=m.v0)
    m.s += m.f0

    assert m.s.init.seek_refs() == [m.v0]
    assert m.dependency_compute_order(inits_order=True) == [m.v0, m.s, m.f0]


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
    assert m.limited.value == 5

    m.change.eq = -1
    m()
    assert m.limited.value == 0

    m.limited.min = -2
    m()
    assert m.limited.value == -1


def test_init_without_explicit_scalar():
    """Creating a stock with an initial condition that is implicitly a scalar should be auto
    converted."""
    m = Model()
    m.thing = Stock(init=5)
    m.inflow = Flow(1)
    m.thing += m.inflow
    ds = m()
    assert ds.thing.values[-1][-1] == 14.0


def test_populate_scalar_w_dims():
    """Populating a variable that is assigned a static single value but provided a dim > 1 should
    correctly broadcast to a full dim sized repeat of that value."""
    thing = Variable(5, dim=4)
    thing.populate(1, 5)
    print(thing.value)
    assert thing.value.shape == (4,)


def test_multidim_scalar_eval():
    """Running eval on a static scalar with additional dim should return the expanded array I think."""
    thing1 = Variable([1, 2, 3, 4], dim=4)
    thing2 = Variable(2, dim=4)
    thing1.populate(1, 1)
    thing2.populate(1, 1)

    assert (thing1.value == [1, 2, 3, 4]).all()
    assert (thing1.eval(0) == [1, 2, 3, 4]).all()

    assert (thing2.value == [2, 2, 2, 2]).all()
    assert (thing2.eval(0) == [2, 2, 2, 2]).all()


def test_multidim_component_shapes(multidim_model_determ):
    """Getting the shapes of all the components in a multidim model should be correct"""
    assert multidim_model_determ.v1.shape == 4
    assert multidim_model_determ.v2.shape == 1
    assert multidim_model_determ.v3.shape == 4
    assert multidim_model_determ.s.shape == 4


def test_multidim_component_shapes_implicit(multidim_model_determ_implicit):
    """Getting the shapes of all the components in a multidim model should be correct"""
    assert multidim_model_determ_implicit.v1.shape == 4
    assert multidim_model_determ_implicit.v2.shape == 1
    assert multidim_model_determ_implicit.v3.shape == 4
    assert multidim_model_determ_implicit.s.shape == 4


def test_unexpected_dim_throws_error():
    """Assigning a dim to a thing with a shape not of that dim should throw an exception."""
    v = Variable([4, 4, 4], dim=7)
    with pytest.raises(Exception):
        v.shape


def test_list_dist_shapes():
    v0 = Variable(ops.List([20.0, 18.0, 16.0, 14.0]))
    v1 = Variable(ops.List([[20, 19], [18, 17], [16, 15], [14, 13]]))
    assert v0.shape == 1
    assert v0.dtype == float
    assert v1.shape == 2
    assert v1.dtype == int


def test_dtype_transfer():
    """Type information should transfer through the equations correctly."""
    v0 = Variable(1)
    v1 = Variable(2.0)
    v2 = Variable([1.0, 2.0])
    v3 = Variable(v0 + v1)
    v4 = Variable(v0 + v2)
    v5 = Variable(v2 > 1.0)
    v6 = Variable(1, dtype=float)
    v7 = Variable(v1, dtype=int)

    assert v0.dtype == int
    assert v1.dtype == float
    assert v2.dtype == float
    assert v3.dtype == float
    assert v4.dtype == float
    assert v5.dtype == bool
    assert v6.dtype == float

    v1.populate(1, 1)
    v7.populate(1, 1)
    assert v7.dtype == int
    assert v7.value == 2


def test_dtype_otf_config_diff_type():
    """A variable that initially has one type, but is then run passed in something
    of a different type, it should be that different type for that different run."""

    m = Model()
    m.v0 = Variable(1)

    m()
    assert m.v0.dtype == int
    m(v0=ops.Normal(10, 5), keep_config=True)
    assert m.v0.dtype == float


def test_piecewise_shape_multidim_eq():
    """Piecewise dimensions of multidim equations should be multidim"""
    v0 = Variable(0)
    assert Piecewise([0, 5], [v0 < 2, v0 >= 2]).shape == 1
    assert Piecewise([[0, 0, 0, 0], 5], [v0 < 2, v0 >= 2]).shape == 4
    assert Piecewise([Variable(0, dim=4), 5], [v0 < 2, v0 >= 2]).shape == 4

    assert (
        Piecewise([Variable(0, dim=4), 5], [v0 < 2, v0 >= 2]).eval() == [0, 0, 0, 0]
    ).all()


def test_piecewise_shape_multidim_condition():
    """Piecewise dimensions of multidim conditions should be multidim"""
    v0 = Variable([0, 1, 2, 3])
    assert (v0 < 2).shape == 4
    assert Piecewise([0, 5], [v0 < 2, v0 >= 2]).shape == 4
    assert Piecewise([[0, 0, 0, 0], 5], [v0 < 2, v0 >= 2]).shape == 4
    assert Piecewise([Variable(0, dim=4), 5], [v0 < 2, v0 >= 2]).shape == 4

    assert (Piecewise([0, 5], [v0 < 2, v0 >= 2]).eval() == [0, 0, 5, 5]).all()


def test_multidim_piecewise():
    """Evaluating a piecewise with a data dimension should operate across
    all dimensions."""

    m = Model()
    m.v0 = Variable([0, 1, 2, 3])
    m.v1 = Variable(Piecewise([0, 5], [m.v0 < 2, m.v0 >= 2]))
    ds = m(n=1, steps=1)
    assert (ds.v1.values == [[0, 0, 5, 5]]).all()


def test_multidim_piecewise_timeseries():
    """Piecewise should still work with a data dim and also the time dim."""
    m = Model()
    t = TimeRef()
    m.v0 = Variable([0, 1, 2, 3])
    m.v1 = Variable(m.v0 + t)
    m.v2 = Variable(Piecewise([0, 5], [m.v1 < 3, m.v1 >= 3]))

    ds = m(1, 3)
    assert (ds.v2.values == [[[0, 0, 0, 5], [0, 0, 5, 5], [0, 5, 5, 5]]]).all()


def test_multidim_piecewise_assignment():
    """A piecewise assignment from a matrix should still work."""
    m = Model()
    t = TimeRef()
    m.v0_b = Variable(ops.List([[20, 19], [16, 15]]))
    m.v0 = Variable(m.v0_b - t)
    m.v1_b = Variable(ops.List([[0, 1], [4, 5]]))
    m.v1 = Variable(m.v1_b + t)

    m.v2 = Variable(Piecewise([0, m.v0], [m.v1 < 5, m.v1 >= 5]))

    ds = m(2, 2)

    assert (ds.v2.values == [[[0, 0], [0, 0]], [[0, 15], [15, 14]]]).all()


def test_nested_piecewise_statics():
    """A piecewise with another piecewise inside of it should still work."""
    m = Model()
    m.v0 = Variable(ops.List([0, 1, 2, 3, 4]))
    m.v1 = Variable(
        Piecewise(
            [0, Piecewise([1, 2], [m.v0 < 3, m.v0 >= 3])],
            [m.v0 < 1, m.v0 >= 1],
        )
    )
    ds = m(5, 1)
    assert (ds.v1.values == [[0, 1, 1, 2, 2]]).all()


# TODO: skip this until multidim/arbitrary taps in pymc and multidim static time refs
# supported in reno
@pytest.mark.skip
def test_multidim_historical_value():
    """A multidim vector used to index a historical of a multidim value should correctly pick
    up the right timestep on an individual data dim entry level"""
    m = Model()
    t = TimeRef()
    m.v0 = Variable(ops.List([0, 1, 2], [10, 11, 12]))
    m.v1 = Variable(m.v0 + t + 1)
    m.index = Variable([1, 2, 1])
    m.v2 = Variable(m.v1.history(t - m.index))

    ds = m(2, 3)

    assert (
        ds.v1.values
        == [
            [[1, 2, 3], [2, 3, 4], [5, 6, 7]],
            [[11, 12, 13], [12, 13, 14], [13, 14, 15]],
        ]
    ).all()
    assert (
        ds.v2.values
        == [[[0, 0, 0], [1, 0, 3], [2, 2, 4]], [[0, 0, 0], [11, 0, 13], [12, 12, 14]]]
    ).all()


# TODO: temp doing away with compute_masks/row_indices, too much complexity
# right now for too little gain
# def test_compute_mask_basic_and_rows():
#     """Evalling with a compute mask requesting specific rows should only return requested rows."""
#     m = Model()
#     m.v0 = Variable()
#     m.v0._static = False
#     m.v0._sample_dim = True
#     m.v0.value = np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]])
#     m.v0.computed_mask = np.array([[True, True, True, True, True]])
#     print(m.v0.eval(0))
#     assert (m.v0.eval(0) == [0, 10]).all()
#
#     assert (m.v0.eval(0, np.array([False, True])) == [10]).all()
#
#
# def test_compute_mask_datadim():
#     """Evalling with a compute mask requesting specific rows and specific entries in data dimension should only return requested data."""
#     m = Model()
#     m.v0 = Variable()
#     m.v0._static = False
#     m.v0._sample_dim = True
#     m.v0.value = np.array([[[0, 1], [2, 3]], [[10, 11], [12, 13]]])
#     m.v0.computed_mask = np.array([[True, True]])
#     print(m.v0.eval(0))
#     assert (m.v0.eval(0) == [[0, 1], [10, 11]]).all()
#
#     assert (m.v0.eval(0, np.array([[False, True], [True, True]])) == [1, 10, 11]).all()


def test_multidim_stock_w_singledim_init():
    """A stock with requested extra dim should extend that dim to the init equation, even
    if the init equation has no specified dim."""
    m = Model()
    m.s = Stock(init=100, dim=5)
    ds = m(n=1, steps=3)
    assert ds.s.values[0][0].shape == (5,)


def test_multidim_stock_w_singledim_init_pymc():
    """A stock with requested extra dim should extend that dim to the init equation, even
    if the init equation has no specified dim."""
    m = Model()
    m.s = Stock(init=100, dim=5)
    ds = m.pymc(n=1, steps=3)
    assert ds.prior.s.values[0][0][0].shape == (5,)


def test_multidim_var_w_singledim_eq_pymc():
    """A variable with requested extra dim should extend that dim to the equation's first eval, even
    if the equation itself has no specified dim."""
    m = Model()
    m.v = Variable(4, dim=5)
    ds = m.pymc(n=1, steps=3)
    assert ds.prior.v.values[0][0].shape == (5,)


def test_non_basic_historical_index():
    """A historical index equation that isn't simply "t - static_var" should still work correctly.
    Specifically, a static integer historial index equation should be zero until that integer
    timestep is reached and should then be the same value for the remainder of the sim.
    """
    m = Model()
    t = TimeRef()
    m.v = Variable(t + 2)
    m.f = Flow(m.v.history(3))
    m.s = Stock()
    m.s += m.f

    ds = m()
    assert (ds.f.values == [[0, 0, 0, 5, 5, 5, 5, 5, 5, 5]]).all()


def test_non_basic_historical_index_pymc():
    """A historical index equation that isn't simply "t - static_var" should still work correctly.
    Specifically, a static integer historial index equation should be zero until that integer
    timestep is reached and should then be the same value for the remainder of the sim.
    """
    m = Model()
    t = TimeRef()
    m.v = Variable(t + 2)
    m.f = Flow(m.v.history(3))
    m.s = Stock()
    m.s += m.f

    ds = m.pymc(compute_prior_only=True)
    assert (ds.prior.f.values[0] == [[0, 0, 0, 5, 5, 5, 5, 5, 5, 5]]).all()


def test_non_basic_dynamic_historical_index():
    """A historical index equation with a changing (but non basic) historical index equation
    should still work correctly."""
    m = Model()
    t = TimeRef()
    m.v = Variable(t + 2)
    m.f = Flow(m.v.history((t / 2).astype(int)))
    m.s = Stock()
    m.s += m.f

    ds = m()
    assert (ds.f.values == [[2, 2, 3, 3, 4, 4, 5, 5, 6, 6]]).all()


def test_non_basic_dynamic_historical_index_pymc():
    """A historical index equation with a changing (but non basic) historical index equation
    should still work correctly in pymc."""
    m = Model()
    t = TimeRef()
    m.v = Variable(t + 2)
    m.f = Flow(m.v.history((t / 2).astype(int)))
    m.s = Stock()
    m.s += m.f

    ds = m.pymc(n=1, compute_prior_only=True)
    assert (ds.prior.f.values[0] == [[2, 2, 3, 3, 4, 4, 5, 5, 6, 6]]).all()


def test_diff_hist_index_across_list_dist():
    """A reno.List distribution used to index different historical values
    should correctly provide different results for each of n"""
    m = Model()
    t = TimeRef()
    m.v1 = Variable(t + 2)
    m.v2 = Variable(ops.List([1, 2]))
    m.f = Flow(m.v1.history(3 + m.v2))

    ds = m(n=2)
    assert (
        ds.f.values == [[0, 0, 0, 0, 6, 6, 6, 6, 6, 6], [0, 0, 0, 0, 0, 7, 7, 7, 7, 7]]
    ).all()


# TODO: test for basic too
def test_multidim_dynamic_hist():
    """A multidim index equation (non-basic) for a variable should work in
    normal reno math."""
    m = Model()
    t = TimeRef()
    m.v1 = Variable(t + 2)
    m.v2 = Variable([1, 2])
    m.f = Flow(m.v1.history(3 + m.v2))

    ds = m()
    assert (
        ds.f.values
        == [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [6, 0],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
            ]
        ]
    ).all()


def test_multidim_dynamic_hist_pymc():
    """A multidim index equation (non-basic) for a variable should work in pymc."""
    m = Model()
    t = TimeRef()
    m.v1 = Variable(t + 2)
    m.v2 = Variable([1, 2])
    m.f = Flow(m.v1.history(3 + m.v2))

    ds = m.pymc(1, compute_prior_only=True)
    assert (
        ds.prior.f.values[0]
        == [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [6, 0],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
                [6, 7],
            ]
        ]
    ).all()


def test_dist_based_basic_hist_pymc():
    """A history indexed by a distribution, even in a basic time-based equation, should work in pymc."""
    m = Model()
    t = TimeRef()
    m.v0 = Variable(t + 1)
    m.v1 = Variable(ops.DiscreteUniform(2, 5))
    m.v2 = Variable(m.v0.history(t - m.v1))
    ds = m.pymc(100, compute_prior_only=True)
    assert 8 in ds.prior.v2[0, :, -1]
    assert 7 in ds.prior.v2[0, :, -1]
    assert 6 in ds.prior.v2[0, :, -1]
    assert 5 in ds.prior.v2[0, :, -1]


def test_rshift_lshift_op_overloading_for_stocks():
    """With the magic of operator overloading, stock >> flow >> stock should correctly
    add a flow as inflows and outflows appropriately"""
    m = Model()
    with m:
        s0, s1, s2 = Stock(), Stock(), Stock()
        f0, f1 = Flow(), Flow()
        s0 << f0 << s1 >> f1 >> s2

    assert f0 in s0.in_flows
    assert f0 in s1.out_flows
    assert f1 in s1.out_flows
    assert f1 in s2.in_flows


def test_implicit_inflow_eq():
    """An equation with a flow given to a stock should correctly create an implicit
    flow."""
    m = Model()
    with m:
        s0, s1 = Stock(), Stock()
        f0 = Flow(3)

        s0 >> f0
        (f0 - 1) >> s1

    assert s1.in_flows[0].implicit

    ds = m()
    assert ds.s1.values[0][1] == 2
    assert ds.s0.values[0][1] == -3


def test_inflow_list():
    """A stock given a list of flows to add should correctly add all of them."""
    m = Model()
    with m:
        s0 = Stock()
        f0, f1 = Flow(1), Flow(2)

        [f0, f1] >> s0

    assert s0.in_flows == [f0, f1]
    ds = m()
    assert ds.s0.values[0][1] == 3
