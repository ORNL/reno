"""Tests for the math operations."""

import numpy as np

import reno
from reno import model, ops
from reno.components import Flow, Metric, Piecewise, Scalar, Stock, TimeRef, Variable


def test_sum_on_matrix():
    """Running ops.sum on a matrix should give you a row-wise sum."""
    v = Variable()
    v.value = np.array([[0, 1, 2], [1, 2, 3]])

    assert (ops.sum(v.timeseries).eval(3) == np.array([[3, 6]])).all()


def test_sum_on_vector():
    """Running ops.sum on a vector (static variable) should give you the row-wise
    "sum" which is just the value times the number of timesteps."""
    v = Variable()
    v.value = np.array([2, 3])

    assert (ops.sum(ops.orient_timeseries(v)).eval(3) == np.array([[8, 12]])).all()


def test_sum_on_matrix_start_stop():
    """Running ops.sum on a matrix should give you a row-wise sum, correctly
    accounting for specified range."""
    v = Variable()
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert (v.timeseries[2:].sum().eval(5) == np.array([[9, 12]])).all()
    assert (v.timeseries[:2].sum().eval(5) == np.array([[1, 3]])).all()
    assert (v.timeseries[1:3].sum().eval(5) == np.array([[3, 5]])).all()


def test_sum_on_vector_start_stop():
    """Running ops.sum on a vector (static variable) should give you the row-wise
    "sum" which is just the value times the number of timesteps. Correctly
    accounting for specified range."""
    v = Variable()
    v.value = np.array([2, 3])

    assert (v.timeseries[2:].sum().eval(4) == np.array([[6, 9]])).all()
    assert (v.timeseries[:2].sum().eval(4) == np.array([[4, 6]])).all()
    assert (v.timeseries[1:3].sum().eval(4) == np.array([[4, 6]])).all()


def test_static_value_sum():
    """By default, the sum of a static value will just be itself, unless you access
    it as a slice.
    NOTE: changed this behavior by having sum automatically insert slice if static detected.
    NOTE: changed again due to separate timeseries vs vec series concerns, no longer
    makes sense to take a sum of a non vector thing (which you only get for a scalar by
    looking at its full timeseries)
    """
    v = Variable(Scalar(5))

    # assert v.sum().eval(4) == np.array([25])
    assert v.timeseries[:].sum().eval(4) == np.array([25])


def test_sum_of_series_inside_model():
    """The sum of a timeseries inside a model should still work the same as
    outside a model."""
    m = model.Model()
    with m:
        v0 = Variable(2)
        v1 = Variable(v0.timeseries[0:4], dim=4)
        v2 = Variable(v1.sum())
    ds = m()
    assert (ds.v1.values[0][4] == [2, 2, 2, 2]).all()
    assert ds.v2.values[0][4] == 8


def test_series_max():
    """Both API forms should return the correct series max."""
    v = Variable()
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert (v.series_max().eval(4) == ops.series_max(v).eval(4)).all()


def test_slice_on_matrix():
    """Slice bounds should work properly on a matrix."""
    v = Variable()
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert (v.timeseries[:2].eval(5) == np.array([[0, 1], [1, 2]])).all()
    assert v.timeseries[:2].eval(5).shape == np.array([[0, 1], [1, 2]]).shape
    assert (v.timeseries[0:2].eval(5) == np.array([[0, 1], [1, 2]])).all()
    assert v.timeseries[0:2].eval(5).shape == np.array([[0, 1], [1, 2]]).shape
    assert (v.timeseries[3:].eval(5) == np.array([[3, 4], [4, 5]])).all()
    assert v.timeseries[3:].eval(5).shape == np.array([[3, 4], [4, 5]]).shape


def test_slice_on_vector():
    """Slice bounds should work properly on a matrix."""
    v = Variable()
    v.value = np.array([2, 3])

    assert (v.timeseries[:2].eval(3) == np.array([[2, 2], [3, 3]])).all()
    assert v.timeseries[:2].eval(3).shape == np.array([[2, 2], [3, 3]]).shape
    assert (v.timeseries[0:2].eval(3) == np.array([[2, 2], [3, 3]])).all()
    assert v.timeseries[0:2].eval(3).shape == np.array([[2, 2], [3, 3]]).shape
    assert (v.timeseries[3:].eval(5) == np.array([[2, 2, 2], [3, 3, 3]])).all()
    assert v.timeseries[3:].eval(5).shape == np.array([[2, 2, 2], [3, 3, 3]]).shape


def test_slice_on_scalar():
    """Slicing a scalar should result in a 2d array. (Otherwise
    series ops won't work correctly on the expanded slice.)"""
    v = Variable(Scalar(5))
    assert (v.timeseries[:].eval(4) == np.array([[5, 5, 5, 5, 5]])).all()
    assert v.timeseries[:].eval(4).shape == np.array([[5, 5, 5, 5, 5]]).shape


def test_slice_staticness_static_slice():
    """A slice with defined bounds (or static bounds) of a non-timeseries multidim should be considered
    static."""
    v = Variable([0, 1, 2, 3, 4])

    assert v[1:3].is_static()

    a = Scalar(1)
    b = Scalar(3)

    assert v[a:b].is_static()

    av = Variable(a)
    bv = Variable(b)

    assert v[av:bv].is_static()


def test_slice_staticness_static_slice_timeseries():
    """A slice with defined bounds (or static bounds) of a timeseries should NOT be considered static."""
    v = Variable(None)
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert not v.timeseries[1:3].is_static()

    a = Scalar(1)
    b = Scalar(3)

    assert not v.timeseries[a:b].is_static()

    av = Variable(a)
    bv = Variable(b)

    assert not v.timeseries[av:bv].is_static()


def test_slice_staticness_none_endpoint():
    """A slice should not be considered static if it's endpoint is None or t."""
    v = Variable(None)
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert not v.timeseries[:].is_static()
    assert not v.timeseries[1:].is_static()
    t = TimeRef()
    assert not v.timeseries[:t].is_static()


def test_slice_staticness_of_timeseries_in_model():
    """A timeseries slice within a model should _not_ be considered static."""
    m = model.Model()
    with m:
        v0 = Variable(2)
        v1 = Variable(v0.timeseries[0:4], dim=4)
        v2 = Variable(v1.sum())

    assert m.v0.is_static()
    assert not m.v1.is_static()
    assert not m.v2.is_static()


def test_sum_of_dist():
    """Sanity check that running a sum on an array from a distribution
    is correctly row-wise and doesn't collapse to single summed value."""
    b = ops.Bernoulli(0.5)
    b.populate(10)
    b.timeseries.sum().eval(5).shape == (10,)


def test_slice_end_t_is_correct_simpler():
    """A slice stop of none should actually technically be t+1 to be inclusive of current timestep? (This test is
    proof of why, getting index 0 shouldn't be nothing)"""
    v = Variable(Scalar(np.array([2, 3])))
    assert (v.timeseries[:].eval(0) == np.array([[2], [3]])).all()
    assert v.timeseries[:].eval(0).shape == np.array([[2], [3]]).shape


def test_slice_end_t_is_correct():
    """A slice stop of none should actually technically be t+1 to be inclusive of current timestep?"""
    t = TimeRef()
    m = model.Model()
    m.v0 = Variable(t + 1)
    m.v1 = Variable(Scalar(5))
    m.m0 = Metric(m.v0[m.v1 :].sum())
    ds = m()
    assert ds.m0.values[0] == 40.0


def test_piecewise_with_int():
    """Piecewise equations should support just directly specifying an integer and
    have it auto-wrapped in a scalar."""
    t = TimeRef()
    m = model.Model()
    m.v0 = Variable(Piecewise([0, 1], [t < 2, t >= 2]))
    m()


def test_interpolation():
    """Interpolation should work in regular reno math."""
    m = model.Model()
    m.v0 = Variable(Scalar([0, 1, 1.5, 2.72, 3.14]))
    m.v1 = Variable(ops.interpolate(m.v0, [1, 2, 3], [3, 2, 0]))
    ds = m(steps=1)
    np.testing.assert_almost_equal(
        ds.v1.values[0], np.array([3.0, 3.0, 2.5, 0.56, 0.0])
    )


def test_interpolation_pymc():
    """Interpolation should work in pymc math."""
    m = model.Model()
    m.v0 = Variable(Scalar([0, 1, 1.5, 2.72, 3.14]))
    m.v1 = Variable(ops.interpolate(m.v0, [1, 2, 3], [3, 2, 0]))
    ds = m.pymc(steps=1, compute_prior_only=True)
    np.testing.assert_almost_equal(
        ds.prior.v1.values[0][0], np.array([3.0, 3.0, 2.5, 0.56, 0.0])
    )


def test_pulse():
    """A pulse should fire at the correct time for the correct amount of time."""
    m = model.Model()
    m.a = Variable(3)
    m.b = Variable(ops.pulse(5, m.a))
    ds = m()
    assert (ds.b.values == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).all()


def test_first_order_delay():
    """The delay1 op should be equivalent to the corresponding manual stock/flow setup
    for a first order delay, and the pymc output should also work."""
    m1 = model.Model()
    m1.delay_time = Variable(5, dtype=float)
    m1.inflow = Flow(ops.pulse(3, 1) * 10, dtype=float)
    m1.delay_stock = Stock()
    m1.outflow = Flow(m1.delay_stock / m1.delay_time)
    m1.delay_stock += m1.inflow
    m1.delay_stock -= m1.outflow
    ds1 = m1(steps=20)

    m2 = model.Model()
    m2.delay_time = Variable(5, dtype=float)
    m2.input = Variable(ops.pulse(3, 1) * 10, dtype=float)
    m2.flow = Flow(ops.delay1(m2.input, m2.delay_time))
    ds2 = m2(steps=20)

    assert (ds1.outflow.values == ds2.flow.values).all()

    ds3 = m2.pymc(steps=20, n=1, compute_prior_only=True)
    np.testing.assert_almost_equal(ds3.prior.flow.values[0], ds2.flow.values)


def test_delays_against_insight_maker():
    """Trying to ensure my delay operation implementations are consistent with how
    insightmaker does them."""

    m = model.Model()
    m.delay_time = Variable(5.0)
    m.hungry = Flow(ops.pulse(3, 1) * 12, dtype=float)
    m.satiated = Flow(ops.delay1(m.hungry, m.delay_time))
    m.satiated3 = Flow(ops.delay3(m.hungry, m.delay_time))

    ds = m(steps=21)

    im_satiated = [
        0,
        0,
        0,
        0,
        2.4,
        1.92,
        1.536,
        1.2288,
        0.98304,
        0.786432,
        0.6291456,
        0.50331648,
        0.402653184,
        0.3221225472,
        0.25769803776,
        0.206158430208,
        0.1649267441664,
        0.13194139533312,
        0.105553116266496,
        0.0844424930131968,
        0.0675539944105575,
    ]

    im_satiated3 = [
        0,
        0,
        0,
        0,
        0,
        0,
        2.592,
        3.1104,
        2.48832,
        1.65888,
        0.995328,
        0.55738368,
        0.297271296,
        0.1528823808,
        0.0764411904,
        0.03737124864,
        0.0179381993472,
        0.00847987605504001,
        0.003957275492352,
        0.001826434842624,
        0.000834941642342401,
    ]

    np.testing.assert_almost_equal(ds.satiated.values.tolist()[0], im_satiated)
    np.testing.assert_almost_equal(ds.satiated3.values.tolist()[0], im_satiated3)


# TODO: add tests for slices and pymc...


def test_multidim_list():
    """Adding a dim to the List distribution should repeat each "row" value to the given number of columns."""
    my_list = ops.List([0, 1, 2])
    my_list.populate(3)
    assert (my_list.value == np.array([0, 1, 2])).all()

    my_list.populate(3, dim=2)
    assert (my_list.value == np.array([[0, 0], [1, 1], [2, 2]])).all()


def test_multidim_categorical():
    """Adding a dim to the Categorical distribution should return a randomized category matrix."""
    my_cats = ops.Categorical([0.25, 0.25, 0.25, 0.25])
    my_cats.populate(4)
    assert my_cats.value.shape == (4,)

    my_cats.populate(4, dim=3)
    assert my_cats.value.shape == (4, 3)


def test_multidim_vector():
    """Specifying a static variable/flow with an array and dims should treat the array as that extra dim."""
    t = TimeRef()
    thing1 = Variable([0, 1, 2, 3, 4], dim=5)
    thing2 = Variable(thing1 + 10 + t, dim=5)
    thing1.populate(1, 2)
    thing2.populate(1, 2)

    thing2.eval(1, save=True)
    assert (thing1.value == [0, 1, 2, 3, 4]).all()
    assert (thing2.value == [[10, 11, 12, 13, 14], [11, 12, 13, 14, 15]]).all()


def test_multidim_vector2():
    """A static variable/flow from a single scalar should still work as above"""
    t = TimeRef()
    thing1 = Variable(4, dim=5)
    thing2 = Variable(thing1 + 10 + t, dim=5)
    thing1.populate(1, 2)
    thing2.populate(1, 2)

    thing2.eval(1, save=True)
    assert (thing1.value == [4, 4, 4, 4, 4]).all()
    assert (thing2.value == [[14, 14, 14, 14, 14], [15, 15, 15, 15, 15]]).all()


def test_normal_population():
    """Populating a normal distribution without specifying dim should correctly populate."""
    norm = ops.Normal(0, 1)
    norm.populate(3)
    assert norm.value.shape == (3,)

    norm.populate(3, dim=2)
    assert norm.value.shape == (3, 2)


def test_basic_normal_pymc():
    """A normal distribution should translate correctly into pymc."""
    m = model.Model()
    m.v0 = Variable(reno.Normal(5, 10))
    m.f0 = Flow(m.v0)
    m.s = Stock()
    m.s += m.f0
    ds = m.pymc(5, compute_prior_only=True)
    ds.prior.v0.values[0].shape == (5,)


def test_normal_not_none():
    """A model with a normal distribution should run corrrectly."""
    m = model.Model()
    m.v1 = Variable(reno.Normal(5, 10))
    m()
    assert m.v1.value[0] is not None


def test_normal_w_dim():
    """A normal distribution on a variable with an extra dimension should populate the
    full matrix as expected."""
    m = model.Model()
    m.v1 = Variable(reno.Normal(10, 5), dim=3)
    m(n=7, steps=4)
    assert m.v1.value.shape == (7, 3)


def test_basic_normal_w_dim_pymc():
    """A normal distribution with a extra dimension should translate correctly into pymc."""
    m = model.Model()
    m.v0 = Variable(reno.Normal(5, 10), dim=3)
    m.f0 = Flow(m.v0)
    m.s = Stock()
    m.s += m.f0
    ds = m.pymc(5, compute_prior_only=True)
    ds.prior.v0.values[0].shape == (5, 3)


def test_normal_w_dim_and_single_sample():
    """A normal distribution on a variable with an extra dimension should populate the
    full matrix as expected, even when num samples is only 1."""
    m = model.Model()
    m.v1 = Variable(reno.Normal(10, 5), dim=3)
    m(n=1, steps=4)
    assert m.v1.value.shape == (1, 3)


def test_normal_w_seq():
    """A normal distribution with per_timestep requested should correctly populate
    a value for each timestep."""
    m = model.Model()
    m.v1 = Variable(reno.Normal(1, 2, per_timestep=True))
    m(n=2, steps=5)
    assert m.v1.value.shape == (2, 5)


def test_normal_w_seq_and_dim():
    """A normal distribution with per_timestep requested as well as an extra dimension
    should correctly populate all the things."""
    m = model.Model()
    m.v1 = Variable(reno.Normal(1, 2, per_timestep=True), dim=4)
    m(n=2, steps=5)
    assert m.v1.value.shape == (2, 5, 4)


def test_normal_w_seq_and_dim_pymc():
    """Running the pymc version of a model with a normal with both timesteps and extra dims
    should work the same way."""
    m = model.Model()
    m.v1 = Variable(reno.Normal(1, 2, per_timestep=True), dim=4)
    ds = m.pymc(n=2, steps=5, compute_prior_only=True)
    assert ds.prior.v1.values[0].shape == (2, 5, 4)


def test_broken_normal_dist_init_example():
    m = model.Model()

    data_dim = 20
    m.mean = Variable(100, dim=data_dim)
    m.sd = Variable(10, dim=data_dim)
    m.v = Variable(reno.Normal(m.mean, m.sd), dim=data_dim)

    m(n=1, steps=4)


def test_normal_w_dim_parameters_works():
    """A normal distribution populating two-valued data dimension with very different values
    should create two different distributions"""
    m = model.Model()
    m.mean = Variable([0, 1000])
    m.sd = Variable(1)
    m.v = Variable(reno.Normal(m.mean, m.sd))

    assert m.v.shape == 2
    ds = m(n=1000, steps=1)
    assert ds.v.values.shape == (1000, 2)
    low_mean = np.mean(ds.v.values[:, 0])
    high_mean = np.mean(ds.v.values[:, 1])

    assert abs(low_mean) < 0.2
    assert abs(high_mean - 1000) < 0.2


# NOTE: I don't think an actual scalar on a computed value is possible, should
# always be a vector
# def test_sum_on_scalar():
#     """Running ops.sum on a scalar (likely from static varible) should give "sum", or the value
#     time the number of timesteps."""
#     v = Variable(Scalar(3))
#     v.value = 3
#     assert (ops.sum(v).eval(3) == 9)


def test_implicit_components_wo_dim():
    """Using extended ops with equations involving multidim shouldn't break (dim
    should somehow extend thru)"""
    m = model.Model()
    m.v0 = Variable(5, dim=3)
    t = TimeRef()
    m.v1 = Variable(m.v0 + t, dim=3)
    m.v2 = Variable(ops.smooth(m.v1, 3))
    m()


def test_implicit_components_wo_dim_pymc():
    """Using extended ops with equations involving multidim shouldn't break (dim
    should somehow extend thru) and the pymc variant should correctly populate
    implicit components even if the normal __call__() hasn't happened yet"""
    m = model.Model()
    m.v0 = Variable(5, dim=3)
    t = TimeRef()
    m.v1 = Variable(m.v0 + t, dim=3)
    m.v2 = Variable(ops.smooth(m.v1, 3))
    m.pymc()


def test_basic_ops_shapes():
    """Operations between components of different shapes should broadcast as expected."""

    v0 = Variable(5, dim=3)
    v1 = Variable(2)
    v2 = Variable(v0 + v1)

    assert (v0 + v1).shape == 3
    assert v2.shape == 3

    assert (v2.eval(0) == [7, 7, 7]).all()


def test_agg_op_shape():
    """An aggregate operation should correctly get the shape down to 1"""
    v0 = Variable(5, dim=3)
    v1 = Variable(v0.sum())

    assert v0.shape == 3
    assert v0.sum().shape == 1
    assert v1.shape == 1
    assert v1.eval(0) == 15


def test_implied_shape_for_scalar():
    """A scalar given a list/array should use that as the shape."""
    assert Scalar([1, 2, 3]).shape == 3

    v = Variable([1, 2, 3])
    assert v.shape == 3


def test_other_agg_op_shapes():
    """The shape after some agg operations should be 1"""
    v0 = Variable([1, 2, 3])
    v1 = Variable(v0.series_min())
    v2 = Variable(v0.series_max())

    assert v1.shape == 1
    assert v1.eval(0) == 1
    assert v2.shape == 1
    assert v2.eval(0) == 3


def test_agg_after_timeseries_shape():
    """Shape of an aggregate on a timeseries should be 1"""
    m = model.Model()
    t = TimeRef()
    m.v0 = Variable(3)
    m.v1 = Variable(m.v0 + t)
    m.post = Metric(m.v1.timeseries.sum())
    assert m.post.eq.shape == 1
    assert m.post.shape == 1
    m(steps=3)
    assert (m.post.value == [[12]]).all()


def test_list_in_op():
    """Using a list distribution variable with math operations should function as expected."""
    m = model.Model()
    t = TimeRef()
    m.v0 = Variable(ops.List([0, 1, 2]))
    m.v1 = Variable(m.v0 + 3 + t)
    ds = m(n=3, steps=1)
    print(ds)
    assert (ds.v1.values == [[3], [4], [5]]).all()


def test_model_w_multidim_list():
    """Initializing values in a model with a list dist with data dim shouldnt' crash."""
    m = model.Model()
    m.v0 = Variable(ops.List([[0, 1], [1, 2], [2, 3]]))
    m(n=3)


def test_multidim_list_in_op():
    """Using a list distribution variable with a datadim with math operations should function
    as expected."""

    m = model.Model()
    t = TimeRef()
    m.v0 = Variable(ops.List([[0, 1], [1, 2], [2, 3]]))
    m.v1 = Variable(m.v0 + 3 + t)

    assert m.v0.shape == 2
    assert m.v1.shape == 2

    ds = m(n=3, steps=1)
    print(m.v0.value)
    print(m.v1.value)
    print(ds)
    assert (ds.v1.values == [[[3, 4]], [[4, 5]], [[5, 6]]]).all()


# TODO: (2025.09.30) tentatively for now I'm okay with this failing - it
# would make it harder to then count free vs independent variables I think
# if you could directly include?
# def test_distribution_inside_op_as_variable_eq():
#     """What happens when you include a distribution as part of an operation directly in
#     equation for a variable?"""
#     m = model.Model()
#     t = TimeRef()
#     m.v0 = Variable(ops.List([[0, 1], [1, 2], [2, 3]]) + t)
#     m()


def test_distribution_with_time_in_eq():
    """What happens when you include a distribution and incorporate a timeref in an
    equation for a variable?"""
    m = model.Model()
    t = TimeRef()
    m.v0 = Variable(ops.List([[0, 1], [1, 2], [2, 3]]))
    m.v1 = Variable(m.v0 - t)
    m(3, 1)


def test_static_astype():
    """astype on a static value should correctly convert the type"""
    m = model.Model()
    m.v0 = reno.Variable(5.5)
    m.v1 = reno.Variable(m.v0.astype(int))
    ds = m()
    assert ds.v1.values[0] == 5


def test_static_astype_pymc():
    """astype on a static value should correctly convert the type in pymc"""
    m = model.Model()
    m.v0 = reno.Variable(5.5)
    m.v1 = reno.Variable(m.v0.astype(int))
    ds = m.pymc(n=1, compute_prior_only=True)
    assert ds.prior.v1.values[0][0] == 5


def test_astype():
    """astype on a non-static value should correctly convert the type"""
    m = model.Model()
    t = TimeRef()
    m.v0 = reno.Variable(5.5 + t)
    m.v1 = reno.Variable(m.v0.astype(int))
    ds = m(steps=3)
    assert (ds.v1.values == [5, 6, 7]).all()


def test_astype_pymc():
    """astype on a static value should correctly convert the type in pymc"""
    m = model.Model()
    t = TimeRef()
    m.v0 = reno.Variable(5.5 + t)
    m.v1 = reno.Variable(m.v0.astype(int))
    ds = m.pymc(n=1, steps=3, compute_prior_only=True)
    assert (ds.prior.v1.values[0] == [5, 6, 7]).all()


def test_stack():
    """Stacking multiple 1dim variables should correctly turn into a multidim value"""
    m = model.Model()
    with m:
        v0, v1, v2 = Variable(1), Variable(2), Variable(3)
        f3 = Flow(ops.stack(v0, v1, v2))

        s0 = Stock()
        f3 >> s0

    assert f3.shape == 3
    ds = m()
    assert (ds.f3.values[0] == [1, 2, 3]).all()
    assert (ds.s0.values[0][2] == [2, 4, 6]).all()


def test_stack_pymc():
    """Stacking multiple 1dim variables should correctly turn into a multidim value in pymc"""
    m = model.Model()
    with m:
        v0, v1, v2 = Variable(1), Variable(2), Variable(3)
        f3 = Flow(ops.stack(v0, v1, v2))

        s0 = Stock()
        f3 >> s0

    assert f3.shape == 3
    ds = m.pymc(1, compute_prior_only=True)
    assert (ds.prior.f3.values[0][0] == [1, 2, 3]).all()
    assert (ds.prior.s0.values[0][0][2] == [2, 4, 6]).all()


def test_timeseries_slice_nonmetric():
    """Using timeseries in a regular component (rather than a metric computed after simulation)
    should still work."""
    m = model.Model()
    t = TimeRef()
    with m:
        v0 = Variable(t + 2)
        v1 = Variable(v0.timeseries[t - 3 : t - 1].sum())
    ds = m()
    assert (ds.v1.values[0][0:4] == [0, 0, 2, 5]).all()


def test_timeseries_slice_nonmetric_pymc():
    """Using timeseries in a regular component (rather than a metric computed after simulation)
    should still work in pymc."""
    m = model.Model()
    t = TimeRef()
    with m:
        v0 = Variable(t + 2)
        v1 = Variable(v0.timeseries[t - 3 : t - 1].sum())
    ds = m.pymc(1, compute_prior_only=True)
    assert (ds.prior.v1.values[0][0][0:4] == [0, 0, 2, 5]).all()


def test_dynamic_timeseries_static_slice():
    """A timeseries on a regular component with static slice endpoints should correctly compute
    as if the underlying component is indeed dynamic."""
    m = model.Model()
    t = TimeRef()
    with m:
        v0 = Variable(2 + t)
        v1 = Variable(v0.timeseries[0:4], dim=4)
        v2 = Variable(v1.sum())

    ds = m()
    assert (ds.v2.values[0][0:4] == [2, 5, 9, 14]).all()


def test_dynamic_timeseries_static_slice_pymc():
    """A timeseries on a regular component with static slice endpoints should correctly compute
    as if the underlying component is indeed dynamic."""
    m = model.Model()
    t = TimeRef()
    with m:
        v0 = Variable(2 + t)
        v1 = Variable(v0.timeseries[0:4], dim=4)
        v2 = Variable(v1.sum())

    ds = m.pymc(1, compute_prior_only=True)
    assert (ds.prior.v2.values[0][0][0:4] == [2, 5, 9, 14]).all()


def test_static_timeseries_static_slice():
    """A timeseries on a static component with static slice endpoints should correctly compute
    as if the underlying component is actually dynamic (filling up zeros over time)."""
    m = model.Model()
    with m:
        v0 = Variable(2)
        v1 = Variable(v0.timeseries[0:4], dim=4)
        v2 = Variable(v1.sum())

    ds = m()
    assert (ds.v2.values[0][0:4] == [2, 4, 6, 8]).all()


def test_static_timeseries_static_slice_pymc():
    """A timeseries on a static component with static slice endpoints should correctly compute
    as if the underlying component is actually dynamic (filling up zeros over time)."""
    m = model.Model()
    t = TimeRef()
    with m:
        v0 = Variable(2)
        v1 = Variable(v0.timeseries[0:4], dim=4)
        v2 = Variable(v1.sum())

    ds = m.pymc(1, compute_prior_only=True)
    assert (ds.prior.v2.values[0][0][0:4] == [2, 4, 6, 8]).all()
