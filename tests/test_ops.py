"""Tests for the math operations."""

import numpy as np

from reno import model, ops
from reno.components import Piecewise, PostMeasurement, Scalar, TimeRef, Variable


def test_sum_on_matrix():
    """Running ops.sum on a matrix should give you a row-wise sum."""
    v = Variable()
    v.value = np.array([[0, 1, 2], [1, 2, 3]])

    assert (ops.sum(v).eval(3) == np.array([[3, 6]])).all()


def test_sum_on_vector():
    """Running ops.sum on a vector (static variable) should give you the row-wise
    "sum" which is just the value times the number of timesteps."""
    v = Variable()
    v.value = np.array([2, 3])

    assert (ops.sum(v).eval(3) == np.array([[8, 12]])).all()


def test_sum_on_matrix_start_stop():
    """Running ops.sum on a matrix should give you a row-wise sum, correctly
    accounting for specified range."""
    v = Variable()
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert (v[2:].sum().eval(5) == np.array([[9, 12]])).all()
    assert (v[:2].sum().eval(5) == np.array([[1, 3]])).all()
    assert (v[1:3].sum().eval(5) == np.array([[3, 5]])).all()


def test_sum_on_vector_start_stop():
    """Running ops.sum on a vector (static variable) should give you the row-wise
    "sum" which is just the value times the number of timesteps. Correctly
    accounting for specified range."""
    v = Variable()
    v.value = np.array([2, 3])

    assert (v[2:].sum().eval(4) == np.array([[6, 9]])).all()
    assert (v[:2].sum().eval(4) == np.array([[4, 6]])).all()
    assert (v[1:3].sum().eval(4) == np.array([[4, 6]])).all()


def test_static_value_sum():
    """By default, the sum of a static value will just be itself, unless you access
    it as a slice.
    NOTE: changed this behavior by having sum automatically insert slice if static detected.
    """
    v = Variable(Scalar(5))

    assert v.sum().eval(4) == np.array([25])
    assert v[:].sum().eval(4) == np.array([25])


def test_series_max():
    """Both API forms should return the correct series max."""
    v = Variable()
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert (v.series_max().eval(4) == ops.series_max(v).eval(4)).all()


def test_slice_on_matrix():
    """Slice bounds should work properly on a matrix."""
    v = Variable()
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert (v[:2].eval(5) == np.array([[0, 1], [1, 2]])).all()
    assert v[:2].eval(5).shape == np.array([[0, 1], [1, 2]]).shape
    assert (v[0:2].eval(5) == np.array([[0, 1], [1, 2]])).all()
    assert v[0:2].eval(5).shape == np.array([[0, 1], [1, 2]]).shape
    assert (v[3:].eval(5) == np.array([[3, 4], [4, 5]])).all()
    assert v[3:].eval(5).shape == np.array([[3, 4], [4, 5]]).shape


def test_slice_on_vector():
    """Slice bounds should work properly on a matrix."""
    v = Variable()
    v.value = np.array([2, 3])

    assert (v[:2].eval(3) == np.array([[2, 2], [3, 3]])).all()
    assert v[:2].eval(3).shape == np.array([[2, 2], [3, 3]]).shape
    assert (v[0:2].eval(3) == np.array([[2, 2], [3, 3]])).all()
    assert v[0:2].eval(3).shape == np.array([[2, 2], [3, 3]]).shape
    assert (v[3:].eval(5) == np.array([[2, 2, 2], [3, 3, 3]])).all()
    assert v[3:].eval(5).shape == np.array([[2, 2, 2], [3, 3, 3]]).shape


def test_slice_on_scalar():
    """Slicing a scalar should result in a 2d array. (Otherwise
    series ops won't work correctly on the expanded slice.)"""
    v = Variable(Scalar(5))
    assert (v[:].eval(4) == np.array([[5, 5, 5, 5, 5]])).all()
    assert v[:].eval(4).shape == np.array([[5, 5, 5, 5, 5]]).shape


def test_slice_staticness_static_slice():
    """A slice with defined bounds (or static bounds) should be considered static."""
    v = Variable(None)
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert v[1:3].is_static()

    a = Scalar(1)
    b = Scalar(3)

    assert v[a:b].is_static()

    av = Variable(a)
    bv = Variable(b)

    assert v[av:bv].is_static()


def test_slice_staticness_none_endpoint():
    """A slice should not be considered static if it's endpoint is None or t."""
    v = Variable(None)
    v.value = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    assert not v[:].is_static()
    assert not v[1:].is_static()
    t = TimeRef()
    assert not v[:t].is_static()


def test_sum_of_dist():
    """Sanity check that running a sum on an array from a distribution
    is correctly row-wise and doesn't collapse to single summed value."""
    b = ops.Bernoulli(0.5)
    b.populate(10)
    b.sum().eval(5).shape == (10,)


def test_slice_end_t_is_correct_simpler():
    """A slice stop of none should actually technically be t+1 to be inclusive of current timestep? (This test is
    proof of why, getting index 0 shouldn't be nothing)"""
    v = Variable(Scalar(np.array([2, 3])))
    assert (v[:].eval(0) == np.array([[2], [3]])).all()
    assert v[:].eval(0).shape == np.array([[2], [3]]).shape


def test_slice_end_t_is_correct():
    """A slice stop of none should actually technically be t+1 to be inclusive of current timestep?"""
    t = TimeRef()
    m = model.Model()
    m.v0 = Variable(t + 1)
    m.v1 = Variable(Scalar(5))
    m.m0 = PostMeasurement(m.v0[m.v1 :].sum())
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
    ds = m(n=5, steps=1)
    np.testing.assert_almost_equal(ds.v1.values, np.array([3.0, 3.0, 2.5, 0.56, 0.0]))


def test_interpolation_pymc():
    """Interpolation should work in pymc math."""
    m = model.Model()
    m.v0 = Variable(Scalar([0, 1, 1.5, 2.72, 3.14]))
    m.v1 = Variable(ops.interpolate(m.v0, [1, 2, 3], [3, 2, 0]))
    ds = m.pymc(steps=1, compute_prior_only=True)
    np.testing.assert_almost_equal(
        ds.prior.v1.values[0][0], np.array([3.0, 3.0, 2.5, 0.56, 0.0])
    )


# TODO: add tests for slices and pymc...


# NOTE: I don't think an actual scalar on a computed value is possible, should
# always be a vector
# def test_sum_on_scalar():
#     """Running ops.sum on a scalar (likely from static varible) should give "sum", or the value
#     time the number of timesteps."""
#     v = Variable(Scalar(3))
#     v.value = 3
#     assert (ops.sum(v).eval(3) == 9)
