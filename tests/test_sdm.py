"""Tests for system dynamics functionality."""

import os

import pytest

import reno
from reno.components import Flow, Metric, Piecewise, Scalar, Stock, TimeRef, Variable
from reno.model import Model
from reno.pymc import find_historical_tracked_refs


def test_tub_against_insight_maker():
    """The stock/flow outputs for a system where flows reference stocks etc.
    should match what insight maker outputs, to verify that our n+1, n, n-1
    functionality for various eval'd objects is working as expected."""

    t = TimeRef()

    m = Model("tub", steps=7)
    m.water = Stock()
    m.faucet = Flow(
        Piecewise([m.water + 5, Scalar(0)], [t <= Scalar(4), t > Scalar(4)])
    )
    m.drain = Flow(Scalar(2))

    m.water += m.faucet
    m.water -= m.drain

    ds = m()
    assert ds.water.values.tolist()[0] == [0.0, 3.0, 9.0, 21.0, 45.0, 93.0, 91.0]
    assert ds.faucet.values.tolist()[0] == [5.0, 8.0, 14.0, 26.0, 50.0, 0.0, 0.0]


def test_reassigning_eq_to_ref_warns():
    """Very easy to accidentally define a Flow and then assign its equation to the
    reference itself rather than the ref.eq, so make sure this displays a warning."""

    m = Model("tub", steps=7)
    m.water = Stock()
    m.faucet = Flow()
    m.drain = Flow()

    with pytest.warns(RuntimeWarning):
        m.drain = Scalar(2)


def test_multimodel():
    # TODO: see notebook 11
    pass


def test_copy_model_works(tub_model):
    """Getting a deepcopy of a model shouldn't fail."""
    tub_model.copy("new_name")


def test_historical_tracked_refs():
    """A basic model with several index equations for historical values should
    correctly construct a taps dictionary for the tracked references."""
    t = TimeRef()

    m = Model()
    m.v0 = Variable(Scalar(1))
    m.v1 = Variable(t)
    m.v2 = Variable(m.v1.history(t - 1))
    m.v3 = Variable(m.v1.history(t - 2))
    m.v4 = Variable(m.v2.history(t - m.v0))

    assert find_historical_tracked_refs(m) == {m.v1: [-2, -1], m.v2: [-1]}


def test_multiple_consecutive_runs(tub_model):
    """Running a model multiple times should work the same each time/each
    consecutive run starting from scratch"""
    run1 = tub_model(n=10, faucet_volume=6)
    run2 = tub_model(n=10, faucet_volume=6)

    for variable in run1.keys():
        assert (run1[variable].values == run2[variable].values).all()


def test_multiple_multimodel_consecutive_runs(tub_multimodel):
    """Running a multimodel multiple times should work the same each time/each
    consecutive run starting from scratch"""
    run1 = tub_multimodel(n=10, faucet_volume=6)
    # NOTE: (2025-09-22) Taking these assertions out at least for now because tentatively
    # allowing variable values to be the raw values themselves if no sample
    # dimension required.
    # assert isinstance(tub_multimodel.after_drain.loss_multiplier.value, np.ndarray)
    # assert isinstance(tub_multimodel.after_drain.loss.value, np.ndarray)
    run2 = tub_multimodel(n=10, faucet_volume=6)
    # assert isinstance(tub_multimodel.after_drain.loss_multiplier.value, np.ndarray)
    # assert isinstance(tub_multimodel.after_drain.loss.value, np.ndarray)

    for variable in run1.keys():
        assert (run1[variable].values == run2[variable].values).all()


def test_multimodel_dataset_uses_submodel_prefixes(tub_multimodel):
    """The keys in the dataset from a multimodel run should prefix submodel names
    for each submodel reference."""
    run = tub_multimodel()
    assert "after_drain_intake" in run.keys()
    assert "tub_water_level" in run.keys()


# NOTE: (2025-09-20) Taking this test out at least for now because tentatively
# allowing variable values to be the raw values themselves if no sample
# dimension required.
# def test_consecutive_scalar_collapse():
#     """For unknown reasons, an evaluated scalar variable after a run seems to return a
#     raw float instead of a numpy array with a single value."""
#     m = Model()
#     m.v = Variable(1.0)
#
#     m()
#     assert isinstance(m.v.value, np.ndarray)
#     m()
#     assert isinstance(m.v.value, np.ndarray)


def test_subsequent_pymc_prior_calls(tub_model):
    """Running the same pymc model multiple times should have the same results."""
    run1 = tub_model.pymc(10, compute_prior_only=True)
    run2 = tub_model.pymc(10, compute_prior_only=True)

    for variable in run1.prior.keys():
        assert (run1.prior[variable].values == run2.prior[variable].values).all()


def test_symmetric_config(tub_model):
    """Restoring a previous configuration should _exactly_ (symmetrically) restore
    values. This is most obvious in what gets passed to pymc models."""
    previous = tub_model.config()
    assert tub_model.water_level.init is None
    tub_model.config(**previous)
    assert tub_model.water_level.init is None


def test_symmetric_config_pymc_str(tub_model):
    """Restoring a previous configuration should _exactly_ (symmetrically) restore
    values. This is most obvious in what gets passed to pymc models."""
    previous = tub_model.config()
    str1 = tub_model.pymc_str()
    tub_model.config(**previous)
    str2 = tub_model.pymc_str()

    assert str1 == str2


def test_to_from_dict_consistency(tub_multimodel):
    """The serialized from of a deserialized serialized model should
    be the same as the original serialization."""
    serialized1 = tub_multimodel.to_dict()

    new_model = Model.from_dict(serialized1)
    serialized2 = new_model.to_dict()

    assert serialized1 == serialized2


def test_from_dict_result_consistency(tub_multimodel):
    """Loading a previously saved model should have the same results when run."""

    serialized = tub_multimodel.to_dict()
    new_model = Model.from_dict(serialized)

    run1 = tub_multimodel(n=10, faucet_volume=6)
    run2 = new_model(n=10, faucet_volume=6)

    for variable in run1.keys():
        assert (run1[variable].values == run2[variable].values).all()


def test_save_load(data_file_loc, tub_multimodel):
    """Saving to and loading from a file should have the same results when run."""
    run1 = tub_multimodel(n=10, faucet_volume=6)
    filename = f"{data_file_loc}/testmodel.json"

    tub_multimodel.save(filename)
    assert os.path.exists(filename)

    new_model = Model.load(filename)
    run2 = new_model(n=10, faucet_volume=6)

    for variable in run1.keys():
        assert (run1[variable].values == run2[variable].values).all()


def test_multidim_model_runs(multidim_model):
    """A model with stocks and flows with a vector dimension should run without errors
    and all outputs in dataset should have correct shape."""

    ds = multidim_model()
    assert len(ds.coords) == 6
    assert "s_vec" in ds.coords
    assert "v1_vec" in ds.coords
    assert "v3_vec" in ds.coords
    assert "v4_vec" in ds.coords

    assert ds.s.shape == (1, 10, 4)
    assert ds.v1.shape == (1, 4)
    assert ds.v2.shape == (1,)
    assert ds.v3.shape == (1, 10, 4)
    assert ds.v4.shape == (1, 6)


def test_multidim_model_w_vectors_runs(multidim_model_determ):
    """A multidim model with a static vector should still run correctly and have correct dims in the dataset output."""
    ds = multidim_model_determ()
    assert (ds.v1.values == [[0.25, 0.35, 0.45, 0.55]]).all()
    assert ds.v1.shape == (1, 4)
    assert ds.v2.shape == (1,)
    assert ds.v3.shape == (1, 10, 4)
    assert ds.s.shape == (1, 10, 4)


def test_model_context_manager_finds_name():
    """Creating references within a model's context manager should appropriately find the variable names and add them
    to the model"""

    m = Model()
    with m:
        var1 = Variable(2)
        var2 = Variable(5)
        combination = Variable(var1 + var2)
        some_metric = Metric(var1.timeseries[-1])

    assert var1.name == "var1"
    assert var2.name == "var2"
    assert combination.name == "combination"
    assert some_metric.name == "some_metric"
    assert var1.model == m
    assert var2.model == m
    assert combination.model == m
    assert some_metric.model == m

    assert m.var1 == var1
    assert m.var2 == var2
    assert m.combination == combination
    assert m.some_metric == some_metric

    ds = m(1, 1)
    assert ds.combination.values[0] == 7


def test_model_context_manager_ignores_preexisting():
    """Creating references within a model's context manager should appropriately find the variable names and add them
    to the model unless it's explicitly added to the model already"""

    m = Model()
    with m:
        var1 = Variable(2)
        m.var2 = Variable(5)
        combination = Variable(var1 + m.var2)

    assert var1.name == "var1"
    assert m.var2.name == "var2"
    assert combination.name == "combination"
    assert var1.model == m
    assert m.var2.model == m
    assert combination.model == m

    assert m.var1 == var1
    assert m.combination == combination

    ds = m(1, 1)
    assert ds.combination.values[0] == 7


def test_model_context_manager_handles_submodels():
    """Creating models within a model's context manager should appropriately find the model name and add to parent
    model."""

    m1 = Model()
    with m1:
        var1 = Variable(2)
        m2 = Model()
        with m2:
            var2 = Variable(5)
        m1.combination = Variable(var1 + m2.var2)

    assert m2.parent == m1
    assert m1.m2 == m2
    assert m1.m2.var2 == var2
    assert m2 in m1.models
    assert m2.name == "m2"

    ds = m1(1, 1)
    assert ds.combination.values[0] == 7


def test_bottleneck_example():
    """A model using the .outflows or .space of a stock to help limit should correctly bottleneck upstream."""
    m = Model()
    with m:
        f_in, f_between, f_out = Flow(), Flow(), Flow(1)
        s0, s1, sfinal = Stock(max=2), Stock(max=3), Stock()

        f_in.eq = reno.minimum(4, s0.space)
        f_between.eq = reno.minimum(2, s1.space)

        s0 += f_in
        s0 -= f_between
        s1 += f_between
        s1 -= f_out
        sfinal += f_out

    ds = m()
    assert (ds.f_in[0][0:4] == [4.0, 2.0, 2.0, 1.0]).all()
    assert (ds.f_between[0][0:4] == [2.0, 2.0, 2.0, 1.0]).all()


def test_bottleneck_example_pymc():
    """A pymc model using the .outflows or .space of a stock to help limit should correctly bottleneck upstream."""
    m = Model()
    with m:
        f_in, f_between, f_out = Flow(), Flow(), Flow(1)
        s0, s1, sfinal = Stock(max=2), Stock(max=3), Stock()

        f_in.eq = reno.minimum(4, s0.space)
        f_between.eq = reno.minimum(2, s1.space)

        s0 += f_in
        s0 -= f_between
        s1 += f_between
        s1 -= f_out
        sfinal += f_out

    ds1 = m()
    ds2 = m.pymc(n=1, compute_prior_only=True)
    assert (ds1.f_in.values == ds2.prior.f_in.sel(chain=0).values).all()
    assert (ds1.f_between.values == ds2.prior.f_between.sel(chain=0).values).all()
    assert (ds1.s0.values == ds2.prior.s0.sel(chain=0).values).all()
    assert (ds1.s1.values == ds2.prior.s1.sel(chain=0).values).all()
