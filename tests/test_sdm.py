"""Tests for system dynamics functionality."""

import os

import numpy as np
import pytest

from reno.components import Flow, Piecewise, Scalar, Stock, TimeRef, Variable
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
    assert isinstance(tub_multimodel.after_drain.loss_multiplier.value, np.ndarray)
    assert isinstance(tub_multimodel.after_drain.loss.value, np.ndarray)
    run2 = tub_multimodel(n=10, faucet_volume=6)
    assert isinstance(tub_multimodel.after_drain.loss_multiplier.value, np.ndarray)
    assert isinstance(tub_multimodel.after_drain.loss.value, np.ndarray)

    for variable in run1.keys():
        assert (run1[variable].values == run2[variable].values).all()


def test_consecutive_scalar_collapse():
    """For unknown reasons, an evaluated scalar variable after a run seems to return a
    raw float instead of a numpy array with a single value."""
    m = Model()
    m.v = Variable(1.0)

    m()
    assert isinstance(m.v.value, np.ndarray)
    m()
    assert isinstance(m.v.value, np.ndarray)


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
