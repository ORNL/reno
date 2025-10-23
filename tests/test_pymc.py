"""Tests for bayesian analysis/SDM -> PYMC compiling/generation"""

# these are needed for pymc_str exec to work
import numpy as np  # noqa: F401
import pymc as pm
import pytensor  # noqa: F401
import pytensor.tensor as pt  # noqa: F401
import pytest  # noqa: F401
from pytensor.ifelse import ifelse  # noqa: F401

from reno import ops
from reno.components import Flow, Metric, Piecewise, Scalar, Stock, TimeRef, Variable
from reno.model import Model


def test_tub_against_base_reno():
    """pymc_model prior samples with basic deterministic tub example should be
    equivalent to the reno version (which in test_sdm shows is equivalent to
    insightmaker)"""

    t = TimeRef()

    m = Model("tub", steps=7)
    m.water = Stock()
    m.faucet = Flow(
        Piecewise([m.water + 5, Scalar(0.0)], [t <= Scalar(4), t > Scalar(4)])
    )
    m.drain = Flow(Scalar(2))

    m.water += m.faucet
    m.water -= m.drain

    with m.pymc_model():  # as pymc_m:
        idata = pm.sample_prior_predictive(1)

    ds = m()
    assert ds.water.values.tolist()[0] == [0.0, 3.0, 9.0, 21.0, 45.0, 93.0, 91.0]
    assert ds.faucet.values.tolist()[0] == [5.0, 8.0, 14.0, 26.0, 50.0, 0.0, 0.0]

    assert (
        idata.prior.water.sel(chain=0).values.tolist()[0] == ds.water.values.tolist()[0]
    )
    assert (
        idata.prior.faucet.sel(chain=0).values.tolist()[0]
        == ds.faucet.values.tolist()[0]
    )


def test_multidim_model_against_base_reno(multidim_model_determ):
    """A model with stuff with vector dimensions should be equivalent to the reno version."""

    ds1 = multidim_model_determ()
    ds2 = multidim_model_determ.pymc(1)

    assert (ds1.s.values == ds2.prior.s.sel(chain=0).values).all()
    assert (ds1.v1.values == ds2.prior.v1.sel(chain=0).values).all()
    assert (ds1.v2.values == ds2.prior.v2.sel(chain=0).values).all()
    assert (ds1.v3.values == ds2.prior.v3.sel(chain=0).values).all()


def test_pymc_str_equiv_to_pymc_model():
    """Evaling and running pymc_str code should be equivalent to pymc_model"""

    t = TimeRef()

    m = Model("tub", steps=7)
    m.water = Stock()
    m.faucet = Flow(
        Piecewise([m.water + 5, Scalar(0.0)], [t <= Scalar(4), t > Scalar(4)])
    )
    m.drain = Flow(Scalar(2))

    m.water += m.faucet
    m.water -= m.drain

    vars = globals()
    exec(m.pymc_str(), vars)
    with vars["tub_pymc_m"]:
        idata1 = pm.sample_prior_predictive(1)

    with m.pymc_model():
        idata2 = pm.sample_prior_predictive(1)

    assert (
        idata1.prior.water.sel(chain=0).values == idata2.prior.water.sel(chain=0).values
    ).all()
    assert (
        idata1.prior.faucet.sel(chain=0).values
        == idata2.prior.faucet.sel(chain=0).values
    ).all()


def test_bernoulli_udpates():
    """From notebook 28, confirming a boolean based RV can correctly update based on downstream
    likelihoods."""

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

    with m.pymc_model() as m1:
        pm.Normal("likelihood", m1["s"][-1], 100, observed=[900])
        prior = pm.sample_prior_predictive(100)
        posterior = pm.sample(100)

    assert (
        prior.prior.decision.values.sum() >= 20
        and prior.prior.decision.values.sum() <= 80
    )
    assert (posterior.posterior.decision.values == 1).all()


def test_bernoulli_updates_with_obs_op():
    """Similar to test_bernoulli_updates, but directly passing observations to pymc_model."""

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

    m.final_s = Metric(ops.index(m.s, Scalar(-1)))

    with m.pymc_model([ops.Observation(m.final_s, 100, [900])]):
        posterior = pm.sample(100)
    assert (posterior.posterior.decision.values == 1).all()


def test_bernoulli_updates_faster_compile():
    """Similar to test_bernoulli_updates_with_obs_op, but ensuring 'lesser compile optimization'
    doesn't break."""

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

    m.final_s = Metric(ops.index(m.s, Scalar(-1)))

    posterior = m.pymc(
        100, compile_faster=True, observations=[ops.Observation(m.final_s, 100, [900])]
    )
    assert (posterior.posterior.decision.values == 1).all()


# TODO: all the testing models from notebook 23 to ensure ordering


def test_pymc_multi_submodels():
    """Evaling and prior sampling from a model with several submodels shouldn't fail."""
    a = Model("a")
    a.s1 = Stock()
    a.f1, a.f2 = Flow(Scalar(3)), Flow(Scalar(2))
    a.s1 += a.f1
    a.s1 -= a.f2
    a.s2 = Stock()

    b = Model("b")
    a.b = b
    b.s1_b = Stock()
    b.f1_b, b.f2_b = Flow(), Flow(Scalar(1))
    b.f1_b.eq = a.f2
    b.s1_b += b.f1_b
    b.s1_b -= b.f2_b

    c = Model("c")
    a.c = c
    c.s1_c = Stock()
    c.f1_c, c.f2_c = Flow(), Flow(Scalar(1))
    c.f1_c.eq = a.f2
    c.s1_c += c.f1_c
    c.s1_c -= c.f2_c

    a.s2 += b.f2_b
    a.s2 += c.f2_c

    vars = globals()
    exec(a.pymc_str(), vars)
    with vars["a_pymc_m"]:
        idata1 = pm.sample_prior_predictive(1)

    with a.pymc_model():
        idata2 = pm.sample_prior_predictive(1)

    assert (
        idata1.prior.s2.sel(chain=0).values == idata2.prior.s2.sel(chain=0).values
    ).all()


def test_pymc_str_historical_refs():
    """PyMC string code for a model referencing historical values should
    correctly run."""

    t = TimeRef()

    m = Model()
    m.v0 = Variable(Scalar(1))
    m.v1 = Variable(t)
    m.v2 = Variable(m.v1.history(t - 1))
    m.v3 = Variable(m.v1.history(t - 2))
    m.v4 = Variable(m.v2.history(t - m.v0))

    ds = m()

    vars = globals()
    exec(m.pymc_str(), vars)
    with vars["pymc_m"]:
        idata = pm.sample_prior_predictive(1)

    assert idata.prior.v0.sel(chain=0).values.tolist() == ds.v0.values.tolist()
    assert idata.prior.v1.sel(chain=0).values.tolist() == ds.v1.values.tolist()
    assert idata.prior.v2.sel(chain=0).values.tolist() == ds.v2.values.tolist()
    assert idata.prior.v3.sel(chain=0).values.tolist() == ds.v3.values.tolist()
    assert idata.prior.v4.sel(chain=0).values.tolist() == ds.v4.values.tolist()


def test_pymc_model_historical_refs():
    """The pymc converted model should run equivalently to the pymc str code. (and
    by transitive property should be equivalent to the normal reno model)"""

    t = TimeRef()

    m = Model()
    m.v0 = Variable(Scalar(1))
    m.v1 = Variable(t)
    m.v2 = Variable(m.v1.history(t - 1))
    m.v3 = Variable(m.v1.history(t - 2))
    m.v4 = Variable(m.v2.history(t - m.v0))

    vars = globals()
    exec(m.pymc_str(), vars)
    with vars["pymc_m"]:
        idata1 = pm.sample_prior_predictive(1)

    with m.pymc_model():
        idata2 = pm.sample_prior_predictive(1)

    assert (
        idata1.prior.v0.sel(chain=0).values == idata2.prior.v0.sel(chain=0).values
    ).all()
    assert (
        idata1.prior.v1.sel(chain=0).values == idata2.prior.v1.sel(chain=0).values
    ).all()
    assert (
        idata1.prior.v2.sel(chain=0).values == idata2.prior.v2.sel(chain=0).values
    ).all()
    assert (
        idata1.prior.v3.sel(chain=0).values == idata2.prior.v3.sel(chain=0).values
    ).all()
    assert (
        idata1.prior.v4.sel(chain=0).values == idata2.prior.v4.sel(chain=0).values
    ).all()


def test_pymc_multi_submodels_with_metrics(tub_multimodel):
    """Make sure submodel metrics correctly get handled in pymc."""
    run = tub_multimodel.pymc(n=1, compute_prior_only=True)
    assert "tub_final_water_level" in run.prior.keys()


def test_pymc_slice_in_metric():
    """Postmeasurements using slices should work the same in pymc as in reno math."""
    t = TimeRef()
    m = Model()
    m.v0 = Variable(t + 1)
    m.m0 = Metric(m.v0[-4:-1].sum())

    run_reno = m()
    run_pymc = m.pymc()
    assert (run_pymc.prior.v0.sel(chain=0).values == run_reno.v0.values).all()
    assert (run_pymc.prior.m0.sel(chain=0).values == run_reno.m0.values).all()


def test_pymc_slice_in_metric_no_upper_bound():
    """Postmeasurements using slices should work the same in pymc as in reno math."""
    t = TimeRef()
    m = Model()
    m.v0 = Variable(t + 1)
    m.m0 = Metric(m.v0[-4:].sum())

    run_reno = m()
    run_pymc = m.pymc()
    assert (run_pymc.prior.v0.sel(chain=0).values == run_reno.v0.values).all()
    assert (run_pymc.prior.m0.sel(chain=0).values == run_reno.m0.values).all()


def test_pymc_slice_with_vars_in_metric():
    """Postmeasurements using slices should work the same in pymc as in reno math."""
    t = TimeRef()
    m = Model()
    m.v0 = Variable(t + 1)
    m.v1 = Variable(Scalar(5))
    m.m0 = Metric(m.v0.timeseries[m.v1 :].sum())

    run_reno = m()
    run_pymc = m.pymc()
    assert (run_pymc.prior.v0.sel(chain=0).values == run_reno.v0.values).all()
    assert (run_pymc.prior.m0.sel(chain=0).values == run_reno.m0.values).all()


def test_pymc_historical_of_multidim():
    """The init should correctly initialize multidim zero vectors for a historical val
    of a multidim variable"""
    t = TimeRef()
    m = Model()
    m.v0 = Variable([2, 3])
    m.v1 = Variable(t * m.v0)
    m.v2 = Variable(m.v1.history(t - 1))
    m.s = Stock()
    m.s += m.v2

    assert m.v0.shape == 2
    assert m.v1.shape == 2
    assert m.v2.shape == 2
    assert m.s.shape == 2

    run = m.pymc(compute_prior_only=True)
    assert (run.prior.s.values[0][0][-1] == [56, 84]).all()


def test_pymc_historical_of_multidim_w_taps():
    """The init should correctly initialize multidim zero vectors for a historical val
    of a multidim variable. (This one actually uses taps)"""
    t = TimeRef()
    m = Model()
    m.v0 = Variable([2, 3])
    m.v1 = Variable(t * m.v0)
    m.v2 = Variable(m.v1.history(t - 2))
    m.s = Stock()
    m.s += m.v2

    assert m.v0.shape == 2
    assert m.v1.shape == 2
    assert m.v2.shape == 2
    assert m.s.shape == 2

    run = m.pymc(compute_prior_only=True)
    assert (run.prior.s.values[0][0][-1] == [42, 63]).all()
