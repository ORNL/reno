"""Tests for basic viz"""

import reno as r


def test_plot_trace_refs_on_compute_prior_only(tub_model):
    ds = tub_model.pymc(n=1000, compute_prior_only=True)
    r.plot_trace_refs(tub_model, [ds.prior], ["faucet_off_time", "faucet", "drain", tub_model.water_level])


def test_plot_trace_refs_on_full_posterior(tub_model):
    ds = tub_model.pymc(n=1000, faucet_shutoff_time=r.Normal(10, 5), observations=[r.Observation(tub_model.final_water_level, 5.0, [12.0])])
    r.plot_trace_refs(tub_model, {"prior": ds.prior, "post": ds.posterior}, ["faucet_off_time", "faucet", "drain", "water_level"])
