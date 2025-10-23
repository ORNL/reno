import os
import shutil

import pytest

from reno import ops
from reno.components import Flow, Metric, Piecewise, Scalar, Stock, TimeRef, Variable
from reno.model import Model


@pytest.fixture
def data_file_loc():
    filepath = "tests/data"
    # remove the filepath if error occurred on cleanup from previous run
    shutil.rmtree(filepath, ignore_errors=True)

    # setup
    os.makedirs(filepath, exist_ok=True)

    yield filepath

    # teardown
    shutil.rmtree(filepath, ignore_errors=True)


@pytest.fixture
def tub_model():
    t = TimeRef()

    m = Model("tub", steps=50, doc="A simple model of water filling a tub.")
    m.faucet_shutoff_time = Variable(15, doc="Timestep at which to stop filling tub.")
    m.faucet_volume = Variable(5, doc="Rate of flow of the tub faucet.")

    m.water_level = Stock(doc="Amount of water in tub basin.")

    m.faucet = Flow(
        Piecewise(
            [m.faucet_volume, Scalar(0)],
            [t < m.faucet_shutoff_time, t >= m.faucet_shutoff_time],
        )
    )

    m.drain = Flow(ops.sin(t) + 2, min=Scalar(0), max=m.water_level)

    m.water_level += m.faucet
    m.water_level -= m.drain

    m.final_water_level = Metric(ops.index(m.water_level, -1))

    return m


@pytest.fixture
def tub_multimodel(tub_model):
    plumbing = Model("plumbing")
    plumbing.intake = Flow()
    plumbing.water_usage = Stock()

    plumbing.loss_multiplier = Variable(1.2)
    plumbing.loss = Variable(plumbing.loss_multiplier * Scalar(2.0))

    plumbing.water_usage += plumbing.intake
    plumbing.water_usage += plumbing.loss

    whole_system = Model("combined")
    whole_system.tub = tub_model
    whole_system.after_drain = plumbing
    whole_system.after_drain.intake.eq = whole_system.tub.drain

    return whole_system


@pytest.fixture
def multidim_model():
    t = TimeRef()
    m = Model()
    m.v1 = Flow(ops.Categorical([0.25, 0.25, 0.25, 0.25]), dim=4)
    m.v2 = Flow(5)
    m.v3 = Variable(m.v1 + 10 + t, dim=4)
    m.v4 = Variable(ops.Categorical([0.25, 0.25, 0.25, 0.25]), dim=6)

    m.s = Stock(dim=4)
    m.s += m.v1
    m.s += m.v2

    return m


@pytest.fixture
def multidim_model_determ():
    t = TimeRef()
    m = Model()
    m.v1 = Flow([0.25, 0.35, 0.45, 0.55], dim=4)
    m.v2 = Flow(5)
    m.v3 = Variable(m.v1 + 10 + t, dim=4)

    m.s = Stock(dim=4)
    m.s += m.v1
    m.s += m.v2

    return m


@pytest.fixture
def multidim_model_determ_implicit():
    t = TimeRef()
    m = Model()
    m.v1 = Flow([0.25, 0.35, 0.45, 0.55])
    m.v2 = Flow(5)
    m.v3 = Variable(m.v1 + 10 + t)

    m.s = Stock()
    m.s += m.v1
    m.s += m.v2

    return m
