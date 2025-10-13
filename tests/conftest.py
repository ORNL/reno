import os
import shutil

import pytest

from reno import ops
from reno.components import (
    Flow,
    Piecewise,
    PostMeasurement,
    Scalar,
    Stock,
    TimeRef,
    Variable,
)
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

    m.final_water_level = PostMeasurement(ops.index(m.water_level, -1))

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
