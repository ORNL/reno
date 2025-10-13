"""Outputs from previous runs should probably still continue to be correct."""

import json

import xarray as xr

from reno import ops
from reno.components import Flow, Scalar, Stock, Variable
from reno.model import Model


def test_bottleneck_model():
    """This model was used to help understand how min/max's interact with throughputs,
    e.g. how do you correctly limit flow mins/maxes to ensure the bottleneck is retained,
    but you don't end up with a really jagged/timestep-oscilating output?

    The xarray datasets used for this are generated from notebook 14_bottleneck_example
    """

    m = Model()
    m.s1 = Stock(doc="Input material stockpile")
    m.s2 = Stock(doc="Internal processing facility temp stock")
    m.s3 = Stock(doc="Output material stockpile")
    m.f1 = Flow()
    m.f2 = Flow()
    m.f3 = Flow(doc="bottleneck that needs a lower throughput than this one")

    m.s2_max = Variable()
    m.process_throughput = Variable()
    m.s3_max = Variable()

    m.s1 -= m.f1
    m.s2 += m.f1
    m.s2 -= m.f2
    m.s3 += m.f2
    m.s3 -= m.f3

    m.f1.eq = m.process_throughput
    m.f1.max = ops.minimum(m.s1, m.s2_max - m.s2)
    m.f1.min = Scalar(0)

    m.f2.eq = m.process_throughput
    m.f2.max = ops.minimum(m.s2, m.s3_max - m.s3)
    m.f2.min = Scalar(0)

    m.f3.eq = m.process_throughput / 2
    m.f3.max = m.s3
    m.f3.min = Scalar(0)

    ds1 = m(s1_0=100, s2_max=10, process_throughput=10, s3_max=30)
    ds2 = m(s1_0=100, s2_max=20, process_throughput=10, s3_max=30)
    ds3 = m(s1_0=100, s2_max=30, process_throughput=10, s3_max=30)
    ds4 = m(s1_0=100, s2_max=15, process_throughput=10, s3_max=30)

    # ds1 = json.dumps(ds1.to_dict(), indent=4, default=lambda x: str(x))
    # ds2 = json.dumps(ds2.to_dict(), indent=4, default=lambda x: str(x))
    # ds3 = json.dumps(ds3.to_dict(), indent=4, default=lambda x: str(x))
    # ds4 = json.dumps(ds4.to_dict(), indent=4, default=lambda x: str(x))
    # ds1 = ds1.to_dict()
    # ds2 = ds2.to_dict()
    # ds3 = ds3.to_dict()
    # ds4 = ds4.to_dict()

    with open("tests/regression_data/ds1.json") as infile:
        ds1_prev = xr.Dataset.from_dict(json.load(infile))
    with open("tests/regression_data/ds2.json") as infile:
        ds2_prev = xr.Dataset.from_dict(json.load(infile))
    with open("tests/regression_data/ds3.json") as infile:
        ds3_prev = xr.Dataset.from_dict(json.load(infile))
    with open("tests/regression_data/ds4.json") as infile:
        ds4_prev = xr.Dataset.from_dict(json.load(infile))

    assert ds1.equals(ds1_prev)
    # assert ds1 == ds1_prev
    assert ds2 == ds2_prev
    assert ds3 == ds3_prev
    assert ds4 == ds4_prev
