"""A very basic tub model, tracking the water level over time as the faucet and drain vary.

See the tub notebook for additional context/instruction."""

import reno as r

t = r.TimeRef()
tub = r.Model(
    "Tub",
    doc="Model the amount of water in a bathtub based on a drain and faucet rate.",
)
with tub:
    faucet, drain = r.Flow(), r.Flow(dtype=float)
    water_level = r.Stock()

    faucet_off_time = r.Variable(
        r.DiscreteUniform(2, 6),
        doc="Which timestep to turn the faucet off in the simulation.",
    )

    faucet >> water_level >> drain

    faucet.eq = r.Piecewise([5, 0], [t < faucet_off_time, t >= faucet_off_time])
    drain.eq = r.sin(t) + 2
    drain.min = 0
    drain.max = water_level

    final_water_level = r.Metric(water_level.timeseries[-1])
