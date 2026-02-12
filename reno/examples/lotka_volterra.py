"""Predator/prey dynamics, defined by the lotka volterra equations

See the predator_prey_model notebook"""

import reno as r

predator_prey = r.Model(
    name="predator_prey",
    steps=200,
    doc="Classic predator-prey interaction model example",
)

with predator_prey:
    # make stocks to monitor the predator/prey populations over time
    rabbits = r.Stock(init=100.0)
    foxes = r.Stock(init=100.0)

    # free variables that can quickly be changed to influence equilibrium
    rabbit_growth_rate = r.Variable(0.1, doc="Alpha")
    rabbit_death_rate = r.Variable(0.001, doc="Beta")
    fox_death_rate = r.Variable(0.1, doc="Gamma")
    fox_growth_rate = r.Variable(0.001, doc="Delta")

    # flows that define how much the stocks change in a timestep
    rabbit_births = r.Flow(rabbit_growth_rate * rabbits)
    rabbit_deaths = r.Flow(rabbit_death_rate * rabbits * foxes, max=rabbits)
    fox_deaths = r.Flow(fox_death_rate * foxes, max=foxes)
    fox_births = r.Flow(fox_growth_rate * rabbits * foxes)

    # hook up inflows/outflows for stocks
    rabbit_births >> rabbits >> rabbit_deaths
    fox_births >> foxes >> fox_deaths

    # define some metrics for observation
    minimum_foxes = r.Metric(foxes.timeseries.series_min())
    maximum_foxes = r.Metric(foxes.timeseries.series_max())
