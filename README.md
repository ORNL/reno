# Reno

Reno is a tool for creating, visualizing, and analyzing system dynamics
models in Python. It additionally has the ability to convert models to PyMC,
allowing Bayesian inference on models with variables that include prior probability
distributions.

Reno models are created by defining the equations for the various stocks, flows,
and variables, and can then be simulated over time similar to something like
[Insight Maker](https://insightmaker.com/), examples of which can be seen below
and in the `notebooks` folder.

Currently, models only support discrete timesteps (technically implementing
difference equations rather than true differential equations.)

## Installation

Install from PyPI via:
```
pip install reno-sd
```

## Example

A classic system dynamics example is the predator-prey population model,
described by the [Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations).

Implementing these in Reno would look something like:

```python
import reno

m = reno.Model(name="m", steps=200, doc="Classic predator-prey interaction model example")

# make stocks to monitor the predator/prey populations over time
m.rabbits = reno.Stock(init=100.0)
m.foxes = reno.Stock(init=100.0)

# free variables that can quickly be changed to influence equilibrium
m.rabbit_growth_rate = reno.Variable(.1, doc="Alpha")
m.rabbit_death_rate = reno.Variable(.001, doc="Beta")
m.fox_death_rate = reno.Variable(.1, doc="Gamma")
m.fox_growth_rate = reno.Variable(.001, doc="Delta")

# flows that define how the stocks are influenced
m.rabbit_births = reno.Flow(m.rabbit_growth_rate * m.rabbits)
m.rabbit_deaths = reno.Flow(m.rabbit_death_rate * m.rabbits * m.foxes, max=m.rabbits)
m.fox_deaths = reno.Flow(m.fox_death_rate * m.foxes, max=m.foxes)
m.fox_births = reno.Flow(m.fox_growth_rate * m.rabbits * m.foxes)

# hook up inflows/outflows for stocks
m.rabbits += m.rabbit_births
m.rabbits -= m.rabbit_deaths

m.foxes += m.fox_births
m.foxes -= m.fox_deaths
```

The stock and flow diagram for this model (obtainable via `m.graph()`) looks
like this (green boxes are variables, white boxes are stocks, the labels between
arrows are the flows):

![stock_and_flow_diagram](./images/predator_prey_model.png)

Once a model is defined, it can be called like a function, optionally specifying
any free variables/initial values (any of which otherwise use the default defined
in the model above.), you can print the output of `m.get_docs()` to see a
docstring showing what this should look like:

```python
>>> print(m.get_docs())
Classic predator-prey interaction model example

Example:
	m(rabbit_growth_rate=0.1, rabbit_death_rate=0.001, fox_death_rate=0.1, fox_growth_rate=0.001, rabbits_0=100.0, foxes_0=100.0)

Args:
	rabbit_growth_rate: Alpha
	rabbit_death_rate: Beta
	fox_death_rate: Gamma
	fox_growth_rate: Delta
	rabbits_0
	foxes_0
```

To run and plot the population stocks:

```python
m(fox_growth_rate=.002, rabbit_death_rate=.002, rabbits_0=120.0)
reno.plot_refs([(m.rabbits, m.foxes)])
```

![basic_run](./images/predator_prey_basic_run.png)

To use Bayesian inference, we define a few metrics that can be observed (can
have defined likelihoods), for instance, maybe we want to find out what the
rabbit population growth rate would need to be for the fox population to
oscillate somewhere between 20-120. Transpiling into PyMC and running is similar
to the normal call, but with ``.pymc()``:

```python
m.minimum_foxes = reno.PostMeasurement(reno.series_min(m.foxes))
m.maximum_foxes = reno.PostMeasurement(reno.series_max(m.foxes))

trace = m.pymc(
    n=1000,
    fox_growth_rate=reno.Normal(.001, .0001),  # specify some variables as distributions to sample from
    rabbit_growth_rate=reno.Normal(.1, .01),   # specify some variables as distributions to sample from
    observations=[
        reno.Observation(m.minimum_foxes, 5, [20]),  # likelihood normally distributed around 20 with SD of 5
        reno.Observation(m.maximum_foxes, 5, [120]), # likelihood normally distributed around 120 with SD of 5
    ]
)
```

To see the shift in prior versus posterior distributions, we can plot the random
variables and some of the relevant stocks using ``plot_trace_refs``:

```python
reno.plot_trace_refs(
    m,
    {"prior": trace.prior, "post": trace.posterior},
    ref_list=[m.minimum_foxes, m.maximum_foxes, m.fox_growth_rate, m.rabbit_growth_rate, m.foxes, m.rabbits],
    figsize=(8, 5),
)
```

![bayes_run](./images/predator_prey_bayes.png)

showing that the `rabbit_growth_rate` needs to be around `0.07` in order for
those observations to be met.

For a more in-depth introduction to reno, see the tub example in the `./notebooks` folder.
