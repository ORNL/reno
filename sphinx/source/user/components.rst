Components
##########


Reno models are based primarily on `stocks and flows
<https://en.wikipedia.org/wiki/Stock_and_flow>`__. A model is created by
defining all of these components and the corresponding equations that make them
up.

The equations themselves and how to construct them are discussed in more depth
on the :ref:`math in reno` page, while this page primarily focuses on the higher
level Flow/Stock/Variable components.


Flows
=====

Flows are equations that define rates of change, or represent how much
material/information moves over time.

Flows are created with the :py:class:`reno.Flow <reno.components.Flow>` class, and the equation can either be
directly provided in the constructor or by setting the ``.eq`` attribute later
on:

.. code-block:: python

    from reno import Flow, TimeRef
    import reno

    # create a flow with an equation of "5"
    faucet = Flow(5)  # 5 units per timestep

    # change the equation to vary sinusoidally with time
    t = TimeRef()  # a TimeRef instance is a special type of variable that
                   # always refers to the current timestep in the simulation
    faucet.eq = reno.sin(t) * 2 + 5


Stocks
======

A stock represents an accumulation of material or information, or some quantity
thereof over time.

Stock equations are defined exclusively in terms of flows, in-flows (rates of
material moving *into* the stock) and out-flows (rates of material moving *out
of* the stock.)

Creating stocks in Reno are done via the ``Stock`` class:

.. code-block:: python

    from reno import Stock

    tub_water_level = Stock()


Defining stock equations
------------------------

Stock equations are defined by setting up in-flows and out-flows. The basic
syntax for doing this uses the ``+=`` operator for in-flows and ``-=`` operator for
outflows:

.. code-block:: python

    from reno import Stock, Flow

    my_inflow, my_outflow = Flow(), Flow()
    my_stock = Stock()

    my_stock += my_inflow
    my_stock -= my_outflow


A slightly more readable syntax that allows constructing whole "chains" of
in-flow/out-flows can be done with the ``>>`` and ``<<`` operators, where the
arrows indicate the direction of a flow in relation to the stock on the other
side:

.. code-block:: python

    from reno import Stock, Flow

    inflow, midflow, outflow = Flow(), Flow(), Flow()
    stock1, stock2 = Stock(), Stock()

    inflow >> stock1 >> midflow >> stock2 >> outflow

Specifically a ``stock >> flow`` or ``flow << stock`` makes ``flow`` an
**out**-flow of ``stock``, and ``stock << flow`` or ``flow >> stock`` makes
``flow`` an **in**-flow to ``stock``.

Chains of these ``>>``/``<<`` operations work because they are
interpreted left to right, and the "return" value of an individual operation
is always the right-most component, e.g. ``component2`` in ``component1 >>
component2``.

As a result,

.. code-block:: python

   inflow >> stock1 >> midflow

is equivalent to:

.. code-block:: python

   inflow >> stock1
   stock1 >> midflow


Implicit stock in-flows
-----------------------

When an in-flow to a stock is set (either through ``+=`` or ``>>``/``<<``)
with an equation rather than just a flow, an **implicit** flow defined by that
equation is created and applied.

(e.g. if there's some loss involved between the outflow of one stock and the
inflow for another, you could of course explicitly model this with two separate
flows as well)

.. code-block:: python

    from reno import Stock, Flow

    inflow, midflow, outflow = Flow(), Flow(), Flow()
    stock1, stock2 = Stock(), Stock()

    inflow >> stock1 >> midflow
    (midflow - 3) >> stock2 >> outflow

by combining operations together on the same line with commas, you can still do
a full chain-like definition when an inflow needs to be a slightly modified version:

.. code-block:: python

    from reno import Stock, Flow

    inflow, midflow, outflow = Flow(), Flow(), Flow()
    stock1, stock2 = Stock(), Stock()

    inflow >> stock1 >> midflow, (midflow - 3) >> stock2 >> outflow


Using stocks in other equations
-------------------------------

(This might need to have its own section at the end to discuss the difference
between circular references involving stocks and those between flows)

Referencing a stock always refers to the stock's value in the *previous*
timestep. This allows a form of circular reference between stocks

.. code-block:: python

    from reno import Stock, Flow


    my_flow = Flow()
    my_stock = Stock()

    my_stock += my_flow
    my_flow.eq = 10 - my_stock

In this example, ``my_stock`` is incremented by the value of ``my_flow``
in the current timestep `t`, while the value of ``my_flow`` for timestep ``t``
is 10 minus the value of ``my_stock`` in timestep ``t - 1``.

In other words, the equations for these would translate to:

* my_stock(t) = my_stock(t-1) + my_flow(t)
* my_flow(t) = 10 - my_stock(t-1)


Variables
=========

A variable is any other equation or value that can be referenced in flow (and
other variable) equations and helps define the user-settable model parameters.
Variables should be used to specify what can be modified about a
simulation/what values you want to experiment with.


.. code-block:: python

    coffee_process = reno.Model(steps=10)
    with coffee_process:
        drip_speed = reno.Variable(3.0)

        water = reno.Stock(init=100.0)
        coffee = reno.Stock()
        coffee_machine = reno.Flow(drip_speed, max=water)

        water >> coffee_machine >> coffee

In the above model, ``drip_speed`` is a variable that directly impacts the
``coffee_machine`` flow/the rate at which ``coffee`` increases. Since it is a
free variable/not defined in terms of any other variables, it can be specified
during a model run to configure the simulation. We can run a couple simulations
and compare the final coffee stock values (at timestep 10):

.. code-block:: python

    >>> coffee_process(drip_speed=4.0).coffee.values[0, -1]
    36.0

    >>> coffee_process(drip_speed=1.0).coffee.values[0, -1]
    9.0


Metrics
=======

TODO


Other arguments for components
==============================

(doc, min/max, init, dim, type)
