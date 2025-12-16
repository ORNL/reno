Components
##########


Reno models are based primarily on `stocks and flows <https://en.wikipedia.org/wiki/Stock_and_flow>`__.
A model is created by defining all of the components and the corresponding
equations that make them up.

TODO: equations (sep page?) - include tips on what can do with equations e.g.
piecewise


Flows
=====

Flows are equations that define rates of change, representing how much material or
information move over time.


Flows are created with the :ref:`reno.Flow <reno.components.Flow>` class, and the equation can either be
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
with an equation rather than just a flow, an implicit flow defined by that
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


(Is this actually a circular reference?)

.. code-block:: python

    from reno import Stock, Flow


    my_flow = Flow()
    my_stock = Stock()

    my_stock += my_flow
    my_flow.eq = 10 - my_stock

In this example, ``my_stock`` is incremented by the value of ``my_flow``
in the current timestep `t`, while the value of ``my_flow`` for timestep ``t``
is 10 minus the value of ``my_stock`` in timestep ``t - 1``.


Variables
=========

A variable is any other equation or value that can be referenced in flow
equations (and other variable equations) and helps define the user-settable
model parameters.


TODO


Metrics
=======

TODO


Other arguments for components
==============================

(doc, min/max, init, dim, type)
