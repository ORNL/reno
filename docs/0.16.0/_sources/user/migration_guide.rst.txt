Migration Guide
###############

This page includes details on how to fix breaking changes from updating
``reno`` versions.

0.14.0 - 0.15.0
===============

Flip :py:class:`reno.ops.Observation` arguments
-----------------------------------------------

Previously an ``Observation`` expected ``(reference, standard_devation, mean)``
parameters. It can now handle any distribution (by passing a class to the
``dist`` argument) and the ordering of the parameters is based on the
distribution's ordering. The default is still :py:class:`reno.ops.Normal`, but
since the normal distribution takes ``(mean, standard_deviation)``, the default
Observation parameter ordering is: ``(reference, mean, standard_deviation)``.


Don't rely on a component's ``.value``
--------------------------------------

Previously, the ``.value`` attribute on any component after a simulation would
hold the entirety of that component's simulation data (all samples and
timesteps.) This attribute is now used internally to hold only a single sample's
data at a time.

Instead, use the XArray dataset that is returned from a model simulation run to
access full data for a component. (This was always possible, but is now the only
option.)

Previously:

.. code-block:: python

    import reno as r

    m = r.Model()
    with m:
        ...
        stock = r.Stock()
        ...

    m(n=100, ...)
    stock_data = stock.value

Change to:

.. code-block:: python

    import reno as r

    m = r.Model()
    with m:
        ...
        stock = r.Stock()
        ...

    ds = m(n=100, ...)
    stock_data = ds.stock.values

(This change comes in part because running multiple samples are now down in series, a
new set of numpy arrays are created per sample rather than creating a matrix of
all of them at once.)
