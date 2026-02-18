Visualizations
##############


Diagrams
========


As already shown in the Stock and flow diagrams ...


Groups/Color groups
-------------------

Every tracked reference optionally has both a :py:attr:`group
<reno.components.TrackedReference.group>` and :py:attr`cgroup
<reno.components.TrackedReference.cgroup>` attribute, which influences how they
appear in the diagrams. The ``group`` attribute is used to encourage graphviz to
visually tighten up/keep elements within the same group closer to each other.
This is primarily done by straightening and shortening any connections between
elements of a group where possible.

In this example, suppose a variable applies to two different flows:

.. code-block:: python

    import reno as r

    m = r.Model()
    with m:
        s1 = r.Stock()
        v1 = r.Variable()
        f1, f2 = r.Flow(v1), r.Flow(v1)
        f1 >> s1 >> f2

The stock/flow diagram looks like this:

.. figure:: ../_static/non_grouped_diagram.png
    :align: center

If we assign the same group name to the variable and the second flow, it
straightens out the connection between ``v1`` and ``v2``:

.. code-block:: python

    import reno as r

    m = r.Model()
    with m:
        s1 = r.Stock()
        v1 = r.Variable(group="test")
        f1, f2 = r.Flow(v1), r.Flow(v1, group="test")
        f1 >> s1 >> f2

.. figure:: ../_static/grouped_diagram.png
    :align: center

``cgroup`` is a "color group" attribute intended to make it easier to change
colors of specific sets of references in the diagram without influencing layout.

Either groups or color groups can be colored from a :py:func:`model.graph() <reno.model.Model.graph>` call with the ``group_colors`` attribute:

.. code-block:: python

    m.graph(group_colors={"test":"#4499AA"})


.. figure:: ../_static/colored_diagram.png
    :align: center


Settings can be defined on models to hide specific groups or set default
colors for designated groups, making them potentially easier to interpret
when given to someone else.

These settings can also be specified manually on a :py:func:`model.graph() <reno.model.Model.graph>` call with the ``hide_groups``, ``show_groups`` (to override a model's ``default_hide_groups`` setting), and ``group_colors``.

Get a list of the groups/cgroups on a model with the :py:attr:`groups
<reno.model.Model.groups>` property.

Universe
--------



Latex
=====


Reloading previous runs
-----------------------

(load_dataset)



Plots
=====


Multi-trace plots
-----------------


Single axis plots
-----------------
