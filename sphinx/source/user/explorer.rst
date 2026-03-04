Model Explorer UI
#################

Reno comes with a `Panel <https://panel.holoviz.org>`_ web application for
experimenting with models and constructing mini narrative dashboards around
specific results.

.. figure:: ../_static/explorer_screenshot.png
   :align: center

Running
=======

Reno installs with a ``reno`` command, which runs the panel server (by default on port 5006)

The full set of CLI args can be found by running ``reno -h``:

.. code-block::

    usage: reno [-h] [--version] [--session-path SESSION_PATH] [--url-root-path ROOT_PATH] [--port PORT] [--address ADDRESS] [--liveness-check] [--websocket-origin WEBSOCKET_ORIGIN]

    options:
      -h, --help            show this help message and exit
      --version             Print out Reno library version.
      --session-path SESSION_PATH
                            Where to store and load saved explorer sessions and models from.
      --url-root-path ROOT_PATH
                            Root path the application is being served on when behind a reverse proxy.
      --port PORT           What port to run the server on.
      --address ADDRESS     What address to listen on for HTTP requests.
      --liveness-check      Flag to host a liveness endpoint at /liveness.
      --websocket-origin WEBSOCKET_ORIGIN
                            Host that can connect to the websocket, localhost by default.


The ``--session-path`` argument refers to a cache or save directory to use for
persisting sessions. (A session here refers to loading and experimenting with a
particular model, not a web session.) By default this is set to
``./work_sessions``. Inside the session path directory, any Reno model files
saved within the ``models/`` directory will populate new session buttons in the
upper left of the interface to allow quickly starting new sessions.

To populate this directory with Reno's packaged example models, one could run:

.. code-block::

    python -c "from reno.examples.lotka_volterra import predator_prey; predator_prey.save('work_sessions/models/predator_prey.json')"
    python -c "from reno.examples.one_compartment import one_compartment_model; one_compartment_model.save('work_sessions/models/one_compartment.json')"
    python -c "from reno.examples.tub import tub; tub.save('work_sessions/models/tub.json')"
    python -c "from reno.examples.urban_growth import urban_growth; urban_growth.save('work_sessions/models/urban_growth.json')"


Interface
=========


Building up a tab
=================


Saving/loading
==============
