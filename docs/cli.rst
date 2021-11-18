.. _cli:

Command Line Interface (CLI)
============================

TODO: Describe major CLI sections.

TODO: Describe use of runtime CLI, how different commands works together to fit, produce, score, evaluate. How parameters are passed.

TODO: Describe top-level CLI arguments shared between all commands. How to use static files or pipeline search path.

Primitives Discovery
--------------------

The :mod:`d3m.index` module also provides a command line interface by
running ``python3 -m d3m primitive``. The following commands are currently
available.

Use ``-h`` or ``--help`` argument to obtain more information about each
command and its arguments.

``python3 -m d3m primitive search``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Searches locally available primitives. Lists registered Python paths for
primitives installed on the system.

``python3 -m d3m primitive discover``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discovers primitives available on PyPi. Lists package names containing
D3M primitives on PyPi.

``python3 -m d3m primitive describe``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates a primitive annotation of a primitive from its metadata.
