.. _installation:

Core Package Installation
-------------------------

This package works with Python 3.6 and pip 19+. On Debian/Ubuntu you should
have the following packages installed on the system:

* ``libyaml-dev``

You can install latest stable version from `PyPI <https://pypi.org/>`__::

    $ pip3 install d3m

To install latest development version::

    $ pip3 install -e git+https://gitlab.com/datadrivendiscovery/d3m.git@devel#egg=d3m

When cloning a repository, clone it recursively to get also git
submodules::

    $ git clone --recursive https://gitlab.com/datadrivendiscovery/d3m.git

Primitives Installation
-----------------------

Installing only the core package is not very useful on its own.
To get started, you should also install `common primitives <https://gitlab.com/datadrivendiscovery/common-primitives>`__
which include many glue primitives::

    $ apt-get install build-essential libopenblas-dev libcap-dev ffmpeg
    $ pip3 install python-prctl
    $ pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@master#egg=common-primitives

To use `sklearn estimators <https://scikit-learn.org/>`__ they have to be wrapped into primitives
(e.g., to document hyper-parameters in a machine-readable way).
Some of them have already been wrapped and are available as `sklearn-wrap primitives <https://gitlab.com/datadrivendiscovery/sklearn-wrap>`__::

    $ pip3 install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@dist#egg=sklearn_wrap

The index of all published primitives is
available `here <https://gitlab.com/datadrivendiscovery/primitives>`__.
You can also use `MARVIN <https://marvin.datadrivendiscovery.org>`__ to explore primitives.

.. note::

    For more information how to install primitives consult :ref:`this HOWTO <primitives_installation>`.

.. note::

    After primitives are installed through a Python package they are automatically available
    to be used in Python (and pipelines). See :ref:`HOWTO <register_primitive>` for details how this works.

Now that you have the core package and basic primitives installed, you can try to :ref:`write and run your own
pipeline <write_pipeline>`.
