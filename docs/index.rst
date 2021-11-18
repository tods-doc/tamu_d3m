D3M Developer Documentation
===========================

:Version: |version|

This is `Data Driven Discovery of Models (D3M)`_ developer documentation.
Its target audience is anyone who wants to build upon technologies developed
as part of the program or extend them. Primarily, developers and researchers
interested in AutoML. If you are not familiar with the program, read about
it on its `main page <https://datadrivendiscovery.org/>`__.

.. _Data Driven Discovery of Models (D3M): https://datadrivendiscovery.org/

Core Package
------------

The D3M core package provides the interface of primitives, data types of values which can be passed between them during execution,
the pipeline language, the metadata associated with values being passed between primitives, provides a reference runtime,
and contains a lot of other useful code to write primitives, generate pipelines, and run them. You are reading its documentation.

AutoML RPC Protocol
-------------------

D3M provides also a standard GRPC protocol for communicating with AutoML systems.
It is used as a standard interface to interact with any D3M-compatible AutoML system.
It is documented in `its own repository <https://gitlab.com/datadrivendiscovery/automl-rpc>`__.

Datasets
--------

D3M program `provides many datasets <https://datasets.datadrivendiscovery.org/d3m/datasets>`__
in an uniform structure.
The format of those datasets is `described in this repository <https://gitlab.com/datadrivendiscovery/data-supply>`__.

Metalearning Database
---------------------

Every pipeline which is run with the reference runtime produces a record of that run, called *pipeline run*.
Those pipeline runs (together with metadata about input datasets and problem description) are stored in centralized
and shared metalearning database, building towards a large metalearning dataset. Ideally, all those pipeline runs
are fully reproducible. :ref:`Documentation is here <metalearning>`.

Docker
------

D3M program has many moving pieces, many primitives, with many dependencies. Putting them all together to work correctly can be tricky.
This is why we provide Docker images with all primitives and dependencies installed, and configured to work both with or without GPUs.
Download a Docker image, datasets, and you are ready to go to run some pipelines. More about :ref:`Docker images here <docker>`.

Indices and Search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Getting Started

   start
   installation

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Guides

   write_pipeline
   write_primitive
   hyperparameters
   cli

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Advanced Topics

   metadata
   metalearning
   primitives
   pipelines
   pipeline_predictions
   data_preparation

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: HOWTOs

   load_datasets
   wrap_sklearn
   good_primitive
   reproduce_run
   static_files
   custom
   primitive_tests
   debug
   copy_primitive
   install_primitives
   register_primitive
   manual_run_primitive
   docker
   use_jupyter

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Contribute

   contribute

.. toctree::
   :hidden:
   :maxdepth: -1
   :caption: Reference

   reference
   Metadata Schemas <https://metadata.datadrivendiscovery.org/>
   D3M AutoML RPC <https://gitlab.com/datadrivendiscovery/automl-rpc/-/tree/dev-dist-python>
   D3M Dataset and Problem Formats <https://gitlab.com/datadrivendiscovery/data-supply/-/tree/shared/schemas>
   Primitives <https://gitlab.com/datadrivendiscovery/primitives>
   Datasets <https://datasets.datadrivendiscovery.org/d3m/datasets>
   MARVIN <https://marvin.datadrivendiscovery.org>
   Docker Images List <https://docs.datadrivendiscovery.org/docker.html>
