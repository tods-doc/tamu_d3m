.. _load_datasets:

Load Datasets and Problems
==========================

.. _dataset:

Datasets
--------

This package also provides a Python class to load and represent datasets
in Python through the :mod:`d3m.container.dataset`
module. This container value can serve as an input to the whole pipeline
and be used as input for primitives which operate on a dataset as a
whole. It allows to register multiple loaders to support different
formats of datasets. You need to pass an URI to a dataset and it automatically
picks the right loader. By default it supports URIs for: D3M datasets, CSV files, OpenML
and Sklearn datasets.

-  D3M datasets. Only ``file://`` URI scheme is supported and URI should
   point to the ``datasetDoc.json`` file.
   Example: ``file:///path/to/datasetDoc.json``

-  CSV files. Many URI schemes are supported, including remote ones like
   ``http://``. URI should point to a file with ``.csv`` extension.
   Example: ``http://example.com/iris.csv``

-  OpenML datasets. You need to provide the URL of the dataset page.
   Example: ``https://www.openml.org/d/31``

-  Some Sklearn datasets from :mod:`sklearn.datasets`.
   Example: ``sklearn://boston``


To load a dataset, you just need to call the method ``load`` from the ``Dataset`` class
passing as a parameter the URI. Bellow, you can see how to load an OpenML dataset:

.. code:: python

    from d3m.container import Dataset

    dataset_uri = 'https://www.openml.org/d/62'
    dataset = Dataset.load(dataset_uri)


You can save the previously loaded dataset in D3M format using the ``save_container`` method.
You just need to provide the path where the dataset will be saved:

.. code:: python

    from d3m.container.utils import save_container

    destination_path = 'path_to_save_the_dataset'
    save_container(dataset, destination_path)


``load`` and ``save_container`` methods automatically convert and save non-D3M datasets
(e.g. CSV files) to D3M format. However, if you want to do this
process manually, `here <https://gitlab.com/datadrivendiscovery/data-supply/-/blob/how-to-guide/documentation/tabularHowtoGuide.md>`__
you can find more information.






TODO: How to write a dataset/problem loader.

TODO: Document OpenML Crawler.





.. _problem:

Problem Descriptions
--------------------

:mod:`d3m.metadata.problem` module provides
a parser for problem description into a normalized Python object.

You can load a problem description and get the loaded object dumped back
by running:

.. code:: bash

    python3 -m d3m problem describe <path to problemDoc.json>

