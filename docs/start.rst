Get a Docker Image
------------------

The easiest to start is by using one of provided `Docker images <https://gitlab.com/datadrivendiscovery/images>`__.
They contain the core package, all published primitives, and their
dependencies. Include CUDA, Tensorflow and other dependencies to
support using GPUs.

We will pull the latest Docker image available::

    $ docker pull registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-devel

Beware, Docker images are not small, but around 10 GB each.

.. note::

    See :ref:`this HOWTO <docker>` to learn more about Docker images available,
    how to build them and how to extend them.

.. note::

    While Docker images contain all primitives and their dependencies, some
    primitives require also additional static files provided to them at runtime
    (e.g., pretrained model weights). See :ref:`static files HOWTO <static_files>`
    for more information.

.. _get_dataset:

Get Datasets
------------

We provide many datasets in an uniform structure `in a git repository <https://datasets.datadrivendiscovery.org/d3m/datasets>`__
using `git LFS <https://git-lfs.github.com/>`__ to store large files.
You can clone the whole repository, but that can take time and disk space. So let's clone just one dataset::

    $ GIT_LFS_SKIP_SMUDGE=1 git clone --recursive git@datasets.datadrivendiscovery.org:d3m/datasets.git
    $ git -C datasets lfs pull -I training_datasets/seed_datasets_archive/185_baseball

Despite the dataset itself being small this will still take around 4 GB of your disk space.

Alongside datasets there are also problem descriptions available, one for each dataset.

.. note::

    Datasets are in D3M dataset format. Similarly problems. You can learn more about them in `this repository <https://gitlab.com/datadrivendiscovery/data-supply>`__.

.. note::

    You can use and load datasets and problems from other sources and in other formats, too.
    See :ref:`loading datasets and problems HOWTO <load_datasets>` for more information.

Get and Run a Pipeline
----------------------

We have a :ref:`metalearning database <metalearning>` containing millions of pipeline runs and associated documents.
You can use `MARVIN <https://marvin.datadrivendiscovery.org/>`__ to explore it, but for now we will just fetch one existing
pipeline from it::

    $ wget -O pipeline.json https://metalearning.datadrivendiscovery.org/es/pipelines_202002/_source/086fd163fa1d845b651478a25929c03e1fe85bf31c3f3367fecd4451acecd25e

There are also other ways to obtain a pipeline: you can use one of `compatible AutoML systems <https://datadrivendiscovery.org/home-2#data>`__
or you could :ref:`write a pipeline by hand <write_pipeline>`.

The core package provides a :ref:`command line interface (CLI) <cli>` to many parts of what it provides. One of them
is also a reference runtime to run pipelines. We can use core package installed inside the Docker image to :ref:`run the pipeline
on the dataset <reference_runtime>`. Furthermore, we can use one of :ref:`standard data preparation pipelines and the scoring pipeline <data_preparation>`
to evaluate the pipeline with 5 fold cross validation::

    $ docker run --rm -v "$(pwd):/data" registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-devel \
      python3 -m d3m runtime evaluate --pipeline /data/pipeline.json \
      --problem /data/datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_problem/problemDoc.json \
      --input /data/datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_dataset/datasetDoc.json \
      --data-pipeline /src/d3m/d3m/contrib/pipelines/c8ed65df-aa68-4ee0-bbb5-c5f76a40bcf8.yml \
      --data-param number_of_folds 5 --data-param shuffle true --data-param stratified true

You should see some logging output and at the end::

    metric,value,normalized,randomSeed,fold
    F1_MACRO,0.7289795918367347,0.7289795918367347,0,0
    F1_MACRO,0.6905913978494623,0.6905913978494623,0,1
    F1_MACRO,0.6130389433838437,0.6130389433838437,0,2
    F1_MACRO,0.5187109187109187,0.5187109187109187,0,3
    F1_MACRO,0.6154471544715447,0.6154471544715447,0,4

Metric to use (and the target column) is specified in the :ref:`problem description <problem>`.
Observe that every time you run this command you get exactly the same results.
D3M works hard to provide full reproducibility.

.. note::

    To really achieve full reproducibility we would have to instruct you to use a Docker image
    at a fixed version and a dataset at a fixed git commit hash. Read :ref:`more about reproducibility <reproduce_run>`.

Running this is interesting, but to develop using technologies available, you should first install
the :ref:`core package and basic primitives locally <installation>`.
