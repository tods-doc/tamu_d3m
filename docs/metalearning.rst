.. _metalearning:

Metalearning Database
---------------------

D3M provides a metalearning database to support research to improve AutoML system capabilities.
This database contains metadata about datasets, :ref:`primitives <overview_of_primitives_and_pipelines>`, :ref:`pipelines <overview_of_primitives_and_pipelines>`, and the results of executing pipelines on datasets (:ref:`pipeline runs <pipeline_run>`).

The metalearning database is powered by `ElasticSearch <https://www.elastic.co/elasticsearch>`__.
All the data is :ref:`publicly available <metalearning_database_downloading_data>`.
The data has been generated primarily during formal D3M system evaluations and by D3M participants, but anyone can :ref:`contribute <metalearning_database_uploading_data>`.
It can be explored using the `Marvin dashboard <https://marvin.datadrivendiscovery.org>`__ or any `ElasticSearch client <https://www.elastic.co/guide/en/elasticsearch/client/index.html>`__.

The metalearning database endpoint is hosted at

.. code-block::

    https://metalearning.datadrivendiscovery.org/es

.. _metalearning_database_structure:

Database Structure
~~~~~~~~~~~~~~~~~~

The Metalearning Database holds five different `normalized <https://en.wikipedia.org/wiki/Database_normalization>`__ documents in separate `Elasticsearch indexes <https://www.elastic.co/guide/en/elasticsearch/reference/current/glossary.html#glossary-index>`__, including
:ref:`datasets <metalearning_database_datasets>`,
:ref:`problems <metalearning_database_problems>`,
:ref:`primitives <metalearning_database_primitives>`,
:ref:`pipelines <metalearning_database_pipelines>`, and
:ref:`pipeline runs <metalearning_database_pipeline_runs>`.
These documents contain only metadata, for example, a natural language description of a dataset or the dataset's source URI.
They do not contain, for example, the actual instances of data in a dataset.
Each of these documents conform to their respective `metadata schemas <https://metadata.datadrivendiscovery.org/devel/>`__.

..
    TODO: should the schema links point to metadata.datadrivendiscovery.org or to internal docs?

.. _metalearning_database_datasets:

**Datasets**

Dataset metadata is stored in the :code:`datasets` index.
These documents contain, for example, a natural language description of the dataset and dataset's source URI.
See the `dataset schema <https://metadata.datadrivendiscovery.org/devel/?container>`__ for a complete description.
Actual datasets curated by D3M can be found in the `datasets repository <https://datasets.datadrivendiscovery.org/d3m/datasets>`__.

.. _metalearning_database_problems:

**Problems**

Problem metatdata is stored in the :code:`problems` index.
A problem describes a type of machine learning task, including referencing to a dataset, identifying the target column(s), giving performance metrics to optimize, and listing task keywords (e.g. classification, image, remote sensing, etc.).
See the `problem schema <https://metadata.datadrivendiscovery.org/devel/?problem>`__ for a complete description.
Note that many problems can reference the same dataset, for example, by identifying different columns as the target.

.. _metalearning_database_primitives:

**Primitives**

Primitive metatdata is stored in the :code:`primitives` index.
A primitive is a high-level machine learning algorithm.
The primitive metadata describes what kind of algorithm the primitive is, what hyperparameters and methods it has, how to install it, and who authored it.
See the :ref:`primitives documentation <overview_of_primitives_and_pipelines>` or the
`primitives schema <https://metadata.datadrivendiscovery.org/devel/?primitive>`__ for more details.
An index of the latest versions of these documents can be found in the `primitives repository <https://gitlab.com/datadrivendiscovery/primitives>`__.
See also the `source code of the common D3M primitives <https://gitlab.com/datadrivendiscovery/common-primitives>`__. Primitives can also be
browsed and filtered by author, algorithm type, primitive family and many other attributes in the `Marvin dashboard <https://marvin.datadrivendiscovery.org/primitives>`__.

.. _metalearning_database_pipelines:

**Pipelines**

Pipeline metatdata is stored in the :code:`pipelines` index.
Pipelines describe precisely which primitives are used (by referencing primitive documents) and how they are composed together to build an end-to-end machine learning model.
D3M provides a :ref:`reference runtime <reference_runtime>` for executing pipelines on a given dataset and problem.
The execution of pipelines on a dataset is carefully recorded into a :ref:`pipeline run <metalearning_database_pipeline_runs>` document.
For more details, see the :ref:`pipeline overview <overview_of_primitives_and_pipelines>` or the
`pipeline schema <https://metadata.datadrivendiscovery.org/devel/?pipeline>`__.
For help on building a pipeline, see :ref:`an example <pipeline_description_example>`.


.. _metalearning_database_pipeline_runs:

**Pipeline Runs**

Pipeline Run metatdata is stored in the :code:`pipeline_runs` index.
Pipeline run documents contain an execution trace of running a particular pipeline on a particular dataset and problem.
In addition to references to the pipeline, dataset, and problem, this document contains information about how the dataset may have been split for evaluation, performance metrics (e.g. accuracy), predictions, primitive hyperparameters, execution start and end timestamps, primitive methods called, random seeds, logging output, execution environment (e.g. CPUs and RAM available), and much more.
See the :ref:`pipeline run documentation <pipeline_run>` or the
`pipeline run schema <https://metadata.datadrivendiscovery.org/devel/?pipeline_run>`__ for more details.

Pipeline runs contain all information necessary to reproduce the execution of the referenced pipeline on the referenced dataset and problem.
Reproducing a pipeline run requires that the user has access to the same dataset, primitive, and runtime versions.
The reference runtime provides basic functionality for reproducing pipeline runs.

**Other Indexes (Beta)**

Other indexes are being designed and populated to simplify usage of the metalearning database.
The simplifications include removing large fields (e.g. especially predictions) and denormalizing references to other documents.

.. _metalearning_database_downloading_data:

Downloading Data
~~~~~~~~~~~~~~~~

The data in the metalearning database is publicly available.
This data can be downloaded from the endpoint

.. code-block::

    https://metalearning.datadrivendiscovery.org/es

For downloading small amounts of data, use any `ElasticSearch client <https://www.elastic.co/guide/en/elasticsearch/client/index.html>`__.
For bulk downloads, see the available `pre-made dumps <https://metalearning.datadrivendiscovery.org/es/dumps>`__.

Custom bulk downloads can be made using an Elasticsearch client such as `elasticsearch-dump <https://github.com/taskrabbit/elasticsearch-dump>`__.
**Warning: the metalearning database is large and growing.**
**Custom bulk downloads make take a long time to run.**
**It is highly recommended that you refine your dump query as much as possible.**

The following is example usage of elasticsearch-dump and requires `node package manager (npm) <https://www.npmjs.com/get-npm>`__.
(Note that starting with elasticsearch-dump 6.32.0, nodejs 10.0.0 or higher is required.)

..
    TODO: verify the elasticdump and nodejs versions

Install elasticsearch-dump

.. code-block:: bash

    npm install elasticdump

Dump all documents within a specific document ingest timestamp range, e.g pipeline runs ingested in January 2020

.. code-block:: bash

    npx elasticdump \
        --input=https://metalearning.datadrivendiscovery.org/es \
        --input-index=pipeline_runs \
        --output=pipeline_runs.json \
        --sourceOnly \
        --searchBody='{ "query": {"range": {"_ingest_timestamp": {"gte": "2020-01-01T00:00:00Z", "lt": "2020-02-01T00:00:00Z"}}}, "_source": {"exclude": ["run.results.predictions", "steps.*.method_calls"]}}'

Pipeline run documents can be very large, especially due to the predictions and method calls fields.
The above example shows how to exclude those fields.
In general, a dump may be made using any `ElasticSearch query <https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html>`__.

..
    TODO: Generating Data section

.. _metalearning_database_uploading_data:

Uploading Data
~~~~~~~~~~~~~~

Uploading new :ref:`documents <metalearning_database_structure>` to the database can be done using the `HTTP API <https://metalearning.datadrivendiscovery.org/1.0/doc>`__.
(In the future, the reference runtime will be able to automatically upload documents for you.)

**Important:**
Requests to upload documents are validated before the documents are ingested.
This validation includes checking that referenced documents have already been uploaded to the database.
Thus, before uploading a new pipeline run document, for example, the referenced dataset, problem, and pipeline and primitive documents must already be uploaded.

**Submitter Tokens**

Optionally, you may request a submitter name and token.
This allows other users of the metalearning database to find documents submitted by a particular person or organization.
The submitter name is publicly available and shows who authored and submitted the document.
The submitter token is your password to authenticate your identity and should be kept private.

..
    TODO: how to request a token

.. _metalearning_database_issues:

Reporting Issues
~~~~~~~~~~~~~~~~

To report issues with the Metalearning Database or coordinate development work, visit the `GitLab repository <https://gitlab.com/datadrivendiscovery/metalearning>`__.
The source code for the HTTP API and document validation is available there too.
