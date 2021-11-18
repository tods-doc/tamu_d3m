.. _write_pipeline:

Write and Run a Pipeline
========================

Generally you would use an `AutoML system <https://datadrivendiscovery.org/home-2#data>`__
to find pipelines, but it is useful to know how to write a pipeline by yourself, too.
Moreover, once you have a pipeline you might want to explore how it works by running it.

Pipeline
--------

A pipeline is described as a `DAG <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`__
consisting of interconnected steps, where
steps can be primitives, or (nested) other (sub)pipelines.
A pipeline has data-flow semantics, which means that steps are not necessary executed
in the order they are listed, but a step can be executed when all its
inputs are available.
Some steps can even be executed in parallel. On
the other hand, each step can use only previously defined outputs from
steps coming before them in the list of steps.

.. note::

    The reference runtime in the core package runs pipeline steps in order
    in which they are listed in the pipeline.
    It expects the steps to be in the order with all step inputs always
    available from prior steps or pipeline inputs.

Pipelines have multiple representations. The core package supports
pipelines as in-memory objects, JSON, and YAML.
In JSON, the following is a sketch of a pipeline description:

.. code:: yaml

    {
      "id": <UUID of the pipeline>,
      # A URI representing a schema and version to which pipeline description conforms.
      "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
      # Digest is generally computed automatically when saving a pipeline.
      "digest": <digest of the pipeline>,
      "created": <timestamp when created, in ISO 8601>,
      "name": <human friendly name of the pipeline, if it exists>,
      "description": <human friendly description of the pipeline, if it exists>,
      # A list of inputs the pipeline takes.
      "inputs": [
        {
          "name": <human friendly name of the inputs>
        }
      ],
      # A list of outputs the pipeline produces.
      "outputs": [
        {
          "name": <human friendly name of the outputs>,
          "data": <data reference, probably of an output of a step>
        }
      ],
      "steps": [
        {
          "type": "PRIMITIVE",
          "primitive": {
            "id": <ID of the primitive used in this step>,
            "version": <version of the primitive used in this step>,
            "python_path": <Python path of this primitive>,
            "name": <human friendly name of this primitive>,
            "digest": <digest of this primitive>
          },
          # Arguments are inputs to the step as a whole which are then
          # passed as necessary to primitive's methods when called.
          "arguments": {
            "inputs": {
              # Type defines how data is passed. CONTAINER means that it is
              # passed as (container) value itself, which is the most common way.
              "type": "CONTAINER",
              "data": <data reference, probably of an output of a step or pipeline input>
            },
            # Despite misleading name, this is in fact the standard name of an input to the primitive,
            # e.g., representing a target column with known values for training.
            "outputs": {
              "type": "CONTAINER",
              "data": <data reference, probably of an output of a step or pipeline input>
            }
          },
          "outputs": [
            {
              # Output data is made available by this step from default "produce" method.
              "id": "produce"
            },
            {
              # Output data is made available by this step from an extra
              # produce method named "produce_score", too.
              "id": "produce_score"
            }
          ],
          # Hyper-parameters are initialization time parameters of the primitive.
          # Every primitive defines its own set of hyper-parameters it accepts,
          # the hyper-parameter class implementing the logic of a hyper-parameter,
          # and possible values the hyper-parameter can have.
          "hyperparams": {
            "column_to_operate_on": {
              # VALUE means a constant value. "data" is in this
              # case the value itself and not a reference.
              "type": "VALUE",
              # Value is converted to a JSON-compatible value by hyper-parameter class.
              # It also knows how to convert it back. For simple values, no conversion might happen.
              "data": 5
            }
          }
        },
        ... more steps ...
      ]
    }

.. note::

    This sketch is not valid JSON because it contains comments and placeholders.
    It makes no logical sense either.

Here we have shown just a subset of possible standard fields.
Moreover, we used only the ``CONTAINER`` data type, while there are also others data types.
Similarly, there are other step types, too.
To learn more :ref:`read the guide on advanced pipelines <advanced_pipelines>`
or consult the `pipeline JSON schema itself <https://metadata.datadrivendiscovery.org/devel/?pipeline>`__.

Pipeline describes how inputs are computed into outputs. For *standard pipelines*,
the input is a :class:`~d3m.container.dataset.Dataset` container value and
the output is a Pandas :class:`~d3m.container.pandas.DataFrame` container
value with predictions in :ref:`standard predictions
structure <pipeline_predictions>`.
The same pipeline is used for both fitting on train data and producing on test data.

.. note::

    Pipelines are defined very generally: number and meaning of pipeline inputs and outputs
    can be arbitrary, even the execution semantic of the pipeline can be redefined.
    For our purposes we focus on *standard pipelines*, for which we use execution semantics
    of the reference runtime.

Primitive steps describe how to run a primitive for that step and map step inputs to primitive
arguments and hyper-parameters and step outputs to primitive produce methods.
Primitives allow reuse and compositionality of logic, but the downside is that then all logic
has to be in primitives and those are :ref:`slightly tedious to write <write_primitive>`.
Using primitives helps with reproducibility but brings overhead for adding new logic into a pipeline,
if this logic does not already exists as a primitive.

Each primitive has a set of arguments it takes as a whole,
combining all the arguments from all its methods. Each argument
(identified by its name) can have only one value associated with it and
any method accepting that argument receives that value. Once all values
for all arguments for a method are available, that method can be called.

Each primitive can have multiple *produce* methods.
These methods can be called after a primitive has been fitted.
In this way a primitive can have multiple outputs, for each *produce* method one.

.. note::

    There are also :ref:`other step types possible <advanced_pipelines>`, e.g., sub-pipelines and placeholders.

Hyper-parameters
~~~~~~~~~~~~~~~~

Hyper-parameters are initialization time parameters of the primitive.
All hyper-parameters from all primitives together form hyper-parameters of the pipeline.

Some hyper-parameters can be provided when pipeline is run and can be different between different runs of the same pipeline,
but for some hyper-parameters that makes no sense because changing them would also change the logic of the pipeline.
We call the latter *control hyper-parameters* while the former are generally *tunable hyper-parameters*.
Control hyper-parameters should generally be fixed as part of the pipeline definition, leaving other hyper-parameters
to be potentially provided when pipeline is run (otherwise default values are used for them).
``column_to_operate_on`` is an example of a control hyper-parameter in the pipeline sketch above.

There are also other types of hyper-parameters (e.g., to control resource usage) and values to hyper-parameters
can be passed as different data types, too.
Moreover, every hyper-parameter belongs to a hyper-parameter class implementing its logic.
Read :ref:`hyper-parameters guide <hyperparameters>` for details.

Data References
~~~~~~~~~~~~~~~

Pipeline descriptions contains *data references*. A data reference is
just a string which identifies an output of a prior step or a pipeline input.
A data reference describes a data-flow connection between data available and an input to
a step. It is recommended to be a string of the following forms:

-  ``steps.<number>.<id>`` — ``number`` identifies the step in the list
   of steps (0-based) and ``id`` identifies the name of a produce method
   of the primitive
-  ``inputs.<number>`` — ``number`` identifies the pipeline input
   (0-based)
-  ``outputs.<number>`` — ``number`` identifies the pipeline output
   (0-based)

.. _pipeline_description_example:

Pipeline Description Example
----------------------------

.. note::

    The example assumes :ref:`core package and basic primitives installed <installation>`.

The following example uses :class:`~d3m.metadata.pipeline.Pipeline` class
to make an in-memory pipeline. This
specific example creates a pipeline for classification.

.. code:: python

    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep

    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 1: dataset_to_dataframe
    # An input to a standard pipeline is a Dataset. Here we assume it contains
    # only one resource and that it is a DataFrame we extract out.
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 2: column_parser
    # All columns in DataFrames inside a Dataset are loaded as string columns. This is to
    # assure that primitives control how columns are parsed and not logic outside of a pipeline.
    # So we parse columns now, based on their types available in metadata.
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 3: extract_columns_by_semantic_types(attributes)
    # Metadata contains also semantic types which can represent column roles.
    # Here we extract only attribute columns.
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(
        name='semantic_types',
        argument_type=ArgumentType.VALUE,
        data=['https://metadata.datadrivendiscovery.org/types/Attribute'],
    )
    pipeline_description.add_step(step_2)

    # Step 4: extract_columns_by_semantic_types(targets)
    # Here we extract only target columns.
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(
        name='semantic_types',
        argument_type=ArgumentType.VALUE,
        data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
    )
    pipeline_description.add_step(step_3)

    # Step 5: imputer
    # We impute attribute columns.
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.2.produce')
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # Step 6: random_forest
    # And train a random forest on attribute and target columns.
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.regression.random_forest.SKlearn'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.4.produce')
    step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data='steps.3.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Step 7: construct_predictions
    # This is a primitive which assures that the output of a standard pipeline has predictions
    # in the correct structure (e.g., there is also a d3mIndex column with index for every row).
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.5.produce')
    # This is a primitive which uses a non-standard second argument, named "reference".
    step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)

    # Final output
    pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

    # Output to YAML
    print(pipeline_description.to_yaml())

As you can see, building a pipeline by hand is pretty tedious and requires one to use correct data references.
Ideally, you would be using other tools (e.g., an AutoML system) to build a pipeline for you. Those tools can do use this API internally.

Values passed around in a D3M pipeline contain also :ref:`metadata <metadata>` and that part of that
metadata are also :ref:`semantic types <semantic_type>` which can provide information about columns like their role.
A :ref:`later guide <metadata>` explains this in more detail.

.. note::

    Some primitives support determining on which columns to operate automatically based on semantic types.
    This includes all sklearn-wrap primitives, too, so the example pipeline above could be simplified to
    not explicitly extract columns by roles.

YAML representation of this pipeline looks like:

.. code:: yaml

    created: '2021-02-25T21:04:43.399478Z'
    digest: d9a06fbd2ba3f7771e703a6a3e455379c692cc4291904d44f86db07f3a5210f2
    id: e70b61e2-6fcd-470b-becc-d3eba7041ab8
    inputs:
    - name: inputs
    outputs:
    - data: steps.6.produce
      name: output predictions
    schema: https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json
    steps:
    - arguments:
        inputs:
          data: inputs.0
          type: CONTAINER
      outputs:
      - id: produce
      primitive:
        digest: aed657e5effa3e313bd0e59c7334100aa8552fc5aba762a959ce4569284a5e63
        id: 4b42ce1e-9b98-4a25-b68e-fad13311eb65
        name: Extract a DataFrame from a Dataset
        python_path: d3m.primitives.data_transformation.dataset_to_dataframe.Common
        version: 0.3.0
      type: PRIMITIVE
    - arguments:
        inputs:
          data: steps.0.produce
          type: CONTAINER
      outputs:
      - id: produce
      primitive:
        digest: 6f73dc863e2cfcbed90757ab26c34ca8df23e24f9a26632f48dc228f2277dc7b
        id: d510cb7a-1782-4f51-b44c-58f0236e47c7
        name: Parses strings into their types
        python_path: d3m.primitives.data_transformation.column_parser.Common
        version: 0.6.0
      type: PRIMITIVE
    - arguments:
        inputs:
          data: steps.1.produce
          type: CONTAINER
      hyperparams:
        semantic_types:
          data:
          - https://metadata.datadrivendiscovery.org/types/Attribute
          type: VALUE
      outputs:
      - id: produce
      primitive:
        digest: 88f0780f5324d4a881d5d51e29f33fdcdc6d2968acf3b927032cf2d832e10504
        id: 4503a4c6-42f7-45a1-a1d4-ed69699cf5e1
        name: Extracts columns by semantic type
        python_path: d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common
        version: 0.4.0
      type: PRIMITIVE
    - arguments:
        inputs:
          data: steps.0.produce
          type: CONTAINER
      hyperparams:
        semantic_types:
          data:
          - https://metadata.datadrivendiscovery.org/types/TrueTarget
          type: VALUE
      outputs:
      - id: produce
      primitive:
        digest: 88f0780f5324d4a881d5d51e29f33fdcdc6d2968acf3b927032cf2d832e10504
        id: 4503a4c6-42f7-45a1-a1d4-ed69699cf5e1
        name: Extracts columns by semantic type
        python_path: d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common
        version: 0.4.0
      type: PRIMITIVE
    - arguments:
        inputs:
          data: steps.2.produce
          type: CONTAINER
      outputs:
      - id: produce
      primitive:
        digest: 84bf94c87a745011023da7074c65e1cee1272843d5a11cce1c64c7f20d42e408
        id: d016df89-de62-3c53-87ed-c06bb6a23cde
        name: sklearn.impute.SimpleImputer
        python_path: d3m.primitives.data_cleaning.imputer.SKlearn
        version: 2020.12.1
      type: PRIMITIVE
    - arguments:
        inputs:
          data: steps.4.produce
          type: CONTAINER
        outputs:
          data: steps.3.produce
          type: CONTAINER
      outputs:
      - id: produce
      primitive:
        digest: 79111615e8d956499bd1c2a8ee16575379da4e666861979ef293ba408f417549
        id: f0fd7a62-09b5-3abc-93bb-f5f999f7cc80
        name: sklearn.ensemble.forest.RandomForestRegressor
        python_path: d3m.primitives.regression.random_forest.SKlearn
        version: 2020.12.1
      type: PRIMITIVE
    - arguments:
        inputs:
          data: steps.5.produce
          type: CONTAINER
        reference:
          data: steps.0.produce
          type: CONTAINER
      outputs:
      - id: produce
      primitive:
        digest: 7ecceddd6bf78f4a8b0719f1aff46fe2e549c0b4b096be035513a92bdb6510de
        id: 8d38b340-f83f-4877-baaa-162f8e551736
        name: Construct pipeline predictions output
        python_path: d3m.primitives.data_transformation.construct_predictions.Common
        version: 0.3.0
      type: PRIMITIVE

The core package populated more information about primitives used and computed digests.
Because pipeline ID was not provided, it was auto-generated, too.
If you prefer, you can write pipelines in YAML or JSON directly, too.

.. note::

    Digest values will be different for you if you run the code above because you will probably have a different
    version of primitives installed and at least a different pipeline's ``created`` timestamp.

.. _reference_runtime:

Reference Runtime
-----------------

:mod:`d3m.runtime` module contains a reference runtime for pipelines. There
is also an extensive :ref:`command line interface (CLI) <cli>` you can access
through ``python3 -m d3m runtime``.

The reference runtime runs the pipeline twice, in two phases, first fitting the
pipeline and then producing.
During fitting each primitive is first fitted and then
produced on train data, in in steps order.
During producing, each primitive is produced on test data.
Before each phase, the reference runtime sets target column role :ref:`semantic type <semantic_type>`
on target column(s) based on the provided problem description.
This is the way :ref:`how information from the problem description is passed to the pipeline and primitives <interaction_with_problem>`.

.. note::

    We choose to use term *producing* and not *predicting* because
    *producing* encompass both *predicting* and *transforming*.

:mod:`d3m.runtime` module exposes both a low-level :class:`~d3m.runtime.Runtime`
class and high-level functions like :func:`~d3m.runtime.fit` and :func:`~d3m.runtime.produce`.
We can use those high-level functions with the pipeline we made above and :ref:`example dataset <get_dataset>`:

.. code:: python

    import sys

    from d3m import runtime
    from d3m.container import dataset
    from d3m.metadata import base as metadata_base, pipeline, problem

    # Loading problem description.
    problem_description = problem.get_problem('datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_problem/problemDoc.json')

    # Loading train and test datasets.
    train_dataset = dataset.get_dataset('datasets/training_datasets/seed_datasets_archive/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json')
    test_dataset = dataset.get_dataset('datasets/training_datasets/seed_datasets_archive/185_baseball/TEST/dataset_TEST/datasetDoc.json')

    # Loading pipeline description from the YAML representation.
    # We could also just use the in-memory object we made above.
    pipeline_description = pipeline.get_pipeline('pipeline.yaml')

    # Fitting pipeline on train dataset.
    fitted_pipeline, train_predictions, fit_result = runtime.fit(
        pipeline_description,
        [train_dataset],
        problem_description=problem_description,
        context=metadata_base.Context.TESTING,
    )
    # Any errors from running the pipeline are captured and stored in
    # the result objects (together with any values produced until then and
    # pipeline run information). Here we just want to know if it succeed.
    fit_result.check_success()

    # Producing predictions using the fitted pipeline on test dataset.
    test_predictions, produce_result = runtime.produce(
        fitted_pipeline,
        [test_dataset],
    )
    produce_result.check_success()

    test_predictions.to_csv(sys.stdout)

To do the same using CLI, you can run::

    $ python3 -m d3m runtime fit-produce \
      --pipeline pipeline.yaml \
      --problem datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_problem/problemDoc.json \
      --input datasets/training_datasets/seed_datasets_archive/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json \
      --test-input datasets/training_datasets/seed_datasets_archive/185_baseball/TEST/dataset_TEST/datasetDoc.json \
      --output predictions.csv \
      --output-run pipeline_run.yaml

For more information about the usage see :ref:`CLI guide <cli>` or run::

    $ python3 -m d3m runtime --help

:mod:`d3m.runtime` module provides also other high-level functions which can help
with :ref:`data preparation (splitting) and scoring <data_preparation>` for evaluating pipelines.
To better understand how error handling is done in the reference runtime and how you can
debug your primitives and pipelines, read :ref:`this HOWTO <debug>`.

``fit_result`` and ``produce_result`` objects above (of :class:`~d3m.runtime.Result` class) contain values which
were asked to be retained and *exposed* during pipeline's execution
(by default only the pipeline's outputs are retained).
You can control which values are *exposed* by using
``expose_produced_outputs`` and ``outputs_to_expose`` arguments of the
high-level functions, or ``--expose-produced-outputs`` CLI argument.

Those objects also contain *pipeline run* information.
We saved it to a file in the CLI call with ``--output-run`` argument, too.
To learn more about *pipeline run* information, read the next section.

.. _pipeline_run:

Pipeline Run
------------

:class:`~d3m.metadata.pipeline_run.PipelineRun` class represents the *pipeline run*. The pipeline run
contains information about many aspects of the pipeline's execution and enables :ref:`metalearning <metalearning>`
and :ref:`reproducibility <reproduce_run>` to duplicate the execution at a later time.

All information for the pipeline run is automatically collected during pipeline's execution.
Moreover, it references also used pipeline, problem description, and input dataset. It contains
hyper-parameter values provided when pipeline was run and information about the environment inside
which the pipeline was run.

In the example above, we saved it to a file using the ``--output-run`` argument. Pipeline runs
we represent in YAML because they can contain multiple documents, one for each execution phase.
The following is a sketch of the pipeline run representation:

.. code:: yaml

    context: <the context for this phase (TESTING for example)>
    datasets:
    - digest: <digest of the input dataset>
      id: <ID of the input dataset>
    end: <timestamp of when this phase ended>
    environment: <details about the machine the phase was performed on>
    id: <UUID of this pipeline run document>
    pipeline:
      digest: <digest of the pipeline>
      id: <ID of the pipeline>
    problem:
      digest: <digest of the problem>
      id: <ID of the problem>
    random_seed: <random seed value, 0 by default>
    run:
      is_standard_pipeline: true
      phase: FIT
      results: <predictions of the fit phase>
    schema: https://metadata.datadrivendiscovery.org/schemas/v0/pipeline_run.json
    start: <timestamp of when this phase started>
    status:
      state: <whether this stage completed successfully or not>
    steps: <details of running each step: hyper-parameters, timestamps, success, etc.>
    --- <this indicates a divider between documents in YAML>
    ... documents for other phases ...

A pipeline run can also contain information about :ref:`data preparation and scoring <data_preparation>` which
might have been done before and after the pipeline was run. In the case of scoring, the pipeline run contains
also scores computed.

Having a pipeline run allows you to :ref:`rerun <reproduce_run>` the pipeline at a later time, reproducing its results.
Being able to rerun a pipeline is a critical step towards :ref:`metalearning <metalearning>`.

Now that we know how to write a pipeline and run it, we might want to add
some custom logic to the pipeline. For that, we have to :ref:`write our own primitive <write_primitive>`.
